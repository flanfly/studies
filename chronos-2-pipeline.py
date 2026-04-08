import kfp
from kfp import dsl

from datetime import datetime
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = "prj-vertexai-test"
REGION = "asia-southeast1"
PIPELINE_ROOT = "gs://kai-vertex-ai-test-data/cr2"
SERVICE_ACCOUNT = "batch-job-sa@prj-vertexai-test.iam.gserviceaccount.com"
REPOSITORY = "flanfly/studies"


def latest_image_tag():
    from ghapi.all import GhApi

    ghpkg = GhApi(token=os.getenv("GITHUB_PAT")).packages

    for pkg in ghpkg.list_packages_for_authenticated_user(package_type="container"):
        if not pkg.repository or pkg.repository.full_name != REPOSITORY:
            continue

        vers = ghpkg.get_all_package_versions_for_package_owned_by_authenticated_user(
            package_type="container", package_name=pkg.name
        )
        for v in vers:
            tags = v.metadata.container.tags if "container" in v.metadata else []
            if "master" in tags:
                return f"@{v.name}"

    return "@sha256:96134c14d7324eb7b9b52e1abc14452f042eff94dbcbe3db66684892d73df20b"


IMAGE = f"asia-southeast1-docker.pkg.dev/prj-vertexai-test/default/studies{latest_image_tag()}"


@dsl.container_component
def synchronize_1m_archives(today: str, status: dsl.OutputPath(str)):
    """Makes sure 1-minute kline data is in sync with Binance."""

    return dsl.ContainerSpec(
        image=IMAGE,
        command=["bash", "-c"],
        args=[
            """
set -euo pipefail

export CLOUDSDK_CORE_PROJECT=$1
export GIT_KEY_SECRET=git-key

mkdir -p "$(dirname "$0")"
./run.sh

uv run sync-datastore.py | tee "$0"
            """,
            status,
            PROJECT_ID,
        ],
    )


@dsl.container_component
def derive_daily_klines(
    status_1m_sync: str,
    jobs: int,
    window: int,
    stables_1d: dsl.Output[dsl.Dataset],
):
    """Resample 1m klines into 1d klines."""

    return dsl.ContainerSpec(
        image=IMAGE,
        command=["bash", "-c"],
        args=[
            """
set -euo pipefail

export CLOUDSDK_CORE_PROJECT="$4"
export GIT_KEY_SECRET=git-key

./run.sh

uv run sync-datastore.py -d \
        --output-daily-file /dev/null \
        --output-stables-file "$3" \
        --stable-coin USDT \
        --window $2 \
        --fill-missing \
        -j $1
            """,
            status_1m_sync,  # 0
            jobs,  # 1
            window,  # 2
            stables_1d.uri,  # 3
            PROJECT_ID,  # 4
        ],
    )


@dsl.container_component
def fit_garch_model(klines: dsl.Input[dsl.Dataset], garch_model: dsl.Output[dsl.Model]):
    """Fit a GARCH model to the daily klines and save the model artifact."""

    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            """
            uv run vertex-ai-main.py garch-forecast-eval.ipynb \
                    --input_file="$0" \
                    --output_file="$1" \
                    --jobs_concurrency=32
            """,
            klines.uri,  # 0
            garch_model.uri,  # 1
        ],
    )


@dsl.container_component
def forecast_returns(
    klines: dsl.Input[dsl.Dataset],
    garch_model: dsl.Input[dsl.Model],
    report: dsl.Output[dsl.HTML],
    predictions: dsl.Output[dsl.Dataset],
):
    """Fit a GARCH model to the daily klines and save the model artifact."""

    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            """
            uv run vertex-ai-main.py "chronos-2-eval.ipynb=$0" \
                    --input_ohlcv_file="$1" \
                    --input_gjr_file="$2" \
                    --output_pred_file="$3" \
                    --device_map=cuda \
                    --reference_pair=BTCUSDT
            """,
            report.uri,  # 0
            klines.uri,  # 1
            garch_model.uri,  # 2
            predictions.uri,  # 3
        ],
    )


@dsl.pipeline(
    name="chronos-2-pipeline",
    description="Long only portfolio forecast",
)
def pipeline():
    sync = (
        synchronize_1m_archives(today=dsl.PIPELINE_JOB_ID_PLACEHOLDER)
        .set_cpu_limit("8")
        .set_memory_limit("16G")
    )

    daily = (
        derive_daily_klines(status_1m_sync=sync.output, window=1000 + 60, jobs=32)
        .set_cpu_limit("8")
        .set_memory_limit("16G")
    )

    gjr = (
        fit_garch_model(klines=daily.outputs["stables_1d"])
        .set_cpu_limit("32")
        .set_memory_limit("24G")
    )

    forecast = (
        forecast_returns(
            klines=daily.outputs["stables_1d"],
            garch_model=gjr.outputs["garch_model"],
        )
        .set_cpu_limit("4")
        .set_memory_limit("16G")
        .set_accelerator_type("NVIDIA_L4")
        .set_accelerator_limit(1)
    )


if __name__ == "__main__":
    from google.cloud import aiplatform
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix=".yaml") as f:
        kfp.compiler.Compiler().compile(pipeline, f.name)

        aiplatform.init(project=PROJECT_ID, location=REGION)

        job = aiplatform.PipelineJob(
            display_name="chronos-2-daily-run",
            template_path=f.name,
            pipeline_root=PIPELINE_ROOT,
            enable_caching=True,
        )

        job.submit(service_account=SERVICE_ACCOUNT)
