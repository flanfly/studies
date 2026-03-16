import kfp
from kfp import dsl

from datetime import datetime
import os

from dotenv import load_dotenv
import logging as l
import sys

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

load_dotenv()

PROJECT_ID = "prj-vertexai-test"
REGION = "asia-southeast1"
PIPELINE_ROOT = "gs://kai-vertex-ai-test-data/cr2"
SERVICE_ACCOUNT = "batch-job-sa@prj-vertexai-test.iam.gserviceaccount.com"
REPOSITORY = "flanfly/studies"


def latest_image_tag():
    from ghapi.all import GhApi

    tag = "@sha256:96134c14d7324eb7b9b52e1abc14452f042eff94dbcbe3db66684892d73df20b"
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
                tag = f"@{v.name}"

    l.info(tag)
    return tag


# 2026-03-13
DATA_PIPELINE_IMAGE = "asia-southeast1-docker.pkg.dev/prj-vertexai-test/default/studies@sha256:a5e43e4a01f9e861ce1e15fbd6cf7126a28cec9a3c07e2357aa3fb9c0d2c9497"
IMAGE = f"asia-southeast1-docker.pkg.dev/prj-vertexai-test/default/studies{latest_image_tag()}"


@dsl.container_component
def synchronize_1m_archives(today: str, status: dsl.OutputPath(str)):
    """Makes sure 1-minute kline data is in sync with Binance."""

    return dsl.ContainerSpec(
        image=DATA_PIPELINE_IMAGE,
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
    spot_1d: dsl.Output[dsl.Dataset],
    stables_1d: dsl.Output[dsl.Dataset],
):
    """Resample 1m klines into 1d klines."""

    return dsl.ContainerSpec(
        image=DATA_PIPELINE_IMAGE,
        command=["bash", "-c"],
        args=[
            """
set -euo pipefail

export CLOUDSDK_CORE_PROJECT="$4"
export GIT_KEY_SECRET=git-key

./run.sh

uv run sync-datastore.py -d \
        --output-daily-file "$3" \
        --output-stables-file "$2" \
        --stable-coin USDT \
        --fill-missing \
        -j $1
            """,
            status_1m_sync,  # 0
            jobs,  # 1
            stables_1d.uri,  # 2
            spot_1d.uri,  # 3
            PROJECT_ID,  # 4
        ],
    )


@dsl.container_component
def forecast_returns(
    klines: dsl.Input[dsl.Dataset],
    report: dsl.Output[dsl.HTML],
    features: dsl.Output[dsl.Dataset],
    embeddings: dsl.Output[dsl.Dataset],
    classifier: dsl.Output[dsl.Dataset],
):
    """Fit a Supervised Contrastive Learning model to forecast returns."""

    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            """
            uv run vertex-ai-main.py "scl.ipynb=$0" \
                    --input_ohlcv_file="$1" \
                    --output_features_file="$2" \
                    --output_nn_file="$3" \
                    --output_clf_file="$4" \
                    --device=cuda \
                    --dev_mode=0 \
                    --market_pair=BTCUSDT
            """,
            report.uri,  # 0
            klines.uri,  # 1
            features.uri,  # 2
            embeddings.uri,  # 3
            classifier.uri,  # 4
        ],
    )


@dsl.pipeline(
    name="hybrid-pipeline",
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

    forecast = (
        forecast_returns(
            klines=daily.outputs["stables_1d"],
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
            display_name="hybrid-eval",
            template_path=f.name,
            pipeline_root=PIPELINE_ROOT,
            enable_caching=True,
        )

        job.submit(service_account=SERVICE_ACCOUNT)
