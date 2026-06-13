import kfp
from kfp import dsl

from google.cloud import aiplatform
from tempfile import NamedTemporaryFile

SHA256 = "24d762926ec86adff5db451bf50520f54665609e0a5288cdf3ca3d3be9ccfc8d"
# SHA256 = "9c5c7892bf8341f0cc12ded603ce136b06e5ecb4510975cb3968c1bf4a508e61"
IMAGE = (
    f"asia-southeast1-docker.pkg.dev/prj-vertexai-test/default/studies@sha256:{SHA256}"
)
PROJECT_ID = "prj-vertexai-test"
REGION = "asia-southeast1"
PIPELINE_ROOT = "gs://kai-vertex-ai-test-data/cr2"
SERVICE_ACCOUNT = "batch-job-sa@prj-vertexai-test.iam.gserviceaccount.com"


@dsl.container_component
def synchronize_1m_archives(status: dsl.OutputPath(str)):
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
    kline_offset: int,
    jobs: int,
    spot_1d: dsl.Output[dsl.Dataset],
    usdt_1d: dsl.Output[dsl.Dataset],
):
    """Resample 1m klines into 1d klines."""

    return dsl.ContainerSpec(
        image=IMAGE,
        command=["bash", "-c"],
        args=[
            """
set -euo pipefail

export CLOUDSDK_CORE_PROJECT=$2
export GIT_KEY_SECRET=git-key

mkdir -p "$(dirname "$0")"
mkdir -p "$(dirname "$1")"
./run.sh

uv run sync-datastore.py -d --output-daily-file "$0" --output-stables-file "$1" \
        --kline-offset $5 \
        -j $6
            """,
            spot_1d.uri,  # 0
            usdt_1d.uri,  # 1
            PROJECT_ID,  # 2
            SHA256,  # 3
            status_1m_sync,  # 4
            kline_offset,  # 5
            jobs,  # 6
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
            klines.uri,
            garch_model.uri,
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
            mkdir -p $(dirname "$0")
            mkdir -p $(dirname "$3")

            uv run vertex-ai-main.py "chronos-2-eval.ipynb=$0" \
                    --input_ohlcv_file="$1" \
                    --input_gjr_file="$2" \
                    --output_pred_file="$3" \
                    --device_map=cuda
            """,
            report.uri,  # 0
            klines.uri,  # 1
            garch_model.uri,  # 2
            predictions.uri,  # 3
        ],
    )


@dsl.pipeline(
    name="chronos-2-pipeline",
    description="Long/short portfolio forecast",
)
def pipeline():
    sync = synchronize_1m_archives().set_cpu_limit("8").set_memory_limit("16G")

    daily = (
        derive_daily_klines(status_1m_sync=sync.output, kline_offset=0, jobs=32)
        .set_cpu_limit("8")
        .set_memory_limit("16G")
    )

    gjr = (
        fit_garch_model(klines=daily.outputs["usdt_1d"])
        .set_cpu_limit("32")
        .set_memory_limit("24G")
    )

    forecast = forecast_returns(
        klines=daily.outputs["usdt_1d"], garch_model=gjr.outputs["garch_model"]
    )
    forecast.set_cpu_limit("4").set_memory_limit("16G")
    forecast.set_accelerator_type("NVIDIA_L4").set_accelerator_limit(1)


if __name__ == "__main__":
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
