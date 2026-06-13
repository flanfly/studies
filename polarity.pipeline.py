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


IMAGE = f"asia-southeast1-docker.pkg.dev/prj-vertexai-test/default/studies{latest_image_tag()}"


@dsl.container_component
def train(
    concurrency: int,
    zscore_win: int,
    gbt_type: str,
    gbt_min_leafs: int,
    gbt_max_depth: int,
    gbt_lr: float,
    report: dsl.Output[dsl.HTML],
):
    """Forecast using Polarity Digital sourced signals"""

    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            """
            uv run vertex-ai-main.py "polarity.ipynb=$0" \
                    --concurrency="$1" \
                    --zscore_win="$2" \
                    --gbt_type="$3" \
                    --gbt_min_leafs="$4" \
                    --gbt_max_depth="$5" \
                    --gbt_lr="$6" \
            """,
            report.uri,  # 0
            concurrency,  # 1
            zscore_win,  # 2
            gbt_type,  # 3
            gbt_min_leafs,  # 4
            gbt_max_depth,  # 5
            gbt_lr,  # 6
        ],
    )


@dsl.pipeline(
    name="polarity",
    description="Forecast using Polarity Digital sourced signals",
)
def pipeline():
    run = (
        train(
            concurrency=32,
            zscore_win=365,
            gbt_type="gbt",
            gbt_min_leafs=20,
            gbt_max_depth=2,
            gbt_lr=0.01,
        )
        .set_cpu_limit("32")
        .set_memory_limit("32G")
    )


if __name__ == "__main__":
    from google.cloud import aiplatform
    from tempfile import NamedTemporaryFile
    from sys import argv

    name = (
        argv[1]
        if len(argv) > 1
        else f"polarity-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    with NamedTemporaryFile(suffix=".yaml") as f:
        kfp.compiler.Compiler().compile(pipeline, f.name)

        aiplatform.init(project=PROJECT_ID, location=REGION)

        job = aiplatform.PipelineJob(
            job_id=name,
            display_name="polarity",
            template_path=f.name,
            pipeline_root=PIPELINE_ROOT,
            enable_caching=True,
            labels={},
        )

        job.submit(service_account=SERVICE_ACCOUNT)
