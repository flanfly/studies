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
def run(
    results: dsl.Dataset,
):
    """Eval crypto momentum strategies."""

    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            """
            crypto-momentum.trail.sh "$0"
            """,
            results.uri,  # 0
        ],
    )


@dsl.pipeline(
    name="crypto-momentum",
    description="Evaluate crypto momentum strategies with various parameters.",
)
def pipeline():
    run().set_cpu_limit("32").set_memory_limit("32G")


if __name__ == "__main__":
    from google.cloud import aiplatform
    from tempfile import NamedTemporaryFile
    from sys import argv

    name = (
        argv[1]
        if len(argv) > 1
        else f"crypto-momentum-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    with NamedTemporaryFile(suffix=".yaml") as f:
        kfp.compiler.Compiler().compile(pipeline, f.name)

        aiplatform.init(project=PROJECT_ID, location=REGION)

        job = aiplatform.PipelineJob(
            job_id=name,
            display_name="crypto momentum",
            template_path=f.name,
            pipeline_root=PIPELINE_ROOT,
            enable_caching=True,
            labels={},
        )

        job.submit(service_account=SERVICE_ACCOUNT)
