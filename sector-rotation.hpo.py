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
BUCKET = "gs://kai-vertex-ai-test-data/hpo"
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

if __name__ == "__main__":
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt

    from sys import argv

    name = (
        argv[1]
        if len(argv) > 1
        else f"sector-rotation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "e2-standard-4",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE,
                "command": ["uv", "run", "vertex-ai-main.py", "sector-rotation.ipynb"],
            },
        }
    ]
    custom_job = aiplatform.CustomJob(
        display_name=name,
        worker_pool_specs=worker_pool_specs,
    )

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=name,
        custom_job=custom_job,
        metric_spec={
            "cagr": "maximize",
        },
        # max_long = "2"
        # max_short = "1"
        # period = "30"
        # stop_long = "0.5"
        # stop_short = "0.3"
        # hard_stop_long = "0.05"
        # hard_stop_short = "0.05"
        # leverage = "1.0"
        # variant_name = "default"
        parameter_spec={
            "signal": hpt.CategoricalParameterSpec(
                values=[
                    "mom1m",
                    "mom2m",
                    "mom3m",
                    "mom6m",
                    "mom12m",
                    "mom12-1m-a",
                    "mom12-1m-b",
                ],
            ),
            "max_long": hpt.IntegerParameterSpec(
                min=0,
                max=3,
                scale="linear",
            ),
            "max_short": hpt.IntegerParameterSpec(
                min=0,
                max=2,
                scale="linear",
            ),
            "period": hpt.DiscreteParameterSpec(
                values=[1, 5, 10, 21, 42],
                scale="linear",
            ),
            "stop_long": hpt.DiscreteParameterSpec(
                values=[0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 1],
                scale="linear",
            ),
            "stop_short": hpt.DiscreteParameterSpec(
                values=[0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 1],
                scale="linear",
            ),
        },
        max_trial_count=150,
        parallel_trial_count=5,
        search_algorithm=None,
    )
    hpt_job.run(service_account=SERVICE_ACCOUNT)
