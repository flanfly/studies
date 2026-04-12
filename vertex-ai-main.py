import papermill as pm
import scrapbook as sb
import nbformat

import hypertune

import polars as pl
import datetime as dt

from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
from traitlets.config import Config

import logging as l
import sys
import argparse
import os
import tempfile
import fsspec
import shutil as sh
import json
from tempfile import TemporaryDirectory

from typing import List, Dict, NamedTuple, Tuple

# setup logging
l.basicConfig(
    level=l.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[l.StreamHandler(sys.stdout)],
)


class Notebook(NamedTuple):
    input_arg: str
    output_arg: str | None
    directory: str


def new_notebook_io(input: str, output: str | None = None) -> Notebook:
    if os.path.isdir(input):
        raise ValueError(f"Input notebook is a directory: {input}")
    if not os.path.isfile(input):
        raise ValueError(f"Input notebook does not exist: {input}")

    if output != "" and output is not None:
        if os.path.isdir(output):
            output = os.path.join(output, os.path.basename(input))
        if os.path.isfile(output):
            l.warning(
                f"Output notebook file already exists and will be overwritten: {output}"
            )
        os.makedirs(os.path.dirname(output), exist_ok=True)

    d = tempfile.mkdtemp(prefix="vertex-ai-wrapper")
    sh.copy(input, os.path.join(d, "input.ipynb"))

    return Notebook(input_arg=input, output_arg=output, directory=d)


parser = argparse.ArgumentParser(
    description="Vertex AI Notebook Wrapper",
    usage="vertex-ai-main.py NOTEBOOK [--verbose] [--html OUTPUT_NOTEBOOK] [--csv OUTPUT_METRICS] [PARAMETERS...]",
)
parser.add_argument("notebook", help="Input notebook path")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
parser.add_argument("--html", help="Output notebook path")
parser.add_argument("--csv", help="Output metrics CSV path")

# Parse known args, then treat the rest as papermill parameters
args, unknown = parser.parse_known_args()

if args.verbose:
    l.getLogger().setLevel(l.DEBUG)

if not os.path.isfile(args.notebook):
    l.error(f"Input notebook does not exist: {args.notebook}")
    sys.exit(1)

if not args.html:
    args.html = args.notebook.removesuffix(".ipynb") + ".output.ipynb"
if not args.csv:
    args.csv = args.notebook.removesuffix(".ipynb") + ".metrics.csv"

# Process unknown arguments into parameters
parameters: Dict[str, str] = {}
i = 0
while i < len(unknown):
    arg = unknown[i]
    if arg.startswith("--"):
        if "=" in arg:
            key, value = arg[2:].split("=", 1)
            parameters[key] = value
        else:
            key = arg[2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                parameters[key] = unknown[i + 1]
                i += 1
            else:
                parameters[key] = "True"
    i += 1

if args.html:
    l.info(f"Executing {args.notebook} and saving output to {args.html}")
else:
    l.info(f"Executing {args.notebook} without saving output results")
l.info(
    f"""Parameters: {", ".join(map(lambda k: f'{k}={parameters[k]}', parameters))}"""
)

c = Config()
c.HTMLExporter.preprocessors = [TagRemovePreprocessor]
c.TagRemovePreprocessor.remove_all_outputs_tags = ("dev_only",)
c.TagRemovePreprocessor.enabled = True
html_exporter = HTMLExporter(config=c)
html_exporter.exclude_input = True

hpt = hypertune.HyperTune()

# run the notebook
with TemporaryDirectory(prefix="vertex-ai-papermill") as dir:
    notebook = f"{dir}/output.ipynb"

    pm.execute_notebook(
        args.notebook,
        notebook,
        parameters=parameters,
        log_output=True,
    )

    # save the executed notebook if requested
    l.info(f"Saving executed notebook to {args.html}")

    # convert to html
    with open(notebook, "r", encoding="utf-8") as f:
        notebook_node = nbformat.read(f, as_version=4)

    body, _ = html_exporter.from_notebook_node(notebook_node)
    with fsspec.open(args.html, "w", encoding="utf-8") as f:
        f.write(body)

    # read scraps and report
    scraps = sb.read_notebook(notebook)
    scrapdf = (
        pl.from_pandas(scraps.scrap_dataframe)
        if scraps.scrap_dataframe is not None and not scraps.scrap_dataframe.empty
        else pl.DataFrame()
    )

    if (
        not scrapdf.is_empty()
        and "name" in scrapdf.columns
        and "data" in scrapdf.columns
    ):
        scrapstr = map(
            lambda r: f"{r['name']}={r['data']}", scrapdf.iter_rows(named=True)
        )
        l.info(f'Metrics: {", ".join(scrapstr)}')

        for row in scrapdf.iter_rows(named=True):
            name = row["name"]
            value = row["data"]

            if not isinstance(value, (int, float)):
                l.warning(
                    f"Scrap '{name}' is not a number and cannot be reported as a hyperparameter tuning metric: {value}"
                )
                continue

            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=name,
                metric_value=float(value),
                global_step=1,
            )

    l.info(f"Saving metrics to {args.csv}")

    # Ensure we only have name and data, and add metadata
    out_df = scrapdf.select("name", "data").with_columns(
        data=pl.col("data").cast(pl.Utf8),
        notebook=pl.lit(os.path.basename(args.notebook)),
        ts=pl.lit(dt.datetime.now()).cast(pl.Datetime),
        parameters=pl.lit(json.dumps(parameters)),
    )
    with fsspec.open(args.csv, "w") as f:
        out_df.write_csv(f)
