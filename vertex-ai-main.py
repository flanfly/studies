import papermill as pm
import scrapbook as sb
import nbformat

import hypertune

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


# key -> value
parameters: Dict[str, str] = {}

# input, output
notebooks: List[Notebook] = []

if len(sys.argv) < 2:
    l.error(
        "Usage: vertex-ai-main.py NOTEBOOK [NOTEBOOK ...] [--PARAMETER VALUE ...] [--PARAMETER=VALUE ...]"
    )
    sys.exit(1)

args = sys.argv[1:]
while len(args) > 0:
    arg = args.pop(0)

    if arg.startswith("--"):
        p = arg[2:].split("=")
        if len(p) >= 2:
            key, value = p[0], "=".join(p[1:])
        elif len(p) == 1:
            key, value = p[0], args.pop(0)
        else:
            raise ValueError(f"Invalid argument: {arg}")

        parameters[key] = value
    else:
        p = arg.split("=")
        if len(p) >= 2:
            nb = new_notebook_io(p[0], "=".join(p[1:]))
        elif len(p) == 1:
            nb = new_notebook_io(p[0])
        else:
            raise ValueError(f"Invalid argument: {arg}")

        notebooks.append(nb)

l.info(
    f"""Running: {", ".join(map(lambda nb: f'{nb.input_arg} (save to {nb.output_arg})', notebooks))}"""
)
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

for nb in notebooks:
    l.info(
        f"""Executing notebook {nb.input_arg}{f" and saving to {nb.output_arg}" if nb.output_arg else ""}"""
    )

    pm.execute_notebook(
        os.path.join(nb.directory, "input.ipynb"),
        os.path.join(nb.directory, "output.ipynb"),
        parameters=parameters,
        log_output=True,
    )

    # read scraps and report
    l.info(f"Reading scraps from {os.path.join(nb.directory, 'output.ipynb')}")
    try:
        scraps = sb.read_notebook(os.path.join(nb.directory, "output.ipynb"))
        l.info("Notebook Scraps (dataframe):\n" + str(scraps.scrap_dataframe))

        scraps_dict = scraps.scraps if scraps.scraps else {}
        for name, scrap in scraps_dict.items():
            if not isinstance(scrap.data, (int, float)):
                l.warning(
                    f"Scrap '{name}' is not a number and cannot be reported as a hyperparameter tuning metric: {scrap.data}"
                )
                continue

            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=name,
                metric_value=float(scrap.data),
                global_step=1,
            )

    except Exception as e:
        l.error(f"Error reading scraps: {e}")

    # convert to html
    with open(os.path.join(nb.directory, "output.ipynb"), "r", encoding="utf-8") as f:
        notebook_node = nbformat.read(f, as_version=4)

    body, _ = html_exporter.from_notebook_node(notebook_node)
    with open(os.path.join(nb.directory, "output.html"), "w", encoding="utf-8") as f:
        f.write(body)

    if nb.output_arg:
        l.info(f"Saving HTML output to {nb.output_arg}")
        with fsspec.open(nb.output_arg, "w", encoding="utf-8") as f:
            f.write(body)

        # Also save the executed notebook itself if the user wants to see metadata/scraps
        if nb.output_arg.endswith(".html"):
            ipynb_out = nb.output_arg.replace(".html", ".ipynb")
        else:
            ipynb_out = nb.output_arg + ".ipynb"

        l.info(f"Saving executed notebook to {ipynb_out}")
        with open(
            os.path.join(nb.directory, "output.ipynb"), "r", encoding="utf-8"
        ) as f_in:
            with fsspec.open(ipynb_out, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
