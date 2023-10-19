import os
import subprocess
from pathlib import Path

import pandas as pd
import requests
from omegaconf import OmegaConf

from dana import (
    LOGGER,
    add_new_optimum_build,
    add_new_optimum_series,
    add_new_project,
    add_new_sample,
    authenticate,
)

VERBOSE = False
OVERRIDE = False
AVERAGE_RANGE = 5
AVERAGE_MIN_COUNT = 3

PROJECT_ID = "Optimum-AMD"
USERNAME = os.environ.get("DANA_USERNAME", "admin")
PASSWORD = os.environ.get("DANA_PASSWORD", "admin")
BEARER_TOKEN = os.environ.get("BEARER_TOKEN", "secret")
PROJECT_DESCRIPTION = "Benchmarking project for AMD devices."
DANA_DASHBOARD_URL = "https://ilyasmoutawwakil-dana.hf.space"
TRANSFORMERS_URL = "https://github.com/huggingface/transformers"
TRANSFORMERS_PATH = os.environ.get("TRANSFORMERS_PATH", "transformers")
BUILD_ID = int(subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=TRANSFORMERS_PATH).decode().strip())

#######################################################################################################################
COMMIT_HASH = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TRANSFORMERS_PATH).decode().strip()
COMMIT_ABBREV_HASH = (
    subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=TRANSFORMERS_PATH,
    )
    .decode()
    .strip()
)
COMMIT_AUTHOR_NAME = (
    subprocess.check_output(
        ["git", "show", "-s", "--format=%an", COMMIT_HASH],
        cwd=TRANSFORMERS_PATH,
    )
    .decode()
    .strip()
)
COMMIT_AUTHOR_EMAIL = (
    subprocess.check_output(
        ["git", "show", "-s", "--format=%ae", COMMIT_HASH],
        cwd=TRANSFORMERS_PATH,
    )
    .decode()
    .strip()
)
COMMIT_SUBJECT = (
    subprocess.check_output(
        ["git", "show", "-s", "--format=%s", COMMIT_HASH],
        cwd=TRANSFORMERS_PATH,
    )
    .decode()
    .strip()
)
COMMIT_URL = f"{TRANSFORMERS_URL}/commit/{COMMIT_HASH}"
#######################################################################################################################

SESSION = requests.Session()


def main():
    LOGGER.info(" + Authenticating to DANA dashboard")
    authenticate(session=SESSION, dana_url=DANA_DASHBOARD_URL, username=USERNAME, password=PASSWORD)

    LOGGER.info(f" + Publishing project {PROJECT_ID}")
    add_new_project(
        session=SESSION,
        bearer_token=BEARER_TOKEN,
        dana_url=DANA_DASHBOARD_URL,
        project_id=PROJECT_ID,
        project_description=PROJECT_DESCRIPTION,
        override=OVERRIDE,
    )

    LOGGER.info(f" + Publishing build {BUILD_ID}")
    add_new_optimum_build(
        session=SESSION,
        bearer_token=BEARER_TOKEN,
        dana_url=DANA_DASHBOARD_URL,
        project_id=PROJECT_ID,
        build_id=BUILD_ID,
        commit_url=COMMIT_URL,
        commit_hash=COMMIT_HASH,
        commit_abbrev_hash=COMMIT_ABBREV_HASH,
        commit_author_name=COMMIT_AUTHOR_NAME,
        commit_author_email=COMMIT_AUTHOR_EMAIL,
        commit_subject=COMMIT_SUBJECT,
        override=OVERRIDE,
    )

    for series_foler in Path("experiments").iterdir():
        configs = list(series_foler.glob("*/hydra_config.yaml"))
        inference_results = list(series_foler.glob("*/inference_results.csv"))

        if len(inference_results) != 1 or len(configs) != 1:
            continue

        inference_results = pd.read_csv(inference_results[0]).to_dict(orient="records")[0]

        # Latency series
        series_id = f"{series_foler.name}_latency"

        LOGGER.info(f"\t + Publishing series {series_id}")
        add_new_optimum_series(
            session=SESSION,
            bearer_token=BEARER_TOKEN,
            dana_url=DANA_DASHBOARD_URL,
            project_id=PROJECT_ID,
            series_id=series_id,
            series_description=OmegaConf.to_yaml(OmegaConf.load(configs[0])),
            better_criterion="lower",
            average_range=AVERAGE_RANGE,
            average_min_count=AVERAGE_MIN_COUNT,
            override=OVERRIDE,
        )

        LOGGER.info(f"\t\t + Adding new sample to series {series_id}")
        add_new_sample(
            session=SESSION,
            bearer_token=BEARER_TOKEN,
            dana_url=DANA_DASHBOARD_URL,
            project_id=PROJECT_ID,
            build_id=BUILD_ID,
            series_id=series_id,
            sample_value=inference_results["forward.latency(s)"] * 1000,
            override=OVERRIDE,
        )

        # Memory series
        series_id = f"{series_foler.name}_memory"

        LOGGER.info(f"\t + Publishing series {series_id}")

        add_new_optimum_series(
            session=SESSION,
            bearer_token=BEARER_TOKEN,
            dana_url=DANA_DASHBOARD_URL,
            project_id=PROJECT_ID,
            series_id=series_id,
            series_description=OmegaConf.to_yaml(OmegaConf.load(configs[0])),
            better_criterion="lower",
            average_range=AVERAGE_RANGE,
            average_min_count=AVERAGE_MIN_COUNT,
            override=OVERRIDE,
        )

        LOGGER.info(f"\t\t + Adding new sample to series {series_id}")

        add_new_sample(
            session=SESSION,
            bearer_token=BEARER_TOKEN,
            dana_url=DANA_DASHBOARD_URL,
            project_id=PROJECT_ID,
            build_id=BUILD_ID,
            series_id=series_id,
            sample_value=inference_results["forward.peak_memory(MB)"],
            override=OVERRIDE,
        )


if __name__ == "__main__":
    main()
