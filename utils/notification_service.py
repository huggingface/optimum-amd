# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import re
import traceback
from typing import Dict, Optional

import requests


import os

# os.environ["CI_EVENT"] = "scheduled"
# os.environ["CI_SHA"] = "scheduled"
# os.environ["CI_WORKFLOW_REF"] = "scheduled"
# os.environ["GITHUB_RUN_ID"] = "7919603663"
from slack_sdk import WebClient


client = WebClient(token=os.environ["CI_SLACK_BOT_TOKEN"])


def get_jobs(workflow_run_id, token=None):
    """Extract jobs in a GitHub Actions workflow run"""

    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    url = f"https://api.github.com/repos/huggingface/optimum-amd/actions/runs/{workflow_run_id}/jobs?per_page=100"
    result = requests.get(url, headers=headers).json()

    jobs = []

    try:
        jobs.extend(result["jobs"])
        pages_to_iterate_over = math.ceil((result["total_count"] - 100) / 100)

        for i in range(pages_to_iterate_over):
            result = requests.get(url + f"&page={i + 2}", headers=headers).json()
            jobs.extend(result["jobs"])

        return jobs
    except Exception:
        print(f"Unknown error, could not fetch links:\n{traceback.format_exc()}")

    return []


def handle_test_results(test_results):
    expressions = test_results.split(" ")

    failed = 0
    success = 0

    # When the output is short enough, the output is surrounded by = signs: "== OUTPUT =="
    # When it is too long, those signs are not present.
    time_spent = expressions[-2] if "=" in expressions[-1] else expressions[-1]

    for i, expression in enumerate(expressions):
        if "failed" in expression:
            failed += int(expressions[i - 1])
        if "passed" in expression:
            success += int(expressions[i - 1])

    return failed, success, time_spent


def handle_stacktraces(test_results):
    # These files should follow the following architecture:
    # === FAILURES ===
    # <path>:<line>: Error ...
    # <path>:<line>: Error ...
    # <empty line>

    total_stacktraces = test_results.split("\n")[1:-1]
    stacktraces = []
    for stacktrace in total_stacktraces:
        try:
            line = stacktrace[: stacktrace.index(" ")].split(":")[-2]
            error_message = stacktrace[stacktrace.index(" ") :]

            stacktraces.append(f"(line {line}) {error_message}")
        except Exception:
            stacktraces.append("Cannot retrieve error message.")

    return stacktraces


class Message:
    def __init__(
        self,
        title: str,
        ci_title: str,
        results: Dict,
    ):
        self.title = title
        self.ci_title = ci_title

        # Failures and success of the modeling tests
        self.n_success = sum(r["success"] for r in results.values())
        self.n_failures = sum(r["failed"] for r in results.values())
        self.n_tests = self.n_failures + self.n_success

        self.model_results = results

        self.thread_ts = None

    @property
    def time(self) -> str:
        all_results = [*self.model_results.values()]
        time_spent = [r["time_spent"].split(", ")[0] for r in all_results if len(r["time_spent"])]
        total_secs = 0

        for time in time_spent:
            time_parts = time.split(":")

            # Time can be formatted as xx:xx:xx, as .xx, or as x.xx if the time spent was less than a minute.
            if len(time_parts) == 1:
                time_parts = [0, 0, time_parts[0]]

            hours, minutes, seconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
            total_secs += hours * 3600 + minutes * 60 + seconds

        hours, minutes, seconds = total_secs // 3600, (total_secs % 3600) // 60, total_secs % 60
        return f"{int(hours)}h{int(minutes)}m{int(seconds)}s"

    @property
    def header(self) -> Dict:
        return {"type": "header", "text": {"type": "plain_text", "text": self.title}}

    @property
    def ci_title_section(self) -> Dict:
        return {"type": "section", "text": {"type": "mrkdwn", "text": self.ci_title}}

    @property
    def no_failures(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": f"🌞 There were no failures: all {self.n_tests} tests passed. The suite ran in {self.time}.",
                "emoji": True,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/optimum-amd/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }

    @property
    def failures(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": (
                    f"There were {self.n_failures} failures, out of {self.n_tests} tests.\n"
                    f"The suite ran in {self.time}."
                ),
                "emoji": True,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/optimum-amd/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }

    @staticmethod
    def get_category_report(report, rjust=6):
        return f"{str(report['failed']).rjust(rjust)} | {str(report['success']).rjust(rjust)}"

    @property
    def category_failures(self) -> Dict:
        category_failures = []
        for key in self.model_results:
            report = self.get_category_report(self.model_results[key])
            category_failures.append((f"{key} | {report}"))

        header = "Category | Failed | Success \n"
        category_failures_report = prepare_reports(
            title="Test Category Results", header=header, reports=category_failures
        )

        return {"type": "section", "text": {"type": "mrkdwn", "text": category_failures_report}}

    @property
    def model_failures(self):
        model_failure_sections = []
        for key, result in self.model_results.items():
            extracted_models = []
            failures = result["failures"]
            for failure in failures:
                line = failure["line"]
                if "amd" in line:
                    match = re.search(r"::test_model_\d+_amd_([a-zA-Z0-9]+)", line)
                    if match:
                        model_id = "amd/" + match.group(1)
                # Extracting TIMM model names
                elif "timm" in line:
                    match = re.search(
                        r"::test_timm_quantization_\d+_default_timm_config_image_classification_timm_([a-zA-Z0-9_]+)_in1k",
                        line,
                    )
                    if match:
                        model_id = "timm/" + match.group(1) + "_in1k"

                # Check if regressed
                extracted_models.append(f"{model_id} | False | 0 | 0 | 0 ")

            model_header = "Model | Regressed | Total Operators | DPU Operators (Current) | DPU Operators (Baseline)\n"
            model_failures_report = prepare_reports(
                title=f"These following {key} tests had failures", header=model_header, reports=extracted_models
            )

            model_failure_sections.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": model_failures_report},
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Check failures", "emoji": True},
                        "url": result["job_link"],
                    },
                },
            )

            model_failures_report = prepare_reports(
                title=f"These following {key} tests had failures",
                header=model_header,
                reports=extracted_models,
                to_truncate=False,
            )
            os.makedirs(os.path.join(os.getcwd(), "prev_ci_results"), exist_ok=True)
            file_path = os.path.join(
                os.getcwd(), "prev_ci_results", f"model_failures_report_{key.replace(' ', '_').replace('-', '_')}.txt"
            )
            with open(file_path, "w", encoding="UTF-8") as fp:
                fp.write(model_failures_report)

        return model_failure_sections

    @property
    def payload(self) -> str:
        blocks = [self.header]

        if self.ci_title:
            blocks.append(self.ci_title_section)

        if self.n_failures > 0:
            blocks.append(self.failures)
            blocks.append(self.category_failures)
            for block in self.model_failures:
                if block["text"]["text"]:
                    blocks.append(block)
        else:
            blocks.append(self.no_failures)

        return json.dumps(blocks)

    @staticmethod
    def error_out(title, ci_title="", runner_not_available=False, runner_failed=False, setup_failed=False):
        blocks = []
        title_block = {"type": "header", "text": {"type": "plain_text", "text": title}}
        blocks.append(title_block)

        if ci_title:
            ci_title_block = {"type": "section", "text": {"type": "mrkdwn", "text": ci_title}}
            blocks.append(ci_title_block)

        offline_runners = []
        if runner_not_available:
            text = "💔 CI runners are not available! Tests are not run. 😭"
            result = os.environ.get("OFFLINE_RUNNERS")
            if result is not None:
                offline_runners = json.loads(result)
        elif runner_failed:
            text = "💔 CI runners have problems! Tests are not run. 😭"
        elif setup_failed:
            text = "💔 Tests are not run. 😭"
        else:
            text = "💔 There was an issue running the tests. 😭"

        error_block_1 = {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": text,
            },
        }

        text = ""
        if len(offline_runners) > 0:
            text = "\n  • " + "\n  • ".join(offline_runners)
            text = f"The following runners are offline:\n{text}\n\n"
        text += "🙏 Let's fix it ASAP! 🙏"

        error_block_2 = {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": text,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/optimum-amd/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }
        blocks.extend([error_block_1, error_block_2])

        payload = json.dumps(blocks)

        print("Sending the following payload")
        print(json.dumps({"blocks": blocks}))

        client.chat_postMessage(
            channel=os.environ["CI_SLACK_CHANNEL_ID"],
            text=text,
            blocks=payload,
        )

    def post(self):
        payload = self.payload
        print("Sending the following payload")
        print(json.dumps({"blocks": json.loads(payload)}))

        text = f"{self.n_failures} failures out of {self.n_tests} tests," if self.n_failures else "All tests passed."

        self.thread_ts = client.chat_postMessage(
            channel=os.environ["CI_SLACK_CHANNEL_ID"],
            blocks=payload,
            text=text,
        )


def retrieve_artifact(artifact_path: str):
    _artifact = {}

    if os.path.exists(artifact_path):
        files = os.listdir(artifact_path)
        for file in files:
            try:
                with open(os.path.join(artifact_path, file)) as f:
                    _artifact[file.split(".")[0]] = f.read()
            except UnicodeDecodeError as e:
                raise ValueError(f"Could not open {os.path.join(artifact_path, file)}.") from e

    return _artifact


def retrieve_available_artifacts():
    class Artifact:
        def __init__(self, name: str):
            self.name = name

        def __str__(self):
            return self.name

        def add_path(self, path: str):
            self.path = path

    _available_artifacts: Dict[str, Artifact] = {}

    directories = filter(os.path.isdir, os.listdir())
    for directory in directories:
        if directory == "reports":
            for artifact_name in os.listdir(directory):
                _available_artifacts[artifact_name] = Artifact(artifact_name)

                _available_artifacts[artifact_name].add_path(os.path.join(directory, artifact_name))

    return _available_artifacts


def prepare_reports(title, header, reports, to_truncate=True):
    report = ""

    MAX_ERROR_TEXT = 3000 - len("[Truncated]")
    if not to_truncate:
        MAX_ERROR_TEXT = float("inf")

    if len(reports) > 0:
        # `text` must be less than 3001 characters in Slack SDK
        # keep some room for adding "[Truncated]" when necessary

        for idx in range(len(reports)):
            _report = header + "\n".join(reports[: idx + 1])
            new_report = f"{title}:\n```\n{_report}\n```\n"
            if len(new_report) > MAX_ERROR_TEXT:
                # `report` here has length <= 3000
                report = report + "[Truncated]"
                break
            report = new_report

    return report


if __name__ == "__main__":
    # runner_not_available = True if runner_status is not None and runner_status != "success" else False
    # runner_failed = True if runner_env_status is not None and runner_env_status != "success" else False
    # Let's keep the lines regarding runners' status (we might be able to use them again in the future)
    runner_not_available = False
    runner_failed = False

    org = "huggingface"
    repo = "optimum-amd"
    repository_full_name = f"{org}/{repo}"

    # This env. variable is set in workflow file (under the job `send_results`).
    ci_event = os.environ["CI_EVENT"]

    # To find the PR number in a commit title, for example, `Add AwesomeFormer model (#99999)`
    pr_number_re = re.compile(r"\(#(\d+)\)$")

    title = f"🤗 Results of the {ci_event} tests."
    # Add Commit/PR title with a link for push CI
    # (check the title in 2 env. variables - depending on the CI is triggered via `push` or `workflow_run` event)
    ci_title_push = os.environ.get("CI_TITLE_PUSH")
    ci_title_workflow_run = os.environ.get("CI_TITLE_WORKFLOW_RUN")
    ci_title = ci_title_push if ci_title_push else ci_title_workflow_run

    ci_sha = os.environ.get("CI_SHA")

    ci_url = None
    if ci_sha:
        ci_url = f"https://github.com/{repository_full_name}/commit/{ci_sha}"

    if ci_title is not None:
        if ci_url is None:
            raise ValueError(
                "When a title is found (`ci_title`), it means a `push` event or a `workflow_run` even (triggered by "
                "another `push` event), and the commit SHA has to be provided in order to create the URL to the "
                "commit page."
            )
        ci_title = ci_title.strip().split("\n")[0].strip()

        # Retrieve the PR title and author login to complete the report
        commit_number = ci_url.split("/")[-1]
        ci_detail_url = f"https://api.github.com/repos/{repository_full_name}/commits/{commit_number}"
        ci_details = requests.get(ci_detail_url).json()
        ci_author = ci_details["author"]["login"]

        merged_by = None
        # Find the PR number (if any) and change the url to the actual PR page.
        numbers = pr_number_re.findall(ci_title)
        if len(numbers) > 0:
            pr_number = numbers[0]
            ci_detail_url = f"https://api.github.com/repos/{repository_full_name}/pulls/{pr_number}"
            ci_details = requests.get(ci_detail_url).json()

            ci_author = ci_details["user"]["login"]
            ci_url = f"https://github.com/{repository_full_name}/pull/{pr_number}"

            merged_by = ci_details["merged_by"]["login"]

        if merged_by is None:
            ci_title = f"<{ci_url}|{ci_title}>\nAuthor: {ci_author}"
        else:
            ci_title = f"<{ci_url}|{ci_title}>\nAuthor: {ci_author} | Merged by: {merged_by}"

    elif ci_sha:
        ci_title = f"<{ci_url}|commit: {ci_sha}>"

    else:
        ci_title = ""

    if runner_not_available or runner_failed:
        Message.error_out(title, ci_title, runner_not_available, runner_failed)
        exit(0)

    github_actions_jobs = get_jobs(
        workflow_run_id=os.environ["GITHUB_RUN_ID"],  # token=os.environ["ACCESS_REPO_INFO_TOKEN"]
    )

    artifact_name_to_job_map = {}
    for job in github_actions_jobs:
        artifact_name = job["name"].split(" ")[0]
        artifact_name_to_job_map[artifact_name] = job
    available_artifacts = retrieve_available_artifacts()

    if len(available_artifacts) == 0:
        Message.error_out(title, ci_title, runner_not_available, runner_failed, setup_failed=True)
        exit(0)

    test_categories = {
        "Pre-Quantized": "run_tests_prequantized_models",
        "Timm Quantization": "run_tests_quantization",
    }

    results = {
        key: {
            "failed": 0,
            "success": 0,
            "time_spent": "",
            "error": False,
            "failures": [],
            "job_link": 0,
        }
        for key in test_categories.keys()
    }

    for key in results.keys():
        if test_categories[key] not in available_artifacts:
            results[key]["error"] = True
            continue

        artifact_path = available_artifacts[test_categories[key]].path

        job = artifact_name_to_job_map[test_categories[key]]
        results[key]["job_link"] = job["html_url"]

        artifact = retrieve_artifact(artifact_path)

        stacktraces = handle_stacktraces(artifact["failures_line"])

        failed, success, time_spent = handle_test_results(artifact["stats"])

        results[key]["failed"] = failed
        results[key]["success"] = success
        results[key]["time_spent"] = time_spent[1:-1] + ", "

        if len(artifact["errors"]):
            results[key]["error"] = True

        if failed:
            for line in artifact["summary_short"].split("\n"):
                if line.startswith("FAILED "):
                    line = line[len("FAILED ") :]
                    line = line.split()[0].replace("\n", "")

                    results[key]["failures"].append({"line": line, "trace": stacktraces.pop(0)})

    message = Message(
        title,
        ci_title,
        results,
    )

    message.post()
