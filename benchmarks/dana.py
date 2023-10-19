import json
import logging
from typing import Any, Dict, Optional

import coloredlogs


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("dana-client")
coloredlogs.install(level="INFO", logger=LOGGER)


def add_new_project(
    session,
    bearer_token: str,
    dana_url: str,
    project_id: str,
    project_description: str,
    override: bool = False,
) -> None:
    project_url = f"{dana_url}/admin/addProject"
    project_payload = {
        "projectId": project_id,
        "description": project_description,
        "users": "admin",
        "override": override,
    }

    post_to_dashboard(
        session=session,
        bearer_token=bearer_token,
        dashboard_url=project_url,
        payload=project_payload,
    )


def add_new_optimum_build(
    session,
    bearer_token: str,
    dana_url: str,
    project_id: str,
    build_id: str,
    commit_hash: str,
    commit_abbrev_hash: str,
    commit_author_name: str,
    commit_author_email: str,
    commit_subject: str,
    commit_url: str,
    override: bool = False,
) -> None:
    build_url = f"{dana_url}/apis/addBuild"
    build_payload = {
        "projectId": project_id,
        "build": {
            "buildId": build_id,
            "infos": {
                "hash": commit_hash,
                "abbrevHash": commit_abbrev_hash,
                "authorName": commit_author_name,
                "authorEmail": commit_author_email,
                "subject": commit_subject,
                "url": commit_url,
            },
        },
        "override": override,
    }

    post_to_dashboard(
        session=session,
        bearer_token=bearer_token,
        dashboard_url=build_url,
        payload=build_payload,
    )


def add_new_optimum_series(
    session,
    bearer_token: str,
    dana_url: str,
    project_id: str,
    series_id: str,
    average_range: int,
    average_min_count: int,
    series_description: Optional[str] = None,
    better_criterion: str = "lower",
    override: bool = False,
) -> None:
    series_url = f"{dana_url}/apis/addSerie"
    series_payload = {
        "projectId": project_id,
        "serieId": series_id,
        "analyse": {
            "benchmark": {
                "range": average_range,
                "required": average_min_count,
                "trend": better_criterion,
            }
        },
        "override": override,
    }

    if series_description is not None:
        series_payload["description"] = series_description

    post_to_dashboard(
        session=session,
        bearer_token=bearer_token,
        dashboard_url=series_url,
        payload=series_payload,
    )


def add_new_sample(
    session,
    bearer_token: str,
    dana_url: str,
    project_id: str,
    build_id: str,
    series_id: str,
    sample_value: int,
    override: bool = False,
) -> None:
    sample_url = f"{dana_url}/apis/addSample"
    sample_payload = {
        "projectId": project_id,
        "serieId": series_id,
        "sample": {"buildId": build_id, "value": sample_value},
        "override": override,
    }

    post_to_dashboard(
        session=session,
        bearer_token=bearer_token,
        dashboard_url=sample_url,
        payload=sample_payload,
    )


def authenticate(session, dana_url: str, username: str, password: str) -> None:
    session.post(
        f"{dana_url}/login",
        data=json.dumps({"username": username, "password": password}),
        headers={"Content-Type": "application/json"},
    )


def post_to_dashboard(
    session,
    bearer_token: str,
    dashboard_url: str,
    payload: Dict[str, Any],
) -> None:
    data = json.dumps(payload)

    response = session.post(
        dashboard_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        },
    )

    code = response.status_code
    LOGGER.info(f"API response code: {code}")

    if code != 200:
        LOGGER.error(f"API response: {response.text}")
        raise RuntimeError("API request failed")
