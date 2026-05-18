from __future__ import annotations

import os
from typing import Any

APIFY_FB_ACTOR_TASK_ID_ENV = "APIFY_FB_ACTOR_TASK_ID"
APIFY_FB_GROUP_ACTOR_TASK_ID_ENV = "APIFY_FB_GROUP_ACTOR_TASK_ID"
APIFY_X_ACTOR_TASK_ID_ENV = "APIFY_X_ACTOR_TASK_ID"
APIFY_TOKEN_ENV = "APIFY_TOKEN"
APIFY_WEBHOOK_SECRET_ENV = "APIFY_WEBHOOK_SECRET"

KIND_FACEBOOK = "facebook"
KIND_FACEBOOK_GROUP = "facebook_group"
KIND_X = "x"
VALID_KINDS = {KIND_FACEBOOK, KIND_FACEBOOK_GROUP, KIND_X}


def require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is not configured.")
    return value


def get_client():
    try:
        from apify_client import ApifyClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "apify-client is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    return ApifyClient(require_env(APIFY_TOKEN_ENV))


def get_run(run_id: str) -> dict[str, Any]:
    run = get_client().run(run_id).get()
    if not run:
        raise RuntimeError(f"Apify run {run_id} was not found.")
    return run


def get_task_id(kind: str) -> str:
    if kind == KIND_FACEBOOK:
        return require_env(APIFY_FB_ACTOR_TASK_ID_ENV)
    if kind == KIND_FACEBOOK_GROUP:
        value = (os.environ.get(APIFY_FB_GROUP_ACTOR_TASK_ID_ENV) or "").strip()
        if value:
            return value
        return require_env(APIFY_FB_ACTOR_TASK_ID_ENV)
    if kind == KIND_X:
        return require_env(APIFY_X_ACTOR_TASK_ID_ENV)
    raise RuntimeError(f"Unsupported Apify import kind: {kind}")



def get_webhook_secret() -> str:
    return require_env(APIFY_WEBHOOK_SECRET_ENV)


def list_dataset_items(dataset_id: str) -> list[dict[str, Any]]:
    dataset = get_client().dataset(dataset_id)
    result = dataset.list_items(clean=True)
    items = getattr(result, "items", None)
    if items is None and isinstance(result, dict):
        items = result.get("items")
    return list(items or [])


def import_dataset_items(kind: str, dataset_id: str):
    items = list_dataset_items(dataset_id)

    if kind not in VALID_KINDS:
        raise RuntimeError(f"Unsupported Apify import kind: {kind}")

    posts = [item for item in items if item.get("_recordType") == "post"]
    comments = [item for item in items if item.get("_recordType") == "comment"]

    if kind in {KIND_FACEBOOK, KIND_FACEBOOK_GROUP}:
        from import_facebook_dataset import import_items as import_post_items
        from import_facebook_comments_dataset import import_items as import_comment_items
    else:
        from import_x_dataset import import_items as import_post_items
        from import_x_comments_dataset import import_items as import_comment_items

    post_summary = import_post_items(posts) if posts else {"total_records_loaded": 0}
    comment_summary = import_comment_items(comments) if comments else {"total_records_loaded": 0}

    return {
        "kind": kind,
        "dataset_id": dataset_id,
        "item_count": len(items),
        "post_count": len(posts),
        "comment_count": len(comments),
        "summary": {
            "posts": post_summary,
            "comments": comment_summary,
        },
    }


def build_ad_hoc_webhook(request_url: str, kind: str, secret: str) -> dict[str, Any]:
    payload = (
        '{'
        f'"kind":"{kind}",'
        f'"secret":"{secret}",'
        '"resource":{{resource}}'
        '}'
    )
    return {
        "event_types": ["ACTOR.RUN.SUCCEEDED"],
        "request_url": request_url,
        "payload_template": payload,
    }


def start_task(kind: str, *, webhook_url: str | None = None, task_input: dict[str, Any] | None = None):
    task_id = get_task_id(kind)
    task_client = get_client().task(task_id)
    webhooks = None
    if webhook_url:
        webhooks = [build_ad_hoc_webhook(webhook_url, kind, get_webhook_secret())]
    run = task_client.start(task_input=task_input or None, webhooks=webhooks)
    return {
        "kind": kind,
        "task_id": task_id,
        "run_id": run.get("id"),
        "status": run.get("status"),
        "default_dataset_id": run.get("defaultDatasetId"),
    }


def extract_dataset_id(payload: dict[str, Any]) -> str | None:
    for key in ("datasetId", "defaultDatasetId"):
        value = payload.get(key)
        if value:
            return str(value)

    resource = payload.get("resource") or {}
    if isinstance(resource, dict):
        for key in ("defaultDatasetId", "datasetId"):
            value = resource.get(key)
            if value:
                return str(value)
    return None


def resolve_dataset_id(payload: dict[str, Any]) -> str | None:
    dataset_id = extract_dataset_id(payload)
    if dataset_id:
        return dataset_id

    resource = payload.get("resource") or {}
    if isinstance(resource, dict):
        run_id = resource.get("id")
        if run_id:
            run = get_run(str(run_id))
            for key in ("defaultDatasetId", "datasetId"):
                value = run.get(key)
                if value:
                    return str(value)
    return None


def extract_kind(payload: dict[str, Any]) -> str | None:
    kind = payload.get("kind")
    if isinstance(kind, str) and kind in VALID_KINDS:
        return kind
    return None


def validate_webhook_secret(secret: str | None) -> bool:
    expected = os.environ.get(APIFY_WEBHOOK_SECRET_ENV)
    return bool(expected) and secret == expected
