from __future__ import annotations

from typing import Iterable
from urllib.parse import urlparse


_AMBIGUOUS = object()
_X_HOST_ALIASES = {
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "mobile.twitter.com",
    "mobile.x.com",
}


def normalize_x_url(url: str | None) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host not in _X_HOST_ALIASES:
        return raw.rstrip("/")

    path = parsed.path.rstrip("/") or "/"
    return f"https://x.com{path}"


def collect_post_match_keys(
    url: str | None = None,
    external_id: str | None = None,
    conversation_id: str | None = None,
) -> list[str]:
    keys = []
    seen = set()

    def push(prefix: str, value: str | None):
        value = (value or "").strip()
        if not value:
            return
        key = f"{prefix}:{value}"
        if key not in seen:
            seen.add(key)
            keys.append(key)

    normalized = normalize_x_url(url)
    if normalized:
        push("url", normalized)
        parsed = urlparse(normalized)
        segments = [segment for segment in parsed.path.split("/") if segment]
        if "status" in segments:
            idx = segments.index("status")
            if idx + 1 < len(segments):
                push("status-id", segments[idx + 1])

    push("external", str(external_id or ""))
    push("conversation", str(conversation_id or ""))
    return keys


def build_post_match_index(posts: Iterable[object]) -> dict[str, object]:
    index: dict[str, object] = {}

    for post in posts:
        keys = collect_post_match_keys(
            url=getattr(post, "source_url", None),
            external_id=getattr(post, "external_id", None),
            conversation_id=getattr(post, "external_id", None),
        )
        keys.extend(
            key
            for key in collect_post_match_keys(url=getattr(post, "account_url", None))
            if key not in keys
        )

        for key in keys:
            existing = index.get(key)
            if existing is None:
                index[key] = post
            elif existing is not post:
                index[key] = _AMBIGUOUS

    return index


def find_post_match(
    index: dict[str, object],
    url: str | None = None,
    external_id: str | None = None,
    conversation_id: str | None = None,
):
    for key in collect_post_match_keys(url=url, external_id=external_id, conversation_id=conversation_id):
        match = index.get(key)
        if match is not None and match is not _AMBIGUOUS:
            return match
    return None
