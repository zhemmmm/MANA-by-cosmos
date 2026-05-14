"""
Shared label-refinement helper.

Used by:
  - backend/routes/pipeline.py  (full pipeline, runs over all posts)
  - backend/import_facebook_dataset.py  (per-Apify-fetch, runs over new posts)

The function below does the two passes that turn heuristic/low-signal labels into
trustworthy ones:

  1. CorEx strong-signal relabel — when unambiguous disaster terms are present in
     the post's tokens, force the cluster to match. Never overwrites a label
     that was set by a human reviewer.

  2. SVM predict — emits a PostCluster row per (post, cluster) pair, and
     updates Post.cluster_id when the SVM's top prediction is confident enough,
     subject to the same protection rules.
"""

from __future__ import annotations

from models import Post, PostCluster, PreprocessedText, db
from services.svm.cluster_classifier import (
    is_model_trained as svm_trained,
    predict_clusters_batch,
    select_top_cluster,
)


STRONG_CLUSTER_SIGNALS: dict[str, set[str]] = {
    "cluster-g": {  # rescue — fire and active rescue language
        "fire", "fire alert", "txtfire", "bfp", "firefighter", "blaze",
        "burning", "arson", "fire truck", "fire department",
        "structure fire", "wildfire", "rescue", "trapped", "stranded",
        "sos", "rescue boat", "search and rescue",
    },
    "cluster-h": {  # dead/missing — fatality language
        "fatality", "casualty", "confirmed dead", "body found",
        "death toll", "missing person", "remains identified",
    },
    "cluster-c": {  # evacuation
        "evacuation center", "evacuees", "displaced families",
    },
}


def _corex_relabel(post_ids: list[str], rows: list[PreprocessedText]) -> int:
    """Force-assign cluster when strong domain signals are present in the post.

    Returns the number of posts whose cluster_id was updated. Skips posts whose
    cluster_label_source is 'reviewed' (human-curated labels are sacred).
    """
    from data import TOPIC_TO_CLUSTER, load_corex_expanded_keywords

    corex_kw = load_corex_expanded_keywords()
    if not corex_kw:
        return 0

    cluster_keyword_sets: dict[str, set[str]] = {
        TOPIC_TO_CLUSTER[topic]: {w.lower() for w in words}
        for topic, words in corex_kw.items()
        if topic in TOPIC_TO_CLUSTER
    }

    posts_map = {p.id: p for p in Post.query.filter(Post.id.in_(post_ids)).all()}
    relabeled = 0

    for row in rows:
        post = posts_map.get(row.raw_id)
        if not post or post.cluster_label_source == "reviewed":
            continue

        tokens_set = set(row.final_tokens)
        raw_text = (post.caption or "").lower()

        forced_cluster = None
        for cid, signals in STRONG_CLUSTER_SIGNALS.items():
            for term in signals:
                if " " in term:
                    if term in raw_text:
                        forced_cluster = cid
                        break
                elif term in tokens_set:
                    forced_cluster = cid
                    break
            if forced_cluster:
                break

        if forced_cluster:
            if post.cluster_id != forced_cluster:
                post.cluster_id = forced_cluster
                post.cluster_label_source = "corex_enriched"
                relabeled += 1
            continue

        if not cluster_keyword_sets:
            continue
        scores = {
            cluster_id: len(tokens_set & kw_set)
            for cluster_id, kw_set in cluster_keyword_sets.items()
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_cluster_id, best_score = sorted_scores[0]
        runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        if (
            best_score > 0
            and (best_score - runner_up_score) >= 2
            and post.cluster_id != best_cluster_id
        ):
            post.cluster_id = best_cluster_id
            post.cluster_label_source = "corex_enriched"
            relabeled += 1

    return relabeled


def _svm_predict(post_ids: list[str], texts: list[str]) -> dict:
    """Run SVM prediction over the given posts and update labels.

    Inserts a fresh PostCluster row per (post, cluster) prediction, replacing
    any existing rows for those posts. Updates Post.cluster_id only when the
    SVM's top prediction clears the (loosened) confidence + margin thresholds
    AND the existing label_source is not in the protected set
    {'reviewed', 'corex_enriched'}.
    """
    posts_map = {p.id: p for p in Post.query.filter(Post.id.in_(post_ids)).all()}
    batch_clusters = predict_clusters_batch(texts)
    inserted = skipped = updated = 0

    for post_id, cluster_list in zip(post_ids, batch_clusters):
        if not cluster_list:
            skipped += 1
            continue
        PostCluster.query.filter_by(post_id=post_id).delete()
        for item in cluster_list:
            db.session.add(PostCluster(
                post_id=post_id,
                cluster_id=item["cluster_id"],
                confidence=item["confidence"],
            ))
            inserted += 1
        top_cluster = select_top_cluster(cluster_list)
        post = posts_map.get(post_id)
        if post and top_cluster and post.cluster_id != top_cluster["cluster_id"]:
            if post.cluster_label_source in ("reviewed", "corex_enriched"):
                continue
            post.cluster_id = top_cluster["cluster_id"]
            post.cluster_label_source = "svm"
            updated += 1

    return {
        "cluster_rows_inserted": inserted,
        "posts_skipped_no_clusters": skipped,
        "post_cluster_id_updated": updated,
    }


def refine_labels(post_ids: list[str]) -> dict:
    """Refine the cluster_id for the given posts using CorEx signals + SVM.

    Loads the preprocessed tokens for the given post IDs, runs the CorEx
    strong-signal relabel pass, then (if the SVM is trained) the SVM
    prediction pass. Caller is responsible for db.session.commit().

    Returns a metrics dict — never raises for "no posts" / "no SVM"; just
    reports what it did.
    """
    if not post_ids:
        return {"posts_processed": 0, "skipped": "no post_ids"}

    rows = (
        PreprocessedText.query
        .filter(PreprocessedText.raw_id.in_(post_ids))
        .filter_by(preprocessing_status="processed", is_relevant=True, record_type="post")
        .filter(PreprocessedText.final_tokens_json != "[]")
        .all()
    )
    if not rows:
        return {"posts_processed": 0, "skipped": "no preprocessed rows"}

    ordered_ids = [row.raw_id for row in rows]
    texts = [" ".join(row.final_tokens) for row in rows]

    metrics: dict = {"posts_processed": len(rows)}
    metrics["corex_relabeled"] = _corex_relabel(ordered_ids, rows)
    db.session.flush()

    if svm_trained():
        metrics.update(_svm_predict(ordered_ids, texts))
    else:
        metrics["svm_skipped"] = "model not trained"
    db.session.flush()

    return metrics
