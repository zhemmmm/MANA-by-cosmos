"""
backfill_priority.py — Retroactively applies the new Priority and Sarcasm 
logic to all existing posts in the database.

Zero-Blast Radius:
- It only modifies the `Post.priority` column (High/Medium/Low).
- It uses the exact same functions the main pipeline uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the backend folder to sys.path so we can import everything normally
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from app import app
from models import Post, PostPriority, PostSentiment, PostTopic, db
from services.priority.priority_scorer import assign_priority_label, compute_priority_score
from services.vader.sentiment_analyzer import analyze_post

def run_backfill():
    with app.app_context():
        print("Fetching all posts from the database...")
        posts = Post.query.all()
        
        updated_count = 0
        sarcasm_flags_found = 0
        
        print(f"Found {len(posts)} posts. Beginning retroactive update...\n")
        
        for post in posts:
            # 1. Check for sarcasm using the thesis rules
            sentiment_result = analyze_post(post.caption, post.cluster_id)
            sarcasm_flag = sentiment_result.get("sarcasm_flag", False)
            if sarcasm_flag:
                sarcasm_flags_found += 1
            
            # 2. Get the existing Random Forest probabilities (if any)
            rf_probs = None
            if post.rf_priority:
                rf_probs = {
                    "High": post.rf_priority.high_probability,
                    "Medium": post.rf_priority.medium_probability,
                    "Low": post.rf_priority.low_probability
                }
                
            # 3. Get the number of CorEx topics matched
            topic_count = PostTopic.query.filter_by(post_id=post.id).count()
            
            # 4. Compute the newly upgraded 0-100% hybrid priority score
            score_0_100 = compute_priority_score(
                post=post,
                db_session=db.session,
                rf_probabilities=rf_probs,
                topic_count=topic_count,
                sarcasm_flag=sarcasm_flag,
                exaggeration_score=0.0  # Kept at 0 per current system architecture
            )
            
            # 5. Convert 0-100 to High/Medium/Low string
            new_label = assign_priority_label(score_0_100)
            
            # 6. Update the Post row (Zero-Blast Radius)
            if post.priority != new_label:
                # print(f"Post [{post.id}]: Priority updated from '{post.priority}' to '{new_label}' (Score: {score_0_100}%)")
                post.priority = new_label
                updated_count += 1
                
        # Commit all changes at once
        db.session.commit()
        
        print("\n=== Backfill Complete ===")
        print(f"Total posts checked: {len(posts)}")
        print(f"Sarcasm posts flagged: {sarcasm_flags_found}")
        print(f"Posts that changed priority level: {updated_count}")

if __name__ == "__main__":
    run_backfill()
