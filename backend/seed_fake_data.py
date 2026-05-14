"""
seed_fake_data.py — Populate the MANA database with 50 realistic Filipino disaster posts
and run the full ML pipeline end-to-end.

Usage:
    cd backend
    python seed_fake_data.py

What it does:
1.  Creates 50 fake social media posts (English + Tagalog + Taglish) covering all 8 NDRRMC clusters
2.  Inserts them through normalize_item() (same path as real Apify imports)
3.  Runs text preprocessing on every post
4.  Trains CorEx topic model on the preprocessed corpus
5.  Runs CorEx topic inference → writes to post_topics
6.  Trains SVM cluster classifier on the labeled corpus
7.  Runs SVM cluster inference → writes to post_clusters
8.  Runs VADER sentiment analysis → writes to sentiments
9.  Trains Random Forest priority classifier (uses Post.priority as bootstrap labels)
10. Runs RF priority inference → writes to post_priorities, updates Post.priority
11. Prints a stage-by-stage summary with pass/fail indicators
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from app import app, ensure_database
from data import (
    PRIORITY_ORDER,
    extract_location,
    infer_cluster,
    infer_priority,
    infer_sentiment_score,
    media_type_for,
    recommendation_payload_for,
)
from models import Post, PostCluster, PostPriority, PostSentiment, PostTopic, PreprocessedText, db
from preprocessing import save_preprocessed_text
from services.corex.topic_modeler import (
    is_model_trained as corex_trained,
    predict_topics_batch,
    train_corex,
)
from services.svm.cluster_classifier import (
    is_model_trained as svm_trained,
    predict_clusters_batch,
    select_top_cluster,
    train_svm,
)
from services.vader.sentiment_analyzer import analyze_post
from services.random_forest.priority_classifier import (
    SEVERITY_MAP,
    predict_priorities_batch,
    train_rf,
)

# ── Fake dataset — 50 posts across 8 NDRRMC clusters ─────────────────────────
# Format matches Apify Facebook Posts Scraper output so normalize_item() works as-is.

def _ts(days_ago: int, hour: int = 9) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()


FAKE_POSTS = [
    # ── Cluster A — Food and Non-food Items (NFIs) ────────────────────────────
    {
        "postId": "fake-a-001",
        "text": "Relief goods distribution at Barangay San Miguel. Food packs, rice, and water refill containers available. Please bring your registration card.",
        "pageName": "NDRRMC Official",
        "url": "https://www.facebook.com/ndrrmc/posts/fake-a-001",
        "time": _ts(1, 10),
        "likes": 340, "comments": 55, "shares": 42, "topReactionsCount": 340, "viewsCount": 2100,
    },
    {
        "postId": "fake-a-002",
        "text": "NDRRMC nangunguna sa pagbibigay ng rice packs at blankets sa mga evacuees sa Marikina. Sana makarating sa lahat ng pamilya.",
        "pageName": "Balita Ngayon PH",
        "url": "https://www.facebook.com/balita/posts/fake-a-002",
        "time": _ts(1, 11),
        "likes": 280, "comments": 40, "shares": 35, "topReactionsCount": 280, "viewsCount": 1800,
    },
    {
        "postId": "fake-a-003",
        "text": "Urgent: Hygiene kits and food pack needed at evacuation sites in Quezon City. Supplies are running low. Please donate or volunteer.",
        "pageName": "QC Disaster Response",
        "url": "https://www.facebook.com/qcdr/posts/fake-a-003",
        "time": _ts(1, 13),
        "likes": 420, "comments": 88, "shares": 67, "topReactionsCount": 420, "viewsCount": 3200,
    },
    {
        "postId": "fake-a-004",
        "text": "Family of 5 nangangailangan ng food pack at malinis na tubig. Nasa evacuation center kami sa Pasig. Sana may makatulong po sa amin.",
        "pageName": "Pasig Emergency Updates",
        "url": "https://www.facebook.com/pasig/posts/fake-a-004",
        "time": _ts(2, 8),
        "likes": 180, "comments": 66, "shares": 22, "topReactionsCount": 180, "viewsCount": 900,
    },
    {
        "postId": "fake-a-005",
        "text": "Relief goods convoy on the way to San Jose. Rice, canned goods, and water refill containers included. ETA 2 hours.",
        "pageName": "DSWD Relief Ops",
        "url": "https://www.facebook.com/dswd/posts/fake-a-005",
        "time": _ts(2, 14),
        "likes": 195, "comments": 30, "shares": 28, "topReactionsCount": 195, "viewsCount": 1100,
    },
    {
        "postId": "fake-a-006",
        "text": "Volunteers distributing hygiene kit and blanket at the covered court evacuation center. Food pack distribution follows at 3PM. #ReliefOps",
        "pageName": "Marikina Relief Team",
        "url": "https://www.facebook.com/marikina/posts/fake-a-006",
        "time": _ts(2, 15),
        "likes": 310, "comments": 44, "shares": 38, "topReactionsCount": 310, "viewsCount": 1900,
    },
    {
        "postId": "fake-a-007",
        "text": "Kailangan namin ng food pack at hygiene kit dito sa aming evacuation center. Maraming pamilya ang naghihirap. Salamat sa tulong ng lahat.",
        "pageName": "Community Updates PH",
        "url": "https://www.facebook.com/comm/posts/fake-a-007",
        "time": _ts(3, 9),
        "likes": 155, "comments": 38, "shares": 18, "topReactionsCount": 155, "viewsCount": 800,
    },

    # ── Cluster B — WASH, Medical and Public Health ───────────────────────────
    {
        "postId": "fake-b-001",
        "text": "Medical team has arrived at the evacuation center. Patients with fever and dehydration are being attended to. Doctor on duty until 10PM.",
        "pageName": "DOH Philippines",
        "url": "https://www.facebook.com/doh/posts/fake-b-001",
        "time": _ts(1, 9),
        "likes": 390, "comments": 72, "shares": 55, "topReactionsCount": 390, "viewsCount": 2800,
    },
    {
        "postId": "fake-b-002",
        "text": "URGENT: Insulin supplies depleted at Pasig General Hospital. Diabetes patients at the evacuation site urgently need medicine. Medical team needed.",
        "pageName": "Pasig Health Alert",
        "url": "https://www.facebook.com/pasig-health/posts/fake-b-002",
        "time": _ts(1, 12),
        "likes": 510, "comments": 120, "shares": 88, "topReactionsCount": 510, "viewsCount": 4500,
    },
    {
        "postId": "fake-b-003",
        "text": "Doctor at the rescue center requesting additional medical team support. Many evacuees showing signs of dehydration and fever. Respond immediately.",
        "pageName": "Manila Rescue Update",
        "url": "https://www.facebook.com/manila/posts/fake-b-003",
        "time": _ts(1, 14),
        "likes": 435, "comments": 95, "shares": 70, "topReactionsCount": 435, "viewsCount": 3600,
    },
    {
        "postId": "fake-b-004",
        "text": "Maraming naghihirap sa washing area. Walang sapat na tubig para sa kalinisan. Kailangan ng medical team dito sa evacuation site.",
        "pageName": "Brgy San Andres Updates",
        "url": "https://www.facebook.com/bsa/posts/fake-b-004",
        "time": _ts(2, 10),
        "likes": 225, "comments": 48, "shares": 32, "topReactionsCount": 225, "viewsCount": 1400,
    },
    {
        "postId": "fake-b-005",
        "text": "Water safety alert: Flood waters in Marikina are contaminated. Do not drink untreated water. Dehydration cases rising. Doctor advisory posted.",
        "pageName": "Marikina Health Office",
        "url": "https://www.facebook.com/mrh/posts/fake-b-005",
        "time": _ts(2, 11),
        "likes": 470, "comments": 80, "shares": 95, "topReactionsCount": 470, "viewsCount": 3900,
    },
    {
        "postId": "fake-b-006",
        "text": "Mental health and psychosocial support team deployed to evacuation sites. Medical team available at tent 3. Nutrition packs for children now distributed.",
        "pageName": "DSWD Psychosocial",
        "url": "https://www.facebook.com/dswd-psych/posts/fake-b-006",
        "time": _ts(3, 8),
        "likes": 300, "comments": 42, "shares": 38, "topReactionsCount": 300, "viewsCount": 2000,
    },

    # ── Cluster C — Camp Coordination, Management and Protection ──────────────
    {
        "postId": "fake-c-001",
        "text": "Evacuation center at covered court is at overcapacity. More than 500 families registered. Additional safe space urgently needed.",
        "pageName": "Marikina CDRRMO",
        "url": "https://www.facebook.com/cdrrmo/posts/fake-c-001",
        "time": _ts(1, 8),
        "likes": 620, "comments": 130, "shares": 110, "topReactionsCount": 620, "viewsCount": 5200,
    },
    {
        "postId": "fake-c-002",
        "text": "Evacuation center registration now ongoing at Barangay Hall. Bring valid ID. Toilet line is long — additional porta-potties being requested.",
        "pageName": "QC DRRMO",
        "url": "https://www.facebook.com/qcdrrmo/posts/fake-c-002",
        "time": _ts(1, 10),
        "likes": 280, "comments": 65, "shares": 45, "topReactionsCount": 280, "viewsCount": 1800,
    },
    {
        "postId": "fake-c-003",
        "text": "Evacuation center registration desk is overwhelmed. Privacy concerns raised as families are packed in tight spaces. Camp management team deployed.",
        "pageName": "Rescue PH Updates",
        "url": "https://www.facebook.com/rescue/posts/fake-c-003",
        "time": _ts(1, 11),
        "likes": 345, "comments": 78, "shares": 52, "topReactionsCount": 345, "viewsCount": 2400,
    },
    {
        "postId": "fake-c-004",
        "text": "Safe space for women and children available at evacuation center. Registration ongoing at gate 2. Please bring all family members.",
        "pageName": "DSWD Protection",
        "url": "https://www.facebook.com/dswd-prot/posts/fake-c-004",
        "time": _ts(2, 9),
        "likes": 255, "comments": 55, "shares": 40, "topReactionsCount": 255, "viewsCount": 1600,
    },
    {
        "postId": "fake-c-005",
        "text": "Ang evacuation center sa Marikina ay puno na. Hinihiling ng mga residente ang karagdagang lugar para sa mga evacuees. Overcapacity na po.",
        "pageName": "Marikina Balita",
        "url": "https://www.facebook.com/marikina-b/posts/fake-c-005",
        "time": _ts(2, 13),
        "likes": 310, "comments": 70, "shares": 48, "topReactionsCount": 310, "viewsCount": 2100,
    },
    {
        "postId": "fake-c-006",
        "text": "NDRRMC monitoring the overcapacity situation at evacuation centers. Overflow sites being identified. Camp registration to continue tomorrow.",
        "pageName": "NDRRMC Updates",
        "url": "https://www.facebook.com/ndrrmc-up/posts/fake-c-006",
        "time": _ts(3, 7),
        "likes": 415, "comments": 88, "shares": 72, "topReactionsCount": 415, "viewsCount": 3300,
    },
    {
        "postId": "fake-c-007",
        "text": "Toilet line at evacuation center is 30 minutes long. Safe space for elderly still inadequate. More facilities requested from city government.",
        "pageName": "Community Watch PH",
        "url": "https://www.facebook.com/cw/posts/fake-c-007",
        "time": _ts(3, 9),
        "likes": 195, "comments": 50, "shares": 28, "topReactionsCount": 195, "viewsCount": 1100,
    },

    # ── Cluster D — Logistics ─────────────────────────────────────────────────
    {
        "postId": "fake-d-001",
        "text": "Blocked road in Antipolo due to landslide. Convoy rerouted via SLEX. Delivery of relief goods delayed by 3 hours.",
        "pageName": "DPWH Road Update",
        "url": "https://www.facebook.com/dpwh/posts/fake-d-001",
        "time": _ts(1, 7),
        "likes": 295, "comments": 60, "shares": 50, "topReactionsCount": 295, "viewsCount": 2000,
    },
    {
        "postId": "fake-d-002",
        "text": "Warehouse in Pasig flooded. Emergency reroute of supplies to alternate hub. Truck convoy on standby at NLEX exit.",
        "pageName": "OCD Logistics",
        "url": "https://www.facebook.com/ocd/posts/fake-d-002",
        "time": _ts(1, 11),
        "likes": 210, "comments": 44, "shares": 35, "topReactionsCount": 210, "viewsCount": 1400,
    },
    {
        "postId": "fake-d-003",
        "text": "DPWH convoy carrying relief goods blocked at Marcos Highway due to road damage. Delivery reroute to alternate route now in effect.",
        "pageName": "DPWH NCR",
        "url": "https://www.facebook.com/dpwh-ncr/posts/fake-d-003",
        "time": _ts(2, 8),
        "likes": 265, "comments": 55, "shares": 42, "topReactionsCount": 265, "viewsCount": 1700,
    },
    {
        "postId": "fake-d-004",
        "text": "Delivery of food packs suspended due to blocked road in Montalban. Trucks waiting for clearance. Convoy update to follow at 6PM.",
        "pageName": "Montalban Updates",
        "url": "https://www.facebook.com/montalban/posts/fake-d-004",
        "time": _ts(2, 13),
        "likes": 180, "comments": 38, "shares": 28, "topReactionsCount": 180, "viewsCount": 1000,
    },
    {
        "postId": "fake-d-005",
        "text": "Logistics team requesting updated road map. Several routes blocked by debris. Convoy movement delayed until road assessment completed.",
        "pageName": "OCD Operations",
        "url": "https://www.facebook.com/ocd-ops/posts/fake-d-005",
        "time": _ts(3, 6),
        "likes": 155, "comments": 32, "shares": 24, "topReactionsCount": 155, "viewsCount": 900,
    },
    {
        "postId": "fake-d-006",
        "text": "Relief goods warehouse report: 2,000 food packs ready for delivery. Blocked road preventing truck dispatch. Reroute clearance awaited.",
        "pageName": "DSWD Logistics Hub",
        "url": "https://www.facebook.com/dswd-log/posts/fake-d-006",
        "time": _ts(3, 10),
        "likes": 200, "comments": 45, "shares": 32, "topReactionsCount": 200, "viewsCount": 1200,
    },

    # ── Cluster E — Emergency Telecommunications ──────────────────────────────
    {
        "postId": "fake-e-001",
        "text": "Signal down in most of Marikina since early morning. No network available in affected barangays. Cell site repair team deployed.",
        "pageName": "PLDT PH Updates",
        "url": "https://www.facebook.com/pldt/posts/fake-e-001",
        "time": _ts(1, 6),
        "likes": 380, "comments": 90, "shares": 65, "topReactionsCount": 380, "viewsCount": 3100,
    },
    {
        "postId": "fake-e-002",
        "text": "Power bank distribution ongoing at evacuation centers. Radio communication active for all NDRRMC units. Connectivity restoration in progress.",
        "pageName": "NDRRMC Comms",
        "url": "https://www.facebook.com/ndrrmc-c/posts/fake-e-002",
        "time": _ts(1, 9),
        "likes": 245, "comments": 50, "shares": 38, "topReactionsCount": 245, "viewsCount": 1600,
    },
    {
        "postId": "fake-e-003",
        "text": "Walang signal at kuryente sa aming barangay. Hindi kami makahingi ng tulong. Cell site hindi pa nai-restore. Sana marinig kami.",
        "pageName": "Brgy Tumana Residents",
        "url": "https://www.facebook.com/tumana/posts/fake-e-003",
        "time": _ts(1, 10),
        "likes": 420, "comments": 105, "shares": 88, "topReactionsCount": 420, "viewsCount": 3600,
    },
    {
        "postId": "fake-e-004",
        "text": "Emergency radio channel activated for Marikina operations. No network in affected areas. Connectivity update expected at 6PM.",
        "pageName": "Marikina DRRMO",
        "url": "https://www.facebook.com/mrh-drrmo/posts/fake-e-004",
        "time": _ts(2, 7),
        "likes": 310, "comments": 62, "shares": 48, "topReactionsCount": 310, "viewsCount": 2200,
    },
    {
        "postId": "fake-e-005",
        "text": "PLDT signal down restoration team deployed to affected areas. Power bank units being distributed to priority evacuation sites.",
        "pageName": "PLDT Service PH",
        "url": "https://www.facebook.com/pldt-svc/posts/fake-e-005",
        "time": _ts(2, 12),
        "likes": 265, "comments": 55, "shares": 40, "topReactionsCount": 265, "viewsCount": 1700,
    },
    {
        "postId": "fake-e-006",
        "text": "Radio is the only working connectivity in flood-affected zones. Cell site signal down across 3 barangays. Repair expected within 24 hours.",
        "pageName": "Emergency Comms PH",
        "url": "https://www.facebook.com/ecomms/posts/fake-e-006",
        "time": _ts(3, 8),
        "likes": 290, "comments": 60, "shares": 45, "topReactionsCount": 290, "viewsCount": 2000,
    },

    # ── Cluster F — Education ─────────────────────────────────────────────────
    {
        "postId": "fake-f-001",
        "text": "DepEd announces class suspension in all public schools in Marikina due to flooding. Students advised to stay home until further notice.",
        "pageName": "DepEd NCR",
        "url": "https://www.facebook.com/deped/posts/fake-f-001",
        "time": _ts(1, 5),
        "likes": 850, "comments": 200, "shares": 185, "topReactionsCount": 850, "viewsCount": 8000,
    },
    {
        "postId": "fake-f-002",
        "text": "School closure extended for another 2 days in Quezon City. Learning materials distribution for affected students postponed. DepEd update follows.",
        "pageName": "DepEd QC Division",
        "url": "https://www.facebook.com/deped-qc/posts/fake-f-002",
        "time": _ts(2, 6),
        "likes": 620, "comments": 145, "shares": 130, "topReactionsCount": 620, "viewsCount": 5500,
    },
    {
        "postId": "fake-f-003",
        "text": "Temporary classroom set up at Barangay Hall for displaced students. Class suspension still in effect for lower ground areas.",
        "pageName": "Brgy Education Watch",
        "url": "https://www.facebook.com/edwatch/posts/fake-f-003",
        "time": _ts(2, 9),
        "likes": 310, "comments": 70, "shares": 55, "topReactionsCount": 310, "viewsCount": 2200,
    },
    {
        "postId": "fake-f-004",
        "text": "DepEd distributing learning materials to families in evacuation centers. School closure continues. Alternative learning kits for students deployed.",
        "pageName": "DepEd Disaster Response",
        "url": "https://www.facebook.com/deped-dr/posts/fake-f-004",
        "time": _ts(3, 7),
        "likes": 280, "comments": 58, "shares": 45, "topReactionsCount": 280, "viewsCount": 1900,
    },
    {
        "postId": "fake-f-005",
        "text": "Class suspension announced in 15 municipalities. Students without devices given printed learning materials. Schools used as evacuation centers.",
        "pageName": "Regional School Info",
        "url": "https://www.facebook.com/rsi/posts/fake-f-005",
        "time": _ts(3, 10),
        "likes": 490, "comments": 98, "shares": 82, "topReactionsCount": 490, "viewsCount": 4000,
    },
    {
        "postId": "fake-f-006",
        "text": "Temporary classroom at Brgy San Miguel now open. Students may proceed with learning materials provided by DepEd volunteers.",
        "pageName": "Brgy San Miguel Info",
        "url": "https://www.facebook.com/bsm/posts/fake-f-006",
        "time": _ts(4, 8),
        "likes": 195, "comments": 40, "shares": 30, "topReactionsCount": 195, "viewsCount": 1200,
    },

    # ── Cluster G — Search, Rescue and Retrieval (SRR) ───────────────────────
    {
        "postId": "fake-g-001",
        "text": "SOS! Family stranded on rooftop at Brgy San Andres. Rescue boat needed urgently. 3 children and 2 elderly trapped. Please respond immediately.",
        "pageName": "Emergency PH",
        "url": "https://www.facebook.com/eph/posts/fake-g-001",
        "time": _ts(0, 8),
        "likes": 1200, "comments": 350, "shares": 280, "topReactionsCount": 1200, "viewsCount": 12000,
    },
    {
        "postId": "fake-g-002",
        "text": "URGENT: Rescue boat needed at Purok 5 Marikina. Family of 7 trapped on 2nd floor. Water still rising. Please send rescue team now.",
        "pageName": "Marikina SOS",
        "url": "https://www.facebook.com/msos/posts/fake-g-002",
        "time": _ts(0, 9),
        "likes": 980, "comments": 290, "shares": 240, "topReactionsCount": 980, "viewsCount": 9500,
    },
    {
        "postId": "fake-g-003",
        "text": "Nastranded kami sa bubong ng aming bahay. Hindi makarating ang rescue team. SOS! Nagtatawag ng tulong. Purok 4 Brgy Sta Elena Marikina.",
        "pageName": "Brgy Sta Elena",
        "url": "https://www.facebook.com/bse/posts/fake-g-003",
        "time": _ts(0, 10),
        "likes": 1500, "comments": 420, "shares": 380, "topReactionsCount": 1500, "viewsCount": 15000,
    },
    {
        "postId": "fake-g-004",
        "text": "Retrieval operation ongoing at Cagayan Valley. 12 residents confirmed rescued. 3 still stranded. SRR team requesting additional rescue boats.",
        "pageName": "Cagayan DRRMO",
        "url": "https://www.facebook.com/cag-drrmo/posts/fake-g-004",
        "time": _ts(1, 6),
        "likes": 680, "comments": 155, "shares": 130, "topReactionsCount": 680, "viewsCount": 6000,
    },
    {
        "postId": "fake-g-005",
        "text": "Search and rescue team deployed to Brgy Tumana. 5 trapped family members located on rooftop. Rescue boat arriving within 30 minutes.",
        "pageName": "Marikina Rescue Ops",
        "url": "https://www.facebook.com/mro/posts/fake-g-005",
        "time": _ts(1, 8),
        "likes": 590, "comments": 130, "shares": 105, "topReactionsCount": 590, "viewsCount": 5200,
    },
    {
        "postId": "fake-g-006",
        "text": "Rescue operation update: All stranded residents in Tumana have been extracted safely. Retrieval of vehicles and belongings ongoing.",
        "pageName": "Marikina DRRMO Update",
        "url": "https://www.facebook.com/mrdu/posts/fake-g-006",
        "time": _ts(1, 14),
        "likes": 820, "comments": 190, "shares": 160, "topReactionsCount": 820, "viewsCount": 7500,
    },
    {
        "postId": "fake-g-007",
        "text": "Rescue boat needed at two-story house on Marcos Highway flood zone. SOS reported via barangay radio. Coordinates forwarded to SRR team.",
        "pageName": "Marikina Flash Updates",
        "url": "https://www.facebook.com/mfu/posts/fake-g-007",
        "time": _ts(2, 7),
        "likes": 450, "comments": 100, "shares": 85, "topReactionsCount": 450, "viewsCount": 4000,
    },

    # ── Cluster H — Management of Dead and Missing (MDM) ─────────────────────
    {
        "postId": "fake-h-001",
        "text": "3 residents reported missing after flash flood in Cagayan. Family tracing now underway. Please contact the coordination desk at City Hall.",
        "pageName": "Cagayan Emergency",
        "url": "https://www.facebook.com/cag-em/posts/fake-h-001",
        "time": _ts(1, 7),
        "likes": 560, "comments": 125, "shares": 100, "topReactionsCount": 560, "viewsCount": 5000,
    },
    {
        "postId": "fake-h-002",
        "text": "Hospital list of identified flood victims now available at the evacuation center registration desk. Family tracing coordination ongoing.",
        "pageName": "OCD Public Info",
        "url": "https://www.facebook.com/ocd-pi/posts/fake-h-002",
        "time": _ts(1, 11),
        "likes": 415, "comments": 90, "shares": 75, "topReactionsCount": 415, "viewsCount": 3700,
    },
    {
        "postId": "fake-h-003",
        "text": "Missing person alert: Jose Dela Cruz, 65 years old, from Brgy Tumana. Last seen near the flood area before rescue operation. Family tracing ongoing.",
        "pageName": "Missing Persons PH",
        "url": "https://www.facebook.com/mpph/posts/fake-h-003",
        "time": _ts(1, 13),
        "likes": 880, "comments": 220, "shares": 195, "topReactionsCount": 880, "viewsCount": 8200,
    },
    {
        "postId": "fake-h-004",
        "text": "Dead and missing coordination desk set up at Marikina City Hall. Bring identification documents for verification. Hospital intake data being cross-checked.",
        "pageName": "Marikina MDM Desk",
        "url": "https://www.facebook.com/mdm/posts/fake-h-004",
        "time": _ts(2, 8),
        "likes": 340, "comments": 75, "shares": 60, "topReactionsCount": 340, "viewsCount": 2900,
    },
    {
        "postId": "fake-h-005",
        "text": "NDRRMC verification process for reported missing persons underway. Hospital list and family tracing coordination desk active at Marikina City Hall.",
        "pageName": "NDRRMC MDM",
        "url": "https://www.facebook.com/ndrrmc-mdm/posts/fake-h-005",
        "time": _ts(2, 10),
        "likes": 295, "comments": 60, "shares": 48, "topReactionsCount": 295, "viewsCount": 2200,
    },

    # ── Taglish sandbox test posts — one per cluster ─────────────────────────
    # Real-world mixed Filipino/English style to validate preprocessing + classification.
    {
        "postId": "test-a-001",
        "text": "Grabe na ang kalagayan dito sa evac center namin sa Marikina. 3 days na kami dito wala pang food pack o relief goods na dumarating. Bata at matanda na halos walang makain. Pls may makapunta dito agad. #ReliefGoods #Marikina #Tulong",
        "pageName": "Community Updates Marikina",
        "url": "https://www.facebook.com/test/posts/test-a-001",
        "time": _ts(0, 7),
        "likes": 580, "comments": 145, "shares": 112, "topReactionsCount": 580, "viewsCount": 4200,
    },
    {
        "postId": "test-b-001",
        "text": "May nagkasakit na dito sa aming lugar!! Yung bata lagnat na at nahihirapan huminga. Ospital lang daw ang solusyon pero naka-block pa rin ang daan. May available bang doctor sa area? Wala kaming gamot dito. Kailangan ng medical team ASAP. #Tulong #Sakuna #DOH",
        "pageName": "Brgy Health Watch",
        "url": "https://www.facebook.com/test/posts/test-b-001",
        "time": _ts(0, 8),
        "likes": 490, "comments": 130, "shares": 95, "topReactionsCount": 490, "viewsCount": 3800,
    },
    {
        "postId": "test-c-001",
        "text": "UPDATE: Puno na talaga ang evacuation center sa covered court ng Brgy. Tumana! 700+ families na dito, overflowing na. Mga bata nasa labas kasi walang lugar sa loob. Registration desk sobrang overwhelmed. Sana magdala ng dagdag na tent. #EvacCenter #Overcrowded #Marikina",
        "pageName": "Brgy Tumana Residents",
        "url": "https://www.facebook.com/test/posts/test-c-001",
        "time": _ts(0, 9),
        "likes": 720, "comments": 195, "shares": 150, "topReactionsCount": 720, "viewsCount": 5800,
    },
    {
        "postId": "test-d-001",
        "text": "DPWH update: Naharang na ang NLEX at Marcos Highway dahil sa pagguho ng lupa sa Montalban. Convoy ng relief goods naka-stranded sa checkpoint. Alternate route via SCTEX lang ang passable ngayon. Mga truck driver mag-reroute na. ETA delayed 4-6 hrs. #Landslide #Convoy #DPWH",
        "pageName": "DPWH Road Updates NCR",
        "url": "https://www.facebook.com/test/posts/test-d-001",
        "time": _ts(0, 6),
        "likes": 310, "comments": 75, "shares": 65, "topReactionsCount": 310, "viewsCount": 2400,
    },
    {
        "postId": "test-e-001",
        "text": "WALANG KURYENTE na sa buong Brgy. San Andres simula kahapon!! Brownout na brownout wala pang balita kung kailan ibabalik. Walang signal din Globe at Smart. Paano na yung mga nangangailangan ng communication?? Pls i-restore na ang cell site namin! #PowerOutage #NoSignal #Brownout",
        "pageName": "Brgy San Andres Updates",
        "url": "https://www.facebook.com/test/posts/test-e-001",
        "time": _ts(0, 5),
        "likes": 650, "comments": 185, "shares": 140, "topReactionsCount": 650, "viewsCount": 5100,
    },
    {
        "postId": "test-f-001",
        "text": "WALANG PASOK bukas!! Official na galing sa DepEd NCR — class suspended sa lahat ng paaralan sa Marikina at QC dahil sa baha. Pati yung modular classes postponed. Ginagamit na ang school bilang evacuation center ngayon. Stay safe mga students at parents! #WalangPasok #ClassSuspended #DepEd",
        "pageName": "DepEd NCR Parents Group",
        "url": "https://www.facebook.com/test/posts/test-f-001",
        "time": _ts(0, 4),
        "likes": 1100, "comments": 280, "shares": 250, "topReactionsCount": 1100, "viewsCount": 9500,
    },
    {
        "postId": "test-g-001",
        "text": "SOS PLEASE HELP KAMI!! Naka-stranded kami sa bubong ng aming bahay dito sa Purok 3 Brgy. Sta. Elena Marikina! 2 pamilya dito, may 4 na bata at isang lola. Pataas pa rin ang tubig. Hindi makarating ang rescue boat!! Tabang tabang!! #SOS #Rescue #Marikina",
        "pageName": "Emergency PH",
        "url": "https://www.facebook.com/test/posts/test-g-001",
        "time": _ts(0, 3),
        "likes": 2100, "comments": 580, "shares": 420, "topReactionsCount": 2100, "viewsCount": 18000,
    },
    {
        "postId": "test-h-001",
        "text": "MISSING si Lola Conching, 72 anyos, mula Brgy. Tumana Marikina. Last seen kahapon ng gabi bago pa lumala ang baha. Kung nakita ninyo siya pls contact kami. Nag-file na kami ng missing person report sa barangay. Family tracing coordination desk bukas sa City Hall. Pakishare! #MissingPerson #FamilyTracing #Marikina",
        "pageName": "Missing Persons PH",
        "url": "https://www.facebook.com/test/posts/test-h-001",
        "time": _ts(0, 10),
        "likes": 1400, "comments": 390, "shares": 340, "topReactionsCount": 1400, "viewsCount": 12000,
    },
]

assert len(FAKE_POSTS) == 58, f"Expected 58 posts, got {len(FAKE_POSTS)}"


# ── Helper: normalize one Apify-style dict → Post fields ─────────────────────

def _parse_iso(value: str):
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def normalize_item(item: dict) -> dict:
    text = (item.get("text") or item.get("caption") or item.get("content") or "").strip()
    cluster, keywords = infer_cluster(text)
    engagement = int(item.get("likes") or 0) + int(item.get("comments") or 0) + int(item.get("shares") or 0)
    priority = infer_priority(text, engagement)
    sentiment_score = infer_sentiment_score(text, engagement)
    post_id = str(item.get("postId") or item.get("url") or item.get("topLevelUrl"))
    return {
        "id": post_id,
        "source": "Facebook",
        "page_source": item.get("pageName") or "Facebook Source",
        "account_url": item.get("facebookUrl") or item.get("inputUrl"),
        "author": (item.get("user") or {}).get("name") or item.get("pageName"),
        "caption": text,
        "source_url": item.get("url") or item.get("topLevelUrl") or item.get("facebookUrl"),
        "external_id": str(item.get("postId") or ""),
        "reactions": int(item.get("topReactionsCount") or item.get("likes") or 0),
        "shares": int(item.get("shares") or 0),
        "likes": int(item.get("likes") or 0),
        "reposts": 0,
        "comments": int(item.get("comments") or 0),
        "views": int(item.get("viewsCount") or 0),
        "media_type": media_type_for(item),
        "priority": priority,
        "sentiment_score": sentiment_score,
        "recommendation": recommendation_payload_for(
            cluster["id"],
            priority,
            sentiment_score=sentiment_score,
            reactions=int(item.get("topReactionsCount") or item.get("likes") or 0),
            likes=int(item.get("likes") or 0),
            comments=int(item.get("comments") or 0),
            shares=int(item.get("shares") or 0),
            post_count=1,
        )["recommendation"],
        "status": "Monitoring",
        "cluster_id": cluster["id"],
        "date": _parse_iso(item.get("time")),
        "keywords_json": json.dumps(keywords),
        "location": extract_location(text),
        "severity_rank": PRIORITY_ORDER[priority],
        "raw_payload_json": json.dumps(item),
    }


# ── Stage functions ───────────────────────────────────────────────────────────

def _step(label: str, ok: bool, detail: str = ""):
    icon = "OK" if ok else "FAIL"
    print(f"  [{icon}] {label}" + (f" - {detail}" if detail else ""))


def run_import(posts: list[dict]) -> dict:
    inserted = updated = preprocessed = skipped = errors = 0
    for item in posts:
        normalized = normalize_item(item)
        post = db.session.get(Post, normalized["id"])
        if post:
            for field, value in normalized.items():
                setattr(post, field, value)
            updated += 1
        else:
            db.session.add(Post(**normalized))
            inserted += 1

        processed_row, result = save_preprocessed_text(
            item=item,
            raw_id=normalized["id"],
            record_type="post",
            fallback_text=normalized["caption"],
        )
        db.session.add(processed_row)
        status = result["preprocessing_status"]
        if status == "processed":
            preprocessed += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1

    db.session.commit()
    return {"inserted": inserted, "updated": updated, "preprocessed": preprocessed,
            "skipped": skipped, "errors": errors}


def run_corex() -> dict:
    rows = (
        PreprocessedText.query
        .filter_by(record_type="post", preprocessing_status="processed", is_relevant=True)
        .filter(PreprocessedText.final_tokens_json != "[]")
        .all()
    )
    texts = [" ".join(r.final_tokens) for r in rows if r.final_tokens]
    result = train_corex(texts)

    topic_rows = 0
    post_ids = [r.raw_id for r in rows]
    batch = predict_topics_batch(texts)
    for post_id, topics in zip(post_ids, batch):
        PostTopic.query.filter_by(post_id=post_id).delete()
        for t in topics:
            db.session.add(PostTopic(post_id=post_id, topic_label=t["topic"], confidence=t["confidence"]))
            topic_rows += 1
    db.session.commit()
    return {**result, "topic_rows": topic_rows}


def run_svm() -> dict:
    rows = (
        db.session.query(PreprocessedText, Post)
        .join(Post, PreprocessedText.raw_id == Post.id)
        .filter(
            PreprocessedText.preprocessing_status == "processed",
            PreprocessedText.is_relevant == True,
            PreprocessedText.final_tokens_json != "[]",
            PreprocessedText.record_type == "post",
        )
        .all()
    )
    texts = [" ".join(pt.final_tokens) for pt, _ in rows]
    labels = [[post.cluster_id] for _, post in rows]
    result = train_svm(texts, labels)

    cluster_rows = 0
    batch = predict_clusters_batch(texts)
    for (pt, post), cluster_list in zip(rows, batch):
        PostCluster.query.filter_by(post_id=pt.raw_id).delete()
        for c in cluster_list:
            db.session.add(PostCluster(post_id=pt.raw_id, cluster_id=c["cluster_id"], confidence=c["confidence"]))
            cluster_rows += 1
        top = select_top_cluster(cluster_list)
        if top:
            post.cluster_id = top["cluster_id"]
            post.cluster_label_source = "svm"
    db.session.commit()
    return {**result, "cluster_rows": cluster_rows}


def run_vader() -> dict:
    posts = Post.query.all()
    pt_map = {
        r.raw_id: (r.vader_text or r.clean_text)
        for r in PreprocessedText.query.filter_by(record_type="post").all()
        if (r.vader_text or r.clean_text)
    }
    inserted = updated = 0
    for post in posts:
        text = pt_map.get(post.id) or post.caption or ""
        result = analyze_post(text, post.cluster_id)
        existing = PostSentiment.query.filter_by(post_id=post.id).first()
        if existing:
            existing.compound = result["compound"]
            existing.positive = result["positive"]
            existing.negative = result["negative"]
            existing.neutral = result["neutral"]
            existing.sarcasm_flag = result["sarcasm_flag"]
            updated += 1
        else:
            db.session.add(PostSentiment(
                post_id=post.id, compound=result["compound"],
                positive=result["positive"], negative=result["negative"],
                neutral=result["neutral"], sarcasm_flag=result["sarcasm_flag"],
            ))
            inserted += 1
        post.sentiment_score = result["sentiment_score"]
        post.sentiment_compound = result["compound"]
    db.session.commit()
    return {"inserted": inserted, "updated": updated}


def run_rf() -> dict:
    from data import recommendation_payload_for

    _RF_TO_REC_PRIORITY = {"High": "Critical", "Medium": "Moderate", "Low": "Monitoring"}

    result = train_rf()

    rows = (
        PreprocessedText.query
        .filter_by(record_type="post", preprocessing_status="processed", is_relevant=True)
        .filter(PreprocessedText.final_tokens_json != "[]")
        .all()
    )
    post_ids = [r.raw_id for r in rows]
    predictions = predict_priorities_batch(post_ids)
    posts_map = {p.id: p for p in Post.query.filter(Post.id.in_(post_ids)).all()}

    inserted = updated = 0
    for pred in predictions:
        pid   = pred["post_id"]
        label = pred["priority"]
        conf  = pred["confidence"]
        probs = pred["probabilities"]
        post  = posts_map.get(pid)
        if not post:
            continue
        post.priority      = label
        post.severity_rank = SEVERITY_MAP.get(label, 2)
        post.recommendation = recommendation_payload_for(
            post.cluster_id,
            _RF_TO_REC_PRIORITY.get(label, "Moderate"),
            sentiment_score=post.sentiment_score,
            reactions=post.reactions,
            likes=post.likes,
            comments=post.comments,
            shares=post.shares,
            reposts=post.reposts,
            post_count=1,
        )["recommendation"]
        existing = PostPriority.query.filter_by(post_id=pid).first()
        if existing:
            existing.priority_label     = label
            existing.confidence         = conf
            existing.high_probability   = probs.get("High",   0.0)
            existing.medium_probability = probs.get("Medium", 0.0)
            existing.low_probability    = probs.get("Low",    0.0)
            updated += 1
        else:
            db.session.add(PostPriority(
                post_id=pid,
                priority_label=label,
                confidence=conf,
                high_probability=probs.get("High",   0.0),
                medium_probability=probs.get("Medium", 0.0),
                low_probability=probs.get("Low",    0.0),
            ))
            inserted += 1
    db.session.commit()

    dist = {lbl: sum(1 for p in predictions if p["priority"] == lbl) for lbl in ["High", "Medium", "Low"]}
    return {**result, "priority_rows": inserted + updated, "distribution": dist}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== MANA Pipeline Seed Test ===\n")
    ensure_database()

    with app.app_context():
        # ── Stage 1: Import + Preprocessing ───────────────────────────────────
        print("Stage 1 — Import & Preprocessing")
        r = run_import(FAKE_POSTS)
        _step("Posts inserted", r["inserted"] > 0, f"{r['inserted']} new, {r['updated']} updated")
        _step("Preprocessed", r["preprocessed"] > 0, f"{r['preprocessed']} processed, {r['skipped']} skipped, {r['errors']} errors")

        relevant = PreprocessedText.query.filter_by(
            record_type="post", preprocessing_status="processed", is_relevant=True
        ).filter(PreprocessedText.final_tokens_json != "[]").count()
        _step("Relevant posts in DB", relevant >= 10, f"{relevant} posts ready for ML")

        # ── Stage 2: CorEx ────────────────────────────────────────────────────
        print("\nStage 2 — CorEx Topic Modeling")
        try:
            corex_r = run_corex()
            _step("Model trained", True, f"corpus={corex_r['corpus_size']}, coherence={corex_r['overall_coherence']:.3f}")
            _step("Topics predicted", corex_r["topic_rows"] > 0, f"{corex_r['topic_rows']} topic rows written")
            if corex_r.get("low_coherence_topics"):
                _step("Low-coherence topics", False, f"Flag: {corex_r['low_coherence_topics']} — consider more training data")
        except Exception as exc:
            _step("CorEx failed", False, str(exc))

        # ── Stage 3: SVM ──────────────────────────────────────────────────────
        print("\nStage 3 — SVM Cluster Classification")
        try:
            svm_r = run_svm()
            _step("Model trained", True, f"corpus={svm_r['corpus_size']}, best_C={svm_r['best_C']}, f1_macro={svm_r['f1_macro']:.3f}")
            _step("F1 target met", svm_r["f1_macro"] >= 0.75, f"{svm_r['f1_macro']:.3f} (target >= 0.75)")
            _step("Clusters predicted", svm_r["cluster_rows"] > 0, f"{svm_r['cluster_rows']} cluster rows written")
        except Exception as exc:
            _step("SVM failed", False, str(exc))

        # ── Stage 4: VADER ────────────────────────────────────────────────────
        print("\nStage 4 — VADER Sentiment Analysis")
        try:
            vader_r = run_vader()
            _step("Sentiment analyzed", (vader_r["inserted"] + vader_r["updated"]) > 0,
                  f"{vader_r['inserted']} inserted, {vader_r['updated']} updated")
            flagged = PostSentiment.query.filter_by(sarcasm_flag=True).count()
            _step("Sarcasm detection ran", True, f"{flagged} posts flagged")
        except Exception as exc:
            _step("VADER failed", False, str(exc))

        # ── Stage 5: Random Forest ────────────────────────────────────────────
        print("\nStage 5 — Random Forest Priority Classification")
        try:
            rf_r = run_rf()
            _step("Model trained", True,
                  f"corpus={rf_r['corpus_size']}, accuracy={rf_r['accuracy']:.3f}")
            _step("Priorities predicted", rf_r["priority_rows"] > 0,
                  f"{rf_r['priority_rows']} rows written")
            dist = rf_r["distribution"]
            _step("Class distribution", True,
                  f"High={dist.get('High',0)}  Medium={dist.get('Medium',0)}  Low={dist.get('Low',0)}")
            high_posts = Post.query.filter_by(priority="High").count()
            _step("Post.priority updated", high_posts > 0,
                  f"{Post.query.filter_by(priority='High').count()} High, "
                  f"{Post.query.filter_by(priority='Medium').count()} Medium, "
                  f"{Post.query.filter_by(priority='Low').count()} Low")
        except Exception as exc:
            _step("RF failed", False, str(exc))

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n=== DB Row Counts ===")
        print(f"  posts              : {Post.query.count()}")
        print(f"  preprocessed_texts : {PreprocessedText.query.count()}")
        print(f"  post_topics        : {PostTopic.query.count()}")
        print(f"  post_clusters      : {PostCluster.query.count()}")
        print(f"  sentiments         : {PostSentiment.query.count()}")
        print(f"  post_priorities    : {PostPriority.query.count()}")
        print()


if __name__ == "__main__":
    main()
