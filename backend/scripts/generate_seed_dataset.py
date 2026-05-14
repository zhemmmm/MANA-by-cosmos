"""
Generate a realistic seed dataset for training the MANA classification models.

Produces backend/seed_dataset.json — ~110 Facebook-style posts in Apify format,
covering all 8 NDRRMC response clusters with a mix of English and Tagalog text.

The file is intentionally NOT imported into the database. Use train_from_seed.py
to train CorEx + SVM directly from this file without touching the DB.

Run from the backend/ directory:
    python scripts/generate_seed_dataset.py
"""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Seed for reproducibility
random.seed(42)

BASE_TIME = datetime(2025, 11, 1, 8, 0, 0, tzinfo=timezone.utc)

PAGES = [
    "Disaster Risk Reduction Office",
    "LGU Public Info",
    "Barangay Alert System",
    "City DRRMO",
    "Provincial PDRRMO",
    "NDRRMC Official",
    "BFP Updates",
    "PNP Public Safety",
    "DSWD Relief Operations",
    "OCD Disaster Response",
]


def _post(text: str, cluster_id: str, idx: int) -> dict:
    dt = BASE_TIME + timedelta(hours=idx * 3)
    r = random.randint(0, 200)
    return {
        "postId": f"seed_{cluster_id}_{idx:04d}_{uuid.uuid4().hex[:8]}",
        "text": text,
        "pageName": random.choice(PAGES),
        "facebookUrl": f"https://www.facebook.com/page/{random.randint(1000, 9999)}",
        "url": f"https://www.facebook.com/post/{random.randint(100000, 999999)}",
        "time": dt.isoformat(),
        "likes": r,
        "topReactionsCount": r + random.randint(0, 50),
        "reactionLikeCount": r,
        "reactionLoveCount": random.randint(0, 20),
        "reactionCareCount": random.randint(0, 10),
        "reactionHahaCount": 0,
        "reactionWowCount": random.randint(0, 5),
        "reactionSadCount": random.randint(0, 30),
        "reactionAngryCount": random.randint(0, 15),
        "shares": random.randint(0, 80),
        "comments": random.randint(0, 60),
        "viewsCount": random.randint(100, 5000),
        # Extra field used by train_from_seed.py only — not in real Apify payloads
        "_seed_cluster_id": cluster_id,
    }


# ── Posts per cluster ─────────────────────────────────────────────────────────

CLUSTER_A_POSTS = [
    "Distribution of relief goods ongoing at the barangay hall. Please bring your family ID.",
    "Food packs and hygiene kits available for affected families. Line up in an orderly manner.",
    "DSWD is distributing rice, canned goods, and blankets to flood victims in our area.",
    "Relief operations underway. Each family will receive 5kg rice and assorted goods.",
    "Nagtitinda ng relief goods ang DSWD sa evacuation center. Magdala ng ID.",
    "Food assistance para sa mga apektadong pamilya. Hatid ng LGU ngayong araw.",
    "Blankets, water containers, and hygiene kits are being distributed now.",
    "Humanitarian aid including non-food items (NFIs) delivered to 200 households today.",
    "Community kitchen is now open. Hot meals available for displaced residents.",
    "Relief goods including rice sacks, sardines, and soap distributed in Barangay 5.",
    "DOLE livelihood kits available for typhoon-affected workers. Register at the municipal hall.",
    "Water rationing in effect. Each household allowed 20 liters daily until supply is restored.",
]

CLUSTER_B_POSTS = [
    "Medical mission today at the evacuation center. Free consultation, medicines available.",
    "DOH reminds residents to observe proper sanitation to prevent disease outbreaks.",
    "Clean water supply disrupted in affected barangays. Boil water advisory in effect.",
    "Health team deployed to monitor for diarrhea, leptospirosis, and other waterborne diseases.",
    "Free medicine and first aid available at the municipal health office.",
    "Leptospirosis alert: avoid wading in floodwater. Wear protective boots if necessary.",
    "Mental health services available for trauma victims. Call the LGU hotline for assistance.",
    "Sanitation teams disinfecting evacuation centers to prevent cholera outbreak.",
    "DOH nagbabala ng posibleng pagkalat ng sakit sa mga apektadong lugar. Mag-ingat.",
    "Libre na konsultasyon sa doktor at gamot sa evacuation center. Bumisita ngayong umaga.",
    "Water purification tablets being distributed. Do not drink untreated floodwater.",
    "Nutrition assistance for children and pregnant women available at the health center.",
]

CLUSTER_C_POSTS = [
    "Evacuation center at the covered court is now open. Displaced families may proceed.",
    "Mga kababayan, lumikas na po kayo. Ang evacuation center ay bukas na sa elementarya.",
    "Over 300 families now sheltered at the covered court. Food and water available.",
    "Mandatory evacuation order issued for low-lying areas near the river.",
    "Displaced families from Barangay 3 are being relocated to the sports complex.",
    "Camp coordination team is managing the evacuation center. Register upon arrival.",
    "Evacuees are reminded to bring important documents, medicine, and enough clothing.",
    "The sports arena evacuation center is at capacity. Overflow to secondary site.",
    "Sheltering operations ongoing. 150 families already evacuated safely.",
    "Barangay officials to conduct door-to-door evacuation in Sitio Riverside tonight.",
    "Lumikas na ang mga pamilya sa mababang lugar. Handa ang evacuation center.",
    "Family Welfare desks set up at evacuation centers. Report missing family members.",
]

CLUSTER_D_POSTS = [
    "DPWH clearing operations ongoing on the national highway blocked by landslide.",
    "Road to Barangay 7 still impassable due to flooding. Alternative route via Highway 12.",
    "Delivery of relief goods delayed due to blocked bridges in the affected area.",
    "DPWH bulldozers working to clear debris from the main road. ETA 6 hours.",
    "Logistics team coordinating trucking of relief commodities to remote barangays.",
    "Bridge on the provincial road washed out. Heavy vehicles diverted.",
    "Supply chain for relief operations affected by road closures. Air delivery being arranged.",
    "Road clearing crews deployed to 3 sections of the national highway. Avoid the area.",
    "Landslide blocking the mountain road cleared. One lane now passable.",
    "Cargo trucks with relief goods rerouted due to damaged bridge in the north.",
    "DPWH nag-aayos ng daan. Sarado pa ang tulay. Gumamit ng alternatibong ruta.",
    "Port operations suspended due to storm. Relief supply boats on standby.",
]

CLUSTER_E_POSTS = [
    "PLDT restoring signal in affected areas. Estimated 24-48 hours for full restoration.",
    "No internet connection in Barangay 4 and 5 due to damaged cell towers.",
    "Power outage in several barangays. Meralco crews deployed for repair.",
    "Signal lost in northern municipalities. Globe and Smart towers damaged by typhoon.",
    "Blackout expected to last 12-24 hours. Generator sets available at evacuation centers.",
    "Emergency communications via radio only. FM station broadcasting updates.",
    "No network signal in our area since last night. PLDT please fix this ASAP.",
    "Power restoration estimated tomorrow morning for most affected households.",
    "Walang kuryente sa aming lugar. Meralco wala pa. Mag-iingat sa LED candles.",
    "Celltower sa bundok natumbag ng bagyo. Satellite phones in use by rescue teams.",
    "Emergency broadcast system activated. Tune to DWRR for disaster updates.",
    "Power lines down on the main road. Do not approach downed wires.",
]

CLUSTER_F_POSTS = [
    "DepEd suspends classes in all levels due to typhoon signal. Check official advisories.",
    "Class suspension announced for tomorrow. LGU order due to flooding in low-lying areas.",
    "Schools in the municipality are open. No class suspension for today.",
    "All public and private schools to remain closed until further notice from DepEd.",
    "Distance learning modules distributed to students who cannot attend face-to-face classes.",
    "Mga estudyante, walang klase bukas dahil sa bagyo. Manood ng balita.",
    "DepEd Region IV announces suspension of classes in all disaster-affected areas.",
    "School buildings being used as evacuation centers. Classes moved online.",
    "Students stranded due to flooding. Parents advised not to send children to school.",
    "DepEd provides learning continuity plan for affected communities.",
    "Class suspension in effect for all grade levels in the province. Stay safe.",
    "School feeding program suspended during displacement. Nutrition packs given to families.",
]

CLUSTER_G_POSTS = [
    "TXTFIRE: Sunog sa Brgy. 8, tulong! BFP patungo na.",
    "SOS! Trapped on the second floor of our house. Floodwater rising fast. Need rescue boat.",
    "Fire reported near the market. BFP units responding. Stay clear of the area.",
    "Rescue team deployed to extract families stranded on rooftop in flooded Barangay 3.",
    "Nasusunog ang bahay sa Sitio Riverside! Tumawag na sa 911. BFP padating.",
    "Coast Guard conducting swift water rescue operations along the river basin.",
    "Stranded residents in Barangay 7 requesting immediate rescue. Water level chest-deep.",
    "Fire alarm at the public market. Firefighters on scene. Market closed until further notice.",
    "Search and rescue team extracted 12 people from collapsed structure.",
    "Rescue boats deployed to flooded communities. Prioritizing elderly and children.",
    "Helicopter rescue for isolated communities cut off by landslide.",
    "BFP on standby for fire incidents during strong winds. Report fires immediately.",
    "Wildfire spotted near forested area in the upland barangay. Fire trucks dispatched.",
    "Stranded hikers rescued from the mountain trail after landslide blocked the path.",
]

CLUSTER_H_POSTS = [
    "NDRRMC confirms 3 fatalities due to flooding in the province.",
    "Missing persons report: John Dela Cruz, 45, last seen in Barangay 2 before the typhoon.",
    "Death toll rises to 7 as rescue operations continue in affected municipalities.",
    "Body found along the riverbank identified as missing resident from Barangay 5.",
    "Casualty count updated: 5 dead, 12 missing, 230 injured as of latest report.",
    "Search operations for missing fishermen ongoing. 3 boats still unaccounted for.",
    "Remains of typhoon victim identified at the provincial morgue. Family notified.",
    "DSWD activates missing persons tracking system. Report missing relatives here.",
    "Death toll from flash flood confirmed at 4. Families to claim remains at the morgue.",
    "Confirmed dead in the landslide: 2 adults, 1 child. Search for more survivors ongoing.",
    "Missing person alert: Maria Santos, 60 years old, last seen evacuating from Sitio Mabolo.",
    "Management of the Dead and Missing (MDM) team deployed to affected areas.",
]

# ── Assemble and write ────────────────────────────────────────────────────────

def main():
    all_posts = []
    idx = 0
    for cluster_id, posts in [
        ("cluster-a", CLUSTER_A_POSTS),
        ("cluster-b", CLUSTER_B_POSTS),
        ("cluster-c", CLUSTER_C_POSTS),
        ("cluster-d", CLUSTER_D_POSTS),
        ("cluster-e", CLUSTER_E_POSTS),
        ("cluster-f", CLUSTER_F_POSTS),
        ("cluster-g", CLUSTER_G_POSTS),
        ("cluster-h", CLUSTER_H_POSTS),
    ]:
        for text in posts:
            all_posts.append(_post(text, cluster_id, idx))
            idx += 1

    random.shuffle(all_posts)

    out_path = Path(__file__).parent.parent / "seed_dataset.json"
    out_path.write_text(json.dumps(all_posts, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(all_posts)} seed posts to {out_path}")
    counts = {}
    for p in all_posts:
        cid = p["_seed_cluster_id"]
        counts[cid] = counts.get(cid, 0) + 1
    for cid in sorted(counts):
        print(f"  {cid}: {counts[cid]} posts")


if __name__ == "__main__":
    main()
