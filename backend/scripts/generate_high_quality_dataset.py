"""
Generate a high-quality realistic seed dataset for training the MANA classification models.

Produces backend/seed_dataset.json — 240 Facebook-style posts in Apify format (30 per cluster),
covering all 8 NDRRMC response clusters with realistic Filipino/Taglish/English disaster content.

Key improvements over generate_seed_dataset.py:
- 30 posts per cluster (vs 12-14) — guarantees test samples for all 8 clusters in train/test split
- 3 language tiers: English (official), Taglish (civilian), Tagalog (pure local)
- 4 urgency levels with correlated engagement metrics
- Parameterized templates with real Philippine place names and agency names
- Vocabulary carefully separated between ambiguous cluster pairs

Run from the backend/ directory:
    python scripts/generate_high_quality_dataset.py
"""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

random.seed(42)

BASE_TIME = datetime(2025, 11, 1, 8, 0, 0, tzinfo=timezone.utc)

BARANGAY_NAMES = [
    "Barangay Marikina Heights", "Sitio Riverside Marikina", "Brgy. San Roque Marikina",
    "Brgy. Tumana Marikina", "Barangay Sto. Nino Cagayan de Oro", "Brgy. Carmen CDO",
    "Sitio Mabolo Albay", "Brgy. Daraga Albay", "Barangay Paco Manila",
    "Brgy. Pandacan Manila", "Brgy. Bagong Silangan Quezon City", "Sitio Kapatagan Leyte",
    "Barangay Anibong Tacloban", "Brgy. San Joaquin Pampanga", "Barangay Sta. Rita Pampanga",
    "Brgy. Calumpit Bulacan", "Sitio Bungkalan Misamis Oriental", "Barangay Agusan CDO",
    "Brgy. Talisay Batangas", "Sitio Kinabukasan Cagayan Valley", "Brgy. Poblacion Iligan",
    "Barangay Balulang CDO", "Sitio Masag Albay", "Brgy. Fatima General Santos",
]

MUNICIPALITIES = [
    "Marikina", "Cagayan de Oro", "Albay", "Leyte", "Pampanga",
    "Bulacan", "Misamis Oriental", "Tacloban", "Batangas", "Iligan",
    "Naga", "General Santos", "Davao", "Zamboanga", "Cavite",
]

PAGE_SOURCES = {
    "cluster-a": [
        "DSWD Field Office NCR", "DSWD Relief Operations", "LGU Marikina Official",
        "Municipal Social Welfare Office", "Red Cross Philippines",
        "Gawad Kalinga Relief Operations", "LGU Disaster Response",
        "Barangay Relief Committee",
    ],
    "cluster-b": [
        "DOH Regional Office", "Department of Health Philippines",
        "Municipal Health Office", "Barangay Health Center",
        "Philippine Red Cross Medical Team", "Health Emergency Management Bureau",
        "DOH CALABARZON", "WHO Philippines",
    ],
    "cluster-c": [
        "LGU Barangay Evacuation Center", "City DRRMO", "Provincial PDRRMO",
        "Barangay Alert System", "UNHCR Philippines", "Camp Coordination Unit",
        "LGU Disaster Response", "NDRRMC Official",
    ],
    "cluster-d": [
        "DPWH Regional Office", "DPWH District Engineering Office",
        "LGU Road Monitoring Team", "Provincial Road Updates",
        "Traffic Management Group PNP", "DPWH Road Clearing Operations",
    ],
    "cluster-e": [
        "Meralco Updates", "Globe Telecom Disaster Response",
        "Smart Communications Service", "PLDT Fiber Restoration",
        "NTC Philippines Advisory", "Barangay Power Watch",
    ],
    "cluster-f": [
        "DepEd Regional Office IV-A", "DepEd NCR Official",
        "Schools Division Office Marikina", "DepEd CARAGA",
        "LGU Class Suspension Advisory", "DepEd Region X CDO",
    ],
    "cluster-g": [
        "Bureau of Fire Protection Region X", "BFP National Capital Region",
        "Philippine Coast Guard", "NDRRMC Rescue Operations",
        "City Rescue Unit", "Barangay Emergency Response Team",
        "PNP Aviation Security Group",
    ],
    "cluster-h": [
        "NDRRMC Official", "Provincial Disaster Risk Reduction Office",
        "Philippine National Police Public Information",
        "Missing Persons Philippines", "DSWD Crisis Intervention Unit",
    ],
}


def _place() -> str:
    return random.choice(BARANGAY_NAMES)


def _muni() -> str:
    return random.choice(MUNICIPALITIES)


def _n() -> int:
    return random.choice([50, 100, 150, 200, 300, 500])


def _post(text: str, cluster_id: str, idx: int, urgency: str = "official") -> dict:
    dt = BASE_TIME + timedelta(hours=idx * 2 + random.randint(0, 4))
    page = random.choice(PAGE_SOURCES[cluster_id])

    if urgency == "panic":
        likes = random.randint(200, 1000)
        shares = random.randint(200, 600)
        comments = random.randint(100, 400)
        views = random.randint(5000, 50000)
    elif urgency == "urgent":
        likes = random.randint(100, 500)
        shares = random.randint(80, 300)
        comments = random.randint(50, 200)
        views = random.randint(2000, 20000)
    elif urgency == "personal":
        likes = random.randint(30, 150)
        shares = random.randint(30, 120)
        comments = random.randint(20, 100)
        views = random.randint(500, 5000)
    else:
        likes = random.randint(50, 200)
        shares = random.randint(20, 80)
        comments = random.randint(10, 50)
        views = random.randint(200, 3000)

    sad = random.randint(5, 30) if urgency in ("panic", "urgent") else random.randint(0, 15)
    care = random.randint(5, 20) if urgency in ("panic", "urgent") else random.randint(0, 10)
    love = random.randint(0, 10)
    top = likes + sad + care + love + random.randint(0, 10)

    return {
        "postId": f"seed_{cluster_id}_{idx:04d}_{uuid.uuid4().hex[:8]}",
        "text": text,
        "pageName": page,
        "facebookUrl": f"https://www.facebook.com/page/{random.randint(1000, 9999)}",
        "url": f"https://www.facebook.com/post/{random.randint(100000, 999999)}",
        "time": dt.isoformat(),
        "likes": likes,
        "topReactionsCount": top,
        "reactionLikeCount": likes,
        "reactionLoveCount": love,
        "reactionCareCount": care,
        "reactionHahaCount": 0,
        "reactionWowCount": random.randint(0, 5),
        "reactionSadCount": sad,
        "reactionAngryCount": random.randint(0, 10),
        "shares": shares,
        "comments": comments,
        "viewsCount": views,
        "_seed_cluster_id": cluster_id,
    }


# ── cluster-a: Food and Non-Food Items (NFIs) ─────────────────────────────────
# Core vocab: relief goods, food pack, canned goods, hygiene kit, water refill,
#             DSWD, NFI, rice, distribution
# Separated from cluster-c by emphasizing ITEMS, not shelter/camp management

def _cluster_a_posts(base_idx: int) -> list[dict]:
    cid = "cluster-a"
    templates = [
        # English official (10)
        (f"DSWD distributes relief goods to {_n()} families in {_place()}. Distribution at the barangay covered court. Bring valid ID.", "official"),
        (f"Food packs containing 5kg rice, sardines, and canned goods available at {_place()} distribution center. One pack per family.", "official"),
        (f"Relief operations underway in {_place()}. DSWD deploying non-food items including hygiene kits, sleeping mats, and tarpaulins.", "official"),
        (f"Community kitchen at {_place()} is now serving hot meals. Queue management in effect. Priority for elderly and children.", "official"),
        (f"Water refill stations activated at {_place()}. Residents may bring containers up to 20 liters per household.", "official"),
        (f"Red Cross distributing {_n()} family food packs in {_place()} today. Donations of canned goods and rice still accepted.", "official"),
        (f"NFI distribution ongoing. Hygiene kits, blankets, and sleeping mats available at {_place()} barangay hall.", "official"),
        (f"DSWD relief convoy arrived at {_place()} with 300 boxes of food packs and non-food items for flood survivors.", "official"),
        (f"Water containers and purification tablets distributed to {_n()} households in {_place()}. Instructions included.", "official"),
        (f"LGU {_muni()} distributes {_n()} food packs and {_n()} hygiene kits to flood-affected families today.", "official"),
        # Taglish civilian (10)
        (f"Nag-distribute na ng relief goods sa brgy hall ng {_place()}. Magdala ng ID para makakuha ng food pack.", "personal"),
        (f"May libreng canned goods at bigas sa {_place()}. Pumunta na kung kailangan ng inyong pamilya.", "personal"),
        (f"DSWD andito na sa amin sa {_place()}! Nagbibigay ng hygiene kit at relief goods. Salamat po!", "personal"),
        (f"Ang relief goods mula sa LGU ay ibinibigay na sa mga affected families sa {_place()}. Isa lang per pamilya.", "official"),
        (f"Nakatanggap na kami ng food pack at tubig mula DSWD sa {_place()}. May kasamang canned goods at blanket.", "personal"),
        (f"Relief goods konti na lang sa {_place()}! Maraming families pa ang wala. Requesting additional supplies mula DSWD.", "urgent"),
        (f"Meron pang food packs sa {_place()} hanggang ngayong hapon. Magdala ng ID at pumunta na agad.", "urgent"),
        (f"Volunteers needed para sa repacking ng relief items sa {_place()}. Report sa DSWD office by 7am bukas.", "personal"),
        (f"Libre baby formula at diapers para sa mga buntis at sanggol sa {_place()} distribution point.", "personal"),
        (f"Relief convoy galing {_muni()} dumating na sa {_place()}. 200 boxes ng relief goods at hygiene kits.", "official"),
        # Tagalog (5)
        (f"Ibinahagi na ang mga pagkain at gamit sa mga apektadong pamilya sa {_place()}. Maraming salamat sa DSWD.", "official"),
        (f"Ang mga relief goods ay naghihintay sa barangay hall ng {_place()}. Pumunta na at magdala ng ID.", "official"),
        (f"Libre ang canned goods at bigas para sa mga nasalanta sa {_place()}. Pumunta na sa distribution point.", "official"),
        (f"Ang DSWD ay nagbibigay ng mga batayang pangangailangan sa mga pamilyang apektado ng baha sa {_place()}.", "official"),
        (f"Tulong po para sa mga pamilyang walang makain sa {_place()}. Makikipag-ugnayan sa DSWD para sa relief goods.", "personal"),
        # Panic/shortage (3)
        (f"URGENT: Wala na pong food pack dito sa {_place()}! Need pa ng bigas at canned goods para sa {_n()} families!", "panic"),
        (f"SOS! Out of stock na ang relief goods sa {_place()}. {_n()} families pa ang walang makain ngayong gabi!", "panic"),
        (f"EMERGENCY: Kulang ang hygiene kits sa {_place()} evacuation distribution. Requesting immediate resupply from DSWD!", "panic"),
        # Donation drives (2)
        (f"Calling all donors! Accepting relief goods donations for {_place()} flood victims. Drop off at the covered court.", "personal"),
        (f"Food donation drive ongoing for {_place()}. Target: {_n()} food packs for displaced families. Help us reach the goal.", "personal"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-b: Health / Medical / WASH ───────────────────────────────────────
# Core vocab: leptospirosis, diarrhea, DOH, health center, contaminated water,
#             first aid, medicine, WASH, cholera, sanitation
# Separated from cluster-a: defined by disease names and medical procedures, not items

def _cluster_b_posts(base_idx: int) -> list[dict]:
    cid = "cluster-b"
    templates = [
        # English official (10)
        (f"DOH issues leptospirosis alert for flood-affected barangays in {_place()}. Avoid wading in floodwater. Wear protective boots.", "official"),
        (f"Health teams deployed to {_place()} to monitor for waterborne diseases including diarrhea and cholera.", "official"),
        (f"Boil water advisory in effect for all households in {_place()}. Tap water may be contaminated by flooding.", "official"),
        (f"Medical mission at {_place()} evacuation center. Free consultation and medicines available 8am to 5pm today.", "official"),
        (f"DOH mobile clinic stationed at {_place()}. Treating fever, wound care, diarrhea, and leptospirosis cases.", "official"),
        (f"Sanitation teams disinfecting evacuation centers in {_place()} to prevent cholera and disease outbreak.", "official"),
        (f"Water purification tablets distributed to {_n()} households in {_place()}. Do not drink untreated floodwater.", "official"),
        (f"WASH cluster deployed to {_place()}. Setting up handwashing stations, portable toilets, and clean water supply.", "official"),
        (f"Red Cross medical team at {_place()} bringing medicine, IV fluids, and first aid kits for flood survivors.", "official"),
        (f"Field hospital set up near {_place()} to handle surge of injured and sick patients from flood-affected areas.", "official"),
        # Taglish health (10)
        (f"ALERT: Maraming bata ang may lagnat sa {_place()} evacuation center! Kailangan ng doktor at gamot dito ngayon!", "urgent"),
        (f"Huwag uminom ng tubig mula sa gripo sa {_place()}. Contaminated na dahil sa baha. Bumili ng mineral water.", "urgent"),
        (f"May leptospirosis na sa aming lugar sa {_place()}. Iwasan ang pagtapak sa maruming tubig-baha. Mag-ingat!", "urgent"),
        (f"Medical team ng DOH nandito na sa {_place()}. Libre ang konsultasyon at gamot para sa lahat ng evacuees.", "personal"),
        (f"Ang health center sa {_place()} ay bukas hanggang 8pm. Libre ang pagtingin sa mga may sakit na apektado.", "official"),
        (f"Diarrhea cases dumami sa {_place()} evacuation center. DOH nagpadala ng karagdagang health personnel.", "urgent"),
        (f"Mental health team available sa {_place()} evacuation center. Libre ang psychosocial support para sa lahat.", "official"),
        (f"Ambulance requested para sa 3 critical patients sa {_place()} evacuation center! Kailangan ng transport ngayon!", "panic"),
        (f"URGENT: Patient na may symptoms ng leptospirosis sa {_place()} covered court. Need first aid team agad!", "panic"),
        (f"Nutrition assistance para sa malnourished children sa {_place()} health center. Magdala ng health card.", "official"),
        # Tagalog (6)
        (f"Mag-ingat sa leptospirosis! Huwag lumakad sa baha nang walang proteksyon sa {_place()} at karatig-lugar.", "official"),
        (f"Libre na gamot para sa mga may sakit sa {_place()} evacuation center. Pumunta sa health tent.", "official"),
        (f"Ang tubig ay nalason na ng baha sa {_place()}. Huwag uminom ng walang pakuluan muna ng limang minuto.", "official"),
        (f"Ang mga bata sa {_place()} ay binibigyan ng bitamina at gamot. Dalhin ang inyong anak sa health tent.", "official"),
        (f"Dengue alert kasabay ng flood advisory sa {_place()}. Gamitin ang mosquito net at alisin ang nakaimbak na tubig.", "official"),
        (f"DOH nagpaalala sa mga evacuees sa {_place()} na inumin ang maintenance medicine. Available ang refills sa clinic.", "official"),
        # Emergency medical (4)
        (f"EMERGENCY: Isang bata ang nalunod sa baha sa {_place()}. Dinala na sa ospital. Mangyaring mag-ingat sa paligid.", "panic"),
        (f"First aid supplies running low at {_place()} evacuation center. Need gauze, antiseptic, and fever medicine urgently.", "urgent"),
        (f"Clean water supply restored to Barangays 4 to 7 in {_place()}. Boil for 5 minutes before drinking as precaution.", "official"),
        (f"Latrines and sanitation facilities installed at {_place()} evacuation center. One unit per 20 persons as per WASH standard.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-c: Evacuation / Camp Coordination (CCCM) ─────────────────────────
# Core vocab: evacuation center, evacuees, displaced families, covered court,
#             mandatory evacuation, camp, overflow, shelter, displaced
# Separated from cluster-a: emphasizes WHERE people are and CAMP MANAGEMENT

def _cluster_c_posts(base_idx: int) -> list[dict]:
    cid = "cluster-c"
    templates = [
        # English official (8)
        (f"MANDATORY EVACUATION: All residents in low-lying areas near {_place()} river must evacuate immediately.", "urgent"),
        (f"Evacuation center at {_place()} covered court now open. Capacity {_n()} families. Bring important documents.", "official"),
        (f"Barangay {_place()} begins door-to-door evacuation along sitio riverside. Officials and BFP assisting families.", "official"),
        (f"Preemptive evacuation order issued for danger zones in {_place()}. LGU providing transport assistance.", "urgent"),
        (f"{_n()} displaced families now sheltered at {_place()} gymnasium. Additional centers being opened tonight.", "official"),
        (f"Camp coordination team at {_place()}: headcount ongoing. {_n()} registered evacuees as of 6pm.", "official"),
        (f"Privacy partitions installed for families at {_place()} gymnasium. Breastfeeding area and child-friendly space also set up.", "official"),
        (f"Evacuation order lifted for parts of {_place()}. Residents may return home after clearance from DRRMO.", "official"),
        # Taglish civilian (12)
        (f"LUMIKAS NA KAYO! Mataas na ang tubig dito sa {_place()}. Bukas na ang evacuation center sa elementarya.", "panic"),
        (f"Mga kababayan, ang evacuation center sa {_place()} brgy hall ay puno na. Pumunta sa secondary site sa sports complex.", "urgent"),
        (f"Nalumikas na kami sa evacuation center sa {_place()}. Safe naman kami dito. Salamat sa LGU {_muni()}.", "personal"),
        (f"Displacement update: {_n()} families na ang naka-shelter sa covered court ng {_place()}. Overflow na ang dami.", "official"),
        (f"Evacuation center ni-register lahat ng displaced families sa {_place()}. Magdala ng ID at gamit para isang linggo.", "official"),
        (f"Family welfare desks naka-set up sa {_place()} evacuation site. I-report dito ang mga nawawalang miyembro ng pamilya.", "official"),
        (f"Overcrowding na sa {_place()} evacuation site. Secondary shelter sa barangay chapel ay bukas na ngayon.", "urgent"),
        (f"Nandito na kami sa evacuation center sa {_place()}. Ikatlong gabi na namin dito. Sana makauwi na kami.", "personal"),
        (f"Safe naman kami sa {_place()} evacuation site. Maraming salamat sa lahat ng nagbigay ng pagkain at tulong.", "personal"),
        (f"URGENT: Puno na ang evacuation center sa {_place()}! Wala nang space para sa mga bagong dating na pamilya!", "panic"),
        (f"Ang aming pamilya ay naka-shelter na sa {_place()} covered court. May relief goods at tubig. Salamat DSWD.", "personal"),
        (f"Decampment process simula na sa {_place()} evacuation center. LGU tumutulong sa mga pamilyang umuuwi na.", "official"),
        # Tagalog (6)
        (f"Lumikas na po ang mga pamilya sa mababang lugar ng {_place()}. Handa ang evacuation center sa covered court.", "official"),
        (f"Ang mga evacuees sa {_place()} covered court ay bibigyan ng pagkain at mga gamit ngayong gabi.", "official"),
        (f"Puno na ang evacuation center sa {_place()}. Pumunta sa bagong shelter sa sports complex ng bayan.", "urgent"),
        (f"Hindi pa pwedeng umuwi ang mga evacuees sa {_place()}. Naghihintay pa ng clearance mula sa barangay.", "official"),
        (f"Ang mga displaced families sa {_place()} ay puwedeng manatili sa evacuation center hanggang ligtas na.", "official"),
        (f"Isinasagawa ang headcount ng lahat ng evacuees sa {_place()} evacuation center. Pumirma sa logbook.", "official"),
        # Camp management (4)
        (f"Child-friendly space opened at {_place()} evacuation center for minors. Activities and psychosocial support ongoing.", "official"),
        (f"75 families allowed to return home after flood receded in {_place()}. 300 families still remaining in shelter.", "official"),
        (f"Warning: Do not return to {_place()} without clearance from DRRMO. Flood waters still being assessed.", "urgent"),
        (f"Displaced families from {_place()} to be temporarily relocated to safer housing pending flood damage assessment.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-d: Logistics ──────────────────────────────────────────────────────
# Core vocab: blocked road, DPWH, bridge, landslide, convoy, road clearing,
#             alternate route, impassable, passable, debris, truck, reroute
# Separated from cluster-g: stranded = vehicles/goods, never people/SOS

def _cluster_d_posts(base_idx: int) -> list[dict]:
    cid = "cluster-d"
    templates = [
        # English official (10)
        (f"DPWH road clearing operations ongoing on national highway at {_place()}. Expect 6 to 8 hour delay.", "official"),
        (f"Road to {_place()} IMPASSABLE due to landslide. DPWH equipment deployed. Alternate route via highway 12.", "urgent"),
        (f"Bridge at {_place()} washed out by flash flood. Heavy vehicles diverted via pontoon bridge.", "urgent"),
        (f"Road clearing convoy proceeding to {_place()}. Backhoes and dump trucks on route now.", "official"),
        (f"National highway flooded in {_place()}. Passable for light vehicles only. Heavy trucks must use alternate route.", "official"),
        (f"Relief goods delivery to {_place()} delayed due to 3 blocked roads. Air delivery being explored by NDRRMC.", "urgent"),
        (f"Staging area for relief operations set up in {_place()}. Supply trucks loading for remote barangays.", "official"),
        (f"Port operations suspended in {_place()}. Supply barges on standby. Coast Guard monitoring conditions.", "official"),
        (f"Provisional bridge at {_place()} constructed by Army engineers. Single lane, 3-ton weight limit.", "official"),
        (f"Landslide at km 23 near {_place()} cleared by DPWH. One lane now passable. Flagman on duty.", "official"),
        # Taglish road updates (10)
        (f"SARADO ang tulay sa {_place()} dahil sa baha! Gumamit ng alternatibong daan papunta sa {_muni()}.", "urgent"),
        (f"Nablock ang kalsada sa {_place()}. Landslide po. DPWH padating na. Iwasan muna ang lugar.", "urgent"),
        (f"Ang convoy ng relief goods ay hindi makarating sa {_place()} dahil sa blocked road. Nagre-reroute na.", "urgent"),
        (f"Mag-ingat sa kalsada papunta sa {_place()}. Maraming potholes at floating debris pagkatapos ng baha.", "personal"),
        (f"Road update sa {_place()}: passable na sa isang lane. Slow traffic pa rin. DPWH nag-aayos pa.", "official"),
        (f"Convoy ng 10 trucks dumating na sa {_place()} pagkatapos ng 12-oras na byahe via alternate mountain road.", "official"),
        (f"Supply truck natigil sa {_place()} junction. Baha masyadong malalim. Naghihintay na bumaba ang tubig.", "personal"),
        (f"Chokepoint sa national highway ng {_place()} dahil sa demolition ng damaged bridge. Abangan ang updates.", "official"),
        (f"DPWH nag-aayos na ng daan sa {_place()}. Tatlong araw pa bago ganap na maabot ang sitio.", "official"),
        (f"Ang alternate route via sitio {_place()} ay passable na para sa medium trucks. 4WD recommended.", "official"),
        # Tagalog (5)
        (f"Naputol ang tulay papunta sa {_place()}. Pag-aralan ang ibang daan bago umalis.", "urgent"),
        (f"Hindi pa passable ang kalsada sa {_place()}. Naghihintay pa ang mga trucks ng relief goods.", "official"),
        (f"Sarado ang pangunahing daan sa {_place()} dahil sa landslide. DPWH nagtatrabaho na.", "official"),
        (f"Ang mga cargo trucks ay hindi makaraan sa {_place()} dahil sa nabagsak na tulay.", "official"),
        (f"Mag-iwas sa kalsada ng {_place()}. Maraming debris at mapanganib ang daan.", "official"),
        # Infrastructure (5)
        (f"Bridge inspection underway at {_place()}. Currently closed pending structural assessment by DPWH engineers.", "official"),
        (f"Fallen tree blocking the access road to {_place()}. Clearing crew on site. Passable in 2 hours.", "official"),
        (f"Road subsidence detected on provincial road of {_place()}. Heavy vehicles must detour via bypass.", "official"),
        (f"Road network update for {_place()}: 3 routes open, 2 blocked by landslide, 1 passable for light vehicles only.", "official"),
        (f"Logistics team coordinating boat transport for communities cut off by flooding in {_place()}.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-e: Emergency Telecommunications ──────────────────────────────────
# Core vocab: power outage, blackout, no signal, cell site, Meralco, PLDT,
#             Globe, Smart, generator, brownout, network, restoration
# Unique domain — telco brand names and power terms appear in no other cluster

def _cluster_e_posts(base_idx: int) -> list[dict]:
    cid = "cluster-e"
    templates = [
        # English official (8)
        (f"Meralco restoring power in {_place()}. Estimated restoration: 18 to 24 hours. Repair crews deployed.", "official"),
        (f"PLDT announces network outage in {_place()} due to damaged cell towers. Engineers on site.", "official"),
        (f"Globe and Smart towers in {_place()} damaged by typhoon. No signal in affected areas.", "urgent"),
        (f"Power restoration underway in {_place()}. 70 percent of households back online as of 6pm.", "official"),
        (f"NTC advisory: Emergency radio frequencies active. Tune to DZRH or local FM for disaster updates.", "official"),
        (f"No power in {_place()} for 48 hours. Generator sets operating at hospitals and evacuation centers.", "urgent"),
        (f"Power lines down on the main road in {_place()}. Do not approach downed wires. Report to Meralco.", "urgent"),
        (f"Cell on wheels temporary tower deployed in {_place()}. Limited signal now available for affected residents.", "official"),
        # Taglish complaints (12)
        (f"Walang kuryente dito sa {_place()} simula kahapon! Meralco saan na kayo? Kailan maibabalik ang power?", "urgent"),
        (f"Wala na pong signal sa buong {_place()}. Globe, Smart, TM lahat wala. Gumagamit na kami ng radio.", "urgent"),
        (f"Blackout na nang 2 araw sa {_place()}. Generator na lang ang naiiwanan namin. Sana mabilis na.", "personal"),
        (f"Cell site sa bundok ng {_place()} natanggal ng bagyo. Satellite phone lang ang gumagana sa lugar.", "official"),
        (f"No network sa amin sa {_place()}! May makaka-abot ba sa {_muni()}? May signal pa ba diyan?", "urgent"),
        (f"WALA PONG SIGNAL DITO SA {_place().upper()}! PLEASE PATAWID NG MENSAHE SA AMING PAMILYA!", "panic"),
        (f"Blackout pa rin sa {_place()}. Running low na sa power bank. Sana maibalik na ang kuryente.", "personal"),
        (f"Meralco please ayusin na ang power sa {_place()}! 72 oras na walang kuryente. Grabe na ito!", "panic"),
        (f"Globe signal bumalik na sa ilang parte ng {_place()}. Smart at PLDT wala pa. Working on restoration.", "official"),
        (f"Power restoration crew nagtatrabaho overnight sa {_place()}. Asahan ang pagbabalik ng kuryente bukas.", "official"),
        (f"Transformer napalitan sa main substation ng {_place()}. Naka-grid na ulit. Thank you Meralco.", "personal"),
        (f"No internet available sa {_place()} upland barangays. PLDT deploying additional equipment.", "official"),
        # Tagalog (5)
        (f"Walang kuryente sa {_place()} mula kahapon. Mag-ingat sa kandila. Sana mabilis pong maibalik.", "personal"),
        (f"Hindi pa rin maibalik ang signal sa {_place()}. Ang komunikasyon ay sa radyo na lang.", "personal"),
        (f"Ang mga cell tower ay nasira ng bagyo sa {_place()}. Kailangan ng bagong equipment para maitayo.", "official"),
        (f"Walang kuryente at walang signal sa {_place()}. Ganap na nahiwalay sa komunikasyon.", "urgent"),
        (f"Ang kuryente sa {_place()} ay naibalik na. Salamat sa Meralco at sa lahat ng nagtrabaho.", "personal"),
        # Emergency comms (5)
        (f"Emergency broadcast activated at local FM station. Broadcasting disaster updates for {_place()} 24 hours.", "official"),
        (f"Charging stations set up at {_place()} evacuation center. Bring your power bank and mobile devices.", "official"),
        (f"Internet access available only at evacuation center in {_place()}. Hotspot available for residents.", "official"),
        (f"No electricity in {_place()}. Complete blackout since typhoon hit. No signal, no internet, no power.", "urgent"),
        (f"Communication restored in {_place()} using emergency satellite link. Contact numbers for LGU updated.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-f: Education ──────────────────────────────────────────────────────
# Core vocab: class suspension, walang pasok, walang klase, DepEd, school closure,
#             no classes, students, school, suspend, academic calendar
# Separated from cluster-c: topic is CLASS DISRUPTION and STUDENTS, not shelter mgmt

def _cluster_f_posts(base_idx: int) -> list[dict]:
    cid = "cluster-f"
    templates = [
        # English official (8)
        (f"OFFICIAL: DepEd suspends classes in all grade levels in {_place()} effective tomorrow. Monitor official advisory.", "official"),
        (f"Class suspension order issued for {_place()} due to Typhoon Signal No. 3. All schools closed today.", "urgent"),
        (f"DepEd Region IV-A announces no classes in public and private schools in {_place()} tomorrow.", "official"),
        (f"School buildings in {_place()} being used as evacuation centers. All classes moved to online learning mode.", "official"),
        (f"DepEd releases learning continuity plan for flood-affected areas in {_place()}. Printed modules distributed.", "official"),
        (f"3 school buildings damaged by flooding in {_place()}. DepEd repair timeline: 2 to 3 weeks.", "official"),
        (f"DepEd {_place()} advisory: No classes for elementary, high school, and college levels until further notice.", "official"),
        (f"Classes resume next Monday for {_place()} schools. Buildings assessed as safe. See you mga estudyante!", "official"),
        # Taglish walang pasok (12)
        (f"WALANG PASOK bukas sa lahat ng paaralan sa {_place()}! DepEd order na. Stay safe mga estudyante.", "urgent"),
        (f"No classes na ngayon sa {_place()} hanggang sa karagdagang paunawa ng DepEd. Manatili sa bahay.", "official"),
        (f"Class suspension confirmed! Walang klase bukas sa lahat ng levels sa {_place()}. Abangan ang advisory.", "urgent"),
        (f"DepEd nag-announce na ng no classes. Mag-stay sa bahay mga bata. Hindi ligtas lumabas ngayon sa {_place()}.", "official"),
        (f"School cancellation alert: Lahat ng paaralan sa {_place()} ay sarado bukas dahil sa bagyo.", "urgent"),
        (f"Walang klase bukas! Masaya ang mga bata sa {_place()} pero nakakatakot ang baha sa labas.", "personal"),
        (f"May klase ba bukas? Hindi pa malinaw ang DepEd advisory sa {_place()}. Abangan ang opisyal na anunsyo.", "personal"),
        (f"Parents, please do not send children to school tomorrow in {_place()}. Roads are flooded and dangerous.", "urgent"),
        (f"Online class na lang bukas para sa lahat ng students sa {_place()}. Tingnan ang Google Classroom.", "official"),
        (f"No face-to-face classes hanggang sa karagdagang abiso. Distance learning mode activated para sa {_place()}.", "official"),
        (f"DepEd nagbibigay ng make-up classes pagkatapos ng baha. School calendar extended ng 2 linggo sa {_place()}.", "official"),
        (f"School supplies at libro nasira ng baha sa {_place()}. DepEd nagre-request ng donations para sa mga estudyante.", "urgent"),
        # Tagalog (6)
        (f"Walang pasok bukas! Ang lahat ng paaralan sa {_place()} ay sarado. Manatili sa bahay at mag-ingat.", "official"),
        (f"Ang mga guro ay nagbibigay ng mga module sa mga estudyanteng hindi makarating sa paaralan sa {_place()}.", "official"),
        (f"Hindi maaaring pumasok ang mga bata sa paaralan sa {_place()}. Sarado pa ang mga daan.", "official"),
        (f"Ang paaralan sa {_place()} ay ginagamit bilang evacuation center. Suspendido ang klase hanggang maalis ang evacuees.", "official"),
        (f"Ang mga guro sa {_place()} ay nagboboluntaryo sa evacuation center habang suspendido ang klase.", "personal"),
        (f"Ipinagpapaliban ang lahat ng school activities sa {_place()} hanggang sa karagdagang paunawa ng DepEd.", "official"),
        # School damage (4)
        (f"DepEd schools in {_place()}: 5 total — 2 being used as evacuation shelters, 3 closed for safety assessment.", "official"),
        (f"Academic calendar suspended for affected schools in {_place()}. DepEd to announce make-up class schedule.", "official"),
        (f"School feeding program suspended during displacement in {_place()}. Nutrition packs distributed to families.", "official"),
        (f"Teachers at {_place()} school-turned-evacuation-center voluntarily assist displaced families after class suspension.", "personal"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-g: Search, Rescue and Retrieval (SRR) ────────────────────────────
# Core vocab: rescue, trapped, stranded (PEOPLE), SOS, BFP, fire, coast guard,
#             rescue boat, swift water, helicopter, TXTFIRE
# Separated from cluster-h: PRESENT TENSE + active danger. Never uses fatality/morgue/death toll

def _cluster_g_posts(base_idx: int) -> list[dict]:
    cid = "cluster-g"
    templates = [
        # Fire emergency TXTFIRE (6)
        (f"TXTFIRE! Sunog sa {_place()}! Tumawag sa BFP ngayon! Lumayas na sa lugar!", "panic"),
        (f"FIRE! Sumalab ang bahay sa {_place()}. BFP patungo na. Lumayo na mga tao sa paligid!", "panic"),
        (f"Structure fire reported at {_place()} public market. BFP units responding. 2-alarm fire declared.", "urgent"),
        (f"Wildfire spotted near upland barangay of {_place()}. BFP dispatched. Residents placed on alert.", "urgent"),
        (f"Fire update: BFP contained the blaze at {_place()} warehouse after 2 hours. Investigation ongoing.", "official"),
        (f"Fire out at {_place()}! BFP controlled the fire. No casualties reported. BFP reminds public to report fires immediately.", "official"),
        # Flood rescue SOS (8)
        (f"SOS! TRAPPED SA BUBONG! Floodwater waist-deep at rising fast! Need rescue boat AGAD sa {_place()}!", "panic"),
        (f"MAYDAY! Naiipit kami sa loob ng bahay sa {_place()}! Tubig hanggang bintana na! Tulong po!", "panic"),
        (f"Rescue needed: 5 persons stranded on rooftop in {_place()}. Water chest-deep and still rising.", "panic"),
        (f"Coast Guard conducting swift water rescue in {_place()} river. 12 families rescued so far. Operations ongoing.", "urgent"),
        (f"Rescue boat deployed to {_place()}. Prioritizing elderly, children, and PWDs stranded in floodwater.", "official"),
        (f"HELP! Our neighbor is trapped on the second floor in {_place()}! Water still rising! 911 not answering!", "panic"),
        (f"Rescue boat needed URGENTLY at {_place()}! Elderly couple stranded on roof. Please respond now!", "panic"),
        (f"All clear at {_place()}: all trapped residents rescued safely. Rescue teams thank community for cooperation.", "official"),
        # Official rescue operations (8)
        (f"NDRRMC rescue teams extract {_n()} families from flood-affected sitio in {_place()}.", "official"),
        (f"Helicopter rescue ongoing for communities in {_place()} cut off by landslide. 50 persons airlifted.", "urgent"),
        (f"Search and rescue operations launched in {_place()} after flash flood. 8 rescue boats deployed.", "official"),
        (f"Navy rescue personnel assisting in evacuation of stranded residents in {_place()} floodwater.", "official"),
        (f"USAR team deployed to collapsed building in {_place()}. 3 persons confirmed trapped inside structure.", "urgent"),
        (f"Volunteer rescue swimmers needed at {_place()} riverbank. Call DRRMO hotline now for coordination.", "urgent"),
        (f"BFP and rescue team arrived at {_place()}. Two children rescued from rising floodwater. Operations ongoing.", "official"),
        (f"Rescue operation in {_place()}: 30 persons rescued this afternoon. Operations continuing for remaining stranded.", "official"),
        # Taglish rescue requests (5)
        (f"Naiipit pa rin ang {_n()} pamilya sa {_place()}. Hindi pa nakakarating ang rescue boat. Tulong po!", "panic"),
        (f"May nastranded na hikers sa {_place()} bundok dahil sa landslide. Search party naalis na.", "urgent"),
        (f"Rescue operation sa {_place()}: dalawang bata ang naligtas sa baha. Nagpapatuloy pa ang paghahanap.", "official"),
        (f"BFP please tumugon sa {_place()}! May sunog at hindi kami makalabas! Naiipit sa loob!", "panic"),
        (f"SOS mula sa {_place()}: nastranded na hikers, 5 tao, 2 sugatan. Mountain trail blocked ng landslide.", "panic"),
        # Tagalog (3)
        (f"Ang mga naiipit ay inililigtas na sa {_place()}. Huwag matakot. Ang rescue team ay patungo na.", "official"),
        (f"Naligtas na ang mga bata sa baha sa {_place()}. Salamat sa BFP at sa lahat ng nagtulungan.", "personal"),
        (f"Ang sunog ay naapula na ng BFP sa {_place()}. Dalawang pamilya ang naapektuhan. Ligtas na lahat.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-h: Management of Dead and Missing (MDM) ──────────────────────────
# Core vocab: missing person, fatality, death toll, body found, morgue,
#             confirmed dead, casualty, family tracing, remains, identified
# Separated from cluster-g: PAST TENSE + confirmed outcome. Never uses rescue boat/SOS/fire

def _cluster_h_posts(base_idx: int) -> list[dict]:
    cid = "cluster-h"
    templates = [
        # Official death toll (8)
        (f"NDRRMC confirms death toll from flooding in {_place()}: 12 fatalities, 3 missing, 45 injured.", "official"),
        (f"Death toll rises to 8 as recovery operations continue in {_place()}. Search for 5 missing persons ongoing.", "official"),
        (f"Casualty report update: 15 dead, 22 missing, 180 displaced in {_place()} as of 8pm today.", "official"),
        (f"Official MDM count: 6 confirmed dead, 11 unaccounted for in flood-affected areas of {_place()}.", "official"),
        (f"NDRRMC confirms 4 fatalities due to flash flood in {_place()}. Families of victims have been notified.", "official"),
        (f"Death toll from flash flood confirmed at {_n()} in {_place()}. Families may claim remains at the morgue.", "official"),
        (f"Confirmed dead in the landslide at {_place()}: 2 adults, 1 child. Search for more survivors ongoing.", "official"),
        (f"Casualty update for {_place()}: {_n()} dead confirmed, bodies recovered and brought to provincial morgue.", "official"),
        # Missing person alerts (6)
        (f"MISSING: Juan dela Cruz, 55 years old, last seen at {_place()} before the typhoon. Wearing blue jacket.", "urgent"),
        (f"Missing person alert: Maria Santos, 62, {_place()} resident. Last seen evacuating Sitio Riverside yesterday.", "urgent"),
        (f"Looking for our lolo. Missing since typhoon hit {_place()}. Name: Pedro Reyes, 70 years old, diabetic.", "urgent"),
        (f"MISSING child: Anak ko nawawala! 8 years old, nahulog sa baha malapit sa {_place()}. Tulong po!", "panic"),
        (f"Search for missing fishermen ongoing. 4 boats from {_place()} still unaccounted for after the storm.", "urgent"),
        (f"Missing person: family tracing for 3 persons from {_place()} who have not contacted family since typhoon.", "urgent"),
        # Body recovered (5)
        (f"Body found along the {_place()} riverbank identified as flood victim. Family notified. Burial arrangements ongoing.", "official"),
        (f"Remains of 2 typhoon victims recovered at {_place()}. Pending identification at provincial morgue.", "official"),
        (f"Body identified as missing resident from {_place()}. Next of kin contacted. Interment on Saturday.", "official"),
        (f"3 bodies recovered from debris in {_place()}. Death toll officially revised upward. Autopsies ordered.", "official"),
        (f"Post-mortem examination ordered for body found in {_place()}. Identity still being confirmed by authorities.", "official"),
        # Taglish MDM (5)
        (f"MISSING: Ang aking tatay ay nawala noong baha sa {_place()}. Mangyaring makipag-ugnayan sa amin.", "urgent"),
        (f"Confirmed dead na ang 3 residente ng {_place()}. Nagsisimula na ng proseso para sa burial at interment.", "official"),
        (f"Patay na ang isang lalaki na nahanap sa {_place()} riverbank. Dinala na sa morgue ng probinsya.", "official"),
        (f"Ang MDM team ng NDRRMC ay aktibo sa {_place()}. Mag-report ng mga nawawalang kamag-anak dito.", "official"),
        (f"Family tracing desk bukas sa {_place()} evacuation center. I-report ang mga nawawalang pamilya ngayon.", "official"),
        # Official MDM procedures (4)
        (f"DSWD activates missing persons tracking system for {_place()}. File reports at the welfare desk now.", "official"),
        (f"Ante-mortem data collection ongoing for {_place()} flood victims. Bring recent photos and descriptions.", "official"),
        (f"Management of the Dead and Missing team deployed to {_place()} by NDRRMC. Operations 24 hours.", "official"),
        (f"Death certificates being processed at {_place()} municipal hall. Bring burial clearance documents.", "official"),
        # Tagalog (2)
        (f"Ang aking kaibigan ay nawawala mula nang bumaha sa {_place()}. Nahanap na ba siya ng sinuman?", "personal"),
        (f"Natuklasan ang katawan ng isa sa mga nawawalang residente sa {_place()}. Ipinagdadasal namin siya.", "personal"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── Extended posts per cluster (real Apify content types) ────────────────────
# These mirror content observed in real MnlCDRRMD / NDRRMC Apify exports:
# heat index advisories, volcano alerts, air quality, typhoon signal updates.

def _cluster_a_posts_ext(base_idx: int) -> list[dict]:
    """30 extra cluster-a posts: relief during heatwave + drought scenarios."""
    cid = "cluster-a"
    templates = [
        (f"DSWD providing water and food to heat-affected communities in {_place()}. Hydration packs included.", "official"),
        (f"Emergency water distribution for drought-hit families in {_place()}. 20L per household, bring container.", "official"),
        (f"Mga kababayan sa {_place()}, may libreng mineral water at juice sa barangay hall ngayon. Kumuha na.", "personal"),
        (f"Panahon ng tag-araw: nagbibigay ang DSWD ng rehydration packs at snacks sa mga batang apektado sa {_place()}.", "official"),
        (f"Libreng pagkain at tubig para sa mga walang trabaho dahil sa tag-init sa {_place()}. Pumunta sa DSWD office.", "personal"),
        (f"Relief packs including bottled water and energy bars distributed to {_n()} families in {_place()} amid heat advisory.", "official"),
        (f"Food assistance ongoing for displaced families in {_place()} following volcanic activity. Bring barangay ID.", "official"),
        (f"Ang DSWD ay naghahanda ng food packs para sa mga pamilyang lumikas dahil sa bulkang aktibidad sa {_place()}.", "official"),
        (f"Red Cross distributing relief goods to volcano-affected families in {_place()}. Operations 7am to 5pm.", "official"),
        (f"Hydration stations set up along evacuation routes from {_place()} due to high heat index. Water and ORS available.", "official"),
        (f"NFI distribution: hygiene kits and sleeping mats for {_n()} displaced families from {_place()} volcanic zone.", "official"),
        (f"EMERGENCY: Tubig! Walang tubig dito sa {_place()}! Grabe ang init tapos wala na ring minom! Tulong po!", "panic"),
        (f"Mga buntis at matatanda sa {_place()} bibigyan ng priority sa food and water distribution ngayong mainit.", "official"),
        (f"Ang mga residente mula sa 4km danger zone ng {_place()} volcano ay tumatanggap na ng relief goods.", "official"),
        (f"Community feeding program active in {_place()} for families unable to cook due to power outage from eruption.", "official"),
        (f"Libreng handa sa {_place()} community kitchen. Priority sa mga bata at matatanda ngayong mainit ang panahon.", "personal"),
        (f"Relief goods convoy arrives at {_place()} from provincial government. 500 families to receive assistance today.", "official"),
        (f"URGENT: Kulang na ang bottled water sa {_place()} relief center! Requesting immediate resupply from LGU!", "urgent"),
        (f"Mga donation ng tubig at pagkain para sa mga evacuees mula {_place()} volcano ay tinatanggap sa barangay hall.", "personal"),
        (f"Oral rehydration salts at energy drinks ibinibigay ng health center sa {_place()} para sa heat-related cases.", "official"),
        (f"Ang mga naapektuhang pamilya sa {_place()} ay bibigyan ng emergency food assistance ngayong linggo.", "official"),
        (f"Food packs with high-energy biscuits and canned goods being distributed in {_place()} volcanic evacuation area.", "official"),
        (f"Ayuda para sa mga walang hanapbuhay dahil sa volcanic unrest sa {_place()}. I-register sa DSWD ngayon.", "official"),
        (f"Nakatanggap na kami ng libreng pagkain at tubig mula sa LGU {_muni()}. Maraming salamat sa aming barangay!", "personal"),
        (f"DSWD mobile relief team deployed to {_place()} communities affected by ashfall from volcanic eruption.", "official"),
        (f"Water refill stations now at 5 strategic points in {_place()}. Free for all residents during heat emergency.", "official"),
        (f"Tagapamahala ng barangay sa {_place()} nagbibigay ng emergency water supply sa bawat pamilya ngayong araw.", "official"),
        (f"Heatwave relief: cold water, ORS, and snacks available at {_place()} community center. Libre para sa lahat.", "official"),
        (f"Ang mga pamilyang lumikas mula sa {_place()} ay tumatanggap ng dalawang food pack at bottled water bawat isa.", "official"),
        (f"Donation drive for {_place()} evacuees: accepting canned goods, rice, bottled water, and hygiene items.", "personal"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


def _cluster_b_posts_ext(base_idx: int) -> list[dict]:
    """30 extra cluster-b posts: heat index advisories, air quality, volcanic health risks."""
    cid = "cluster-b"
    templates = [
        (f"HEAT INDEX ADVISORY: Apparent temperature in {_place()} reaches 45°C. DANGER level. Avoid outdoor activities 10am-4pm.", "urgent"),
        (f"MnlCDRRMD heat advisory: Heat index sa {_place()} ay nasa DANGER level na. Umiwas sa panlabas na gawain.", "urgent"),
        (f"DOH nagbababala: Dehydration at heat stroke ang pangunahing panganib sa {_place()} ngayong araw. Uminom ng maraming tubig.", "official"),
        (f"Extreme heat alert for {_place()}. Heat index forecast at 43°C to 46°C. At-risk: elderly, children, outdoor workers.", "urgent"),
        (f"PAGASA issues heat index warning for {_place()} and surrounding areas. Feels like 47°C. Stay indoors if possible.", "urgent"),
        (f"Air quality advisory: {_place()} classified VERY UNHEALTHY due to volcanic ashfall. Wear N95 mask outdoors.", "urgent"),
        (f"DOST-EMB air quality index sa {_place()} ay 201-300 VERY UNHEALTHY. Mag-stay sa loob ng bahay.", "urgent"),
        (f"Volcanic ash from {_place()} volcano detected. Air quality hazardous. Wear masks and close windows.", "urgent"),
        (f"Heat exhaustion cases reported at {_place()} outdoor market. DOH team dispatched. Symptoms: dizziness, weakness.", "urgent"),
        (f"Sunstroke warning for outdoor workers in {_place()}. Wear hat, drink water every 15 minutes. Avoid direct sunlight.", "official"),
        (f"Pediatric advisory: Huwag palabasin ang mga bata sa labas ng bahay sa {_place()} dahil sa matinding init.", "official"),
        (f"Cool centers open at {_place()} covered court and gymnasium. Free air conditioning for senior citizens and children.", "official"),
        (f"ALERT: Ashfall advisory for {_place()}. Keep children indoors. Ashfall causes respiratory irritation and eye problems.", "urgent"),
        (f"Ang air quality sa {_place()} ay nasa hazardous level. Huwag lumabas nang walang mask. DOH nag-aalerto.", "urgent"),
        (f"Leptospirosis risk remains HIGH in {_place()} post-flood areas. Still avoid wading even after waters recede.", "official"),
        (f"Heat stroke emergency: 3 patients from {_place()} admitted to hospital. Walang pasok para sa mga outdoor workers.", "urgent"),
        (f"UV index EXTREME sa {_place()} ngayon. Magbihis ng makapal at magsuot ng sombrero kapag lumabas.", "official"),
        (f"DOH issues dengue alert alongside heat advisory in {_place()}. Remove stagnant water. Use mosquito repellent.", "official"),
        (f"WASH advisory: Boil all drinking water in {_place()} following ashfall contamination of water sources.", "official"),
        (f"Medical tents with cooling stations deployed at {_place()} evacuation center for heat-stressed evacuees.", "official"),
        (f"Huwag kalimutang uminom ng tubig! Heat index sa {_place()} ay nakakatakot na mainit ngayon. Mag-ingat.", "personal"),
        (f"Ang mga may asthma at respiratory conditions ay dapat manatili sa loob ng bahay sa {_place()} dahil sa abo.", "official"),
        (f"Eye wash stations set up at {_place()} health center for residents affected by ashfall irritation.", "official"),
        (f"Respiratory health alert: Fine particulate matter levels in {_place()} exceeding safe limits due to volcanic emissions.", "urgent"),
        (f"Cool center at {_place()} gymnasium open 6am to 10pm. Libreng tubig, electric fan, at first aid on standby.", "official"),
        (f"WHO at DOH nagtutulungan sa {_place()} para tugunan ang heat-related illnesses. Pumunta sa pinakamalapit na klinika.", "official"),
        (f"Heat advisory from PAGASA: {_place()} and nearby municipalities to experience 'feels like' temperature of 48°C tomorrow.", "urgent"),
        (f"Nag-aalaga ng lolo naming may hypertension sa {_place()}. Sobrang init. Saan kami makakakuha ng libre na konsultasyon?", "personal"),
        (f"Volcanic gases detected near {_place()} communities. Sulfur dioxide levels elevated. Residents advised to wear masks.", "urgent"),
        (f"DOH-{_muni()} nagpapadala ng health team para subaybayan ang mga heat stroke at dehydration cases sa {_place()}.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


def _cluster_c_posts_ext(base_idx: int) -> list[dict]:
    """30 extra cluster-c posts: volcanic eruption evacuations, typhoon signal evacuations."""
    cid = "cluster-c"
    templates = [
        (f"PHIVOLCS raises Alert Level 3 for {_place()} volcano. Residents within 6km danger zone must evacuate NOW.", "panic"),
        (f"MANDATORY EVACUATION: Alert Level 4 declared for {_place()} volcano. All residents must leave immediately.", "panic"),
        (f"Ang mga nakatira sa loob ng 6km permanent danger zone ng {_place()} bulkan ay KAILANGANG LUMIKAS NA.", "panic"),
        (f"Evacuation centers for {_place()} volcano evacuees now open at {_muni()} covered court and gymnasium.", "official"),
        (f"Volcanic eruption imminent at {_place()}. PHIVOLCS orders total evacuation of all communities near the volcano.", "panic"),
        (f"Typhoon Signal No. 4 raised over {_place()}. Mandatory pre-emptive evacuation of coastal and low-lying areas.", "panic"),
        (f"PAGASA raises Typhoon Signal No. 3 in {_place()}. LGU orders evacuation of flood-prone barangays tonight.", "urgent"),
        (f"Lumikas na agad ang mga nakatira sa coastal areas ng {_place()} dahil sa Signal No. 4 na idineklara.", "panic"),
        (f"{_n()} families evacuated from {_place()} volcanic danger zone. Temporary shelter at {_muni()} Sports Complex.", "official"),
        (f"Evacuation ongoing: residents from ashfall-heavy areas of {_place()} being moved to secondary evacuation centers.", "official"),
        (f"LGU nagbubukas ng 5 dagdag na evacuation centers sa {_place()} para sa mga lumikas mula sa bulkan.", "official"),
        (f"Naglumikas na po kami dahil sa bulkan sa {_place()}. Nandito na kami sa covered court. Ligtas naman.", "personal"),
        (f"Puno na ang unang evacuation center sa {_place()}. Pumunta na sa Sports Hub bilang secondary shelter.", "urgent"),
        (f"Ang mga evacuees mula sa {_place()} volcanic zone ay maaaring mag-register sa DSWD para sa assistance.", "official"),
        (f"Typhoon update: {_place()} covered court evacuation center now accommodating {_n()} displaced families.", "official"),
        (f"PHIVOLCS advisory: Lava flows possible at {_place()} within 24 hours. Do not return to danger zone.", "urgent"),
        (f"LGU nagba-bawal ng pagbabalik ng mga evacuees sa {_place()} hanggang mababa ang alert level ng bulkan.", "official"),
        (f"Camp management team deploying to {_place()} evacuation site to handle {_n()} volcano displaced families.", "official"),
        (f"NDRRMC activates full response for {_place()} eruption. Pre-positioned relief goods and rescue teams on standby.", "official"),
        (f"Hindi pa pwedeng umuwi ang mga taga-{_place()}. Aktibo pa ang bulkan. Manatili sa evacuation center.", "official"),
        (f"Nagpapadala ng karagdagang cots at kumot sa {_place()} evacuation centers para sa mga bagong dating na evacuees.", "official"),
        (f"Families from {_place()} danger zones transported via LGU buses to inland evacuation shelters.", "official"),
        (f"Typhoon-induced flooding in {_place()} forces {_n()} additional families into evacuation. Centers now at capacity.", "urgent"),
        (f"Volcano alert in {_place()}: Barangay captains conducting door-to-door checks to ensure all residents evacuated.", "official"),
        (f"Overcrowding reported at {_place()} evacuation site. DSWD opening additional facility at municipal plaza.", "urgent"),
        (f"PHIVOLCS confirms lava fountaining at {_place()} crater. Lahars possible in river channels. Evacuate river zones.", "urgent"),
        (f"Signal No. 2 pa rin sa {_place()}. LGU nagtatayo ng karagdagang evacuation center para sa mga hindi pa lumikas.", "official"),
        (f"Mga evacuees sa {_place()} covered court ay binibigyan ng sleeping area, kumot, at pagkain. Maayos naman.", "personal"),
        (f"Evacuation update: All residents within permanent danger zone of {_place()} volcano accounted for. Operations ongoing.", "official"),
        (f"DRRMO {_muni()} coordinating with PHIVOLCS for updated danger zone maps for {_place()} volcano evacuation.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


def _cluster_d_posts_ext(base_idx: int) -> list[dict]:
    """30 extra cluster-d posts: ashfall road closures, lahar logistics, typhoon aftermath roads."""
    cid = "cluster-d"
    templates = [
        (f"Ashfall rendering roads in {_place()} slippery and dangerous. DPWH deploying water trucks for road washing.", "urgent"),
        (f"Lahar flow blocking the main highway near {_place()}. DPWH monitoring. Alternate route via mountain road.", "urgent"),
        (f"Road advisory: thick ashfall deposit on {_place()} national highway. Reduce speed. Use headlights. Wear mask.", "official"),
        (f"DPWH road clearing underway in {_place()} after ashfall. Roads expected to be fully passable by tomorrow morning.", "official"),
        (f"Ang lahar mula sa bulkan ay tumatawid sa kalsada ng {_place()}. Sarado na ang daan. Gumamit ng alternative route.", "urgent"),
        (f"Relief convoy to {_place()} delayed by ashfall-covered roads. DPWH is clearing. ETA revised to 6pm.", "urgent"),
        (f"Road visibility near zero in {_place()} due to heavy ashfall. Motorists advised to pull over and wait.", "urgent"),
        (f"Typhoon aftermath: fallen trees block 12 roads in {_place()} municipality. Clearing operations underway.", "official"),
        (f"DPWH heavy equipment clearing lahar deposits from the access road to {_place()}. 4 hours estimated clearance time.", "official"),
        (f"Supply delivery to isolated barangays in {_place()} by boat due to all land routes blocked by typhoon debris.", "official"),
        (f"Ang lahar ay dumating na sa bayan ng {_place()}. Huwag na lumabas. Sarado lahat ng daan.", "panic"),
        (f"Road update: {_place()} national road now passable for light vehicles after DPWH ashfall clearing operations.", "official"),
        (f"Bridge at {_place()} flooded with lahar debris. Engineers inspecting structural integrity before reopening.", "official"),
        (f"Debris flow from {_place()} volcano closes 3 major roads. Relief goods delivery using air transport.", "urgent"),
        (f"DPWH ash-clearing trucks deployed overnight in {_place()}. Expect limited visibility and road narrowing.", "official"),
        (f"Nablock na lahat ng daan sa {_place()} dahil sa lahar. Mga relief trucks nagde-detour na sa kabila.", "urgent"),
        (f"Typhoon road damage report for {_place()}: 5 roads closed, 8 bridges assessed for structural damage.", "official"),
        (f"Logistics team coordinating chopper delivery to cut-off communities in {_place()} after roads washed out.", "official"),
        (f"Ash deposits 10cm thick on {_place()} highway. DPWH recommends complete avoidance until cleared.", "urgent"),
        (f"Road to {_place()} volcano observation post closed due to volcanic hazard. Only PHIVOLCS personnel allowed.", "official"),
        (f"Alternate road to {_place()} via provincial bypass now operational. DPWH opens route for relief convoys.", "official"),
        (f"Ferry service suspended in {_place()} due to rough seas from typhoon. Supply barges on standby.", "official"),
        (f"Ang kalsada papunta sa {_place()} ay natatakpan ng makapal na abo. DPWH nag-aayos. Huwag muna pumunta.", "official"),
        (f"Lahar monitoring in {_place()}: flow detected at 2m height in river channel. Road closures preemptive.", "official"),
        (f"Road safety alert: {_place()} national highway has 15cm ashfall layer. Motorcycles banned from this route.", "official"),
        (f"Evacuation route from {_place()} volcanic danger zone reopened after DPWH cleared debris from main road.", "official"),
        (f"DPWH nagtatrabaho nang walang tigil para linisin ang kalsada ng {_place()} mula sa lahar at abo ng bulkan.", "official"),
        (f"Air drops of relief goods initiated for {_place()} communities inaccessible by land due to lahar flow.", "official"),
        (f"Ang tulay sa {_place()} ay nasira ng baha at lahar. Kailangan ng ilang linggo bago maayos.", "official"),
        (f"Road passability update: 6 of 9 routes to {_place()} now open. 3 still blocked by debris and ashfall.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


def _cluster_e_posts_ext(base_idx: int) -> list[dict]:
    """30 extra cluster-e posts: power/comms disruption from eruptions, heat, typhoons."""
    cid = "cluster-e"
    templates = [
        (f"Ashfall causes power outage in {_place()}. Meralco isolating affected lines. Restoration estimate: 12 hours.", "official"),
        (f"Volcanic eruption damaged power infrastructure near {_place()}. Blackout expected for 2 to 3 days.", "urgent"),
        (f"Cell towers coated with volcanic ash in {_place()}. Globe and Smart signal degraded. Engineers deploying.", "official"),
        (f"Walang kuryente sa {_place()} dahil sa abo ng bulkan. Meralco nag-iinspeksyon ng mga linya.", "urgent"),
        (f"Typhoon downed 12 power poles in {_place()}. Meralco crew working overnight restoration. ETA 24 hours.", "official"),
        (f"No signal and no power in {_place()} volcano evacuation zone. Emergency radio only communication channel.", "urgent"),
        (f"PLDT fiber cut in {_place()} due to ashfall weight on cables. Internet service disrupted for area.", "official"),
        (f"Heat wave causing power demand surge in {_place()}. Meralco warns of possible brownout 1pm to 6pm.", "urgent"),
        (f"Brownout schedule sa {_place()} ngayong araw dahil sa overloading ng grid dahil sa mainit na panahon.", "official"),
        (f"Wala na kaming signal dito sa evacuation center ng {_place()}! Globe at Smart patay na. Radio lang.", "urgent"),
        (f"Power restored to 60% of {_place()} after typhoon damage repairs. Remaining areas by end of day.", "official"),
        (f"Emergency satellite communication deployed in {_place()} for disaster coordination after cell tower damage.", "official"),
        (f"NTC advisory: Temporary frequency spectrum allocation for disaster communications in {_place()} area.", "official"),
        (f"Ang mga generator ng ospital sa {_place()} ay gumagana nang maayos kahit walang kuryente sa buong lungsod.", "official"),
        (f"Cell on Wheels (COW) unit deployed to {_place()} evacuation center. Signal now available for residents.", "official"),
        (f"Power interruption sa {_place()}: Meralco nagtatayo ng temporary lines para sa evacuation centers muna.", "official"),
        (f"Walang internet ang mga evacuees sa {_place()}. May libre-charging station sa evacuation center.", "official"),
        (f"Smart at Globe nagsasagawa ng emergency network restoration sa {_place()} matapos ang bagyo.", "official"),
        (f"Ashfall-induced brownout in {_place()}: short circuit from ash accumulation on transformer. Clearing ongoing.", "urgent"),
        (f"Emergency radio broadcast activated for {_place()}: tune to 702 DZAS for 24-hour disaster updates.", "official"),
        (f"Generators distributed to {_place()} evacuation centers by NDRRMC. Fuel supply for 72 hours.", "official"),
        (f"Power demand too high sa {_place()} dahil lahat gumagamit ng aircon. Posibleng brownout ngayong hapon.", "personal"),
        (f"Meralco linemen nagtatrabaho sa tag-araw na init para maibalik ang kuryente sa {_place()}. Salamat!", "personal"),
        (f"Typhoon blew out main transformer serving {_place()}. Replacement unit being transported from {_muni()}.", "official"),
        (f"Solar charging stations set up by DSWD in {_place()} evacuation center for displaced residents.", "official"),
        (f"NTC coordinating with telcos for priority restoration of {_place()} DRRMO communication lines.", "official"),
        (f"Internet access via satellite terminal available at {_place()} municipal hall for disaster coordination.", "official"),
        (f"Walang kuryente at signal sa {_place()} nang 3 araw na. Sana mabilis maibalik ng Meralco at Globe.", "personal"),
        (f"Temporary cell tower installed at {_place()} evacuation area. Signal strength 3G. Voice calls operational.", "official"),
        (f"All communication lines to {_place()} restored after 36-hour outage from volcanic eruption damage.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


def _cluster_f_posts_ext(base_idx: int) -> list[dict]:
    """30 extra cluster-f posts: class suspensions due to typhoon/flood/eruption.
    IMPORTANT: Do NOT use 'air quality', 'unhealthy', 'advisory', 'landfill', 'facemask',
    'respiratory', 'hika' — those belong to cluster-b and cause CorEx bleed.
    Anchor vocab: walang pasok, walang klase, DepEd, school, paaralan, estudyante, guro."""
    cid = "cluster-f"
    templates = [
        (f"WALANG PASOK bukas sa {_place()}! DepEd nag-suspend ng klase dahil sa malakas na ulan at baha.", "urgent"),
        (f"DepEd suspends classes in {_place()} due to Typhoon Signal No. 3. All schools closed tomorrow.", "official"),
        (f"Walang klase bukas sa lahat ng paaralan sa {_place()}. DepEd order dahil sa bagyo.", "urgent"),
        (f"Schools in {_place()} closed due to ashfall from volcanic eruption. DepEd to announce return date.", "urgent"),
        (f"WALANG PASOK: Ashfall from {_place()} volcano covers school grounds. Students must stay home.", "urgent"),
        (f"DepEd issues class suspension for {_place()} schools. Ashfall makes roads impassable for students.", "official"),
        (f"School closure in {_place()}: buildings covered with volcanic ash. Cleaning needed before reopening.", "official"),
        (f"No face-to-face classes in {_place()} due to Typhoon Signal No. 2. Modular learning activated.", "official"),
        (f"DepEd advisory sa {_place()}: lahat ng paaralan ay sarado bukas dahil sa Signal No. 3.", "official"),
        (f"Schools used as evacuation centers in {_place()} during typhoon. Classes suspended indefinitely.", "official"),
        (f"Walang pasok! Malakas ang bagyo sa {_place()}. Hindi ligtas ang mga estudyante. DepEd nag-suspend.", "personal"),
        (f"DepEd suspends classes in {_place()} for 3 days due to typhoon damage to school buildings.", "official"),
        (f"DepEd distributing ashfall cleanup kits to schools in {_place()}. Brooms and disinfectant included.", "official"),
        (f"Academic year extended for {_place()} schools disrupted by volcanic activity. DepEd announcement.", "official"),
        (f"Mga guro sa {_place()} nagboboluntaryo sa evacuation center habang walang klase.", "personal"),
        (f"No classes na naman bukas! Bagyo ulit sa {_place()}. DepEd suspends ulit ang pasok.", "personal"),
        (f"Schools in {_place()} to remain closed for 1 week due to typhoon and flood damage.", "official"),
        (f"Typhoon-displaced students from {_place()} given printed modules while evacuation centers active.", "official"),
        (f"Class suspension in {_place()}: schools need 3 days to clean ashfall before reopening.", "official"),
        (f"DepEd releases distance learning materials for {_place()} students during school closure.", "official"),
        (f"Walang klase sa {_place()} hanggang mababa ang signal number. DepEd nagmo-monitor ng sitwasyon.", "official"),
        (f"School principals in {_place()} checking buildings for flood damage before students return.", "official"),
        (f"Return to school date for {_place()} announced: Monday next week after DepEd building inspection.", "official"),
        (f"Mga estudyante sa {_place()} ay nakatanggap ng printed modules habang sarado ang kanilang paaralan.", "official"),
        (f"No classes in {_place()} tomorrow. Typhoon signal raised. DepEd prioritizes student safety.", "official"),
        (f"DepEd {_place()} says schools will reopen once floodwaters recede and buildings are cleared.", "official"),
        (f"College students in {_place()} told to stay home. Typhoon signal raised. Exams moved to next week.", "official"),
        (f"DepEd suspends enrollment activities in {_place()} due to ongoing typhoon. Rescheduled next week.", "official"),
        (f"School canteen operations suspended in {_place()} evacuation center schools. Students receive packed meals.", "official"),
        (f"Teachers in {_place()} checking Google Classroom attendance during typhoon-related school suspension.", "personal"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


def _cluster_g_posts_ext(base_idx: int) -> list[dict]:
    """30 extra cluster-g posts: lahar rescues, heat stroke emergencies, volcanic rescue ops."""
    cid = "cluster-g"
    templates = [
        (f"RESCUE NEEDED: Family trapped by lahar flow in {_place()}! Water and mud at rooftop level! SOS!", "panic"),
        (f"Lahar rescue operations ongoing in {_place()} river channel. Army and BFP boats deployed for stranded residents.", "urgent"),
        (f"SOS! Naiipit ang aming pamilya sa lahar sa {_place()}! Hindi na kami makalabas sa bahay! Tulong!", "panic"),
        (f"Philippine Coast Guard rescuing fishermen stranded at sea near {_place()} after typhoon signal upgrade.", "urgent"),
        (f"BFP and NDRRMC rescue team extracting {_n()} families from lahar-inundated areas of {_place()}.", "official"),
        (f"Heat stroke emergency: 5 outdoor workers collapsed in {_place()} needing immediate medical rescue.", "urgent"),
        (f"RESCUE: Elderly resident found unconscious due to heat stroke at {_place()}. Ambulance dispatched.", "urgent"),
        (f"Hikers rescued from {_place()} mountain trail after being stranded by volcanic activity. All safe.", "official"),
        (f"Rescue swimmer needed at {_place()} lahar crossing! Vehicle swept away! Occupants still inside!", "panic"),
        (f"Philippine Air Force helicopter rescuing communities isolated by lahar flows in {_place()} volcano slopes.", "official"),
        (f"SOS mula sa {_place()}: isang lolo at dalawang bata ay naiipit sa lahar! Kailangan ng rescue agad!", "panic"),
        (f"Ang rescue team ay nakarating na sa {_place()}. Nagsisimula na ang pagliligtas sa mga naiipit sa lahar.", "official"),
        (f"USAR team deployed to {_place()} after lahar buried 3 houses. Rescue dogs and thermal cameras in use.", "urgent"),
        (f"5 fishermen missing after boat capsized near {_place()} coast. PCG search and rescue ongoing.", "urgent"),
        (f"River rescue at {_place()}: swift water rescue team recovering family stranded on mid-river debris.", "urgent"),
        (f"BFP responds to fire at {_place()} barangay. 2 families rescued from burning structure. Fire under control.", "official"),
        (f"Lahar rescue complete in {_place()}. All 23 stranded families evacuated. No casualties reported.", "official"),
        (f"Mountain rescue: SARDA team deployed for 8 missing hikers in {_place()} after trail buried by landslide.", "urgent"),
        (f"TXTFIRE! May sunog sa {_place()}! Naiipit ang tatlong tao! Tumawag sa BFP ngayon! 911!", "panic"),
        (f"Rescue boat needed at {_place()} lahar danger zone! Nanay at tatlong bata stranded. Rescue team padating.", "panic"),
        (f"PCG conducting search for missing boat with {_n()} passengers near {_place()} during typhoon.", "urgent"),
        (f"Rescue team ng NDRRMC nagliligtas ng mga pamilya sa {_place()} mula sa banta ng lahar ng bulkan.", "official"),
        (f"All stranded residents in {_place()} volcanic zone have been safely evacuated. Rescue operations complete.", "official"),
        (f"Heat-related rescue in {_place()}: elderly woman in distress found on street during extreme heat advisory.", "urgent"),
        (f"Swift water rescue team training deployed to actual operations at {_place()} river during lahar event.", "official"),
        (f"Volunteer rescue swimmers responding to {_place()} lahar zone. Requesting additional boats from LGU.", "urgent"),
        (f"HELP! Stuck sa bubong ng bahay sa {_place()} habang tumataas ang lahar! May kasama akong sanggol!", "panic"),
        (f"Rescue operations in {_place()}: BFP, PNP, and Army coordinating extraction of 40 stranded residents.", "official"),
        (f"Naligtas na ang pamilya mula sa lahar sa {_place()}. Salamat sa Coast Guard at Army rescue teams.", "personal"),
        (f"Rescue helicopter from {_place()} airlifts 12 persons stranded by volcanic lahar in riverside community.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


def _cluster_h_posts_ext(base_idx: int) -> list[dict]:
    """30 extra cluster-h posts: volcanic fatalities, heat-related deaths, typhoon death tolls."""
    cid = "cluster-h"
    templates = [
        (f"NDRRMC reports 3 fatalities from lahar flow in {_place()}. Families notified. Bodies at provincial morgue.", "official"),
        (f"Death toll from volcanic eruption in {_place()} rises to 7. Recovery teams still searching for missing.", "official"),
        (f"Heat-related death confirmed in {_place()}. DOH investigating heat stroke fatality. Public warned.", "urgent"),
        (f"2 persons confirmed dead after lahar swept through {_place()} community. 4 still missing.", "official"),
        (f"NDRRMC casualty update: {_n()} dead, 12 missing from typhoon in {_place()}. Recovery operations ongoing.", "official"),
        (f"Body of missing fisherman from {_place()} recovered at {_muni()} shoreline. PCG coordinating with family.", "official"),
        (f"Death toll from heat stroke in {_place()} province rises to 4. DOH urges extreme caution during heat wave.", "official"),
        (f"Remains of 2 victims of {_place()} volcanic lahar identified at provincial morgue. Burial rites ongoing.", "official"),
        (f"Missing persons report: 5 residents of {_place()} unaccounted for since volcanic eruption 3 days ago.", "official"),
        (f"Ang bilang ng patay mula sa baha sa {_place()} ay tumaas na sa 9. Naghahanap pa ng 6 na nawawala.", "official"),
        (f"MISSING: Tatay ko si Rolando, 67, wala pa rin simula nang erupsyon ng bulkan sa {_place()}. Tulong!", "urgent"),
        (f"Family tracing for {_place()} volcanic eruption: 15 persons still unaccounted for. Report to DSWD MDM desk.", "official"),
        (f"Official death toll update for typhoon in {_place()}: 18 confirmed dead, 7 missing, 250 injured.", "official"),
        (f"Post-mortem results for heat stroke victim from {_place()}: cause of death heat hyperthermia. Third case this week.", "official"),
        (f"Missing person alert: Maria Dela Cruz, 45, last seen fleeing lahar in {_place()}. Wearing red blouse.", "urgent"),
        (f"Body found in lahar deposit near {_place()} river. Identity being confirmed. Family may approach MDM desk.", "official"),
        (f"Natuklasan ang 2 katawan sa ilalim ng lahar ng bulkan sa {_place()}. Dinala na sa probinsyal na morgue.", "official"),
        (f"Ang DSWD MDM team ay aktibo sa {_place()} para sa family tracing at death certificate processing.", "official"),
        (f"Typhoon death toll in {_place()} confirmed at {_n()} by NDRRMC. Recovery operations continuing.", "official"),
        (f"Heat wave claims 2 lives in {_place()}. Both elderly residents. DOH urges public to check on seniors.", "official"),
        (f"MISSING: Lola Nena, 78, nawawala mula nang lumikas sa volcanic eruption sa {_place()}. Pink duster.", "urgent"),
        (f"Identification of volcanic eruption victims in {_place()} ongoing. NDRRMC using ante-mortem data records.", "official"),
        (f"Typhoon casualty: body of missing resident from {_place()} found 5km from home. Family contacted.", "official"),
        (f"Death toll from flash flood in {_place()} confirmed at 6. NDRRMC concludes search after 7 days.", "official"),
        (f"Missing fishermen from {_place()} declared dead after 30-day search. Bodies never recovered. Certificates issued.", "official"),
        (f"Ang pamilya ng mga biktima ng lahar sa {_place()} ay maaaring pumunta sa MDM desk para sa tulong.", "official"),
        (f"1 dead, 3 missing after landslide in {_place()}. NDRRMC rescue team recovers body from debris.", "official"),
        (f"NDRRMC confirms 0 typhoon fatalities in {_place()} due to preemptive evacuation. Officials commend LGU action.", "official"),
        (f"Post-disaster mortality review for {_place()}: 22 total deaths, 8 from drowning, 6 from trauma, 8 disease-related.", "official"),
        (f"Family tracing desk at {_place()} evacuation center: 47 persons reunited with families so far this week.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-c extended v3: PHIVOLCS alert level / volcano evacuation ──────────
# Mirrors real NDRRMC/PHIVOLCS posts: Alert Level 3/4, permanent danger zone,
# lava flows, pyroclastic density currents, Mayon/Kanlaon, effusive eruption.

def _cluster_c_posts_phivolcs(base_idx: int) -> list[dict]:
    cid = "cluster-c"
    templates = [
        ("PHIVOLCS raises Alert Level 3 over Mayon Volcano. Entry into the 6-km Permanent Danger Zone (PDZ) is strictly prohibited.", "urgent"),
        ("Alert Level 4 declared for Kanlaon Volcano. All residents within 8-km danger zone must evacuate immediately.", "panic"),
        ("NDRRMC advisory: Effusive eruption at Mayon Volcano persists. Lava flows and rockfalls ongoing. Danger zone strictly enforced.", "urgent"),
        ("PHIVOLCS-DOST: Strombolian activity observed at Mayon summit crater. Pyroclastic density currents possible. Stay outside danger zone.", "urgent"),
        ("Mayon Volcano Alert Level 3 remains in effect. Incandescent lava flows and PDC activity recorded. Evacuation order stands.", "urgent"),
        ("Kanlaon Watch: Alert Level 2 raised. Residents within 4-km radius must pre-emptively evacuate. NDRRMC on full alert.", "urgent"),
        ("PHIVOLCS nagpapanatili ng Alert Level 3 sa Mayon Bulkan. Ipinagbabawal ang pagpasok sa 6-km Permanent Danger Zone.", "urgent"),
        ("Ang effusive eruption ng Mayon Volcano ay nagpapatuloy. Lava flows, PDC, at rockfalls aktibo. Manatiling malayo sa danger zone.", "urgent"),
        ("NDRRMC activates full disaster response for Mayon Volcano eruption. Evacuation of 6-km PDZ ongoing. Relief operations begin.", "official"),
        ("Volcano bulletin: Kanlaon shows increased unrest. Alert Level 3 raised. Mandatory evacuation within 6-km radius now in effect.", "panic"),
        ("Mayon Volcano lava flow reaches 3.2 km from summit. PHIVOLCS warns of possible pyroclastic surge. Danger zone strictly enforced.", "urgent"),
        ("Sulphur dioxide emission from Mayon Volcano at hazardous levels. All residents within PDZ must have evacuated by tonight.", "urgent"),
        ("PHIVOLCS update: Mayon crater glow intensifying. Possibility of explosive eruption cannot be ruled out. Alert Level 3 maintained.", "urgent"),
        ("Lahar alert in Mayon river channels during rain. Residents in river valleys must evacuate regardless of alert level.", "urgent"),
        ("NDRRMC at PHIVOLCS nagbababala: Huwag pumasok sa Permanent Danger Zone ng Mayon. Aktibo ang lava flow at PDC ngayon.", "urgent"),
        ("Kanlaon Volcano Alert Level 3: LGU buses ferrying evacuees from danger zone to designated evacuation centers in lowland areas.", "official"),
        ("Pyroclastic density currents or 'uson' recorded at Mayon today. Extreme danger within 6-km PDZ. Do not return to danger zone.", "panic"),
        ("Alert Level 3 sa Mayon: Mahigpit na ipinagbabawal ang pagpasok sa 6-km PDZ. Ang mga sumusuway ay pananagutin.", "urgent"),
        ("PHIVOLCS volcano bulletin: Mayon showing continued effusive activity for the 110th consecutive day. Danger zone remains enforced.", "official"),
        ("NDRRMC confirms all residents within Mayon 6-km PDZ have been evacuated. Monitoring continues 24 hours.", "official"),
        # Real NDRRMC/PHIVOLCS post style — using exact vocabulary from real Apify exports
        ("LOOK: Close-up footage of minor Strombolian activity at the summit crater of Mayon Volcano captured by PHIVOLCS Quick Response Team. Effusive eruption persists producing incandescent lava flows pyroclastic density currents PDC known as uson and frequent rockfalls. Alert Level 3 remains in effect. Entry into the 6-km Permanent Danger Zone strictly prohibited. Source PHIVOLCS-DOST NDRRMC MayonVolcano.", "urgent"),
        ("KANLAON WATCH: Close-up footage of minor Strombolian activity at summit crater of Mayon Volcano captured by Mayon Volcano Observatory. Effusive eruption persists for the 110th consecutive day producing incandescent lava flows pyroclastic density currents PDC uson and frequent rockfalls. Alert Level 3 remains in effect. Entry into the 6-km Permanent Danger Zone strictly prohibited. PHIVOLCS-DOST NDRRMC.", "urgent"),
        ("PHIVOLCS bulletin: Effusive eruption at Mayon Volcano persists for the 111th consecutive day. Incandescent lava flows and rockfalls observed. Pyroclastic density currents uson possible. Alert Level 3. Entry into PDZ strictly prohibited.", "urgent"),
        ("Mayon Volcano update: Strombolian activity recorded at summit crater. Lava fountain and incandescent rockfalls observed at 6pm tonight. Alert Level 3 maintained. Permanent Danger Zone entry prohibited. Source PHIVOLCS-DOST.", "urgent"),
        ("NDRRMC BawatSegundoMahalaga: Effusive eruption ng Mayon Volcano ay nagpapatuloy. Incandescent lava flows pyroclastic density currents at rockfalls ang naitala. Alert Level 3 sa Mayon. Ipinagbabawal ang pagpasok sa 6-km PDZ.", "urgent"),
        ("Kanlaon Watch PHIVOLCS: Minor Strombolian activity observed at Kanlaon summit crater. Lava fragments and incandescent rockfalls recorded. Alert Level 3 raised. Evacuation of 6-km danger zone ordered by NDRRMC.", "urgent"),
        ("PHIVOLCS-DOST volcano bulletin: Mayon effusive eruption day 112. Lava flow advancing 3km southeast. Pyroclastic density currents PDC generated. Rockfalls frequent. Alert Level 3 in effect. Danger zone access strictly prohibited.", "urgent"),
        ("Mayon Volcano Observatory: Incandescent lava fountaining recorded at summit crater. PDC uson generated along gullies. Sulfur dioxide emissions elevated. Alert Level 3 remains. Do not enter 6-km Permanent Danger Zone.", "urgent"),
        ("LOOK: Mayon Volcano lava flow and rockfall footage captured tonight. Effusive eruption continues for consecutive days. Pyroclastic density currents active. PHIVOLCS maintains Alert Level 3. Permanent Danger Zone strictly enforced.", "urgent"),
        ("NDRRMC advisory on Mayon Volcano: Effusive eruption producing lava flows uson pyroclastic currents and rockfalls. Alert Level 3. All residents within 6-km PDZ must remain evacuated. Do not return without PHIVOLCS clearance.", "urgent"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── cluster-b extended v2: Manila DRRM air quality advisory style ─────────────
# Mirrors real MnlCDRRMD posts: air quality sensors, facemask advisory,
# vulnerable groups (bata, senior, buntis, hika), landfill fire smoke, hotlines.

def _cluster_b_posts_ext2(base_idx: int) -> list[dict]:
    cid = "cluster-b"
    templates = [
        ("ADVISORY: MAGSUOT NG FACEMASK. Naitala ang VERY UNHEALTHY sa Air Quality Sensors ng Lungsod ng Maynila. Iwasan ang paglabas kung hindi kinakailangan.", "urgent"),
        ("Air quality advisory: VERY UNHEALTHY level detected in Manila and nearby cities. Ang mga bata, senior citizens, buntis, at may hika ay dapat manatili sa loob.", "urgent"),
        ("MAGSUOT NG FACEMASK! Mababa ang kalidad ng hangin ngayon dahil sa usok mula sa sunog sa Navotas Landfill. Huwag lumabas nang walang proteksyon.", "urgent"),
        ("MnlCDRRMD advisory: Air quality sensors sa Maynila ay nagpapakita ng VERY UNHEALTHY. Mga may respiratory illness ay ipinagbabawal lumabas.", "urgent"),
        ("ALERTO: Ang usok mula sa sunog sa landfill ay nakakaapekto sa kalidad ng hangin sa Maynila at karatig-siyudad. Magsuot ng N95 mask.", "urgent"),
        ("Air quality update sa Maynila: VERY UNHEALTHY level. Possibleng dahilan ang usok mula sa Navotas Landfill. Manatiling alerto.", "official"),
        ("Health advisory: Ang mga bata, matatanda, buntis, at may hika ay huwag lumabas habang mababa ang air quality sa Maynila.", "official"),
        ("Facemask reminder: Ang air quality sa Lungsod ng Maynila ay VERY UNHEALTHY na. Protektahan ang inyong kalusugan. Iwasan ang outdoor activities.", "official"),
        ("DRRM advisory: Monitored air quality sensors sa apat na lokasyon sa Maynila ay nagpapakita ng mapanganib na antas ng polusyon ngayong araw.", "official"),
        ("Ang sunog sa Navotas Landfill ay nagdudulot ng masamang kalidad ng hangin sa buong Maynila. Sarado ang ilang outdoor venues bilang pag-iingat.", "official"),
        ("Hotline ng Manila City DRRM Department para sa emergency: (02) 8463-3295 / 0950-700-3710. Tumawag kung may health emergency dahil sa polusyon.", "official"),
        ("Air quality VERY UNHEALTHY sa Puregold Tayuman area at karatig-lugar. Mahigpit na inirerekomenda na huwag lumabas nang walang facemask.", "urgent"),
        ("Mababa ang kalidad ng hangin sa Maynila simula ngayong gabi. Mga may asthma at respiratory conditions: huwag lumabas. Gamitin ang inhaler.", "urgent"),
        ("ADVISORY: Fine particulate matter (PM2.5) sa mapanganib na antas sa buong Maynila. Magsuot ng facemask lalo na ang mahihina ang kalusugan.", "urgent"),
        ("Ang Manila DRRM ay nagmo-monitor ng air quality sa 4 na sensor locations. Kasalukuyang lahat ay nasa VERY UNHEALTHY hanggang HAZARDOUS na antas.", "official"),
        ("Air quality warning: Huwag mag-exercise sa labas ng bahay ngayon sa Maynila. Ang masamang hangin ay mapanganib sa puso at baga.", "official"),
        ("Senior citizens at bata sa Maynila at karatig-siyudad: manatili sa loob ng bahay. Ang air quality ay nasa VERY UNHEALTHY level ngayon.", "official"),
        ("Usok mula sa Navotas Landfill fire ay patuloy na nakakaapekto sa air quality ng Maynila, Malabon, Navotas, at Caloocan. Mag-ingat.", "urgent"),
        ("DOH at Manila DRRM nagbababala tungkol sa masamang kalidad ng hangin. Mga sintomas ng polusyon: ubo, pangangati ng mata, kahirapan sa paghinga.", "official"),
        ("Mga residente ng Maynila: Isara ang mga bintana at pinto. Ang outdoor air quality ay VERY UNHEALTHY dahil sa usok at particulate matter.", "urgent"),
        ("Air quality advisory para sa mga may sakit sa puso: Huwag lumabas habang ang air quality index ay nasa 201-300 range sa Maynila ngayon.", "official"),
        ("Manila air quality update: Lahat ng 4 monitoring stations ay nagpapakita ng VERY UNHEALTHY. Manatiling alerto at sundan ang susunod na update.", "official"),
        ("REMINDER: Ang paggamit ng facemask ay mandatory sa labas habang masamang kalidad ng hangin sa Maynila. N95 o surgical mask ang inirerekomenda.", "official"),
        ("Ang mga eskwelahan sa Maynila ay may special advisory: huwag ituloy ang outdoor activities ngayon dahil sa poor air quality at init ng panahon.", "official"),
        ("Air quality sensor sa Sampaloc at Tondo ay nagpapakita ng VERY UNHEALTHY. Iwasan ang lugar na may mataas na polusyon ngayon.", "official"),
        ("Buntis na mga nanay sa Maynila: Ang pagkakalantad sa masamang hangin ay mapanganib sa inyong sanggol. Manatili sa loob ng bahay ngayon.", "urgent"),
        ("Health alert sa Maynila: Ang mga nagtatrabaho sa labas ng bahay ay dapat magbihis ng makapal at magsuot ng N95 mask laban sa polusyon.", "official"),
        ("DRRM update: Air quality sa Maynila ay bahagyang bumababa mula HAZARDOUS papunta VERY UNHEALTHY. Patuloy na mag-ingat at magsuot ng mask.", "official"),
        ("Hangin mula sa burning landfill ay nagtutulak ng usok sa direksyon ng Maynila. AQI forecast: VERY UNHEALTHY hanggang gabi. Manatiling alerto.", "official"),
        ("Manila City DRRM nagbibigay ng libreng facemasks sa mga barangay hall. Pumunta na ngayon para protektahan ang inyong pamilya mula sa polusyon.", "official"),
    ]
    return [_post(text, cid, base_idx + i, urgency) for i, (text, urgency) in enumerate(templates)]


# ── Assemble and write ────────────────────────────────────────────────────────

def main():
    all_posts: list[dict] = []
    idx = 0

    for fn, cluster_id in [
        (_cluster_a_posts, "cluster-a"),
        (_cluster_a_posts_ext, "cluster-a"),
        (_cluster_b_posts, "cluster-b"),
        (_cluster_b_posts_ext, "cluster-b"),
        (_cluster_b_posts_ext2, "cluster-b"),
        (_cluster_c_posts, "cluster-c"),
        (_cluster_c_posts_ext, "cluster-c"),
        (_cluster_c_posts_phivolcs, "cluster-c"),
        (_cluster_d_posts, "cluster-d"),
        (_cluster_d_posts_ext, "cluster-d"),
        (_cluster_e_posts, "cluster-e"),
        (_cluster_e_posts_ext, "cluster-e"),
        (_cluster_f_posts, "cluster-f"),
        (_cluster_f_posts_ext, "cluster-f"),
        (_cluster_g_posts, "cluster-g"),
        (_cluster_g_posts_ext, "cluster-g"),
        (_cluster_h_posts, "cluster-h"),
        (_cluster_h_posts_ext, "cluster-h"),
    ]:
        posts = fn(idx)
        all_posts.extend(posts)
        idx += len(posts)

    random.shuffle(all_posts)

    out_path = Path(__file__).parent.parent / "seed_dataset.json"
    out_path.write_text(json.dumps(all_posts, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(all_posts)} posts to {out_path}")
    counts: dict[str, int] = {}
    for p in all_posts:
        cid = p["_seed_cluster_id"]
        counts[cid] = counts.get(cid, 0) + 1
    for cid in sorted(counts):
        print(f"  {cid}: {counts[cid]} posts")
    print("\nNext step: python scripts/train_from_seed.py")


if __name__ == "__main__":
    main()
