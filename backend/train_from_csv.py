#!/usr/bin/env python3
"""
train_from_csv.py — Standalone model training for MANA from a CSV post export.

Usage (from the backend/ directory):
    python train_from_csv.py "C:/path/to/posts_rows.csv"

Pipeline:
  1. Load raw captions, drop empty rows.
  2. Translate Tagalog text to English (matches production pipeline).
  3. Score each post against cluster keyword tables (score >= 1 = relevant).
     Posts that score < 1 are treated as irrelevant and excluded from training.
     Posts mentioning only non-Manila regions (no Manila landmark/term) are also excluded.
  4. Assign priority label (High / Medium / Low) via cluster + keyword + engagement.
  5. Augment any cluster below MIN_PER_CLUSTER with synthetic Philippine-disaster posts.
  6. Train CorEx (iterative, up to 5 passes) — save 4 artifacts.
  7. Train SVM (TF-IDF + LinearSVC OvR) — save 5 artifacts.
  8. Compute VADER scores + CorEx topic predictions for RF feature engineering.
  9. Train Random Forest priority classifier — save 3 artifacts.
 10. Report per-class metrics; warn / exit-1 if any fall below passing thresholds.

Contract:
  - Training data is NEVER written to disk — only the 11 model artifacts are saved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_BACKEND_DIR))

# ---------------------------------------------------------------------------
# NLTK
# ---------------------------------------------------------------------------
import nltk
for _pkg in ("wordnet", "stopwords", "omw-1.4"):
    nltk.download(_pkg, quiet=True)
from nltk.corpus import stopwords as _nltk_stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# MANA service imports (no Flask / DB context needed for training functions)
# ---------------------------------------------------------------------------
from services.corex.topic_modeler import train_iteratively as _corex_train_iteratively
from services.svm.cluster_classifier import train_svm as _svm_train
from services.vader.sentiment_analyzer import analyze_sentiment as _vader_analyze
from services.random_forest.priority_classifier import (
    _build_feature_matrix,
    PRIORITY_LABELS,
    RF_MODEL_PATH,
    RF_COLUMNS_PATH,
    RF_META_PATH,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MIN_PER_CLUSTER     = 100   # augment any cluster below this with synthetic posts
RELEVANCE_THRESHOLD = 1     # posts scoring below this are excluded as irrelevant
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 15

# ---------------------------------------------------------------------------
# Tagalog detection
# ---------------------------------------------------------------------------
_TAGALOG_MARKERS = frozenset({
    "ang","mga","sa","ng","para","nag","may","na","ay","ito","ako","sila",
    "namin","kayo","po","rin","din","pero","kaya","dahil","kung","nang",
    "ba","pa","lang","lamang","kasi","tayo","natin","ninyo","nila","akin",
    "iyo","aming","kanila","siya","kami","iyon",
})

# ---------------------------------------------------------------------------
# Stop-words (keep disaster/negation terms)
# ---------------------------------------------------------------------------
_KEEP_TOKENS = frozenset({
    "no","not","never","fire","help","missing","dead","trapped","need",
    "rescue","call","now","alert","warning","urgent","emergency","flood",
    "typhoon","storm","relief","water","road","school","class","hospital",
    "medical","health","power","signal","death","damage","blocked","closed",
})
_STOPS      = set(_nltk_stopwords.words("english")) - _KEEP_TOKENS
_lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Cluster keyword tables
# KEY FIX: "fire" and "firealert" are separate entries in cluster-g so that
# #FireAlert! posts (which clean to "firealert") score >= 2:
#   "firealert" → single word, score 1
#   "fire"      → substring of "firealert", score 1   →  total = 2  ✓
# ---------------------------------------------------------------------------
CLUSTER_KEYWORDS: dict[str, list[str]] = {

    # cluster-a : relief (Food and Non-food Items)
    "cluster-a": [
        "relief goods","relief donation","food donation","food pack",
        "relief distribution","relief operations","relief convoy",
        "relief pack","relief items","relief package","relief drive",
        "packed relief","aid package","food assistance",
        "non food items","ready to eat",
        "rice","noodles","sardines","canned goods","water sachet",
        "family pack","drinking water","tarpaulin","sleeping mat",
        "mosquito net","jerry can","water container","blanket",
        "hygiene kit","diaper","baby food","formula milk",
        "soap","toothbrush","feminine hygiene","sanitary napkin",
        "repacking","food distribution","hot meal","community kitchen",
        "supply drop","dswd","red cross",
    ],

    # cluster-b : health_medical
    "cluster-b": [
        "hospital","medical center","health center","clinic",
        "emergency room","doctor","nurse","patient","injured","wound",
        "medicine","medication","ambulance","first aid","triage",
        "leptospirosis","diarrhea","cholera","dengue","rabies",
        "hepatitis","tetanus","measles","pneumonia","respiratory",
        "outbreak","epidemic","disease","illness","infection",
        "dehydration","malnutrition","mental health","psychosocial",
        "contaminated water","water borne disease","health risk",
        "medical team","medical mission","medical supply","health advisory",
        "trauma","drowning","heat stroke",
        # COVID-19 posts → health/medical
        "covid","covid-19","coronavirus","covid case","covid monitoring",
        "covid positive","covid alert","covid update","covid patient",
    ],

    # cluster-c : evacuation (CCCM)
    "cluster-c": [
        "evacuation center","evacuation site","evacuation order",
        "evacuate","evacuation","evacuees","evacuated",
        "displaced","displaced families","displaced residents",
        "temporary shelter","safe haven","safe space",
        "tent","covered court","gymnasium","barangay hall",
        "welfare desk","welfare center",
        "preemptive evacuation","mandatory evacuation",
        "forced evacuation","mass evacuation","return home","decampment",
        "stranded families","headcount",
    ],

    # cluster-d : logistics
    "cluster-d": [
        "road closed","road blocked","road flooded","not passable",
        "road clearing","clearing operation","road damage",
        "road subsidence","road condition","road update",
        "bridge","landslide","debris","impassable","alternate route",
        "detour","diversion","sinkhole","fallen tree",
        "dpwh","heavy equipment","backhoe","crane",
        "blocked road","access road","access blocked","route blocked",
        "convoy","supply route","delivery route","supply delivery",
        "logistics team","staging area","distribution point",
    ],

    # cluster-e : telecom_power (Emergency Telecom)
    "cluster-e": [
        "power outage","power failure","power cut","power restoration",
        "blackout","brownout","no electricity","electricity cut",
        "no power","grid down",
        "signal loss","no signal","no internet","mobile data down",
        "network outage","network down","telecoms down",
        "communication blackout","communication cut",
        "meralco","pldt","globe","smart",
        "generator","backup power","charging station",
        "typhoon bulletin","tropical cyclone bulletin",
        "weather forecast","rainfall warning","impact-based warning",
        "pagasa","habagat","amihan","monsoon","low pressure area",
        # thunderstorm advisories → telecom/weather
        "thunderstorm","thunderstorm advisory","thunderstorm watch",
        "thunderstorm warning","thunderstorm watch ncr","rainfall advisory",
        "severe thunderstorm","lightning advisory",
    ],

    # cluster-f : education
    "cluster-f": [
        "class suspension","class suspended","no classes",
        "school suspended","school closed","school closure",
        "class cancellation","classes cancelled","school cancelled",
        "deped advisory","deped orders",
        "school","classroom","students","learners",
        "teacher","academic","enrollment",
        "online class","modular","distance learning","remote learning",
        "blended learning","face to face","school building",
        "school damage","school flooded",
        "school as evacuation center","school used as shelter",
        "deped",
    ],

    # cluster-g : rescue (SRR)
    # IMPORTANT: "fire" (single word) + "firealert" (single word) are BOTH listed
    # so that raw #FireAlert! posts (→ "firealert") score 2 in total.
    # "sunog" is Tagalog for fire; "firesafety" catches #FireSafety posts.
    "cluster-g": [
        "fire","firealert","txtfire","sunog","firesafety","fire safety","ongoing fire",
        "fire alert","fire alarm","fire incident",
        "fire reported","fire out","fire update","active fire",
        "fire rescue","fire response","fire department",
        "bfp","bureau of fire","firefighter","fire truck",
        "1st alarm","2nd alarm","3rd alarm","4th alarm","5th alarm",
        "two alarm","three alarm","four alarm","five alarm",
        "arson","blaze","burning","engulfed","wildfire",
        "structure fire","residential fire",
        "rescue","search and rescue","rescue operation",
        "rescue team","rescue personnel","rescue boat",
        "trapped","rooftop","submerged",
        "chest deep","neck deep","waist deep","sos",
        "save us","help us","need rescue","rescue needed",
        "coast guard","helicopter","swift water",
        "flash flood","swept away","washed away",
        "collapsed structure","building collapse","pinned under",
    ],

    # cluster-h : dead_missing (MDM)
    # "missing" (single word) catches posts that just say "MISSING" without multi-word phrases.
    # "nawawala" is Tagalog for missing.
    "cluster-h": [
        "missing","nawawala",
        "missing person","declared missing","officially missing",
        "confirmed missing","reported missing","still missing",
        "search for missing","missing after flood","missing child",
        "whereabouts unknown","last seen","last contact",
        "has not returned","missing family",
        "death toll","confirmed dead","declared dead",
        "found dead","body found","bodies recovered",
        "death certificate","confirmed fatality",
        "casualty report","fatalities","deceased",
        "morgue","autopsy","burial","interment",
        "remains identified","victim identified",
        "feared dead","death report",
    ],
}

# ---------------------------------------------------------------------------
# Manila geography filter
# Posts mentioning at least one Manila term are kept.
# Posts mentioning ONLY non-Manila regions (and no Manila term) are excluded.
# Posts with neither → assumed to be Manila-based (keep).
# ---------------------------------------------------------------------------
_MANILA_TERMS = frozenset({
    "manila","maynila","ncr","metro manila","manileno",
    "tondo","binondo","paco","sampaloc","malate","ermita",
    "intramuros","pandacan","navotas","malabon","caloocan",
    "pasay","makati","mandaluyong","pasig","marikina",
    "quezon city","paranaque","las pinas","muntinlupa",
    "valenzuela","taguig","san juan","baseco","divisoria",
    "luneta","delpan","nagtahan","lawton","arroceros",
    "gagalangin","sta ana","sta cruz","quiapo","mmda",
    "mdrrmo","mcdrmd","manila bay","estero","north harbor",
    "manila north","manila south","espana","taft","roxas",
    "edsa","c3 road","c4 road","c5","marcos highway",
})

_NON_MANILA_REGIONS = frozenset({
    # PAGASA regional bulletin codes
    "vprsd","mprsd","nvrsd","cvrsd","sprsd",
    # Visayas
    "cebu","negros","iloilo","bacolod","tacloban",
    "visayas","western visayas","central visayas","eastern visayas",
    "bohol","leyte","samar","aklan","antique","capiz",
    # Mindanao
    "mindanao","davao","zamboanga","cagayan de oro",
    "general santos","cotabato","iligan","butuan",
    "caraga","armm","bangsamoro",
    # Bicol
    "bicol","albay","mayon","sorsogon","camarines","legazpi",
    # Others outside NCR
    "surigao","cordillera","benguet","baguio","isabela",
    "cagayan valley","ilocos","pangasinan",
    "pampanga","bulacan","laguna","batangas","cavite",
    "rizal province","quezon province",
})


def is_manila_relevant(clean_text: str) -> bool:
    """Return False only if the post mentions a non-Manila region and NO Manila term."""
    tl = clean_text.lower()
    if any(term in tl for term in _MANILA_TERMS):
        return True
    if any(term in tl for term in _NON_MANILA_REGIONS):
        return False
    return True   # no region signal → assume Manila-based account


# Priority keyword sets
_HIGH_RESCUE_KW  = frozenset({"trapped","sos","save us","help us","rooftop","chest deep","neck deep","rescue needed","swift water","pinned under","swept away","washed away","please rescue","requesting rescue"})
_HIGH_DEATH_KW   = frozenset({"confirmed dead","death toll","bodies recovered","morgue","confirmed fatality","feared dead","declared dead","found dead"})
_HIGH_HEALTH_KW  = frozenset({"outbreak","epidemic","cholera","leptospirosis","contaminated water","disease outbreak"})
_MED_WARNING_KW  = frozenset({"impact-based warning","typhoon bulletin","warning","tropical cyclone","habagat"})

# ---------------------------------------------------------------------------
# Synthetic posts — 80 templates per thin cluster
# ---------------------------------------------------------------------------
_SYNTHETIC: dict[str, list[str]] = {

    # ── cluster-b : health_medical (80 templates) ─────────────────────────
    "cluster-b": [
        "Ospital ng Tondo now accepting flood-related injuries. ER on standby 24/7.",
        "Gat Andres Bonifacio Medical Center emergency room prepared for typhoon casualties.",
        "DOH deploys medical team to evacuation centers in Tondo and Binondo.",
        "Leptospirosis cases rising in flood-affected areas of Manila. Public advised to avoid wading.",
        "Manila Health Department warns of waterborne diseases after flooding. Boil water advisory.",
        "Medical volunteers from Philippine Red Cross deployed to Baseco flood area.",
        "Patients being transferred from flooded hospital ward to higher floors.",
        "Dengue cases surge in flood-affected barangays. Fogging operations ongoing.",
        "Field hospital set up at Nagtahan gymnasium for flood-related medical emergencies.",
        "Cholera outbreak alert in Tondo. DOH distributes oral rehydration salts to evacuees.",
        "Hospital emergency room flooded. Patients evacuated to second floor.",
        "Psychosocial support teams deployed to evacuation centers for typhoon trauma victims.",
        "Mental health counseling available at Arroceros evacuation center.",
        "Malnutrition risk rising among long-term evacuees. Nutritional support being distributed.",
        "First aid stations set up along C4 road for flood rescue operations.",
        "Medical standby at Luneta for mass evacuation support operations.",
        "Drowning victim rescued near Estero de Vitas. Rushed to Manila General Hospital.",
        "Heat stroke cases reported at evacuation centers. Medical teams on standby.",
        "DOH issues health advisory for flood survivors: watch for fever, skin lesions, diarrhea.",
        "Community health worker deployed in Barangay 105 for post-flood disease monitoring.",
        "Medicine shortage at district health center serving flood evacuees. DOH coordinating.",
        "Contaminated water supply reported in Tondo. Manila Water advises not to drink tap water.",
        "Medical mission conducted at Navotas evacuation center. 200 patients attended to.",
        "Snake bite case reported in flood area of Paco. Antivenom administered at clinic.",
        "Measles vaccination campaign launched in evacuation centers after typhoon.",
        "Water safety advisories issued as floodwaters contaminated by sewage overflow.",
        "Flood injuries treated at mobile clinic near Divisoria. Minor cuts and infections prevalent.",
        "Hepatitis A risk elevated in overcrowded evacuation centers. Sanitation teams deployed.",
        "Ambulance routes rerouted due to flooded roads. Emergency response time affected.",
        "Post-flood respiratory illness cases increasing in evacuation centers. Masks distributed.",
        "Mental health support hotline activated for typhoon survivors in Metro Manila.",
        "Medical supply drop delivered to isolated barangay via rescue boat.",
        "Wound care kits distributed to flood victims with foot injuries from debris.",
        "DOH monitoring 15 suspected leptospirosis cases in flood-affected Manila barangays.",
        "Triage center set up at covered court for flood-related injuries.",
        "Medical team braved floods to reach stranded patients in Navotas.",
        "Baby formula and pediatric medicines distributed at evacuation center in Binondo.",
        "Health risk warning: standing floodwaters may carry disease-causing organisms.",
        "Dehydration cases among elderly evacuees. ORS and clean water being distributed.",
        "Clinic in Paco submerged. Medical staff relocated to barangay hall as backup facility.",
        "Emergency dental care provided to flood survivors at makeshift clinic.",
        "Tetanus shots administered to flood rescue workers and wading evacuees.",
        "Prenatal check-up conducted for pregnant evacuees at Delpan sports complex.",
        "Midwife deployed to evacuation center as pregnant woman nears delivery.",
        "Pneumonia cases rising in damp evacuation centers. Antibiotics being distributed.",
        "DOH urges public to seek medical attention for any skin wounds after flood exposure.",
        "Infection control measures put in place at evacuation centers. Handwashing stations set up.",
        "Medical mission at Malate evacuation center: 300 consultations in one day.",
        "Trauma counseling for children who witnessed fire or flood at Arroceros center.",
        "Health team deployed to remote evacuation center via boat due to flooded access roads.",
        "Post-flood eye infection cases reported. Eye drops distributed at health center.",
        "DOH coordinates with private hospitals for overflow medical capacity during typhoon.",
        "Medical alert: avoid contact with floodwater — risk of leptospirosis, skin infections.",
        "Diarrhea outbreak contained at Baseco evacuation center after ORS and antibiotic distribution.",
        "Doctor volunteers needed at Manila City Hall evacuation center this weekend.",
        "Medical standby team deployed at Estero rehabilitation site following flooding.",
        "Water purification tablets distributed with water advisory to flood evacuees.",
        "Health worker confirms 3 suspected cholera cases in Gagalangin. DOH investigating.",
        "Ambulance on standby at rescue staging area near San Nicolas flood zone.",
        "Elderly patient with hypertension needs medication resupply at evacuation shelter.",
        "DOH mobile laboratory conducting rapid diagnostic tests in flood-affected Manila.",
        "Nutritional relief packs with vitamins distributed to malnourished evacuees.",
        "Wound infections from flood debris on the rise. Antibiotics requested from DOH.",
        "Mental health kits distributed to children at evacuation centers after typhoon.",
        "Tetanus outbreak risk following flood — immunization drive starts tomorrow in NCR.",
        "Post-flood skin disease cases being monitored by Manila Health Office.",
        "Sick evacuees transported to hospital from covered court via MMDA ambulance.",
        "DOH urges flu vaccination for evacuees with compromised immunity in crowded shelters.",
        "Over 200 medical consultations provided at Tondo evacuation center today.",
        "Psychosocial workers supporting grieving families at dead and missing coordination desk.",
        "Medical triage at Baseco compound: 40 patients treated, 5 referred to hospital.",
        "Community health volunteers monitor for disease symptoms at evacuation sites.",
        "Dialysis patients in flood area need urgent assistance reaching medical centers.",
        "Water-borne disease alert issued for Sampaloc flood zone. Avoid wading.",
        "Hospital on standby: trauma and orthopedic teams ready for flood-related injuries.",
        "DOH health advisory: wash hands frequently, especially in evacuation centers.",
        "Suspected measles case reported in evacuation center. Containment measures activated.",
        "Nurses volunteer overtime at Manila General Hospital during typhoon emergency.",
        "Health emergency hotline: 8527-9999 for medical assistance during flooding.",
        "DOH health advisory: wash hands frequently to prevent typhoid after flooding in Manila.",
        "Manila Health Department monitoring 45 suspected cases of leptospirosis post-flood.",
        "Dengue surge following typhoon: 3 Manila hospitals on high alert.",
        "Diarrhea outbreak contained in Tondo with ORS distribution and water purification.",
        "Medical volunteers from UERM College of Medicine providing free consultations at Delpan.",
        "Skin lesion cases rising post-flood. Antibiotic cream being distributed by DOH.",
        "Pneumonia risk elevated for children in damp evacuation centers. Parents advised to seek care.",
        "DOH deploys 10 medical teams to typhoon-affected barangays across Manila.",
        "Mental health services: NCMH deployed counselors to Manila evacuation centers.",
        "Food poisoning cases reported at Baseco compound. DOH investigating food preparation.",
        "Typhoid fever risk after contaminated water exposure. DOH advises boiling water.",
        "Post-flood health monitoring: DOH tracking 20 diseases in Manila flood zones.",
        "Hepatitis A immunization drive at Tondo evacuation centers this Thursday.",
        "Nutritional assessment conducted by DOH for malnourished children in flood shelters.",
        "Maternal and child health services deployed to Manila shelters for pregnant women.",
        "Doctor on duty at Santa Cruz evacuation center from 8 AM to 8 PM daily.",
        "Wound treatment clinic set up at barangay hall for flood survivors with foot injuries.",
        "DOH hotline for medical consultation: 1555. Active during typhoon emergency.",
        "Cancer patient evacuation: Manila Medical Center evacuating patients to safer floor.",
        "Dialysis patients affected by flooding rerouted to Ospital ng Tondo for treatment.",
        "DOH issues notice: avoid self-medication after flood exposure. Consult a doctor.",
        "Schistosomiasis risk after flooding. DOH distributes treatment to affected communities.",
        "Community health workers conduct home visits in flood zones for disease monitoring.",
        "Medical supply request from evacuation center: paracetamol, amoxicillin, oral rehydration salts.",
        "Tetanus prophylaxis clinic open at Paco district health center for flood wading victims.",
        "Post-flood respiratory illness reports on the rise. N95 masks being distributed by DOH.",
        "Pediatrician volunteers needed at Nagtahan gymnasium where 300 children are sheltering.",
        "Medical examination for flood survivors at Sta. Ana district health center. Free of charge.",
        "Health kit distribution: oral rehydration salts, water purification tablets, betadine.",
        "DOH confirms no cholera cases in Manila but maintains heightened surveillance.",
        "Insulin supply requested for diabetic patients at Baseco evacuation center.",
        "Occupational health hazards from floodwater: DOH advises rubber boots and gloves.",
        "Nurse volunteers needed at Malate covered court. Contact Manila Health Department.",
        "Heat exhaustion cases treated at Lawton evacuation center. Cooling stations deployed.",
        "DOH mental health program: trauma-informed care workshops at Manila evacuation centers.",
        "Philippine Childrens Medical Center donating pediatric medicines to flood relief operations.",
        "DOH confirms 2 cases of leptospirosis in Tondo. Contact tracing and monitoring ongoing.",
        "Immunocompromised patients at evacuation centers receiving specialized medical support.",
        "DOH distributes 10,000 water purification tablets to flood-affected households in Manila.",
        "Eye irrigation kits distributed after reports of eye irritation from contaminated floodwater.",
        "Wound debridement services at Arroceros clinic for flood victims with laceration injuries.",
        "Medical oxygen supply secured for patients in evacuation centers with respiratory conditions.",
        "DOH posts disease surveillance data: 15 new leptospirosis cases in Manila post-flood.",
        "Gastroenteritis cases increasing in flood shelter. ORS and antibiotic distribution ongoing.",
        "Flu vaccination drive at evacuation center: 500 adults and children vaccinated today.",
        "DOH deploying disease surveillance officers to monitor health conditions in flood areas.",
        "Mobile medical unit providing free check-ups to flood victims in Navotas barangay.",
        "Doctor reports surge in wound infections due to debris contact during flooding.",
        "Health education campaign on proper wound care and hygiene for flood survivors.",
    ],

    # ── cluster-c : evacuation (80 templates) ────────────────────────────
    "cluster-c": [
        "Residents of Barangay 132 ordered to evacuate to covered court due to rising floodwaters.",
        "Evacuation center at Delpan Sports Complex now at capacity with over 300 families.",
        "MDRRMO advises mandatory evacuation for families in low-lying areas near Estero de Vitas.",
        "Evacuees at Barangay Hall 105 need sleeping mats and food packs. Over 200 families displaced.",
        "Pre-emptive evacuation ordered for communities near Marikina River due to water level rise.",
        "Mass evacuation underway in coastal barangays. Families moved to Nagtahan gymnasium.",
        "Evacuation notice issued for Baseco compound. Residents urged to leave immediately.",
        "Stranded families in Tondo moved to welfare center. Headcount ongoing. 450 evacuees.",
        "MMDA reports 12 evacuation centers now open in Manila. Total of 2100 evacuees.",
        "Return home allowed in Barangay 101 after flood waters recede. MDRRMO monitoring.",
        "Forced evacuation ongoing in flood-prone zones of Paco and Pandacan.",
        "Barangay gymnasium at full capacity. Overflow accommodated in nearby covered court.",
        "Mandatory evacuation order for residents near Pasig River tributaries.",
        "Welfare desk activated at Manila City Hall for displaced families needing assistance.",
        "Community shelter opened at Epifanio Delos Santos Sports Center for evacuees.",
        "Decampment operations begin as flood waters recede. Families allowed to return home.",
        "Evacuation center in Tondo reports no more available sleeping space.",
        "Children and elderly evacuated first from high-risk areas near estero.",
        "Evacuation order lifted in 3 barangays. Residents return home after typhoon.",
        "Night shelter established for families who lost homes to fire in Tondo.",
        "Breastfeeding area set up in evacuation center for lactating mothers.",
        "Child-friendly space established at evacuation center with UNICEF support.",
        "DSWD distributes hot meals to 500 evacuees at covered court in Binondo.",
        "Over 1500 families displaced in Navotas. Evacuation centers at 90% capacity.",
        "Evacuation site at Jose Rizal Memorial Stadium opened for Typhoon DomengPH evacuees.",
        "Residents in danger zones alerted. Mandatory evacuation for low-lying communities.",
        "Welfare center activated in Santa Cruz for flood-displaced families from estero zones.",
        "Curfew imposed in flooded barangays. Residents advised to stay in evacuation centers.",
        "Evacuation order for high-risk areas near San Juan River. Families move to shelter.",
        "Displaced residents from Divisoria area moved to Gat Andres Bonifacio Sports Center.",
        "MMDA facilitates evacuation of 800 families from Tondo flood zone to safety.",
        "Evacuation center at Mehan Garden now accommodating 1,200 displaced residents.",
        "DILG orders barangay officials to enforce mandatory evacuation in danger zones.",
        "Families refusing to evacuate reminded of danger. MDRRMO to conduct forced evacuation.",
        "Headcount at Intramuros evacuation center: 340 families, 1,200 individuals.",
        "Temporary shelter at San Andres Sports Complex reaches full capacity.",
        "Safe haven declared at Arroceros Forest Park for evacuees from Binondo flooding.",
        "Last batch of evacuees from Estero de San Miguel moved to covered court.",
        "Decampment advisory issued: 3 barangays cleared for return as flood subsides.",
        "Barangay hall converted to emergency shelter as gymnasium overflows.",
        "Evacuation bus service provided by MMDA for residents unable to walk to shelter.",
        "Pregnant women and persons with disabilities prioritized in evacuation.",
        "MMDA: 15 evacuation centers operational, total capacity 5,000 families.",
        "Welfare assistance desk at Manila City Hall processing displaced families.",
        "Evacuation route map released by MDRRMO for northern Tondo residents.",
        "Evacuation boats deployed by MMDA for residents in neck-deep flood areas.",
        "Evacuation center at Malate gym now hosts 450 families from coastal barangays.",
        "Relief goods distribution inside evacuation center ongoing. Lines forming.",
        "Displaced vendors from Divisoria market temporarily housed at school gymnasium.",
        "Overflow from evacuation centers being managed at SM City Manila parking area.",
        "Residents evacuated from informal settlements along Estero de Paco.",
        "Families with livestock face challenge evacuating. MAFC assists with animal shelter.",
        "Last residents evacuated from Isla Puting Bato before storm surge arrival.",
        "Evacuation team doing final sweep of flood-prone zone before nightfall.",
        "Elderly evacuees transported by MMDA ambulance to covered court shelter.",
        "Barangay 649 fully evacuated. 220 families now at San Andres Sports Complex.",
        "DSWD activates social protection measures for families in evacuation centers.",
        "Informal settlers evacuated from danger zone along Intramuros walls.",
        "Local government sets up family registration at each evacuation center.",
        "Night patrols deployed to ensure no resident remains in flood-prone zones.",
        "Children's activities organized at evacuation center to reduce distress.",
        "Evacuation center at Jones Bridge underpass cleared and set up for displaced families.",
        "MMDA to open additional evacuation facilities as storm approaches.",
        "Mandatory preemptive evacuation for communities near Marikina River.",
        "Welfare package including hygiene kits distributed to evacuees at gymnasium.",
        "Evacuation operations completed in 12 barangays before typhoon landfall.",
        "Temporary bunk beds and mats provided to 600 families at Manila City Hall grounds.",
        "MDRRMO announces zero casualty goal — mandatory evacuation strictly enforced.",
        "Barangay tanod assisting in door-to-door evacuation in flood zone.",
        "Community center used as safe space for 80 families from fire-affected area.",
        "Return home advisory cancelled — flood levels rising again. Evacuees to remain.",
        "Search ongoing for 5 families who refused evacuation in Tondo flood zone.",
        "Evacuation center personnel increased to manage influx of displaced families.",
        "Rescue boats ferrying evacuees from rooftops to covered court shelter.",
        "Cot and sleeping mat distribution ongoing at San Lazaro grounds evacuation center.",
        "Evacuation update: 8 centers open in Manila. Capacity 80%. No overflow yet.",
        "Forced evacuation of 50 families refusing to leave river bank areas.",
        "MMDA reports smooth evacuation operations in Sampaloc flood area.",
        "Evacuation order extended to include riverside communities in Navotas.",
        # Real-world style posts — use anchor words consistently for CorEx coherence
        "Barangay 166, Tondo: ALL residents in danger zone must evacuate to covered court NOW.",
        "MDRRMO Sampaloc: mandatory evacuation in effect for all households near flood canal. Evacuees report to barangay hall.",
        "Evacuation center update: 180 displaced families now sheltering at Delpan. Welfare desk open 24/7.",
        "Pre-emptive evacuation of residents along Estero de Paco started 6 AM. Evacuees moved to temporary shelter.",
        "Covered court capacity exceeded. Overflow evacuees directed to nearby gymnasium. Displaced count: 620 families.",
        "Temporary shelter at Tondo barangay hall now housing 200 evacuees. Sleeping mats distributed.",
        "URGENT: residents of low-lying flood-prone areas must vacate to safe space before 8 PM tonight.",
        "Forced evacuation executed in Navotas coastal zone due to storm surge warning. 300 evacuees displaced.",
        "Evacuees at Delpan sports complex: 450 adults, 120 children, 35 senior citizens. All safe and sheltered.",
        "Welfare center operations running 24/7. Social workers attending to 800 displaced residents at evacuation site.",
        "Return home advisory for Paco flood evacuees: floodwaters have receded. Official decampment begins 8 AM.",
        "Decampment from Nagtahan gymnasium begins. 220 displaced families return to homes after flood subsides.",
        "Stranded families from Barangay 412 brought to evacuation site via rubber boat. 65 persons now sheltered.",
        "Tondo MDRRMO: do not return to evacuated danger zone until official decampment order is issued.",
        "All residents in flood-prone, high-risk areas ordered to evacuate immediately to evacuation center.",
        "Overcrowded evacuation center at Sta. Ana gymnasium. Second shelter opened at Pandacan barangay hall.",
        "Evacuee headcount update 4 PM: 1,200 displaced families across 6 evacuation sites in Manila.",
        "Night shelter now open at Arroceros gymnasium. Bring sleeping mat and personal items. Free entry for evacuees.",
        "Mass evacuation of coastal barangays underway. MMDA deploying additional boats for displaced families.",
        "Community shelter at Divisoria gymnasium accepting displaced residents from Tondo evacuation order.",
        "Evacuation notice: Barangays 1–10 along Pasig River must evacuate to designated shelter before midnight.",
        "Safe haven at Rizal High School gymnasium for 300 displaced residents from flood-prone zone.",
        "Preemptive evacuation recommended for all families living in low-lying flood prone barangays of Manila.",
        "City Mayor signs mandatory evacuation order for all barangays at Flood Alert Level 3.",
        "Tent city established at Luneta park as overflow evacuation site for 500 additional displaced families.",
        "DSWD welfare desk at evacuation center: sleeping mats, food packs for newly arrived evacuees.",
        "Manila DRRMO: 8 evacuation sites open, total shelter capacity 3,000 families for displaced residents.",
        "Barangay captain: residents in danger zone must cooperate with mandatory evacuation order. No exceptions.",
        "Manila Bay coastal communities: preemptive evacuation recommended due to storm surge threat.",
        "Evacuation site at Nagtahan: running water, portable toilets, and cooking area available for evacuees.",
        "Overcrowded shelters: Manila DSWD requesting additional temporary evacuation sites for displaced families.",
        "Last 30 displaced families return home after 5-day evacuation. Evacuation center Paco officially closed.",
        "Evacuation center Delpan: child-friendly space now operational for young evacuees and displaced children.",
        "Warning level raised: all families within 100 meters of flooded canal must evacuate to safe shelter today.",
        "MDRRMO confirms all danger-zone evacuees accounted for. 1,450 displaced persons in evacuation centers.",
        "Families displaced by landslide in Tondo transferred to covered court evacuation center. Food provided.",
        "Evacuation of elderly and PWDs from danger zone prioritized. 45 senior evacuees transported to shelter.",
        "Pasig River water level at critical. Preemptive evacuation of 500 families from flood-prone banks begins.",
        "Evacuation center sleeping mat shortage. LGU requests donations for 300 displaced families at Baseco shelter.",
        "Mandatory evacuation order lifted in Sampaloc. Displaced evacuees may return home after safety inspection.",
    ],

    # ── cluster-d : logistics (80 templates) ─────────────────────────────
    "cluster-d": [
        "DPWH road clearing operations underway along Quirino Highway. Expect significant delays.",
        "Bridge at Nagtahan Interchange closed due to flood damage. Use Pasig and Makati routes.",
        "Road subsidence reported at C5 Highway near Taguig. DPWH deployed heavy equipment.",
        "Convoy of relief goods blocked by landslide on Marcos Highway. Rerouting via EDSA.",
        "Fallen trees obstruct Espana Boulevard. MMDA clearing teams with chainsaws deployed.",
        "Marikina-Infanta Road impassable due to mudslide. No access to Quezon City side.",
        "DPWH clears debris from Tandang Sora Avenue. Road now partially passable.",
        "Sinkhole reported on Taft Avenue. Road closed for repairs. Use alternate via Roxas.",
        "Road flooded in Sampaloc. Vehicles diverted to Quezon Boulevard and Espana.",
        "Supply route to evacuation center via Delpan Bridge blocked. Alt via Jones Bridge.",
        "MMDA road condition update: EDSA slow-moving due to flooding in Kamuning.",
        "Distribution point set up at Lawton Plaza for relief goods logistics.",
        "Logistics team coordinates supply delivery to Tondo evacuation centers via barge.",
        "Access road to Payatas blocked by debris flow. Heavy equipment en route.",
        "Route blocked near Commonwealth Avenue due to flash flood. Pumping trucks deployed.",
        "Road update: C4 road in Malabon impassable. Alternate via MacArthur Highway.",
        "Clearing operations by DPWH at Balintawak road cut. One lane open.",
        "Staging area set up near Rizal Park for relief goods distribution.",
        "Bottleneck at Nagtahan due to flood. Traffic rerouted via Intramuros.",
        "Cargo delivery to Navotas blocked. Rerouting via Malabon-Navotas Road.",
        "DPWH backhoe clearing landslide blocking Antipolo Road to Marikina.",
        "Road condition: Aurora Boulevard flooded at underpass. Not passable.",
        "Diversion set up by MMDA along Quirino Avenue. Follow alternate route signs.",
        "Access blocked in North Harbor area. Barge operations suspended due to rough seas.",
        "Chokepoint at Balintawak market flooded. Relief convoy delayed 3 hours.",
        "Road update: Espana Extension now passable after clearing operations.",
        "Infrastructure damage: 3 bridges affected by Typhoon DomengPH in Manila.",
        "Impassable road in Marikina due to debris. DPWH requests more heavy equipment.",
        "Supply chain disruption due to flooded roads. OCD coordinating relief delivery.",
        "Route to Caloocan evacuation center blocked. Rerouted via Monumento.",
        "MMDA deploys road clearing team to 10 flood-affected intersections in Manila.",
        "Flood debris blocking Quezon Bridge approach. One lane operational.",
        "DPWH heavy equipment deployed to clear landslide on Marcos Highway.",
        "Relief goods convoy rerouted from flooded Delpan Bridge to Jones Bridge.",
        "Road update: Roxas Boulevard partly submerged at Luneta. Avoid area.",
        "Bridge inspection ongoing at Guadalupe. Structural assessment after typhoon.",
        "MMDA confirms: 7 roads impassable in Manila, clearing operations in progress.",
        "Staging area for disaster response goods established at SM San Lazaro parking.",
        "Supply drop via helicopter to isolated community in Rizal Province.",
        "Road condition update: Balintawak to Monumento — intermittently flooded.",
        "Delivery route for medical supplies changed due to flooded access road.",
        "DPWH opens new alternate route via Governor Forbes to bypass flood on España.",
        "Access to Baseco compound blocked by debris. MMDA clearing on one side.",
        "Logistics hub at Manila North Harbor coordinating sea-borne relief distribution.",
        "Relief goods being repackaged and loaded for barge delivery to Navotas.",
        "Chokepoint at Quiapo underpass. Vehicles diverted to Legarda Ave.",
        "MMDA: 5-ton truck stuck in floodwaters near Delpan. Recovery team dispatched.",
        "Road subsidence at Tayuman Street creates 2-meter wide crater. Road closed.",
        "Logistics coordination meeting held at MMDA ops center for flood response.",
        "Distribution convoy with 200 family food packs departed Manila City Hall.",
        "DPWH fast-tracks emergency road repairs on C3 Road flood-damaged section.",
        "Road clearing priority given to routes leading to evacuation centers.",
        "Traffic update: South Luzon Expressway entry point flooded. Use NLEX.",
        "Alternative route advisory: Magsaysay to P. Campa to reach España.",
        "Heavy equipment deployed to clear fallen tree blocking Sampaloc main road.",
        "MMDA crane lifts flooded vehicles to restore traffic flow on Taft Avenue.",
        "Bridge load limit reduced on flooded Napindan bridge pending inspection.",
        "Stockpile of emergency supplies moved to elevated warehouse in anticipation of flooding.",
        "Convoy of 10 trucks with relief goods delayed due to multiple road closures.",
        "Road repair crews working through the night on critical supply routes.",
        "Disaster logistics coordination call between MMDA, DSWD, and OCD at 8 PM.",
        "Passable road update: EDSA Balintawak to Ortigas now accessible — use extreme caution.",
        "Road clearance completed on España Ext. Normal traffic flow resuming slowly.",
        "MMDA confirms main artery from Manila to Quezon City now passable.",
        "Road damage assessment team deployed to 12 flood-affected areas in Manila.",
        "Supply route via Marcos Highway partially restored. One lane operational.",
        "Landslide on Antipolo Road cleared. Road now passable to light vehicles.",
        "MMDA sets up road signage for detour routes around flooded sections.",
        "Emergency repair of Nagtahan Bridge underway. Completion expected in 2 days.",
        "Road flooding in Mandaluyong. EDSA MRT stations serving as evacuation holding areas.",
        "Road subsidence near SM Quiapo. Vehicles advised to avoid the area.",
        "Supply delivery to Tondo relief center restored after road clearing.",
        "DPWH deploys 30 equipment units for road clearing in flood-hit NCR.",
        "Route reconnaissance team confirms: no passable road to barangay 184 — barge needed.",
        "MMDA updates: all major routes in Manila operational. Minor flooding in side streets.",
        "Debris clearance completed in 5 barangays. Road access to evacuation centers restored.",
        "Night road clearing operations ongoing to prepare routes for morning relief convoys.",
    ],

    # ── cluster-f : education (80 templates) ─────────────────────────────
    "cluster-f": [
        "DepEd orders class suspension in Metro Manila tomorrow due to typhoon. No classes at all levels.",
        "Class suspension announced for all public schools in Manila. DepEd advisory issued due to heavy rains.",
        "School building at Paco Elementary used as evacuation center. Classes suspended.",
        "Students stranded by flood. School cancels face-to-face classes, shifts to modular learning.",
        "City Mayor cancels all classes due to heavy rainfall warning. DepEd advisory issued.",
        "No classes tomorrow in NCR. DepEd orders suspension due to Typhoon DomengPH.",
        "School flooded in Tondo. Classrooms damaged. Academic activities postponed.",
        "Online class suspended. No internet connectivity due to power outage.",
        "DepEd issues learning continuity plan for students affected by flooding.",
        "Modular learning continues despite typhoon. Distribution of printed modules halted.",
        "Teachers unable to reach school due to flooded roads. Distance learning activated.",
        "Academic calendar adjusted due to typhoon disruptions. Classes to resume next week.",
        "Enrollment temporarily suspended. School offices closed due to flooding.",
        "School suspension extended in flood-prone barangays. Face-to-face classes cancelled.",
        "DepEd advisory: class resumption Monday pending flood water recession.",
        "Emergency suspension of classes in all levels due to flash flood warning.",
        "Learners advised to stay home. School grounds submerged by flood water.",
        "Class disruption in Sampaloc schools. Roof damaged by typhoon.",
        "Blended learning mode activated as school building used for relief operations.",
        "DepEd suspends classes in Quezon City due to heavy rains and flooding.",
        "No school today in Manila. DepEd announcement: class cancellation citywide.",
        "School damage report: windows broken, classrooms flooded in multiple Manila schools.",
        "Make-up classes scheduled after typhoon. Resumed face to face after suspension.",
        "School supplies damaged by flood. DepEd requests emergency funds for replacement.",
        "Remote learning tools distributed to students in evacuation centers.",
        "Class suspension lifted after 3 days. Students return to school on Monday.",
        "DepEd urges parents to update emergency contacts before school reopening.",
        "Classroom shortage reported after storm. Multiple schools serving as shelters.",
        "TXTFIRE QC Commonwealth High School 1st Alarm. School building evacuated.",
        "School used as evacuation shelter for displaced families from Tondo flooding.",
        "DepEd calls for inspection of all school buildings affected by typhoon.",
        "Students in evacuation centers given tablets for continued distance learning.",
        "School feeding program suspended during class suspension. DSWD covers nutrition.",
        "Teachers deploy printed modules to students in flood-hit areas.",
        "Graduation postponed due to flooding. New schedule to be announced by DepEd.",
        "DepEd confirms: 45 schools in NCR used as evacuation centers.",
        "Academic calendar suspended. DepEd to assess damage before resuming.",
        "School bus routes cancelled due to flooded roads. Parents advised.",
        "Learning centers in evacuation sites set up by DepEd teachers voluntarily.",
        "School buildings inspected for damage before students allowed to return.",
        "DepEd issues guidelines for schools resuming after typhoon disruption.",
        "Emergency school supplies distributed to flood-affected learners.",
        "School counselors deployed to handle trauma among students post-typhoon.",
        "Online learning platforms overloaded as all Manila schools shift to distance learning.",
        "DepEd reminds parents: check school's official social media for suspension updates.",
        "Class suspension in effect for all private and public schools in Cavite and Manila.",
        "No face-to-face classes at Ateneo, La Salle, UST, and UP today due to typhoon.",
        "DepEd Region 4A: class suspension in effect in flood-affected municipalities.",
        "Teacher housing adjacent to school damaged by typhoon. DepEd coordinates repairs.",
        "DepEd activates BPR (Basic Performance Recovery) classes after typhoon days.",
        "School in Navotas submerged. Students not expected to return for 2 weeks.",
        "Distance learning kits prepared for distribution to students without devices.",
        "School grounds serving as staging area for relief goods. Classes relocated.",
        "DepEd extends school year to make up for typhoon suspension days.",
        "Academic performance tracking suspended during disaster period. DepEd memo.",
        "Students in temporary shelters given printed self-learning modules.",
        "Classroom rehabilitation begins at flood-damaged schools across Manila.",
        "DepEd sends letter to parents: no classes — ensure safety of children.",
        "School principal confirms: 10 classrooms flooded at Paco National High School.",
        "TESDA suspends vocational classes in flood-affected areas of NCR.",
        "DepEd volunteers helping clean flooded classrooms before re-opening.",
        "No classes order issued by Mayor for all Kindergarten to Grade 12 levels.",
        "School suspension in effect: Tuesday and Wednesday, Manila and nearby LGUs.",
        "DepEd issues LCP (Learning Continuity Plan) for all schools in affected areas.",
        "Private schools reminded by CHED to follow government suspension orders.",
        "Kindergarten learners excused from attendance for entire typhoon week.",
        "DepEd teacher attends to students in evacuation center as part of duty.",
        "School health facilities damaged by typhoon wind and flooding.",
        "Class suspension advisory: check DepEd NCR official page for school updates.",
        "School sports facilities converted into evacuation centers by barangay.",
        "Emergency procurement of replacement school supplies approved by DepEd.",
        "Classes suspended in 6 municipalities due to flooding from typhoon DomengPH.",
        "Schools serving as evacuation sites must be vacated before class resumption.",
        "School repair budget from calamity fund approved for flood-damaged buildings.",
        "DepEd distributes replacement textbooks to flood-affected schools.",
        "No classes proclamation: NCR-wide suspension effective immediately.",
        "Flood-damaged school declared uninhabitable. Students temporarily moved.",
        "DepEd typhoon response: 120 schools assessed, 35 with major flood damage.",
    ],

    # ── cluster-h : dead_missing (80 templates) ───────────────────────────
    "cluster-h": [
        "Missing person report: Maria Santos, 45 years old, last seen in Tondo during flood.",
        "Death toll from Typhoon DomengPH rises to 12. Confirmed by NDRRMC.",
        "Body recovered from Estero de San Miguel. Identity under investigation.",
        "Confirmed dead: 3 residents from Baseco compound swept away by flashflood.",
        "MMDA coordination desk for missing persons opened at Manila City Hall.",
        "Death certificate issued for flood victim recovered in Binondo yesterday.",
        "Declared missing: 5 fishermen from Navotas overdue since storm passed.",
        "Bodies recovered: 2 adults and 1 child found in debris near Intramuros.",
        "Whereabouts unknown: elderly man separated from family during Tondo evacuation.",
        "Feared dead: 3 people last seen on rooftop before floodwaters rose.",
        "Search for missing family from Gagalangin area continues. Day 2.",
        "Morgue at San Lazaro Hospital receiving unidentified flood victims.",
        "Autopsy scheduled for flood victim recovered near Nagtahan Bridge.",
        "Burial assistance provided to families of typhoon fatalities by DSWD.",
        "Ante mortem data collection ongoing for missing persons from typhoon.",
        "Dental records requested for identification of flood victim remains.",
        "Death notice: 7 confirmed fatalities in Manila from Typhoon DomengPH.",
        "Family tracing services activated at Manila City Hall for displaced persons.",
        "Victim identified: flood casualty from Tondo confirmed via fingerprints.",
        "Casualty report from NDRRMC: 15 dead, 8 missing from NCR typhoon.",
        "Missing child alert: 9-year-old separated from parents during Paco flooding.",
        "Interment assistance available through DSWD for typhoon victims.",
        "Confirmed fatality: woman swept away by current in Marikina River.",
        "Last contact was Saturday evening. Family of 4 still missing after flood.",
        "Remains identified as missing construction worker from San Andres.",
        "Death toll confirmed: Typhoon DomengPH claims 10 lives in Manila alone.",
        "Search and retrieval operation for 2 missing persons in Divisoria flood zone.",
        "Bodies recovered from collapsed structure in Pandacan. 3 fatalities confirmed.",
        "Missing family report: Dela Cruz family, 5 members, last seen in Tondo.",
        "Officially missing: 10 persons from coastal barangays after storm surge.",
        "Victim identified through dental record. Cause of death: drowning.",
        "Death report submitted to NSO by Manila Civil Registry after flood fatalities.",
        "Reported dead: 4 persons from informal settler area along Pasig River.",
        "Casualty update: 6 dead, 12 missing in Manila after 24 hours of flooding.",
        "Next of kin being notified for flood victims recovered in Navotas.",
        "Has not returned: woman who went to check her house in flood zone on Sunday.",
        "Search for missing persons ongoing in Sampaloc. MMDA rescue boats deployed.",
        "Feared dead: elderly couple who refused to evacuate from low-lying area.",
        "Post mortem examination ordered for recovered bodies from Typhoon DomengPH.",
        "DNA testing requested for unidentified flood victims at San Lazaro morgue.",
        "Remains recovered near Laguna Lake identified as missing Taguig resident.",
        "NDRRMC: death toll now 20. Separate 5 still reported missing after typhoon.",
        "Missing after flood: teenager last seen swimming near Estero to retrieve belongings.",
        "Cadaver transported to Manila North Cemetery for identification and burial.",
        "Body found under debris at collapsed building in Pandacan industrial area.",
        "Death notice: 2 children among the 8 typhoon fatalities in Manila.",
        "Cremation waiver filed by family of unidentified flood victim.",
        "Missing person found alive after 3 days on rooftop in Navotas flood zone.",
        "Photograph released by family for missing person last seen in Tondo.",
        "Wearing a red shirt and jeans: missing elderly man from Sta. Cruz Barangay.",
        "Bodies recovered from riverbank by coast guard after floodwaters recede.",
        "NDRRMC final report: 25 dead, 10 missing from Typhoon DomengPH in NCR.",
        "Family still searching for mother missing since typhoon hit Malate.",
        "Person missing after flood: last contact via text 2 AM Sunday before lines died.",
        "Casualty count rises as relief teams reach isolated communities in NCR.",
        "Confirmed missing: 8 residents of Barangay 649 not accounted for after flood.",
        "Bodies recovered in river near Marikina during post-typhoon search operation.",
        "Death certificate processing expedited for typhoon victims at Civil Registry.",
        "Victim identified as resident of Navotas — family notified.",
        "Missing family traced to evacuation center in Caloocan after 3 days.",
        "Confirmed dead: 2 rescue workers swept away during flood response operations.",
        "Death toll update 6 AM: 18 confirmed dead, 14 still missing in NCR.",
        "Search party deployed for reported missing persons in flooded Tondo barangay.",
        "Morgue capacity reached at Manila hospitals. Coordinating with NSO.",
        "Interment of 3 typhoon victims conducted at North Cemetery with DSWD assistance.",
        "Officially declared dead: 3 persons from Estero area after 30-day missing status.",
        "Search called off for 2 missing persons in Marikina after 7 days.",
        "Death toll rises to 30 as bodies recovered from remote flood zones.",
        "Missing person found: confirmed survivor after 4 days trapped on second floor.",
        "Report of 2 confirmed fatalities from collapsed structure in Ermita.",
        "Family searching for loved one missing since Sunday flood. Description: elderly female.",
        "NSO records 35 flood-related deaths from Typhoon DomengPH in Metro Manila.",
        "Body discovered by rescue team near Quinta Market flood zone.",
        "Feared dead: 6 people from coastal community in Navotas swept away by storm surge.",
        "Casualty report verified by LGU: 4 dead, 3 missing in Paco flood zone.",
        "Post-flood search ongoing. Rescue boats scanning remaining floodwaters for survivors.",
        "Remains of flood victim handed over to family for burial with full DSWD assistance.",
        "Formally declared missing: 11 residents from high-risk zones in Manila.",
    ],

    # ── cluster-a : relief / food and non-food items ─────────────────────
    "cluster-a": [
        "Relief goods distribution starts at 8 AM at Barangay Hall 105 in Tondo.",
        "DSWD delivers 500 family food packs to evacuation centers in Manila.",
        "Food donation drive at San Andres Sports Complex. Drop-off until 5 PM.",
        "Relief convoy of 10 trucks departs Manila City Hall loaded with rice and canned goods.",
        "Community kitchen set up near Nagtahan to serve hot meals to flood evacuees.",
        "Sleeping mats, blankets, and mosquito nets distributed to overnight evacuees.",
        "Water sachet distribution ongoing at Delpan evacuation center. 2 liters per family.",
        "Relief pack contains: 5kg rice, 10 canned goods, 1 bottle water per family.",
        "Red Cross packs relief goods for distribution to 300 affected families in Binondo.",
        "DSWD calls for additional supply drop of hygiene kits to evacuation sites.",
        "Non-food items including tarps and jerry cans being delivered to Paco flood zone.",
        "Ready-to-eat meal packs distributed by Army at flooded barangay in Navotas.",
        "Baby food and diapers urgently needed at Intramuros evacuation center.",
        "Food pack repacking volunteers needed at Lawton gymnasium. Saturday 7 AM.",
        "Hot meal cooked by community kitchen: 1,000 servings for evacuees today.",
        "Relief goods loaded from DSWD warehouse and dispatched to 5 evacuation centers.",
        "Aid package includes hygiene kit: soap, toothbrush, sanitary napkin, shampoo.",
        "Blankets and sleeping mats urgently needed at covered court in Gagalangin.",
        "Family pack distribution extended to 3 more barangays as floodwaters rise.",
        "Relief drive organized by BFP Station 7 for flood victims in Tondo.",
        "Formula milk and baby food being repacked for delivery to nursing mothers at shelter.",
        "Water container and purification tablets distributed to families without clean water.",
        "Packed relief distributed to 800 families at Delpan Sports Complex.",
        "Red Cross Manila chapter distributes relief goods to 200 flood-affected households.",
        "Supply drop of tarpaulins for temporary roofing of flood-damaged homes.",
        "DSWD confirms 10,000 family food packs reserved for Typhoon DomengPH response.",
        "Drinking water delivery truck deployed to evacuation centers running out of supply.",
        "Relief goods for senior citizens prioritized — includes adult diapers and vitamins.",
        "NFI (non-food items) kit distribution: 1 mosquito net, 1 tarpaulin, 1 jerrycan per family.",
        "Community donation center at Arroceros accepting canned goods, rice, and bottled water.",
        "Operation Bigay Buhay: Tondo LGU launches major relief distribution for typhoon survivors.",
        "1,000 food packs prepared by Manila DSWD for flood victims in 10 barangays.",
        "Rice distribution line stretches 200 meters at Delpan sports complex this morning.",
        "Manila Red Cross chapter sorting and packing relief goods for affected communities.",
        "Drop-off points for relief donations: Lawton parking, SM San Lazaro, Robinsons Manila.",
        "Humanitarian aid from partner NGOs arrives in Manila for flood relief operations.",
        "LGU distributes 5 kg rice, 2 cans sardines, noodles per family at distribution center.",
        "DSWD relief trucks arrive at 4 barangays in Malate. Items include canned food and water.",
        "Volunteer center at Manila City Hall accepting blankets, sleeping mats, and bottled water.",
        "Relief packs for 300 families loaded at DSWD warehouse. Departure at 8 AM.",
        "Corporate donations of instant noodles and bottled water received for flood victims.",
        "Community kitchen volunteers needed: help prepare hot meals at Lawton gymnasium.",
        "Water refilling station set up at Tondo public market for flood evacuees.",
        "Bulk delivery of rice and canned goods dispatched to 8 evacuation centers in Manila.",
        "DSWD field operatives report relief goods inventory adequate for 5 days in Manila.",
        "Second batch of relief packs ready for distribution to coastal barangays in Navotas.",
        "Emergency repacking: 400 volunteers preparing food packs at Manila Hotel ballroom.",
        "Drinking water sachets and purification tablets delivered to isolated flood zones via boat.",
        "PDRRMO coordinates relief distribution schedule across 16 barangays in flood zone.",
        "Relief goods from private sector: 10 tons of rice, 5,000 canned goods for Manila floods.",
        "Hygiene kits distributed to female evacuees at Arroceros park evacuation center.",
        "Distribution of non-food items: tarpaulins, jerry cans, mosquito nets at 7 AM tomorrow.",
        "Relief for senior citizens: priority queue for elderly at distribution center in Tondo.",
        "Excess relief goods from flooded barangays redistributed to areas with greater need.",
        "NFI distribution complete in 5 barangays. Moving to next distribution point in Paco.",
        "Ready-to-eat meal pouches delivered by military to remote flood-affected communities.",
        "Formula milk and baby food urgently needed. Drop-off at Manila City Hall relief center.",
        "Two delivery trucks blocked by floodwater. DSWD rerouting relief via rescue boats.",
        "Call for donation: 500 more family food packs needed for overflow evacuees in Manila.",
        "Relief convoy from Quezon City arrives with rice, blankets, and cleaning supplies.",
        "LGU confirms all 19 barangays in flood zone have received at least one food pack.",
        "Water supply crisis at evacuation center. Requesting additional water containers ASAP.",
        "Relief items checklist per family: 5 kg rice, 3 noodles, 3 sardines, 1 liter water.",
        "Repacking operation completed. 800 family food packs ready for distribution tomorrow.",
        "Aid from national government: 5 truckloads of relief goods arrive in Manila overnight.",
        "Distribution of blankets and raincoats to families living near floodprone estero areas.",
        "DSWD central office releases relief goods worth P5 million for Metro Manila typhoon response.",
        "Manila City Hall relief operations center open 24 hours for donation drop-off and pickup.",
        "Relief goods for persons with disabilities: wheelchair-accessible distribution lane set up.",
        "Donations of diapers, adult and infant, needed for flood evacuees at Delpan center.",
        "Leftover relief goods from earlier distribution being repacked and dispatched to Navotas.",
        "Community pantry set up by barangay residents to supplement DSWD relief operations.",
        "Relief goods arrival confirmed. Sorting begins at warehouse. Distribution 7 AM tomorrow.",
        "All relief packs at Luneta distribution point given out. Next batch expected Friday.",
        "LGU thanks donors: 2,000 family packs distributed to typhoon victims in 3 days.",
        "Canned goods and instant noodles most needed relief items per DSWD advisory.",
        "Family food pack includes cooked rice, noodles, sardines, biscuits and mineral water.",
        "DSWD encourages cash donations instead of goods to streamline relief distribution.",
        "Relief operations update: 15,000 families served in Manila since typhoon landfall.",
    ],
}

# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)   # #FireAlert → firealert (preserves as single word)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_tagalog(text: str, threshold: int = 3) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return sum(1 for w in words if w in _TAGALOG_MARKERS) >= threshold


def _tokenize_lemmatize(text: str) -> str:
    tokens = re.findall(r"[a-z][\w'-]*", text.lower())
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if len(t) >= 2]
    tokens = [t for t in tokens if (t not in _STOPS) or (t in _KEEP_TOKENS)]
    return " ".join(tokens)


def preprocess(text: str, translator) -> tuple[str, str]:
    """Returns (clean_text, final_tokens_string). Translates Tagalog if available."""
    clean = _clean_text(text)
    if translator and _is_tagalog(clean):
        try:
            translated = translator.translate(clean)
            if translated and len(translated.strip()) > 5:
                clean = _clean_text(translated)
            time.sleep(0.10)
        except Exception as exc:
            log.debug("Translation skipped: %s", exc)
    return clean, _tokenize_lemmatize(clean)


# ---------------------------------------------------------------------------
# Labeling — with relevance filter
# ---------------------------------------------------------------------------

def score_and_label(clean_text: str) -> tuple[str | None, int]:
    """
    Returns (cluster_id, best_score).
    cluster_id is None when best_score < RELEVANCE_THRESHOLD (post is irrelevant).
    """
    text_lower = clean_text.lower()
    scores: dict[str, int] = {}
    for cid, kws in CLUSTER_KEYWORDS.items():
        scores[cid] = sum(2 if " " in kw else 1 for kw in kws if kw in text_lower)
    best_cluster = max(scores, key=scores.get)
    best_score   = scores[best_cluster]
    if best_score < RELEVANCE_THRESHOLD:
        return None, best_score
    return best_cluster, best_score


def assign_priority(
    clean_text: str, cluster_id: str,
    engagement: int, p40: float, p75: float,
) -> str:
    t = clean_text.lower()
    if cluster_id == "cluster-g" and any(kw in t for kw in _HIGH_RESCUE_KW):
        return "High"
    if cluster_id == "cluster-h" and any(kw in t for kw in _HIGH_DEATH_KW):
        return "High"
    if cluster_id == "cluster-b" and any(kw in t for kw in _HIGH_HEALTH_KW):
        return "High"
    if engagement >= p75:
        return "High"
    if cluster_id == "cluster-g":
        return "Medium"
    if cluster_id == "cluster-c":
        return "Medium"
    if cluster_id == "cluster-e" and any(kw in t for kw in _MED_WARNING_KW):
        return "Medium"
    if cluster_id == "cluster-b":
        return "Medium"
    if engagement >= p40:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(csv_path: str) -> None:
    log.info("=" * 64)
    log.info("MANA Model Training — %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 64)

    # Step 1: Load CSV
    log.info("Loading: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("  Loaded %d rows", len(df))

    for col in ("reactions", "comments", "shares", "reposts", "likes", "views"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df[df["caption"].notna() & (df["caption"].str.strip() != "")].copy()
    df = df.reset_index(drop=True)
    log.info("  After dropping empty captions: %d posts", len(df))

    # Step 2: Translation
    translator = None
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="tl", target="en")
        log.info("  Translator: ready (Tagalog -> English)")
    except Exception as exc:
        log.warning("  Translator unavailable (%s) — Tagalog posts not translated", exc)

    # Step 3: Preprocess
    log.info("Preprocessing %d captions ...", len(df))
    clean_texts:   list[str] = []
    token_strings: list[str] = []
    for i, row in df.iterrows():
        if i > 0 and i % 100 == 0:
            log.info("  ... %d / %d", i, len(df))
        c, t = preprocess(row["caption"], translator)
        clean_texts.append(c)
        token_strings.append(t)

    # Step 4: Label + relevance filter
    log.info("Labeling and filtering irrelevant posts (threshold >= %d) ...", RELEVANCE_THRESHOLD)

    reactions = df.get("reactions", pd.Series([0]*len(df))).fillna(0).astype(int)
    comments  = df.get("comments",  pd.Series([0]*len(df))).fillna(0).astype(int)
    shares    = df.get("shares",    pd.Series([0]*len(df))).fillna(0).astype(int)
    reposts   = df.get("reposts",   pd.Series([0]*len(df))).fillna(0).astype(int)
    engage_all = (reactions + comments + shares + reposts).values
    p40 = float(np.percentile(engage_all, 40))
    p75 = float(np.percentile(engage_all, 75))
    log.info("  Engagement P40=%.0f  P75=%.0f", p40, p75)

    filt_clean:   list[str] = []
    filt_tokens:  list[str] = []
    filt_cluster: list[str] = []
    filt_priority:list[str] = []
    filt_reactions:list[int] = []
    filt_comments: list[int] = []
    filt_shares:   list[int] = []
    filt_reposts:  list[int] = []

    n_irrelevant = 0
    n_geo_filtered = 0
    for i in range(len(df)):
        cid, score = score_and_label(clean_texts[i])
        if cid is None:
            n_irrelevant += 1
            continue
        if not is_manila_relevant(clean_texts[i]):
            n_irrelevant += 1
            n_geo_filtered += 1
            continue
        eng = int(engage_all[i])
        filt_clean.append(clean_texts[i])
        filt_tokens.append(token_strings[i])
        filt_cluster.append(cid)
        filt_priority.append(assign_priority(clean_texts[i], cid, eng, p40, p75))
        filt_reactions.append(int(reactions.iloc[i]))
        filt_comments.append(int(comments.iloc[i]))
        filt_shares.append(int(shares.iloc[i]))
        filt_reposts.append(int(reposts.iloc[i]))

    log.info("  Irrelevant (filtered): %d / %d  (geo-filtered: %d)", n_irrelevant, len(df), n_geo_filtered)
    log.info("  Relevant (kept)      : %d", len(filt_clean))
    log.info("  Cluster distribution : %s", dict(Counter(filt_cluster)))
    log.info("  Priority distribution: %s", dict(Counter(filt_priority)))

    # Step 5: Augment thin clusters
    log.info("Augmenting thin clusters (target >= %d each) ...", MIN_PER_CLUSTER)
    cluster_dist = Counter(filt_cluster)

    aug_clean:    list[str] = []
    aug_tokens:   list[str] = []
    aug_clusters: list[str] = []
    aug_priority: list[str] = []
    aug_r: list[int] = []; aug_c: list[int] = []
    aug_s: list[int] = []; aug_p: list[int] = []

    for cid, templates in _SYNTHETIC.items():
        count  = cluster_dist.get(cid, 0)
        needed = max(0, MIN_PER_CLUSTER - count)
        if needed == 0:
            continue
        added = 0
        for tmpl in templates:
            if added >= needed:
                break
            c, t = preprocess(tmpl, None)
            aug_clean.append(c); aug_tokens.append(t)
            aug_clusters.append(cid)
            # synthetic priority: b/h = Medium (health/death incidents), c = Medium,
            # d/f = Low (logistics/education updates)
            # cluster-a (relief distribution), cluster-d (logistics), cluster-f (education)
            # are informational/non-urgent → Low priority when 0 engagement.
            # cluster-b (health), cluster-c (evacuation), cluster-h (dead/missing) → Medium.
            aug_priority.append("Low" if cid in ("cluster-a", "cluster-d", "cluster-f") else "Medium")
            aug_r.append(0); aug_c.append(0); aug_s.append(0); aug_p.append(0)
            added += 1
        log.info("  %-12s  %d -> %d  (+%d synthetic)", cid, count, count + added, added)

    n_real = len(filt_clean)
    all_clean    = filt_clean    + aug_clean
    all_tokens   = filt_tokens   + aug_tokens
    all_clusters = filt_cluster  + aug_clusters
    all_priority = filt_priority + aug_priority
    all_r        = filt_reactions + aug_r
    all_c        = filt_comments  + aug_c
    all_s        = filt_shares    + aug_s
    all_p        = filt_reposts   + aug_p

    log.info("  Total: %d  (real=%d  synthetic=%d)", len(all_clean), n_real, len(aug_clean))
    log.info("  Final cluster dist : %s", dict(Counter(all_clusters)))
    log.info("  Final priority dist: %s", dict(Counter(all_priority)))

    # Step 6: Train CorEx
    log.info("")
    log.info("--- CorEx: iterative anchored topic modelling (up to 5 passes) ---")
    corex_result = _corex_train_iteratively(all_tokens, max_iterations=5, target_coherence=3.0)
    log.info("  Corpus size      : %d", corex_result["corpus_size"])
    log.info("  Best iteration   : %d / %d",
             corex_result["best_iteration"], len(corex_result["iterations"]))
    log.info("  Overall coherence: %.4f", corex_result["best_overall_coherence"])
    for it in corex_result["iterations"]:
        low = it["low_coherence_topics"] or "none"
        log.info("    iter %d  coherence=%.4f  low=%s",
                 it["iteration"], it["overall_coherence"], low)

    # Step 7: Train SVM
    log.info("")
    log.info("--- SVM: TF-IDF + LinearSVC One-vs-Rest ---")
    svm_result = _svm_train(all_tokens, [[c] for c in all_clusters])
    log.info("  corpus=%d  best_C=%s  f1_macro=%.4f",
             svm_result["corpus_size"], svm_result["best_C"], svm_result["f1_macro"])
    log.info("  Per-class report:")
    for label, s in svm_result["per_class_report"].items():
        warn = "  <-- BELOW TARGET" if s["f1"] < 0.70 and s["support"] > 0 else ""
        log.info("    %-12s  P=%.3f  R=%.3f  F1=%.3f  sup=%d%s",
                 label, s["precision"], s["recall"], s["f1"], s["support"], warn)

    # Step 8: VADER
    log.info("")
    log.info("--- VADER: scoring %d posts ---", len(all_clean))
    vader_scores = [_vader_analyze(c) for c in all_clean]

    # Step 9: CorEx topic predictions for RF
    log.info("--- CorEx: topic predictions for RF features ---")
    from services.corex.topic_modeler import predict_topics_batch as _corex_pred_batch
    topic_preds = _corex_pred_batch(all_tokens)

    # Step 10: Build RF records
    log.info("--- RF: building feature matrix ---")
    records: list[dict] = []
    for i in range(len(all_clean)):
        v = vader_scores[i]
        c = v["compound"]
        records.append({
            "post_id":        str(i),
            "priority_label": all_priority[i],
            "compound":       c,
            "positive":       v["positive"],
            "negative":       v["negative"],
            "neutral":        v["neutral"],
            "sentiment_label": ("Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")),
            "reactions": all_r[i], "comments": all_c[i],
            "shares":    all_s[i], "reposts":  all_p[i],
            "clean_text":   all_clean[i],
            "topic_labels": [p["topic"] for p in topic_preds[i]],
        })

    X, feature_columns = _build_feature_matrix(records)
    y = np.array([r["priority_label"] for r in records])
    log.info("  Shape: %s   Classes: %s", X.shape, dict(Counter(y.tolist())))

    # Step 11: Train RF
    log.info("--- RF: RandomForest (n=%d, max_depth=%d, balanced) ---",
             RF_N_ESTIMATORS, RF_MAX_DEPTH)
    unique_cls, cls_counts = np.unique(y, return_counts=True)
    can_strat = len(unique_cls) > 1 and all(c >= 2 for c in cls_counts)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=(y if can_strat else None))
    except ValueError:
        X_tr, X_te, y_tr, y_te = X, X, y, y
        log.warning("  Stratified split failed — using full set for train/test.")

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, class_weight="balanced",
        max_depth=RF_MAX_DEPTH, min_samples_leaf=1,
        random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    rf_report   = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    rf_accuracy = float((np.array(y_pred) == np.array(y_te)).mean())
    rf_wf1      = float(rf_report.get("weighted avg", {}).get("f1-score", 0.0))

    log.info("  Accuracy    : %.4f", rf_accuracy)
    log.info("  Weighted F1 : %.4f", rf_wf1)
    for cls in ["High", "Medium", "Low"]:
        if cls in rf_report:
            r = rf_report[cls]
            warn = "  <-- BELOW TARGET" if r["f1-score"] < 0.70 and r["support"] > 0 else ""
            log.info("    %-8s  P=%.3f  R=%.3f  F1=%.3f  sup=%d%s",
                     cls, r["precision"], r["recall"], r["f1-score"], r["support"], warn)

    # Step 12: Save RF
    log.info("--- Saving RF artifacts ---")
    joblib.dump(clf, RF_MODEL_PATH)
    RF_COLUMNS_PATH.write_text(json.dumps(feature_columns))
    rf_meta = {
        "trained_at":         datetime.now(timezone.utc).isoformat(),
        "corpus_size":        len(records),
        "n_estimators":       RF_N_ESTIMATORS,
        "accuracy":           round(rf_accuracy, 4),
        "class_distribution": {lbl: int(np.sum(y == lbl)) for lbl in PRIORITY_LABELS},
        "feature_columns":    feature_columns,
    }
    RF_META_PATH.write_text(json.dumps(rf_meta, indent=2))
    log.info("  Saved: %s", RF_MODEL_PATH)
    log.info("  Saved: %s", RF_COLUMNS_PATH)
    log.info("  Saved: %s", RF_META_PATH)

    # Step 13: Summary + metric check
    log.info("")
    log.info("=" * 64)
    log.info("Training complete.")
    log.info("  Irrelevant posts filtered : %d / %d (%.1f%%)  geo-filtered=%d",
             n_irrelevant, len(df), 100*n_irrelevant/len(df), n_geo_filtered)
    log.info("  CorEx overall coherence   : %.4f", corex_result["best_overall_coherence"])
    log.info("  SVM   f1_macro            : %.4f", svm_result["f1_macro"])
    log.info("  RF    accuracy            : %.4f", rf_accuracy)
    log.info("  RF    weighted F1         : %.4f", rf_wf1)
    log.info("=" * 64)

    warnings_list: list[str] = []
    if svm_result["f1_macro"] < 0.70:
        warnings_list.append(f"SVM f1_macro={svm_result['f1_macro']:.4f} < 0.70")
    for label, s in svm_result["per_class_report"].items():
        if s["f1"] < 0.70 and s["support"] > 0:
            warnings_list.append(f"SVM {label} F1={s['f1']:.4f} < 0.70")
    if rf_wf1 < 0.70:
        warnings_list.append(f"RF weighted F1={rf_wf1:.4f} < 0.70")
    for cls in ["High", "Medium", "Low"]:
        r = rf_report.get(cls, {})
        if r.get("f1-score", 1.0) < 0.70 and r.get("support", 0) > 0:
            warnings_list.append(f"RF {cls} F1={r['f1-score']:.4f} < 0.70")

    if warnings_list:
        log.warning("Metric warnings:")
        for w in warnings_list:
            log.warning("  WARNING: %s", w)
        sys.exit(1)
    else:
        log.info("All metrics meet targets (F1 >= 0.70 for every class).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all MANA ML models from a posts CSV export.")
    parser.add_argument("csv_path", help="Path to posts_rows.csv")
    args = parser.parse_args()
    if not os.path.isfile(args.csv_path):
        log.error("File not found: %s", args.csv_path)
        sys.exit(1)
    main(args.csv_path)
