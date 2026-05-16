"""
Text preprocessing helpers for imported Apify records.

The original extraction, cleaning, and tokenization stages remain intact.
Everything below tokenization extends the pipeline for downstream NLP models.
"""

from __future__ import annotations

import json as _json
import os as _os
import re
from html import unescape

from models import PreprocessedText

try:
    from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional dependency
    GoogleTranslator = None

try:
    from nltk.stem import WordNetLemmatizer
except Exception:  # pragma: no cover - optional dependency
    WordNetLemmatizer = None

# Set to True the first time Google Translate returns a quota/rate-limit error.
# All subsequent translate_text() calls return the input unchanged instead of
# hammering the API with requests that will keep failing.
_TRANSLATE_QUOTA_EXHAUSTED = False

TEXT_FIELDS = ("text", "caption", "content", "comment", "message", "postText", "body")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_]+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[a-zA-Z]")
EMOJI_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
VADER_STRIP_RE = re.compile(
    r"[^a-zA-Z0-9\s\!\?\.\,\'\:"
    r"\U0001F000-\U0001FFFF"   # misc symbols+pictographs, emoticons
    r"\U00002600-\U000027BF"   # misc symbols, dingbats
    r"\U00002300-\U000023FF"   # misc technical
    r"\U000025A0-\U000025FF"   # geometric shapes
    r"]",
    re.UNICODE,
)

# ── Dual-dictionary setup for Tagalog detection ───────────────────────────────
# Loaded once at import time. Falls back gracefully when files/libraries are absent.

_TAGALOG_DICT_PATH = _os.path.join(_os.path.dirname(__file__), "models", "tagalog_wordlist.json")
try:
    with open(_TAGALOG_DICT_PATH, encoding="utf-8") as _f:
        _tagalog_words: frozenset = frozenset(w.lower() for w in _json.load(_f))
except Exception:
    _tagalog_words: frozenset = frozenset()

try:
    import enchant as _enchant
    _en_dict = _enchant.Dict("en_US")
except Exception:
    _en_dict = None

NEGATION_WORDS = {
    "no", "not", "never", "walang", "hindi", "wala", "cannot", "cant", "can't", "dont", "don't",
}
NEGATION_CONNECTORS = {"na", "ng", "the", "a", "an", "very", "too", "so"}
DISASTER_TERMS = {
    "flood", "flooding", "flooded", "rescue", "rescues", "rescued", "evacuation", "evacuate", "evacuated",
    "center", "centre", "shelter", "relief", "goods", "water", "road", "damage", "power", "outage",
    "emergency", "storm", "typhoon", "landslide", "earthquake", "fire", "signal", "medical", "hospital",
    "alert", "warning", "stranded", "trapped", "missing", "dead", "telecommunications", "logistics",
    "wash", "nutrition", "school", "class", "suspension", "boat", "roof", "camp",
}
TAGALOG_HINTS = {
    # Core negations / function words unique to Filipino
    "walang", "wala", "hindi", "huwag",
    # Disaster-domain Tagalog nouns
    "tubig", "pagkain", "gamot", "kuryente", "putik",
    "baha", "ulan", "bagyo", "lindol", "sunog",
    "kalsada",
    # Geographic / community terms
    "barangay", "sityo", "purok",
    # Aid / rescue (Tagalog forms only)
    "tulong", "saklolo", "sagipin",
    # Question / location words
    "nasaan", "saan", "kailan", "paano", "bakit",
    # Verbal disaster forms (Tagalog morphology)
    "nastranded", "naligtas", "nailigtas", "nailikas",
    "nawalan", "naputol", "binaha", "bumaha", "nagbabaha",
    "nakakulong", "nalunod",
    "patay", "sugat", "namatay", "nasaktan",
    # High-frequency Filipino discourse markers (near-certain Filipino signal)
    "mga", "po",
    # Pronouns absent from English
    "namin", "natin", "nila", "kami", "sila", "tayo", "kayo",
    "nandito", "nandiyan", "nandoon",
    # Common Filipino intensifiers / exclamations
    "grabe", "grabeh", "grabi",
    "matindi", "malakas", "marami",
    # na- prefix past-tense Tagalog verbs (extremely common in disaster reports)
    "nabaha", "nasalba", "nasira", "naipit", "naiipit",
    "natapos", "natigil", "naiwan", "nasugatan", "nawasak",
    # Additional disaster-specific Tagalog nouns
    "pagbaha", "pagguho", "sakuna", "pinsala", "biktima",
    # Cebuano/Bisaya — covers Visayas + Mindanao disaster posts (~30% of PH)
    "tabang", "lunod", "naanod", "napukan", "buhawi",
    "gubot", "asa", "unsay", "ngano",
}
INFORMAL_WORD_MAP = {
    # Filipino negation shortcuts → standard Tagalog (helps translation + negation handling)
    "di": "hindi",
    "dili": "hindi",
    "wag": "huwag",
    "hwag": "huwag",
    # Compressed Tagalog particles → standard forms (cleaner translator input)
    "lng": "lang",
    "nman": "naman",
    "nmn": "naman",
    # Location abbreviations
    "brgy": "barangay",
    "bgy": "barangay",
    "bldg": "building",
    # Informal spelling variants of common Filipino words
    "grabeh": "grabe",
    "grabi": "grabe",
    # Taglish compound disaster words → English equivalents (VADER + CorEx friendly)
    "nastranded": "stranded",
    "naflood": "flooded",
    "nabaha": "flooded",
    "binaha": "flooded",
    # English abbreviations safe to expand in this domain
    "pls": "please",
    "plss": "please",
    "tnx": "thanks",
    "ty": "thanks",
    "thx": "thanks",
}
EMOTION_ONLY_WORDS = {
    "grabe", "grabi", "grabeh", "hala", "hay", "hays", "omg", "aww", "aw", "argh", "sad", "cry", "iyak", "wow",
    "lol", "lmao", "help", "pls", "please", "prayers", "pray", "rip",
}
BIGRAM_ALLOWLIST = {
    # original 15
    "flood_water", "rescue_team", "road_damage", "power_outage",
    "evacuation_center", "relief_goods", "emergency_shelter",
    "class_suspension", "signal_loss", "medical_team", "clean_water",
    "food_pack", "safe_space", "blocked_road", "rescue_boat",
    # cluster-a (Food/NFI)
    # *_good variants guard against heuristic lemmatizer stripping the -s from "goods"
    "relief_pack", "canned_goods", "canned_good", "relief_good", "hygiene_kit", "water_refill",
    # cluster-b (WASH/Medical)
    "health_center", "first_aid",
    # cluster-c (CCCM)
    "evacuation_site", "displaced_families", "covered_court",
    # cluster-d (Logistics)
    "blocked_bridge", "alternate_route", "road_clearing",
    # cluster-e (ETC)
    "no_signal", "no_network", "power_bank", "cell_site",
    # cluster-f (Education)
    "walang_pasok", "school_closure", "no_classes",
    # cluster-g (SRR)
    "search_party", "coast_guard", "swift_water",
    # cluster-h (MDM)
    "missing_person", "family_tracing", "body_identified",
}
STOP_WORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "are", "as", "at", "be", "been",
    "before", "being", "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "of", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves", "sa", "ang",
    "mga", "si", "ni", "ng", "na", "po", "pa", "lang", "din", "rin", "ito", "iyan", "yun", "yan", "may",
    "meron", "naman", "kasi", "pero", "daw", "raw", "nga", "nasa", "dito", "doon",
    # Additional Tagalog pronouns
    "ako", "ikaw", "ka", "siya", "niya", "namin", "natin", "nila",
    "kami", "tayo", "kayo", "sila", "akin", "iyo", "kanya",
    # Additional Tagalog demonstratives / locatives
    "iyon", "diyan", "yung", "ung",
    # Common Tagalog discourse fillers
    "ngayon", "pala", "nang", "muna", "ba", "eh", "ha",
    "talaga", "siguro", "sana", "kaya", "kahit",
    "kapag", "kung", "habang", "ganun", "ganito",
}
LEMMA_OVERRIDES = {
    "flooding": "flood",
    "flooded": "flood",
    "rescues": "rescue",
    "rescued": "rescue",
    "rescuing": "rescue",
    "stranded": "strand",
    "supplies": "supply",
    "children": "child",
    "people": "person",
    "roads": "road",
    "centers": "center",
    "centres": "center",
    "shelters": "shelter",
}
RELEVANCE_TERMS = DISASTER_TERMS | {
    "baha", "saklolo", "tulong", "gamot", "kuryente", "lindol", "bagyo", "ulan", "landslide", "ashfall",
    "volcano", "relief_goods", "evacuation_center", "power_outage", "road_damage", "signal_loss",
}
STRONG_DISASTER_TERMS = {
    "flood", "flooding", "flooded", "rescue", "rescue_team", "stranded", "trapped", "sos", "evacuation",
    "evacuation_center", "relief_goods", "landslide", "earthquake", "typhoon", "storm", "volcano",
    "ashfall", "power_outage", "signal_loss", "medical", "hospital", "missing", "dead", "fatality",
    "casualty", "family tracing", "body identified", "missing person",
    # PHIVOLCS volcano bulletin vocabulary
    "strombolian", "effusive eruption", "lava flow", "pyroclastic", "uson", "rockfall",
    "alert level", "permanent danger zone", "phivolcs", "danger zone", "volcanic eruption",
    "incandescent", "lava fountain", "sulfur dioxide", "mayon", "kanlaon", "taal",
}
PUBLIC_SERVICE_MAINTENANCE_TERMS = {
    "cleanliness", "kalinisan", "flushing", "flushing operation", "bugahan", "washdown", "road washing",
    "cleanup", "clean up", "street cleaning", "canal cleaning", "declogging", "traffic advisory",
    "sanitation drive", "beautification", "road clearing", "public works",
    # Political / governance PR — not disaster response
    "motorcycle donation", "motor pool", "flag-raising ceremony", "flag raising",
    "donation ceremony", "turnover ceremony", "city hall", "board of trustees",
    "housing program", "vertical housing", "binondominium", "base community",
    "real estate", "nrea", "dhsud", "national convention", "landbank", "development bank",
    "college president", "vice president", "institutional advancement",
    "producer to consumer", "p2c", "ilocano trader", "good governance",
    "bilis kilos", "self-made man", "private sector effort", "operational use",
    "police district", "motor pool", "mpd", "manila police",
    "catheterization laboratory", "angioplasty", "college of medicine",
    "squatter", "informal settlement", "relocation", "housing development board",
    "monthly fee", "ownership transfer", "public service announcement",
    "news alert", "manila pio", "city mayor", "lgu program", "lgu project",
    # Cultural / religious events — not disaster response
    "flores de mayo", "sagala", "prusisyon", "hermano mayor", "hermana mayor",
    "fashion designer", "fashion designers association", "fdap",
    "department of tourism", "national parks development", "npdc",
    "anibersaryo", "selebrasyon", "pagdiriwang", "debosyon", "tradisyon",
    "manila hotel", "cultural event", "cultural show", "beauty pageant",
    "fiesta", "pista", "parade", "festival", "float", "carnival",
    "concert", "sports event", "basketball", "pba", "uaap",
    "award ceremony", "awarding", "recognition ceremony", "graduation",
    "inauguration", "ribbon cutting", "groundbreaking ceremony", "groundbreaking",
    # School construction / inspection PR — not disaster-related school closure
    "school building", "school construction", "construksiyon", "konstruksiyon",
    "gusali", "palapag", "fully air-conditioned", "air-conditioned classrooms",
    "e-library", "learning environment", "inspeksiyunin", "progreso ng konstruksiyon",
    "modernong pasilidad", "auditorium", "gymnasium", "laborator",
}


def extract_raw_text(item: dict, fallback_text: str | None = None):
    for field in TEXT_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    if isinstance(fallback_text, str) and fallback_text.strip():
        return fallback_text.strip(), "fallback"
    return None, None


def clean_text(text: str):
    value = unescape(text or "")
    value = HTML_TAG_RE.sub(" ", value)
    value = URL_RE.sub(" ", value)
    value = MENTION_RE.sub(" ", value)
    value = HASHTAG_RE.sub(r" \1 ", value)
    value = value.lower()
    value = NON_WORD_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    return value


def clean_text_for_vader(text: str) -> str:
    """Branch 2B (Sentiment Track): minimal cleaning for VADER.

    Preserves casing (ALL CAPS is a VADER intensity signal) and sentiment
    punctuation (!!!, ? carry weight). Unicode emoji are preserved so VADER
    can score them from its built-in lexicon (e.g. 😭 → negative signal).
    """
    value = unescape(text or "")
    value = HTML_TAG_RE.sub(" ", value)
    value = URL_RE.sub(" ", value)
    value = MENTION_RE.sub(" ", value)
    value = HASHTAG_RE.sub(r" \1 ", value)
    # VADER_STRIP_RE keeps letters, digits, spaces, !?.,': and unicode emoji ranges.
    value = VADER_STRIP_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    return value


def tokenize_text(cleaned_text: str):
    if not cleaned_text:
        return []
    return [token for token in cleaned_text.split(" ") if token]


def tokenize_preserving_apostrophes(text: str):
    normalized = WHITESPACE_RE.sub(" ", (text or "").strip().lower())
    return re.findall(r"[a-z]+(?:'[a-z]+)?", normalized)


def build_location_terms(item: dict):
    location_terms = set()
    for key in ("location", "pageName", "page_source", "author"):
        value = item.get(key)
        if not isinstance(value, str):
            continue
        for token in tokenize_text(clean_text(value)):
            if len(token) > 2:
                location_terms.add(token)
    return location_terms


def normalize_informal_tokens(tokens: list[str]) -> list[str]:
    """Collapse repeated-char emphasis and map informal Filipino/Taglish words to standard forms.

    Applied only on the ML path — raw_text and clean_text are never touched.
    Collapse 3+ repeated chars to 2 so "flooood" → "flood" without breaking
    legitimate double-letter words like "good" or "need".
    """
    result = []
    for token in tokens:
        t = _REPEATED_CHAR_RE.sub(r"\1\1", token)
        t = INFORMAL_WORD_MAP.get(t, t)
        result.append(t)
    return result


def _is_tagalog_word(word: str) -> bool:
    """Check both the scraped Tagalog wordlist AND TAGALOG_HINTS (union).

    TAGALOG_HINTS is always consulted — it covers na- morphological forms and
    disaster-domain vocabulary that may be absent from the scraped wordlist.
    The scraped dictionary extends coverage beyond the 94-word curated set.
    """
    w = word.lower()
    return w in TAGALOG_HINTS or w in _tagalog_words


def _is_english_word(word: str) -> bool:
    """Return True if PyEnchant confirms the word is valid English."""
    if _en_dict is None:
        return False
    try:
        return _en_dict.check(word)
    except Exception:
        return False


# Distinctive Filipino verb-form prefixes. Only applied to words NOT confirmed English.
# Covers: na-past (nasalba, naharang), nag-past (nagbaha), naka-resultative (nakalusot).
_TAGALOG_MORPHO_PREFIXES = frozenset({"na", "nag", "naka"})

# Fraction of meaningful (len≥3) tokens unknown to both dicts before Rule 3 triggers.
# Raise to 0.40–0.50 if English posts with acronyms get over-translated.
_UNKNOWN_RATIO_THRESHOLD = 0.40


def should_translate(cleaned_text: str, tokens: list[str]):
    if not cleaned_text or not tokens:
        return False

    meaningful = [t for t in tokens if len(t) >= 3]
    if not meaningful:
        return False

    # Rule 1: any confirmed Tagalog word → translate
    for token in meaningful:
        if _is_tagalog_word(token):
            return True

    # Rules 2 & 3 require PyEnchant to safely distinguish English from unknown.
    # Without it, only Rule 1 fires (fallback to TAGALOG_HINTS behaviour).
    if _en_dict is not None:
        for token in meaningful:
            if _is_english_word(token):
                continue
            # Rule 2: Filipino morphological prefix on a non-English word → likely Tagalog
            w = token.lower()
            if any(w.startswith(p) and len(w) > len(p) + 1 for p in _TAGALOG_MORPHO_PREFIXES):
                return True

        # Rule 3: high ratio of words unknown to both dicts → likely Taglish
        unknown = sum(1 for t in meaningful if not _is_english_word(t))
        if unknown / len(meaningful) > _UNKNOWN_RATIO_THRESHOLD:
            return True

    return False


def translate_text(cleaned_text: str, translator=None):
    global _TRANSLATE_QUOTA_EXHAUSTED
    if not cleaned_text:
        return "", "skipped", None
    if _TRANSLATE_QUOTA_EXHAUSTED:
        return cleaned_text, "skipped", "Google Translate quota exhausted — skipping translation."
    if translator is None:
        if GoogleTranslator is None:
            return cleaned_text, "skipped", "Translator dependency unavailable."
        translator = GoogleTranslator(source="auto", target="en")
    try:
        translated = translator.translate(cleaned_text)
        translated = (translated or cleaned_text).strip()
        status = "skipped" if translated == cleaned_text.strip() else "translated"
        return translated, status, None
    except Exception as exc:
        err_lower = str(exc).lower()
        if any(sig in err_lower for sig in ("429", "quota", "too many", "rate limit", "limit exceeded")):
            _TRANSLATE_QUOTA_EXHAUSTED = True
            import sys
            print("[MANA] Google Translate quota hit — disabling translation for this run.", file=sys.stderr)
        return cleaned_text, "error", str(exc)


def apply_negation_handling(tokens: list[str]):
    handled = []
    changed = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in NEGATION_WORDS:
            j = i + 1
            while j < len(tokens) and tokens[j] in NEGATION_CONNECTORS:
                handled.append(tokens[j])
                j += 1
            if j < len(tokens):
                handled.append(f"{token}_{tokens[j]}")
                changed = True
                i = j + 1
                continue
        handled.append(token)
        i += 1
    return handled, changed


def heuristic_lemmatize(token: str):
    if token in LEMMA_OVERRIDES:
        return LEMMA_OVERRIDES[token]
    if "_" in token:
        parts = [heuristic_lemmatize(part) for part in token.split("_")]
        return "_".join(parts)
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if token.endswith("ed") and len(token) > 4:
        base = token[:-2]
        if base.endswith("u"):
            return base + "e"
        return base
    if token.endswith("es") and len(token) > 4 and not token.endswith("ses"):
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def lemmatize_tokens(tokens: list[str]):
    lemmatizer = None
    if WordNetLemmatizer is not None:
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("tests")
        except Exception:
            lemmatizer = None

    lemmatized = []
    changed = False
    for token in tokens:
        if "_" in token:
            parts = token.split("_")
            lemma_parts = [lemmatize_tokens([part])[0][0] for part in parts]
            lemma = "_".join(lemma_parts)
        elif lemmatizer is not None:
            try:
                lemma = lemmatizer.lemmatize(token, "v")
                lemma = lemmatizer.lemmatize(lemma, "n")
            except Exception:
                lemma = heuristic_lemmatize(token)
        else:
            lemma = heuristic_lemmatize(token)
        lemmatized.append(lemma)
        changed = changed or lemma != token
    return lemmatized, changed


def detect_bigrams(tokens: list[str]):
    bigrams = []
    for left, right in zip(tokens, tokens[1:]):
        candidate = f"{left}_{right}"
        if candidate in BIGRAM_ALLOWLIST:
            bigrams.append(candidate)
    return bigrams


def remove_stop_words(tokens: list[str], location_terms: set[str] | None = None):
    location_terms = location_terms or set()
    final_tokens = []
    for token in tokens:
        if token in NEGATION_WORDS or token in DISASTER_TERMS or token in location_terms:
            final_tokens.append(token)
            continue
        if "_" in token:
            final_tokens.append(token)
            continue
        if token in STOP_WORDS:
            continue
        final_tokens.append(token)
    return final_tokens


def is_emotion_only_text(raw_text: str, clean_text_value: str, tokens: list[str]):
    raw = (raw_text or "").strip()
    cleaned = (clean_text_value or "").strip()
    if raw and not LETTER_RE.search(raw) and EMOJI_ONLY_RE.match(raw):
        return True
    if len(tokens) <= 2 and tokens and all(token in EMOTION_ONLY_WORDS for token in tokens):
        return True
    if len(cleaned) <= 12 and cleaned in EMOTION_ONLY_WORDS:
        return True
    return False


def is_relevant_text(final_tokens: list[str], bigrams: list[str], clean_text_value: str, parent_context_text: str | None = None):
    combined = set(final_tokens) | set(bigrams)
    scan_text = " ".join([clean_text_value or "", parent_context_text or ""]).strip().lower()

    maintenance_hits = {term for term in PUBLIC_SERVICE_MAINTENANCE_TERMS if term in scan_text}
    strong_hits = {term for term in STRONG_DISASTER_TERMS if term in combined or term in scan_text}
    if maintenance_hits and not strong_hits:
        return False

    if combined & RELEVANCE_TERMS:
        return True
    return any(term.replace("_", " ") in scan_text for term in RELEVANCE_TERMS)


def merge_error(existing: str | None, new_message: str | None):
    if not new_message:
        return existing
    if not existing:
        return new_message
    if new_message in existing:
        return existing
    return f"{existing}; {new_message}"


def preprocess_record(
    raw_id: str,
    item: dict,
    record_type: str,
    fallback_text: str | None = None,
    parent_post_id: str | None = None,
    parent_context_text: str | None = None,
    translator=None,
):
    result = {
        "raw_id": str(raw_id or ""),
        "raw_text": None,
        "clean_text": None,
        "translated_text": None,
        "vader_text": None,
        "translation_status": "skipped",
        "tokens": [],
        "negation_handled_tokens": [],
        "lemmatized_tokens": [],
        "bigrams": [],
        "final_tokens": [],
        "is_emotion_only": False,
        "is_relevant": True,
        "parent_post_id": parent_post_id,
        "preprocessing_stage": "tokenized",
        "preprocessing_status": "processed",
        "error_message": None,
        "stats": {
            "translated": 0,
            "translation_failed": 0,
            "negation_handled": 0,
            "lemmatized": 0,
            "bigrams_detected": 0,
            "emotion_only_flagged": 0,
            "irrelevant_flagged": 0,
            "errors": 0,
        },
    }

    try:
        raw_text, _field = extract_raw_text(item or {}, fallback_text=fallback_text)
        if not raw_text:
            result["preprocessing_status"] = "skipped"
            result["error_message"] = "No usable text field found in source record."
            return result

        cleaned = clean_text(raw_text)
        tokens = tokenize_text(cleaned)
        result["raw_text"] = raw_text
        result["clean_text"] = cleaned
        result["tokens"] = tokens
        if not cleaned:
            is_emotion_only = record_type == "comment" and is_emotion_only_text(raw_text, cleaned, tokens)
            result["translated_text"] = ""
            result["is_emotion_only"] = is_emotion_only
            result["is_relevant"] = bool(parent_post_id) if is_emotion_only else False
            result["preprocessing_stage"] = "finalized" if is_emotion_only else "tokenized"
            result["preprocessing_status"] = "processed" if is_emotion_only else "skipped"
            result["error_message"] = "Text became empty after preprocessing."
            if is_emotion_only:
                result["stats"]["emotion_only_flagged"] = 1
                if not result["is_relevant"]:
                    result["stats"]["irrelevant_flagged"] = 1
            return result

        # Normalize informal/slang tokens for the ML path only.
        # raw_text and clean_text are stored unchanged above and are never modified.
        normalized_tokens = normalize_informal_tokens(tokens)
        normalized_clean = " ".join(normalized_tokens)
        vader_clean = clean_text_for_vader(raw_text)

        # Single translation gate: detect once, translate once on vader_clean so that
        # sentiment cues (caps, emojis, !?.) survive for VADER. ML derives its input
        # by running clean_text() on the same translated string.
        if should_translate(normalized_clean, normalized_tokens):
            try:
                translated_vader, translation_status, translation_error = translate_text(
                    vader_clean, translator=translator
                )
                result["vader_text"] = translated_vader or vader_clean
                result["translation_status"] = translation_status
                if translation_status == "translated":
                    result["stats"]["translated"] = 1
                if translation_error:
                    result["error_message"] = merge_error(result["error_message"], translation_error)
                result["translated_text"] = clean_text(translated_vader) if translated_vader else normalized_clean
            except Exception as exc:
                result["translated_text"] = normalized_clean
                result["vader_text"] = vader_clean
                result["translation_status"] = "error"
                result["error_message"] = merge_error(result["error_message"], f"Translation failed: {exc}")
                result["stats"]["translation_failed"] = 1
        else:
            result["translated_text"] = normalized_clean
            result["vader_text"] = vader_clean

        # Never store empty string — None is the sentinel for "not yet computed".
        if not result.get("vader_text"):
            result["vader_text"] = None

        translation_tokens = tokenize_preserving_apostrophes(result["translated_text"] or cleaned)
        negation_tokens, negation_changed = apply_negation_handling(translation_tokens)
        lemmatized_tokens, lemmatized_changed = lemmatize_tokens(negation_tokens)
        bigrams = detect_bigrams(lemmatized_tokens)
        location_terms = build_location_terms(item or {})
        final_tokens = remove_stop_words(lemmatized_tokens, location_terms=location_terms)
        for bigram in bigrams:
            if bigram not in final_tokens:
                final_tokens.append(bigram)

        is_emotion_only = record_type == "comment" and is_emotion_only_text(raw_text, cleaned, tokens)
        relevant = is_relevant_text(final_tokens, bigrams, cleaned, parent_context_text=parent_context_text)
        if is_emotion_only and parent_post_id:
            relevant = True

        result["negation_handled_tokens"] = negation_tokens
        result["lemmatized_tokens"] = lemmatized_tokens
        result["bigrams"] = bigrams
        result["final_tokens"] = final_tokens
        result["is_emotion_only"] = is_emotion_only
        result["is_relevant"] = relevant
        result["preprocessing_stage"] = "finalized"

        if negation_changed:
            result["stats"]["negation_handled"] = 1
        if lemmatized_changed:
            result["stats"]["lemmatized"] = 1
        if bigrams:
            result["stats"]["bigrams_detected"] = len(bigrams)
        if is_emotion_only:
            result["stats"]["emotion_only_flagged"] = 1
        if not relevant:
            result["stats"]["irrelevant_flagged"] = 1
        return result
    except Exception as exc:
        result["preprocessing_status"] = "error"
        result["preprocessing_stage"] = "error"
        result["error_message"] = merge_error(result["error_message"], str(exc))
        result["stats"]["errors"] = 1
        return result


def save_preprocessed_text(
    item: dict,
    raw_id: str,
    record_type: str,
    fallback_text: str | None = None,
    parent_post_id: str | None = None,
    parent_context_text: str | None = None,
    translator=None,
):
    # Skip re-processing posts that are already fully preprocessed in the DB.
    # Translation quota is only spent on genuinely new posts; re-importing the
    # same dataset a second time will not trigger any Google Translate calls.
    _existing = PreprocessedText.query.filter_by(
        record_type=record_type, raw_id=str(raw_id or "")
    ).first()
    if _existing and _existing.preprocessing_status == "processed":
        return _existing, {
            "preprocessing_status": "processed",
            "is_relevant": _existing.is_relevant,
            "stats": {
                "translated": 0, "translation_failed": 0, "negation_handled": 0,
                "lemmatized": 0, "bigrams_detected": 0, "emotion_only_flagged": 0,
                "irrelevant_flagged": 0, "errors": 0,
            },
        }

    processed = preprocess_record(
        raw_id=raw_id,
        item=item,
        record_type=record_type,
        fallback_text=fallback_text,
        parent_post_id=parent_post_id,
        parent_context_text=parent_context_text,
        translator=translator,
    )
    row = PreprocessedText.query.filter_by(record_type=record_type, raw_id=processed["raw_id"]).first()
    if not row:
        row = PreprocessedText(record_type=record_type, raw_id=processed["raw_id"])

    row.raw_text = processed["raw_text"]
    row.clean_text = processed["clean_text"]
    row.translated_text = processed["translated_text"]
    row.vader_text = processed["vader_text"]
    row.translation_status = processed["translation_status"]
    row.set_tokens(processed["tokens"])
    row.set_negation_handled_tokens(processed["negation_handled_tokens"])
    row.set_lemmatized_tokens(processed["lemmatized_tokens"])
    row.set_bigrams(processed["bigrams"])
    row.set_final_tokens(processed["final_tokens"])
    row.is_emotion_only = processed["is_emotion_only"]
    row.is_relevant = processed["is_relevant"]
    row.parent_post_id = processed["parent_post_id"]
    row.preprocessing_stage = processed["preprocessing_stage"]
    row.preprocessing_status = processed["preprocessing_status"]
    row.error_message = processed["error_message"]
    return row, processed
