"""
Multilingual translation support for the VeriTrust AI fact-checking engine.

Detects non-English text across 100+ languages including Marathi, Hindi, Spanish,
French, German, Arabic, Bengali, etc., and translates it to English before claim
extraction, retrieval, and NLI stance detection.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Human-readable names for common languages
LANGUAGE_NAMES: Dict[str, str] = {
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "th": "Thai",
    "fa": "Persian",
    "en": "English",
}

# Distinctive markers for Marathi (written in Devanagari script)
MARATHI_KEYWORDS = {
    "ळ", "आहे", "आहेत", "नाही", "नाहीत", "झाला", "झाली", "झाले",
    "केला", "केली", "केले", "म्हणाले", "यांनी", "त्यांनी", "त्यांच्या",
    "च्या", "मध्ये", "शेतकरी", "शेतात", "माणूस", "आणि", "असला", "असून",
    "कसा", "कशी", "कसे", "नुकताच", "कळवले", "सांगितले", "पोहोचले",
    "होता", "होती", "होते", "करावे", "जावे", "येथे", "करणार", "मिळाली",
    "याबाबत", "असल्याचे", "झालेला", "केलेला", "गेल्या", "वर्ष", "दिवस"
}

# Distinctive markers for Hindi (written in Devanagari script)
HINDI_KEYWORDS = {
    "है", "हैं", "नहीं", "किया", "कहा", "था", "थी", "थे",
    "में", "और", "हुआ", "हुई", "हुए", "ने", "को", "से", "पर", "का", "की", "के",
    "गया", "गई", "गए", "बारे", "लेकिन", "मगर", "क्योंकि", "होगा", "होगी"
}

# Regex to detect non-Latin scripts
_NON_LATIN_RE = re.compile(
    r"[\u0900-\u097F"   # Devanagari (Hindi, Marathi)
    r"\u0980-\u09FF"   # Bengali
    r"\u0A00-\u0A7F"   # Gurmukhi / Punjabi
    r"\u0A80-\u0AFF"   # Gujarati
    r"\u0B80-\u0BFF"   # Tamil
    r"\u0C00-\u0C7F"   # Telugu
    r"\u0C80-\u0CFF"   # Kannada
    r"\u0D00-\u0D7F"   # Malayalam
    r"\u0600-\u06FF"   # Arabic / Urdu / Persian
    r"\u0400-\u04FF"   # Cyrillic (Russian, etc.)
    r"\u4E00-\u9FFF"   # CJK unified ideographs (Chinese, Japanese)
    r"\u3040-\u30FF"   # Hiragana & Katakana
    r"\uAC00-\uD7AF]"  # Hangul (Korean)
)

_ACCENTED_LATIN_RE = re.compile(r"[áéíóúñ¿¡àèìòùâêîôûäëïöüßç]", re.IGNORECASE)


def detect_source_language(text: str, hint: str = "") -> str:
    """
    Accurately detect the source language.
    Specifically disambiguates Devanagari script between Marathi (mr) and Hindi (hi).
    """
    if hint:
        h = hint.lower().split("-")[0].split("_")[0].strip()
        if h in ("mr", "marathi"):
            return "mr"
        if h in ("hi", "hindi"):
            return "hi"
        if len(h) == 2 and h in LANGUAGE_NAMES:
            return h

    # Non-Devanagari unique scripts
    if re.search(r"[\u0980-\u09FF]", text): return "bn"   # Bengali
    if re.search(r"[\u0A80-\u0AFF]", text): return "gu"   # Gujarati
    if re.search(r"[\u0A00-\u0A7F]", text): return "pa"   # Punjabi
    if re.search(r"[\u0B80-\u0BFF]", text): return "ta"   # Tamil
    if re.search(r"[\u0C00-\u0C7F]", text): return "te"   # Telugu
    if re.search(r"[\u0C80-\u0CFF]", text): return "kn"   # Kannada
    if re.search(r"[\u0D00-\u0D7F]", text): return "ml"   # Malayalam
    if re.search(r"[\u0600-\u06FF]", text): return "ur"   # Urdu / Arabic
    if re.search(r"[\u0400-\u04FF]", text): return "ru"   # Russian
    if re.search(r"[\u4E00-\u9FFF]", text): return "zh"   # Chinese
    if re.search(r"[\u3040-\u30FF]", text): return "ja"   # Japanese
    if re.search(r"[\uAC00-\uD7AF]", text): return "ko"   # Korean

    # Devanagari script: Disambiguate Marathi vs Hindi
    if re.search(r"[\u0900-\u097F]", text):
        if "ळ" in text:
            return "mr"
        words = set(re.findall(r"[\u0900-\u097F]+", text))
        mr_matches = len(words & MARATHI_KEYWORDS)
        hi_matches = len(words & HINDI_KEYWORDS)
        if mr_matches > 0 and mr_matches >= hi_matches:
            return "mr"
        if hi_matches > mr_matches:
            return "hi"
        return "mr" if mr_matches > 0 else "hi"

    return "auto"


def is_likely_non_english(text: str) -> bool:
    """Quick check to see if text is noticeably non-English."""
    if _NON_LATIN_RE.search(text):
        return True
    if len(_ACCENTED_LATIN_RE.findall(text)) >= 2:
        return True
    return False


def get_language_name(code: str) -> str:
    """Return the human-readable language name from an ISO code."""
    if not code:
        return "Unknown"
    norm = code.lower().split("-")[0].split("_")[0].strip()
    return LANGUAGE_NAMES.get(norm, code.upper())


def _translate_with_google(
    text: str, source_lang: str = "auto", target_lang: str = "en"
) -> tuple[Optional[str], Optional[str]]:
    """
    Translates text to target_lang using Google Translate's public endpoint (client=gtx) via POST.
    Handles long inputs by chunking appropriately.
    Returns (translated_text, detected_source_lang) or (None, None) on failure.
    """
    if not text.strip():
        return "", source_lang

    sl = source_lang if source_lang and source_lang != "auto" else "auto"

    # Split text into chunks of <= 2500 chars if large
    chunks: List[str] = []
    if len(text) <= 2500:
        chunks = [text]
    else:
        paragraphs = text.split("\n")
        curr: List[str] = []
        curr_len = 0
        for p in paragraphs:
            if curr_len + len(p) + 1 > 2000 and curr:
                chunks.append("\n".join(curr))
                curr = [p]
                curr_len = len(p)
            else:
                curr.append(p)
                curr_len += len(p) + 1
        if curr:
            chunks.append("\n".join(curr))

    translated_chunks: List[str] = []
    detected_lang: Optional[str] = None

    for chunk in chunks:
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue
        try:
            url = (
                f"https://translate.googleapis.com/translate_a/single?"
                f"client=gtx&sl={urllib.parse.quote(sl)}&tl={urllib.parse.quote(target_lang)}&dt=t"
            )
            data = urllib.parse.urlencode({"q": chunk}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                },
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
                if raw_json and isinstance(raw_json, list) and len(raw_json) > 0 and isinstance(raw_json[0], list):
                    chunk_trans = "".join([seg[0] for seg in raw_json[0] if seg and len(seg) > 0 and seg[0]])
                    translated_chunks.append(chunk_trans)
                    if not detected_lang and len(raw_json) > 2 and isinstance(raw_json[2], str):
                        detected_lang = raw_json[2].lower()
                else:
                    return None, None
        except Exception as exc:
            logger.warning("Google Translate API chunk error: %s", exc)
            return None, None

    return "\n".join(translated_chunks), detected_lang


def _translate_with_mymemory(
    text: str, source_lang: str = "auto", target_lang: str = "en"
) -> tuple[Optional[str], Optional[str]]:
    """
    Fallback translation using api.mymemory.translated.net.
    Returns (translated_text, detected_source_lang) or (None, None) on failure.
    """
    if source_lang != "auto" and source_lang != "en":
        langpair = f"{source_lang}|{target_lang}"
    else:
        langpair = f"autodetect|{target_lang}"

    words = text.split()
    chunks: List[str] = []
    curr: List[str] = []
    curr_len = 0
    for w in words:
        if curr_len + len(w) + 1 > 350:
            chunks.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
        else:
            curr.append(w)
            curr_len += len(w) + 1
    if curr:
        chunks.append(" ".join(curr))

    translated_chunks: List[str] = []
    final_detected_lang = source_lang if source_lang != "auto" else "en"

    try:
        for chunk in chunks:
            url = (
                "https://api.mymemory.translated.net/get?"
                + urllib.parse.urlencode({
                    "q": chunk,
                    "langpair": langpair,
                })
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "VeriTrustAI-Multilingual/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                res_data = data.get("responseData", {}) or {}
                trans = (res_data.get("translatedText") or "").strip()

                if trans == "PLEASE SELECT TWO DISTINCT LANGUAGES":
                    return text, "en"

                if trans:
                    translated_chunks.append(trans)
                else:
                    translated_chunks.append(chunk)

                if res_data.get("detectedLanguage") and final_detected_lang in ("auto", "en"):
                    final_detected_lang = res_data["detectedLanguage"]

        return " ".join(translated_chunks).strip(), final_detected_lang
    except Exception as exc:
        logger.warning("MyMemory translation failed: %s", exc)
        return None, None


def translate_to_english(
    text: str, target_lang: str = "en", source_hint: str = ""
) -> Dict[str, any]:
    """
    Detect language and translate input to English if needed.

    Returns a dict with:
      - text: English text (or original if already English)
      - original_text: the original text
      - source_language: ISO language code (e.g. 'mr', 'hi')
      - source_language_name: human-readable name (e.g. 'Marathi', 'Hindi')
      - was_translated: True if translation took place, False otherwise
    """
    raw = (text or "").strip()
    if not raw:
        return {
            "text": raw,
            "original_text": raw,
            "source_language": "en",
            "source_language_name": "English",
            "was_translated": False,
        }

    is_non_eng = is_likely_non_english(raw)
    detected_source = detect_source_language(raw, hint=source_hint)

    # If the text is already English, skip translation entirely
    if not is_non_eng and detected_source in ("auto", "en"):
        return {
            "text": raw,
            "original_text": raw,
            "source_language": "en",
            "source_language_name": "English",
            "was_translated": False,
        }

    try:
        # Try Google Translate first (robust, fast, handles full articles without 429)
        translated_text, google_detected = _translate_with_google(
            raw,
            source_lang=detected_source if detected_source != "auto" else "auto",
            target_lang=target_lang,
        )

        final_detected_lang = detected_source
        if not translated_text:
            # Fallback to MyMemory if Google Translate fails
            logger.info("Google Translate unavailable, attempting MyMemory fallback...")
            translated_text, mymem_detected = _translate_with_mymemory(
                raw,
                source_lang=detected_source if detected_source != "auto" else "auto",
                target_lang=target_lang,
            )
            if mymem_detected and final_detected_lang in ("auto", "en"):
                final_detected_lang = mymem_detected
        else:
            if google_detected and final_detected_lang in ("auto", "en"):
                final_detected_lang = google_detected

        if not translated_text:
            # Translation could not complete
            logger.warning("All translation backends failed, falling back to original text")
            return {
                "text": raw,
                "original_text": raw,
                "source_language": "unknown" if is_non_eng else "en",
                "source_language_name": "Unknown" if is_non_eng else "English",
                "was_translated": False,
            }

        translated_full = translated_text.strip()
        # If our regex specifically determined Marathi ('mr'), stick with 'mr' even if generic API said 'hi'
        if detected_source == "mr":
            final_detected_lang = "mr"
        elif final_detected_lang in ("auto", ""):
            final_detected_lang = "en"

        was_translated = (
            bool(translated_full)
            and translated_full.lower() != raw.lower()
            and final_detected_lang.lower().split("-")[0] != "en"
        )

        lang_code = final_detected_lang.lower().split("-")[0].split("_")[0]
        return {
            "text": translated_full if was_translated else raw,
            "original_text": raw,
            "source_language": lang_code,
            "source_language_name": get_language_name(lang_code),
            "was_translated": was_translated,
        }

    except Exception as exc:
        logger.warning("Translation failed (falling back to original text): %s", exc)
        return {
            "text": raw,
            "original_text": raw,
            "source_language": "unknown" if is_non_eng else "en",
            "source_language_name": "Unknown" if is_non_eng else "English",
            "was_translated": False,
            "translation_error": str(exc),
        }
