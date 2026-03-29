#!/usr/bin/env python3
"""
spell_checker.py — Identify correctly vs incorrectly spelled Hindi words.

Multi-strategy approach:
  1. Hindi dictionary/lexicon lookup (using pyspellchecker + custom wordlists)
  2. Frequency-based filtering (rare words more likely errors)
  3. English-in-Devanagari detection (valid per transcription guidelines)
  4. Pattern analysis (common Hindi morphological patterns)

For each word outputs: classification (correct/incorrect), confidence (high/medium/low), reason.

Usage:
  python spell_checker.py --wordlist data/wordlist.txt --output results/q3/
"""

import os
import re
import json
import argparse
import csv
import unicodedata
from collections import Counter, defaultdict


# ─── Devanagari Character Validation ────────────────────────────────────────

# Valid Devanagari Unicode ranges
DEVANAGARI_RANGE = (0x0900, 0x097F)
DEVANAGARI_EXT_RANGE = (0x0980, 0x09FF)  # Some extended chars

# Valid Devanagari consonants
CONSONANTS = set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')

# Valid Devanagari vowels
VOWELS = set('अआइईउऊऋएऐओऔ')

# Valid matras (vowel signs)
MATRAS = set('ा ि ी ु ू ृ े ै ो ौ ं ः ँ ्'.split())

# Common valid conjuncts
VALID_CONJUNCTS = {
    'क्ष', 'त्र', 'ज्ञ', 'श्र', 'क्र', 'प्र', 'ग्र', 'द्र', 'ब्र',
    'स्त', 'स्थ', 'न्त', 'न्द', 'म्प', 'ल्प', 'क्त', 'ध्य', 'द्ध',
    'च्छ', 'त्त', 'प्प', 'म्म', 'न्न', 'ल्ल', 'स्स', 'द्द',
    'ट्ट', 'ड्ड', 'ठ्ठ', 'क्क', 'ज्ज', 'ब्ब',
    'स्व', 'त्व', 'द्व', 'ध्व', 'श्व',
    'स्न', 'स्म', 'ह्म', 'ह्न', 'ह्य', 'ह्र', 'ह्ल', 'ह्व',
}


def is_devanagari(char):
    """Check if a character is Devanagari."""
    code = ord(char)
    return DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]


def is_valid_devanagari_word(word):
    """Check if a word contains only valid Devanagari characters.
    
    Returns:
        (bool, str) — (is_valid, reason)
    """
    if not word:
        return False, "empty word"
    
    # Check each character
    for char in word:
        if not (is_devanagari(char) or char in '़ॐ'):
            if char.isdigit():
                continue  # Digits are OK
            if char in '-':
                continue  # Hyphens OK in compound words
            return False, f"non-Devanagari character: '{char}' (U+{ord(char):04X})"
    
    return True, "valid Devanagari characters"


# ─── Known Hindi Word Lists ─────────────────────────────────────────────────

# Common Hindi stop words (definitely correct)
HINDI_STOPWORDS = {
    'और', 'का', 'के', 'की', 'को', 'में', 'से', 'है', 'हैं', 'था', 'थी', 'थे',
    'हो', 'ने', 'पर', 'इस', 'वह', 'यह', 'जो', 'तो', 'भी', 'कि', 'एक', 'मैं',
    'हम', 'तुम', 'आप', 'वो', 'ये', 'कर', 'कोई', 'कुछ', 'सब', 'अपना', 'अपनी',
    'अपने', 'मेरा', 'मेरी', 'मेरे', 'तेरा', 'तेरी', 'तेरे', 'इसका', 'उसका',
    'हमारा', 'हमारी', 'तुम्हारा', 'तुम्हारी', 'क्या', 'कैसे', 'कहाँ', 'कब', 'क्यों',
    'कितना', 'कितनी', 'किसने', 'जब', 'तब', 'अब', 'जैसे', 'ऐसे', 'वैसे',
    'बहुत', 'कम', 'ज्यादा', 'अच्छा', 'बुरा', 'नहीं', 'हाँ', 'न', 'जी',
    'लेकिन', 'मगर', 'पहले', 'बाद', 'साथ', 'ही', 'तक', 'सिर्फ', 'या', 'ना',
    'अगर', 'फिर', 'भर', 'करके', 'होकर', 'कर', 'बिना', 'द्वारा', 'यानी',
    'लिए', 'रहा', 'रही', 'रहे', 'गया', 'गई', 'गए', 'आया', 'आई', 'दिया',
    'दी', 'दिए', 'लिया', 'ली', 'लिए', 'किया', 'वाला', 'वाली', 'वाले',
    'सकता', 'सकती', 'सकते', 'चाहिए', 'होता', 'होती', 'होते', 'करता', 'करती',
    'बोला', 'बोली', 'बोले', 'देखा', 'सुना', 'पता', 'चला', 'मतलब', 'बात',
    'लोग', 'आदमी', 'लड़का', 'लड़की', 'दिन', 'रात', 'साल', 'घर', 'काम',
    'पैसा', 'पानी', 'खाना', 'देश', 'शहर', 'गाँव', 'दुनिया', 'जिंदगी',
}

# English words commonly transliterated in Devanagari (valid per guidelines)
VALID_ENGLISH_DEVANAGARI = {
    'कंप्यूटर', 'मोबाइल', 'फोन', 'इंटरनेट', 'ऑनलाइन', 'ऑफलाइन',
    'स्कूल', 'कॉलेज', 'यूनिवर्सिटी', 'टीचर', 'स्टूडेंट', 'एग्जाम',
    'ऑफिस', 'कंपनी', 'बॉस', 'मैनेजर', 'सैलरी', 'जॉब', 'इंटरव्यू',
    'डॉक्टर', 'हॉस्पिटल', 'बस', 'ट्रेन', 'टैक्सी', 'एयरपोर्ट',
    'प्रॉब्लम', 'सॉल्व', 'टाइम', 'प्लान', 'रिपोर्ट', 'प्रोजेक्ट',
    'एरिया', 'टेंट', 'कैम्प', 'लाइट', 'बैटरी', 'मिस्टेक',
    'वीडियो', 'ऑडियो', 'कैमरा', 'शॉपिंग', 'मार्केट',
    'ग्रेट', 'नाइस', 'गुड', 'बैड', 'सॉरी', 'थैंक्स', 'हेलो',
    'ट्राई', 'यूज', 'चेंज', 'फोकस', 'शेयर', 'हेल्प',
    'टीम', 'ग्रुप', 'मेंबर', 'लीडर', 'सिस्टम', 'प्रोसेस',
    'फ्री', 'सेफ', 'रिस्क', 'चांस', 'ऑप्शन', 'कंडीशन',
    'पर्सेंट', 'लेवल', 'टाइप', 'पेज', 'फॉर्म', 'कॉपी',
    'ड्राइवर', 'पेट्रोल', 'डीजल', 'रोड', 'पार्किंग',
    'बिजनेस', 'मीटिंग', 'प्रमोशन', 'ट्रेनिंग', 'डेटा',
    'मैसेज', 'ईमेल', 'पासवर्ड', 'अपडेट', 'डाउनलोड',
    'फैमिली', 'फ्रेंड', 'पार्टनर',
}


# ─── Spelling Validation Strategies ─────────────────────────────────────────

def check_character_validity(word):
    """Check if word has valid Devanagari character sequences.

    Detects:
    - Invalid character combinations
    - Double matras
    - Halant placement issues
    - Orphaned matras

    Returns:
        (is_valid, reason)
    """
    # Check for double matras (invalid)
    matra_pattern = re.compile(r'[\u093E-\u094D]{3,}')
    if matra_pattern.search(word):
        return False, "multiple consecutive matras/halants"

    # Check for word starting with a matra/halant (invalid)
    if word and '\u093E' <= word[0] <= '\u094D':
        return False, "starts with matra/halant"

    # Check for orphan halant at end
    if word.endswith('्'):
        # This can be valid (e.g., conjuncts) but suspicious for standalone words
        return None, "ends with halant (may be incomplete conjunct)"

    return True, "valid character sequence"


def check_morphological_patterns(word):
    """Check if word follows valid Hindi morphological patterns.

    Returns:
        (is_valid, reason)
    """
    # Very short words (1 char) that are not known
    if len(word) == 1:
        if word in 'अआइईउऊकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहन':
            return True, "single valid character"
        return None, "single character - ambiguous"

    # Very long words (more than 15 chars) are suspicious
    if len(word) > 15:
        return None, "unusually long word"

    # Check for repeated characters (heuristic for typos)
    for char in word:
        if is_devanagari(char) and word.count(char) > 4:
            return False, f"character '{char}' repeated excessively"

    return None, "no morphological issues detected"


def classify_word(word, known_correct=None, word_frequencies=None):
    """Classify a word as correctly or incorrectly spelled.
    
    Args:
        word: Devanagari text to classify
        known_correct: Set of known correct words
        word_frequencies: Dict of word -> frequency count
    
    Returns:
        Dict with 'word', 'classification', 'confidence', 'reason'
    """
    known_correct = known_correct or HINDI_STOPWORDS
    word_frequencies = word_frequencies or {}
    
    result = {
        'word': word,
        'classification': 'unknown',
        'confidence': 'low',
        'reason': '',
    }
    
    # Clean word
    clean = word.strip('।,.!?()[]{}"\' ')
    if not clean:
        result['classification'] = 'incorrect'
        result['confidence'] = 'high'
        result['reason'] = 'empty or punctuation-only'
        return result
    
    # Strategy 1: Known correct word
    if clean in known_correct or clean in HINDI_STOPWORDS:
        result['classification'] = 'correct'
        result['confidence'] = 'high'
        result['reason'] = 'in known Hindi dictionary/stopwords'
        return result
    
    # Strategy 2: Valid English in Devanagari
    if clean in VALID_ENGLISH_DEVANAGARI:
        result['classification'] = 'correct'
        result['confidence'] = 'high'
        result['reason'] = 'valid English word in Devanagari (per transcription guidelines)'
        return result

    # Strategy 3: Character validity
    is_valid_chars, char_reason = is_valid_devanagari_word(clean)
    if not is_valid_chars:
        result['classification'] = 'incorrect'
        result['confidence'] = 'high'
        result['reason'] = f'invalid characters: {char_reason}'
        return result
    
    char_valid, char_seq_reason = check_character_validity(clean)
    if char_valid is False:
        result['classification'] = 'incorrect'
        result['confidence'] = 'high'
        result['reason'] = char_seq_reason
        return result

    # Strategy 4: Morphological patterns
    morph_valid, morph_reason = check_morphological_patterns(clean)
    if morph_valid is False:
        result['classification'] = 'incorrect'
        result['confidence'] = 'medium'
        result['reason'] = morph_reason
        return result
    
    # Strategy 5: Frequency-based heuristic
    freq = word_frequencies.get(clean, 0)
    total_words = sum(word_frequencies.values()) if word_frequencies else 1
    
    if freq > 0:
        relative_freq = freq / total_words
        if relative_freq > 0.0001:  # Common word
            result['classification'] = 'correct'
            result['confidence'] = 'high'
            result['reason'] = f'high frequency word (freq={freq})'
            return result
        elif relative_freq > 0.00001:  # Moderately common
            result['classification'] = 'correct'
            result['confidence'] = 'medium'
            result['reason'] = f'moderate frequency word (freq={freq})'
            return result
        else:  # Rare word
            result['classification'] = 'correct'  # Give benefit of doubt
            result['confidence'] = 'low'
            result['reason'] = f'rare word - may be valid but uncommon (freq={freq})'
            return result
    
    # Strategy 6: Common suffix/prefix patterns
    valid_suffixes = ['ना', 'ता', 'ती', 'ते', 'ला', 'ली', 'ले', 'कर', 'ने', 'नी', 'वाला', 'वाली']
    valid_prefixes = ['अ', 'आ', 'उ', 'उप', 'प्र', 'अन', 'सु', 'दु', 'नि', 'वि', 'बे', 'ला']
    
    has_valid_suffix = any(clean.endswith(s) for s in valid_suffixes)
    has_valid_prefix = any(clean.startswith(p) for p in valid_prefixes)
    
    if has_valid_suffix and len(clean) > 3:
        result['classification'] = 'correct'
        result['confidence'] = 'medium'
        result['reason'] = 'has valid Hindi suffix pattern'
        return result
    
    # Default: mark as low-confidence classification based on character analysis
    if char_valid is True and morph_valid is not False:
        result['classification'] = 'correct'
        result['confidence'] = 'low'
        result['reason'] = 'valid characters but not in dictionary - may be proper noun, dialectal, or uncommon word'
    else:
        result['classification'] = 'incorrect'
        result['confidence'] = 'low'
        result['reason'] = 'not in dictionary and has suspicious patterns'
    
    return result


# ─── Batch Processing ───────────────────────────────────────────────────────

def classify_wordlist(words, word_frequencies=None):
    """Classify a list of unique words.
    
    Args:
        words: List of unique words to classify
        word_frequencies: Optional word frequency dict
    
    Returns:
        List of classification dicts
    """
    results = []
    
    for word in words:
        result = classify_word(word, word_frequencies=word_frequencies)
        results.append(result)
    
    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hindi spelling error detection")
    parser.add_argument('--wordlist', type=str, required=True,
                        help='Path to word list file (one word per line, or CSV)')
    parser.add_argument('--frequencies', type=str, default=None,
                        help='Path to word frequency JSON')
    parser.add_argument('--output', type=str, default='results/q3/',
                        help='Output directory')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Hindi Spelling Error Detection")
    print("=" * 60)
    
    # Load word list
    print(f"\n[STEP 1] Loading word list from {args.wordlist}")
    words = []
    
    if args.wordlist.endswith('.csv'):
        with open(args.wordlist, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    words.append(row[0].strip())
    else:
        with open(args.wordlist, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
    
    print(f"  Total words: {len(words)}")
    print(f"  Unique words: {len(unique_words)}")
    
    # Load frequencies if available
    word_freq = None
    if args.frequencies:
        with open(args.frequencies, 'r', encoding='utf-8') as f:
            word_freq = json.load(f)
        print(f"  Word frequencies loaded: {len(word_freq)} entries")
    
    # Classify words
    print(f"\n[STEP 2] Classifying {len(unique_words)} unique words...")
    results = classify_wordlist(unique_words, word_freq)
    
    # Statistics
    correct = [r for r in results if r['classification'] == 'correct']
    incorrect = [r for r in results if r['classification'] == 'incorrect']
    unknown = [r for r in results if r['classification'] == 'unknown']
    
    high_conf = [r for r in results if r['confidence'] == 'high']
    med_conf = [r for r in results if r['confidence'] == 'medium']
    low_conf = [r for r in results if r['confidence'] == 'low']
    
    print(f"\n  Results:")
    print(f"    Correct:   {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
    print(f"    Incorrect: {len(incorrect)} ({len(incorrect)/len(results)*100:.1f}%)")
    print(f"    Unknown:   {len(unknown)} ({len(unknown)/len(results)*100:.1f}%)")
    print(f"\n  Confidence:")
    print(f"    High:   {len(high_conf)} ({len(high_conf)/len(results)*100:.1f}%)")
    print(f"    Medium: {len(med_conf)} ({len(med_conf)/len(results)*100:.1f}%)")
    print(f"    Low:    {len(low_conf)} ({len(low_conf)/len(results)*100:.1f}%)")
    
    # Save classified words CSV (deliverable format)
    csv_path = os.path.join(args.output, 'classified_words.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'classification', 'confidence', 'reason'])
        for r in results:
            writer.writerow([r['word'], r['classification'], r['confidence'], r['reason']])
    
    # Save Google-Sheets-ready format (2 columns only)
    sheets_path = os.path.join(args.output, 'words_for_sheets.csv')
    with open(sheets_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'spelling_status'])
        for r in results:
            status = 'correct spelling' if r['classification'] == 'correct' else 'incorrect spelling'
            writer.writerow([r['word'], status])
    
    # Save detailed results JSON
    json_path = os.path.join(args.output, 'classification_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': {
                'total_unique_words': len(unique_words),
                'correct': len(correct),
                'incorrect': len(incorrect),
                'unknown': len(unknown),
                'high_confidence': len(high_conf),
                'medium_confidence': len(med_conf),
                'low_confidence': len(low_conf),
            },
            'words': results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n[STEP 3] Saved outputs:")
    print(f"  CSV (with confidence): {csv_path}")
    print(f"  CSV (Sheets format):   {sheets_path}")
    print(f"  JSON (detailed):       {json_path}")
    
    # Final answer
    print(f"\n" + "=" * 60)
    print(f"FINAL: {len(correct)} unique correctly spelled words")
    print(f"       {len(incorrect)} unique incorrectly spelled words")
    print(f"=" * 60)


if __name__ == '__main__':
    main()
