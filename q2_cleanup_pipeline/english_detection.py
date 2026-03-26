#!/usr/bin/env python3
"""
english_detection.py — Detect English words in Hindi (Devanagari) transcripts.

Identifies English words that are transliterated in Devanagari script and tags them.

Methods:
  1. Dictionary lookup (common English words in Devanagari)
  2. Phonetic pattern matching (English loanword patterns)
  3. Script analysis (Roman script detection)

Output format: [EN]word[/EN] tags around detected English words.

Usage:
  python english_detection.py --input "मेरा इंटरव्यू बहुत अच्छा गया"
  python english_detection.py --file transcripts.json --output results/q2/
"""

import re
import json
import argparse
import os
from collections import defaultdict


# ─── English-to-Devanagari Dictionary ────────────────────────────────────────

# Common English words frequently used in Hindi conversation (transliterated)
ENGLISH_DEVANAGARI_DICT = {
    # Technology
    'कंप्यूटर': 'computer', 'लैपटॉप': 'laptop', 'मोबाइल': 'mobile',
    'फोन': 'phone', 'इंटरनेट': 'internet', 'वेबसाइट': 'website',
    'सॉफ्टवेयर': 'software', 'हार्डवेयर': 'hardware', 'अपडेट': 'update',
    'डाउनलोड': 'download', 'अपलोड': 'upload', 'ऑनलाइन': 'online',
    'ऑफलाइन': 'offline', 'डेटा': 'data', 'पासवर्ड': 'password',
    'ईमेल': 'email', 'मैसेज': 'message', 'ऐप': 'app',
    'गूगल': 'Google', 'यूट्यूब': 'YouTube', 'व्हाट्सएप': 'WhatsApp',
    'इंस्टाग्राम': 'Instagram', 'फेसबुक': 'Facebook',
    'वीडियो': 'video', 'ऑडियो': 'audio', 'कैमरा': 'camera',
    
    # Education
    'स्कूल': 'school', 'कॉलेज': 'college', 'यूनिवर्सिटी': 'university',
    'क्लास': 'class', 'टीचर': 'teacher', 'स्टूडेंट': 'student',
    'एग्जाम': 'exam', 'रिजल्ट': 'result', 'सर्टिफिकेट': 'certificate',
    'डिग्री': 'degree', 'कोर्स': 'course', 'सिलेबस': 'syllabus',
    'ट्यूशन': 'tuition', 'प्रोजेक्ट': 'project',
    'असाइनमेंट': 'assignment', 'प्रेजेंटेशन': 'presentation',
    
    # Work
    'ऑफिस': 'office', 'कंपनी': 'company', 'बॉस': 'boss',
    'मैनेजर': 'manager', 'सैलरी': 'salary', 'जॉब': 'job',
    'इंटरव्यू': 'interview', 'रिज्यूमे': 'resume', 'बिजनेस': 'business',
    'मीटिंग': 'meeting', 'प्रमोशन': 'promotion', 'ट्रेनिंग': 'training',
    'एक्सपीरियंस': 'experience', 'प्रोफेशनल': 'professional',
    
    # Daily Life
    'शॉपिंग': 'shopping', 'मार्केट': 'market', 'मॉल': 'mall',
    'रेस्टोरेंट': 'restaurant', 'होटल': 'hotel', 'हॉस्पिटल': 'hospital',
    'डॉक्टर': 'doctor', 'नर्स': 'nurse', 'मेडिसिन': 'medicine',
    'टैक्सी': 'taxi', 'बस': 'bus', 'ट्रेन': 'train',
    'एयरपोर्ट': 'airport', 'टिकट': 'ticket', 'पार्किंग': 'parking',
    'ड्राइवर': 'driver', 'पेट्रोल': 'petrol', 'डीजल': 'diesel',
    
    # Common words
    'प्रॉब्लम': 'problem', 'सॉल्व': 'solve', 'रिजॉल्व': 'resolve',
    'टाइम': 'time', 'स्टार्ट': 'start', 'एंड': 'end',
    'टॉपिक': 'topic', 'पॉइंट': 'point', 'लिस्ट': 'list',
    'प्लान': 'plan', 'गोल': 'goal', 'टारगेट': 'target',
    'रिपोर्ट': 'report', 'फाइनल': 'final', 'फर्स्ट': 'first',
    'लास्ट': 'last', 'नेक्स्ट': 'next', 'बेस्ट': 'best',
    'ग्रेट': 'great', 'नाइस': 'nice', 'गुड': 'good',
    'बैड': 'bad', 'हैप्पी': 'happy', 'सैड': 'sad',
    'सॉरी': 'sorry', 'थैंक्स': 'thanks', 'प्लीज': 'please',
    'हेलो': 'hello', 'बाय': 'bye', 'ओके': 'OK',
    'यस': 'yes', 'नो': 'no', 'श्योर': 'sure',
    'एक्चुअली': 'actually', 'बेसिकली': 'basically',
    'इम्पॉर्टेंट': 'important', 'इंटरेस्टिंग': 'interesting',
    'डिफरेंट': 'different', 'स्पेशल': 'special',
    'फैमिली': 'family', 'फ्रेंड': 'friend', 'पार्टनर': 'partner',
    
    # Verbs (commonly used in Hindi)
    'ट्राई': 'try', 'यूज': 'use', 'चेंज': 'change',
    'मैनेज': 'manage', 'हैंडल': 'handle', 'फोकस': 'focus',
    'अवॉइड': 'avoid', 'अचीव': 'achieve', 'डिसाइड': 'decide',
    'शेयर': 'share', 'हेल्प': 'help', 'सपोर्ट': 'support',
    
    # Miscellaneous
    'पर्सेंट': 'percent', 'लेवल': 'level', 'टाइप': 'type',
    'साइज': 'size', 'फॉर्म': 'form', 'कॉपी': 'copy',
    'पेज': 'page', 'लाइन': 'line', 'ग्रुप': 'group',
    'टीम': 'team', 'मेंबर': 'member', 'लीडर': 'leader',
    'सेंटर': 'center', 'एरिया': 'area', 'जोन': 'zone',
    'सिस्टम': 'system', 'प्रोसेस': 'process', 'मेथड': 'method',
    'फ्री': 'free', 'सेफ': 'safe', 'रिस्क': 'risk',
    'चांस': 'chance', 'ऑप्शन': 'option', 'चॉइस': 'choice',
    'कंडीशन': 'condition', 'सिचुएशन': 'situation',
    'मिस्टेक': 'mistake', 'प्रैक्टिस': 'practice',
    'टेंट': 'tent', 'कैम्प': 'camp', 'कैम्पिंग': 'camping',
    'लाइट': 'light', 'पावर': 'power', 'बैटरी': 'battery',
}

# Build reverse lookup
DEVANAGARI_TO_ENGLISH = ENGLISH_DEVANAGARI_DICT.copy()

# Also detect when English words appear in Roman script within Hindi text
ROMAN_ENGLISH_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')


# ─── Phonetic Pattern Detection ─────────────────────────────────────────────

# Suffixes common in Devanagari transliterations of English
ENGLISH_SUFFIX_PATTERNS = [
    r'\w+शन$',      # -tion (प्रेजेंटेशन, सिचुएशन)
    r'\w+मेंट$',     # -ment (असाइनमेंट, मैनेजमेंट)
    r'\w+नेस$',      # -ness (हैप्पीनेस, बिजनेस)
    r'\w+इंग$',      # -ing (ट्रेनिंग, शॉपिंग)
    r'\w+ली$',       # -ly (एक्चुअली, बेसिकली) — but also Hindi words
    r'\w+टी$',       # -ty (क्वालिटी, सेफ्टी)
    r'\w+टर$',       # -ter (कंप्यूटर, कैरेक्टर)
    r'\w+टिव$',      # -tive (पॉजिटिव, क्रिएटिव)
    r'\w+बल$',       # -ble (पॉसिबल, कम्फर्टेबल)
    r'\w+फुल$',      # -ful (ब्यूटीफुल, सक्सेसफुल)
]

# Hindi-origin words that end with these suffixes (false positives to avoid)
HINDI_SUFFIX_EXCEPTIONS = {
    'शन': ['उषन', 'भूषन'],  # Hindi names/words
    'ली': ['सहेली', 'बियली', 'दीवाली', 'गली', 'थाली', 'डाली', 'साली'],
}


def detect_english_by_suffix(word):
    """Check if a Devanagari word has English loanword suffixes.
    
    Returns:
        bool indicating likely English origin
    """
    for pattern in ENGLISH_SUFFIX_PATTERNS:
        if re.match(pattern, word):
            # Check exceptions
            for suffix, exceptions in HINDI_SUFFIX_EXCEPTIONS.items():
                if word.endswith(suffix) and word in exceptions:
                    return False
            return True
    return False


# ─── Main Detection ─────────────────────────────────────────────────────────

def detect_english_words(text, method='all'):
    """Detect English words in Hindi text.
    
    Args:
        text: Hindi text (may contain Devanagari and/or Roman script)
        method: Detection method - 'dict', 'suffix', 'roman', or 'all'
    
    Returns:
        List of dicts with 'word', 'english_equivalent', 'method', 'start', 'end'
    """
    detections = []
    words = text.split()
    
    for i, word in enumerate(words):
        clean_word = word.strip('।,."\'!?()[]{}')
        
        if not clean_word:
            continue
        
        detected = False
        english_equiv = None
        detect_method = None
        
        # Method 1: Dictionary lookup
        if method in ('dict', 'all') and clean_word in DEVANAGARI_TO_ENGLISH:
            detected = True
            english_equiv = DEVANAGARI_TO_ENGLISH[clean_word]
            detect_method = 'dictionary'
        
        # Method 2: Suffix pattern matching
        if not detected and method in ('suffix', 'all'):
            if detect_english_by_suffix(clean_word):
                detected = True
                detect_method = 'suffix_pattern'
        
        # Method 3: Roman script detection
        if not detected and method in ('roman', 'all'):
            if ROMAN_ENGLISH_PATTERN.match(clean_word):
                detected = True
                english_equiv = clean_word
                detect_method = 'roman_script'
        
        if detected:
            detections.append({
                'word': clean_word,
                'original': word,
                'english_equivalent': english_equiv,
                'method': detect_method,
                'position': i,
            })
    
    return detections


def tag_english_words(text, method='all'):
    """Tag English words in Hindi text with [EN]...[/EN] markers.
    
    Args:
        text: Input Hindi text
        method: Detection method
    
    Returns:
        Tagged text string
    """
    detections = detect_english_words(text, method)
    
    if not detections:
        return text
    
    # Mark positions to tag
    words = text.split()
    for det in detections:
        pos = det['position']
        if pos < len(words):
            original = words[pos]
            clean = det['word']
            # Preserve any surrounding punctuation
            prefix = original[:original.find(clean)] if clean in original else ''
            suffix = original[original.find(clean) + len(clean):] if clean in original else ''
            words[pos] = f"{prefix}[EN]{clean}[/EN]{suffix}"
    
    return ' '.join(words)


# ─── Batch Processing ───────────────────────────────────────────────────────

def process_transcripts(transcripts, method='all'):
    """Process a list of transcripts and tag English words.
    
    Args:
        transcripts: List of text strings or dicts with 'text' key
    
    Returns:
        List of dicts with 'original', 'tagged', 'english_words'
    """
    results = []
    all_english_words = defaultdict(int)
    
    for item in transcripts:
        text = item if isinstance(item, str) else item.get('text', '')
        
        detections = detect_english_words(text, method)
        tagged = tag_english_words(text, method)
        
        for det in detections:
            all_english_words[det['word']] += 1
        
        results.append({
            'original': text,
            'tagged': tagged,
            'english_words': [d['word'] for d in detections],
            'english_count': len(detections),
            'total_words': len(text.split()),
            'english_ratio': len(detections) / max(len(text.split()), 1),
        })
    
    return results, dict(all_english_words)


# ─── Demo ────────────────────────────────────────────────────────────────────

def demonstrate():
    """Show detection examples."""
    
    print("\n" + "=" * 70)
    print("ENGLISH WORD DETECTION — EXAMPLES")
    print("=" * 70)
    
    examples = [
        "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        "ये प्रॉब्लम सॉल्व नहीं हो रहा",
        "कंप्यूटर पर ऑनलाइन क्लास चल रही है",
        "मैंने अपना रिज्यूमे अपडेट किया और कंपनी को ईमेल किया",
        "प्रोजेक्ट भी था कि जो जन जाती पाई जाती है उधर की एरिया में",
        "लोग घूमने जाते हैं तो लाइट वगैरा लेकर जाने चाहिए हम ने मिस्टेक किए",
    ]
    
    for i, text in enumerate(examples, 1):
        tagged = tag_english_words(text)
        detections = detect_english_words(text)
        
        print(f"\n  Example {i}:")
        print(f"    Input:    {text}")
        print(f"    Output:   {tagged}")
        print(f"    Detected: {[d['word'] for d in detections]}")
        if detections:
            methods = set(d['method'] for d in detections)
            print(f"    Methods:  {', '.join(methods)}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="English word detection in Hindi transcripts")
    parser.add_argument('--input', type=str, default=None,
                        help='Single Hindi text to process')
    parser.add_argument('--file', type=str, default=None,
                        help='JSON file with transcripts')
    parser.add_argument('--output', type=str, default='results/q2/',
                        help='Output directory')
    parser.add_argument('--method', type=str, default='all',
                        choices=['dict', 'suffix', 'roman', 'all'],
                        help='Detection method')
    parser.add_argument('--demo', action='store_true',
                        help='Run demonstration')
    
    args = parser.parse_args()
    
    if args.demo or (not args.input and not args.file):
        demonstrate()
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.input:
        tagged = tag_english_words(args.input, args.method)
        print(f"Input:    {args.input}")
        print(f"Tagged:   {tagged}")
    
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        transcripts = [item.get('text', item.get('transcription', '')) for item in data]
        results, english_vocab = process_transcripts(transcripts, args.method)
        
        # Save results
        output_path = os.path.join(args.output, 'english_tagged.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save English vocabulary
        vocab_path = os.path.join(args.output, 'english_vocabulary.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(dict(sorted(english_vocab.items(), key=lambda x: -x[1])),
                     f, ensure_ascii=False, indent=2)
        
        total_english = sum(r['english_count'] for r in results)
        total_words = sum(r['total_words'] for r in results)
        print(f"Processed {len(results)} transcripts")
        print(f"Total English words found: {total_english} / {total_words} ({total_english/max(total_words,1)*100:.1f}%)")
        print(f"Unique English words: {len(english_vocab)}")
        print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
