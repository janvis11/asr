#!/usr/bin/env python3
"""
number_normalization.py — Convert spoken Hindi number words into digits.

Handles:
  - Simple numbers: दो → 2, दस → 10, सौ → 100
  - Compound numbers: तीन सौ चौवन → 354, पच्चीस → 25, एक हज़ार → 1000
  - Edge cases: idiomatic usage (दो-चार बातें stays as-is)

Usage:
  python number_normalization.py --input "तीन सौ चौवन लोग आए" --output results/q2/
  python number_normalization.py --file transcripts.json --output results/q2/
"""

import re
import json
import argparse
import os


# ─── Hindi Number Word Mapping ──────────────────────────────────────────────

# Basic units (0-19)
UNITS = {
    'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4,
    'पांच': 5, 'पाँच': 5, 'छह': 6, 'छः': 6, 'सात': 7,
    'आठ': 8, 'नौ': 9, 'दस': 10,
    'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14,
    'पंद्रह': 15, 'सोलह': 16, 'सत्रह': 17, 'अठारह': 18,
    'उन्नीस': 19,
}

# Tens (20-90)
TENS = {
    'बीस': 20, 'तीस': 30, 'चालीस': 40, 'पचास': 50,
    'साठ': 60, 'सत्तर': 70, 'अस्सी': 80, 'नब्बे': 90,
}

# Special compound numbers (21-99 that have unique Hindi forms)
SPECIAL_COMPOUNDS = {
    'इक्कीस': 21, 'बाईस': 22, 'तेईस': 23, 'चौबीस': 24,
    'पच्चीस': 25, 'छब्बीस': 26, 'सत्ताईस': 27, 'अट्ठाईस': 28,
    'उनतीस': 29, 'उन्तीस': 29,
    'इकतीस': 31, 'इकत्तीस': 31, 'बत्तीस': 32, 'तैंतीस': 33,
    'चौंतीस': 34, 'पैंतीस': 35, 'छत्तीस': 36, 'सैंतीस': 37,
    'अड़तीस': 38, 'उनतालीस': 39, 'उन्तालीस': 39,
    'इकतालीस': 41, 'बयालीस': 42, 'तैंतालीस': 43, 'चवालीस': 44,
    'पैंतालीस': 45, 'छियालीस': 46, 'सैंतालीस': 47, 'अड़तालीस': 48,
    'उनचास': 49, 'उनचास': 49,
    'इक्यावन': 51, 'बावन': 52, 'तिरपन': 53, 'चौवन': 54,
    'पचपन': 55, 'छप्पन': 56, 'सत्तावन': 57, 'अट्ठावन': 58,
    'उनसठ': 59,
    'इकसठ': 61, 'बासठ': 62, 'तिरसठ': 63, 'चौंसठ': 64,
    'पैंसठ': 65, 'छियासठ': 66, 'सड़सठ': 67, 'अड़सठ': 68,
    'उनहत्तर': 69,
    'इकहत्तर': 71, 'बहत्तर': 72, 'तिहत्तर': 73, 'चौहत्तर': 74,
    'पचहत्तर': 75, 'छिहत्तर': 76, 'सतहत्तर': 77, 'अठहत्तर': 78,
    'उनासी': 79, 'उन्यासी': 79,
    'इक्यासी': 81, 'बयासी': 82, 'तिरासी': 83, 'चौरासी': 84,
    'पचासी': 85, 'छियासी': 86, 'सत्तासी': 87, 'अट्ठासी': 88,
    'नवासी': 89,
    'इक्यानबे': 91, 'बानबे': 92, 'तिरानबे': 93, 'चौरानबे': 94,
    'पंचानबे': 95, 'छियानबे': 96, 'सत्तानबे': 97, 'अट्ठानबे': 98,
    'निन्यानबे': 99,
}

# Multipliers
MULTIPLIERS = {
    'सौ': 100,
    'हज़ार': 1000, 'हजार': 1000,
    'लाख': 100000,
    'करोड़': 10000000,
    'अरब': 1000000000,
}

# Combine all number words
ALL_NUMBER_WORDS = {}
ALL_NUMBER_WORDS.update(UNITS)
ALL_NUMBER_WORDS.update(TENS)
ALL_NUMBER_WORDS.update(SPECIAL_COMPOUNDS)
ALL_NUMBER_WORDS.update(MULTIPLIERS)


# ─── Idiomatic / Edge Case Detection ────────────────────────────────────────

# Phrases where numbers should NOT be converted to digits
IDIOMATIC_PATTERNS = [
    r'दो-चार\s+\w+',           # दो-चार बातें → keep as-is
    r'दो\s*चार\s+\w+',         # दो चार बातें → keep as-is
    r'एक-दो\s+\w+',            # एक-दो दिन → keep as-is
    r'एक\s*दो\s+\w+',
    r'चार-पांच\s+\w+',
    r'दो-तीन\s+\w+',
    r'एक\s+न\s+एक',            # एक न एक दिन
    r'एक\s+से\s+बढ़कर\s+एक',   # एक से बढ़कर एक
    r'एक\s+दूसरे',             # एक दूसरे (each other)
    r'पहली\s+बार',              # पहली बार (first time - ordinal)
    r'दूसरी\s+बार',
    r'तीसरी\s+बार',
    r'एक\s+तरह',               # एक तरह से
    r'एक\s+बार',               # एक बार (once)
    r'चारों\s+तरफ',            # चारों तरफ (all around)
    r'दोनों',                   # दोनों (both)
    r'तीनों',                   # तीनों (all three)
    r'चारों',                   # चारों (all four)
]


def is_idiomatic(text, match_start, match_end):
    """Check if a number word at the given position is part of an idiom.
    
    Args:
        text: Full input text
        match_start: Start index of the number word
        match_end: End index of the number word
    
    Returns:
        True if the number is used idiomatically (should not convert)
    """
    # Check surrounding context
    context = text[max(0, match_start - 20):min(len(text), match_end + 30)]
    
    for pattern in IDIOMATIC_PATTERNS:
        if re.search(pattern, context):
            return True
    
    return False


# ─── Number Parsing ─────────────────────────────────────────────────────────

def parse_number_sequence(words):
    """Parse a sequence of Hindi number words into a single number.
    
    Examples:
        ['तीन', 'सौ', 'चौवन'] → 354
        ['एक', 'हज़ार'] → 1000
        ['पच्चीस'] → 25
        ['दो', 'लाख', 'तीन', 'हज़ार', 'पाँच', 'सौ'] → 203500
    
    Args:
        words: List of Hindi number word strings
    
    Returns:
        Integer value, or None if not parseable
    """
    if not words:
        return None
    
    total = 0
    current = 0
    
    for word in words:
        if word in UNITS or word in TENS or word in SPECIAL_COMPOUNDS:
            value = ALL_NUMBER_WORDS[word]
            current += value
        
        elif word in MULTIPLIERS:
            multiplier = MULTIPLIERS[word]
            if current == 0:
                current = 1  # "सौ" alone means 100
            current *= multiplier
            
            # For large multipliers (lakh, crore), add to total
            if multiplier >= 1000:
                total += current
                current = 0
        else:
            return None  # Unknown word
    
    total += current
    return total if total > 0 else None


def find_number_sequences(text):
    """Find sequences of Hindi number words in text.
    
    Returns:
        List of tuples (start_idx, end_idx, word_list, parsed_value)
    """
    words = text.split()
    sequences = []
    i = 0
    
    while i < len(words):
        # Clean word for lookup (remove punctuation)
        clean_word = words[i].strip('।,.')
        
        if clean_word in ALL_NUMBER_WORDS:
            # Start of a number sequence
            seq_start = i
            seq_words = [clean_word]
            
            j = i + 1
            while j < len(words):
                next_clean = words[j].strip('।,.')
                if next_clean in ALL_NUMBER_WORDS:
                    seq_words.append(next_clean)
                    j += 1
                else:
                    break
            
            parsed = parse_number_sequence(seq_words)
            if parsed is not None:
                sequences.append((seq_start, j, seq_words, parsed))
                i = j
                continue
        
        i += 1
    
    return sequences


# ─── Main Normalization Function ────────────────────────────────────────────

def normalize_numbers(text, preserve_idioms=True):
    """Convert Hindi number words to digits in text.
    
    Args:
        text: Input Hindi text
        preserve_idioms: If True, preserve idiomatic number usage
    
    Returns:
        Normalized text with numbers as digits
    """
    sequences = find_number_sequences(text)
    
    if not sequences:
        return text
    
    # Process sequences in reverse order (to preserve indices)
    words = text.split()
    
    for start_idx, end_idx, seq_words, value in reversed(sequences):
        # Build the original text span for idiom checking
        original_span = ' '.join(words[start_idx:end_idx])
        span_start = text.find(original_span)
        
        if preserve_idioms and span_start >= 0:
            if is_idiomatic(text, span_start, span_start + len(original_span)):
                continue  # Skip idiomatic usage
        
        # Replace number words with digit
        words[start_idx:end_idx] = [str(value)]
    
    return ' '.join(words)


# ─── Examples & Demonstration ───────────────────────────────────────────────

def demonstrate():
    """Show before/after examples from actual data patterns."""
    
    print("\n" + "=" * 70)
    print("NUMBER NORMALIZATION — EXAMPLES")
    print("=" * 70)
    
    # Correct conversion examples
    correct_examples = [
        ("तीन सौ चौवन लोग आए", "354 लोग आए"),
        ("पच्चीस साल पहले", "25 साल पहले"),
        ("एक हज़ार रुपये", "1000 रुपये"),
        ("दो लाख तीन हज़ार पाँच सौ", "203500"),
        ("मेरी उम्र अठारह साल है", "मेरी उम्र 18 साल है"),
    ]
    
    print("\n--- Correct Conversions ---")
    for i, (input_text, expected) in enumerate(correct_examples, 1):
        result = normalize_numbers(input_text)
        status = "✅" if result.strip() == expected.strip() or str(expected) in result else "⚠️"
        print(f"\n  Example {i}: {status}")
        print(f"    Input:    {input_text}")
        print(f"    Output:   {result}")
        print(f"    Expected: {expected}")
    
    # Edge cases
    edge_cases = [
        {
            'input': "दो-चार बातें करनी हैं",
            'expected': "दो-चार बातें करनी हैं",
            'reasoning': "Idiomatic usage — 'दो-चार' means 'a few', not literally '2-4'"
        },
        {
            'input': "एक दूसरे से बात करो",
            'expected': "एक दूसरे से बात करो",
            'reasoning': "'एक दूसरे' means 'each other', not the number 1"
        },
        {
            'input': "चारों तरफ लोग खड़े थे",
            'expected': "चारों तरफ लोग खड़े थे",
            'reasoning': "'चारों तरफ' means 'all around', used as determiner not cardinal number"
        },
    ]
    
    print("\n--- Edge Cases ---")
    for i, case in enumerate(edge_cases, 1):
        result = normalize_numbers(case['input'])
        status = "✅" if result == case['expected'] else "⚠️"
        print(f"\n  Edge Case {i}: {status}")
        print(f"    Input:     {case['input']}")
        print(f"    Output:    {result}")
        print(f"    Expected:  {case['expected']}")
        print(f"    Reasoning: {case['reasoning']}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hindi number normalization")
    parser.add_argument('--input', type=str, default=None,
                        help='Single Hindi text to normalize')
    parser.add_argument('--file', type=str, default=None,
                        help='JSON file with transcripts to normalize')
    parser.add_argument('--output', type=str, default='results/q2/',
                        help='Output directory')
    parser.add_argument('--demo', action='store_true',
                        help='Run demonstration with examples')
    
    args = parser.parse_args()
    
    if args.demo or (not args.input and not args.file):
        demonstrate()
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.input:
        result = normalize_numbers(args.input)
        print(f"Input:  {args.input}")
        print(f"Output: {result}")
    
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            text = item.get('text', item.get('transcription', ''))
            normalized = normalize_numbers(text)
            results.append({
                'original': text,
                'normalized': normalized,
                'changed': text != normalized,
            })
        
        output_path = os.path.join(args.output, 'number_normalized.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        changed = sum(1 for r in results if r['changed'])
        print(f"Processed {len(results)} transcripts, {changed} changed")
        print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
