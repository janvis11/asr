#!/usr/bin/env python3
"""
fix_implementation.py — Implement proposed fixes for top error types.

Implements:
  1. Text Normalization Post-processor (punctuation, spacing, Unicode)
  2. Repetition Suppression Filter
  3. Phonetic Confusion Resolver (common Devanagari confusables)

Shows before/after results on a targeted error subset.

Usage:
  python fix_implementation.py --sampled_errors results/sampled_errors.json \
                                --model_dir models/whisper-small-hi --output results/
"""

import os
import json
import re
import argparse
import unicodedata
from collections import defaultdict


# ─── Fix 1: Text Normalization Post-processor ────────────────────────────────

class TextNormalizer:
    """Normalize ASR output text to reduce superficial errors.
    
    Operations:
      - Unicode NFC normalization
      - Remove zero-width characters
      - Standardize punctuation (।/. normalization)
      - Normalize whitespace
      - Normalize Devanagari nukta variants
      - Normalize common transliteration variants
    """
    
    # Common Devanagari nukta character mappings
    NUKTA_MAP = {
        'क़': 'क',  # Can normalize if needed
        'ख़': 'ख',
        'ग़': 'ग',
        'ज़': 'ज',
        'ड़': 'ड',
        'ढ़': 'ढ',
        'फ़': 'फ',
    }
    
    # Common spelling variants in Hindi ASR
    VARIANT_MAP = {
        'मैने': 'मैंने',
        'हुऐ': 'हुए',
        'गयी': 'गई',
        'गया': 'गया',
        'मे': 'में',
    }
    
    def __init__(self, normalize_nukta=False, normalize_variants=True):
        self.normalize_nukta = normalize_nukta
        self.normalize_variants = normalize_variants
    
    def normalize(self, text):
        """Apply all normalizations."""
        # Unicode NFC
        text = unicodedata.normalize('NFC', text)
        
        # Remove zero-width chars
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        
        # Standardize punctuation
        text = text.replace('|', '।')
        text = re.sub(r'\.{2,}', '।', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove extra punctuation at boundaries
        text = text.strip(' ।,.')
        
        # Normalize variants if enabled
        if self.normalize_variants:
            for old, new in self.VARIANT_MAP.items():
                text = text.replace(old, new)
        
        # Normalize nukta if enabled (lossy)
        if self.normalize_nukta:
            for old, new in self.NUKTA_MAP.items():
                text = text.replace(old, new)
        
        return text.strip()


# ─── Fix 2: Repetition Suppression ──────────────────────────────────────────

class RepetitionSuppressor:
    """Detect and remove unintended word/phrase repetitions in ASR output.
    
    Strategy:
      - Detect consecutive identical words that weren't in reference patterns
      - Detect repeated n-gram patterns (2-3 word phrases repeating)
      - Be conservative: some repetitions are intentional (e.g., "बार बार")
    """
    
    # Known intentional repetitions in Hindi
    INTENTIONAL = {
        'बार बार', 'बार-बार', 'धीरे धीरे', 'धीरे-धीरे',
        'कभी कभी', 'कभी-कभी', 'अलग अलग', 'अलग-अलग',
        'जगह जगह', 'साथ साथ', 'एक एक', 'थोड़ा थोड़ा',
        'आगे आगे', 'पीछे पीछे', 'ऊपर ऊपर',
        'हां हां', 'नहीं नहीं', 'अच्छा अच्छा',
    }

    def suppress(self, text):
        """Remove unintended repetitions."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Check for intentional repetitions first
        for pattern in self.INTENTIONAL:
            if pattern in text:
                return text  # Don't modify if intentional pattern found
        
        # Remove consecutive duplicate words
        result = [words[0]]
        for i in range(1, len(words)):
            if words[i] != words[i - 1]:
                result.append(words[i])
            # Allow up to 2 consecutive same words if they look intentional
        
        # Remove repeated bigram/trigram patterns
        result = self._remove_repeated_ngrams(result, n=2)
        result = self._remove_repeated_ngrams(result, n=3)
        
        return ' '.join(result)
    
    def _remove_repeated_ngrams(self, words, n=2):
        """Remove immediately repeated n-grams."""
        if len(words) < 2 * n:
            return words
        
        result = list(words)
        i = 0
        while i < len(result) - n:
            ngram = result[i:i + n]
            next_ngram = result[i + n:i + 2 * n]
            if ngram == next_ngram:
                # Remove the duplicate
                result = result[:i + n] + result[i + 2 * n:]
            else:
                i += 1
        
        return result


# ─── Fix 3: Phonetic Confusion Resolver ─────────────────────────────────────

class PhoneticConfusionResolver:
    """Resolve common phonetic confusions in Hindi ASR output.
    
    Common confusable pairs in Devanagari:
      ब/व, श/स/ष, ण/न, ड/ड़, ट/त, भ/ब, etc.
    
    Uses context and frequency to choose the more likely form.
    """
    
    # Confusable character pairs (source -> target mapping, with context hints)
    CONFUSABLES = [
        ('व', 'ब'),   # va/ba confusion
        ('ब', 'व'),
        ('श', 'स'),   # sha/sa confusion
        ('स', 'श'),
        ('ष', 'श'),   # sha variants
        ('ण', 'न'),   # retroflex/dental nasal
        ('न', 'ण'),
        ('ड', 'ड़'),  # retroflex + nukta
        ('ट', 'त'),   # retroflex/dental stop
        ('त', 'ट'),
    ]
    
    def __init__(self, word_freq=None):
        """
        Args:
            word_freq: Dict mapping word -> frequency count. 
                       Used to prefer more common spellings.
        """
        self.word_freq = word_freq or {}
    
    def resolve(self, text):
        """Try to resolve phonetic confusions using frequency heuristic."""
        words = text.split()
        resolved = []
        
        for word in words:
            if word in self.word_freq:
                resolved.append(word)
                continue
            
            # Try all confusable substitutions
            best_word = word
            best_freq = self.word_freq.get(word, 0)
            
            for src, tgt in self.CONFUSABLES:
                if src in word:
                    candidate = word.replace(src, tgt, 1)
                    candidate_freq = self.word_freq.get(candidate, 0)
                    if candidate_freq > best_freq:
                        best_word = candidate
                        best_freq = candidate_freq
            
            resolved.append(best_word)
        
        return ' '.join(resolved)


# ─── Pipeline ───────────────────────────────────────────────────────────────

class FixPipeline:
    """Combined fix pipeline applying all fixes in sequence."""
    
    def __init__(self, word_freq=None):
        self.normalizer = TextNormalizer()
        self.repetition_suppressor = RepetitionSuppressor()
        self.phonetic_resolver = PhoneticConfusionResolver(word_freq)
    
    def apply(self, text):
        """Apply all fixes in order."""
        text = self.normalizer.normalize(text)
        text = self.repetition_suppressor.suppress(text)
        text = self.phonetic_resolver.resolve(text)
        return text


# ─── Before/After Evaluation ────────────────────────────────────────────────

def evaluate_fixes(sampled_errors, pipeline):
    """Evaluate fixes on sampled error utterances.
    
    Returns:
        List of dicts with 'reference', 'original_pred', 'fixed_pred', 
        'original_wer', 'fixed_wer'
    """
    from jiwer import wer as compute_wer
    
    results = []
    improved = 0
    unchanged = 0
    degraded = 0
    
    for error in sampled_errors:
        ref = error['reference']
        orig_pred = error['prediction']
        
        # Apply fix pipeline
        fixed_pred = pipeline.apply(orig_pred)
        
        # Also normalize reference for fair comparison
        normalizer = TextNormalizer()
        ref_norm = normalizer.normalize(ref)
        orig_norm = normalizer.normalize(orig_pred)
        
        try:
            orig_wer = compute_wer(ref_norm, orig_norm)
            fixed_wer = compute_wer(ref_norm, fixed_pred)
        except:
            orig_wer = error.get('wer', 1.0)
            fixed_wer = orig_wer
        
        result = {
            'reference': ref,
            'original_prediction': orig_pred,
            'fixed_prediction': fixed_pred,
            'original_wer': orig_wer,
            'fixed_wer': fixed_wer,
            'wer_change': fixed_wer - orig_wer,
        }
        results.append(result)
        
        if fixed_wer < orig_wer:
            improved += 1
        elif fixed_wer == orig_wer:
            unchanged += 1
        else:
            degraded += 1
    
    return results, improved, unchanged, degraded


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Implement and evaluate error fixes")
    parser.add_argument('--sampled_errors', type=str, default='results/sampled_errors.json',
                        help='Path to sampled errors JSON')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--word_freq', type=str, default=None,
                        help='Path to word frequency JSON (optional)')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Fix Implementation — Before/After Evaluation")
    print("=" * 60)
    
    # Load sampled errors
    with open(args.sampled_errors, 'r', encoding='utf-8') as f:
        sampled_errors = json.load(f)
    print(f"  Loaded {len(sampled_errors)} error samples")
    
    # Load word frequency if available
    word_freq = {}
    if args.word_freq and os.path.exists(args.word_freq):
        with open(args.word_freq, 'r', encoding='utf-8') as f:
            word_freq = json.load(f)
        print(f"  Loaded {len(word_freq)} word frequencies")
    
    # Build pipeline
    pipeline = FixPipeline(word_freq)
    
    # Evaluate
    print("\n[STEP 1] Applying fixes and evaluating...")
    results, improved, unchanged, degraded = evaluate_fixes(sampled_errors, pipeline)
    
    print(f"\n  Results:")
    print(f"    Improved:  {improved} ({improved/len(results)*100:.1f}%)")
    print(f"    Unchanged: {unchanged} ({unchanged/len(results)*100:.1f}%)")
    print(f"    Degraded:  {degraded} ({degraded/len(results)*100:.1f}%)")
    
    # Generate before/after report
    report_path = os.path.join(args.output, "fix_before_after.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Fix Implementation — Before/After Results\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Metric | Count | Percentage |\n")
        f.write(f"|--------|-------|------------|\n")
        f.write(f"| Improved | {improved} | {improved/len(results)*100:.1f}% |\n")
        f.write(f"| Unchanged | {unchanged} | {unchanged/len(results)*100:.1f}% |\n")
        f.write(f"| Degraded | {degraded} | {degraded/len(results)*100:.1f}% |\n")
        f.write(f"| **Total** | **{len(results)}** | **100%** |\n\n")
        
        f.write("## Fixes Applied\n\n")
        f.write("1. **Text Normalizer**: Unicode NFC, zero-width removal, punctuation standardization, spelling variant normalization\n")
        f.write("2. **Repetition Suppressor**: Remove unintended word/phrase repetitions\n")
        f.write("3. **Phonetic Confusion Resolver**: Resolve common Devanagari confusable characters using frequency\n\n")
        
        f.write("## Before/After Examples\n\n")
        for i, r in enumerate(results[:25], 1):
            status = "✅ Improved" if r['wer_change'] < 0 else ("⚠️ Degraded" if r['wer_change'] > 0 else "➖ Unchanged")
            f.write(f"### Example {i} — {status}\n\n")
            f.write(f"- **Reference**: {r['reference']}\n")
            f.write(f"- **Before (Original)**: {r['original_prediction']}\n")
            f.write(f"- **After (Fixed)**: {r['fixed_prediction']}\n")
            f.write(f"- **WER**: {r['original_wer']:.2f} → {r['fixed_wer']:.2f} (Δ {r['wer_change']:+.2f})\n\n")
    
    # Save detailed results
    results_path = os.path.join(args.output, "fix_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[DONE]")
    print(f"  Report: {report_path}")
    print(f"  Results: {results_path}")


if __name__ == '__main__':
    main()
