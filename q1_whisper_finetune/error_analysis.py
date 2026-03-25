#!/usr/bin/env python3
"""
error_analysis.py — Systematic error analysis of fine-tuned Whisper-small on Hindi.

Steps:
  1. Load detailed evaluation results (per-utterance WER)
  2. Systematically sample 25+ error utterances (stratified by severity)
  3. Build error taxonomy from observed patterns
  4. Provide 3-5 examples per category
  5. Propose fixes for top 3 error types

Usage:
  python error_analysis.py --results_file results/detailed_results.json --output results/
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re


# ─── Error Classification Heuristics ────────────────────────────────────────

def classify_error(reference, prediction):
    """Classify the type of error between reference and prediction.
    
    Returns a list of error categories detected.
    """
    errors = []
    ref_words = reference.split()
    pred_words = prediction.split()
    
    # 1. Code-mixing / English words in Devanagari
    english_in_devanagari = re.findall(r'[a-zA-Z]+', reference + ' ' + prediction)
    if english_in_devanagari:
        errors.append('code_mixing')
    
    # 2. Number-related errors
    hindi_numbers = ['शून्य', 'एक', 'दो', 'तीन', 'चार', 'पांच', 'छह', 'सात', 'आठ', 'नौ', 'दस',
                     'ग्यारह', 'बारह', 'तेरह', 'चौदह', 'पंद्रह', 'सोलह', 'सत्रह', 'अठारह', 'उन्नीस', 'बीस',
                     'सौ', 'हज़ार', 'लाख', 'करोड़']
    ref_has_numbers = any(w in ref_words for w in hindi_numbers) or any(c.isdigit() for c in reference)
    pred_has_numbers = any(w in pred_words for w in hindi_numbers) or any(c.isdigit() for c in prediction)
    if ref_has_numbers or pred_has_numbers:
        if ref_has_numbers != pred_has_numbers or set(ref_words) != set(pred_words):
            errors.append('number_error')
    
    # 3. Disfluency / filler words
    fillers = ['हम्म', 'उम्म', 'अह', 'ऐसे', 'मतलब', 'बोलो', 'पता', 'हां', 'हाँ']
    if any(f in prediction and f not in reference for f in fillers):
        errors.append('disfluency_insertion')
    if any(f in reference and f not in prediction for f in fillers):
        errors.append('disfluency_deletion')
    
    # 4. Homophone / phonetically similar confusion
    # Look for single-character differences in words (matra changes)
    if len(ref_words) == len(pred_words):
        for rw, pw in zip(ref_words, pred_words):
            if rw != pw and _levenshtein(rw, pw) <= 2:
                errors.append('phonetic_confusion')
                break
    
    # 5. Word repetition errors
    for i in range(len(pred_words) - 1):
        if pred_words[i] == pred_words[i + 1] and (i >= len(ref_words) - 1 or ref_words[i] != ref_words[min(i + 1, len(ref_words) - 1)]):
            errors.append('repetition_error')
            break
    
    # 6. Insertion errors (extra words in prediction)
    if len(pred_words) > len(ref_words) * 1.3:
        errors.append('insertion_error')
    
    # 7. Deletion errors (missing words in prediction)
    if len(pred_words) < len(ref_words) * 0.7:
        errors.append('deletion_error')
    
    # 8. Punctuation / formatting differences
    ref_clean = re.sub(r'[।,.!?\-]', '', reference)
    pred_clean = re.sub(r'[।,.!?\-]', '', prediction)
    if ref_clean.split() == pred_clean.split() and reference != prediction:
        errors.append('punctuation_only')
    
    # 9. Conjunct consonant errors (common in Hindi)
    conjuncts = ['क्ष', 'त्र', 'ज्ञ', 'श्र', 'क्र', 'प्र', 'ग्र', 'द्र', 'ब्र']
    for conj in conjuncts:
        if conj in reference and conj not in prediction:
            errors.append('conjunct_consonant_error')
            break
        if conj in prediction and conj not in reference:
            errors.append('conjunct_consonant_error')
            break
    
    # 10. If no specific error found, mark as general substitution
    if not errors and reference != prediction:
        errors.append('general_substitution')
    
    return errors


def _levenshtein(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


# ─── Sampling Strategy ──────────────────────────────────────────────────────

def sample_errors(per_utterance, n_samples=25):
    """Systematically sample error utterances.
    
    Strategy:
      - Sort by WER
      - Define 3 severity buckets: low (0-33%), medium (33-66%), high (66-100%)
      - Sample proportionally from each bucket + every Nth error
    
    Args:
        per_utterance: List of dicts with 'reference', 'prediction', 'wer'
        n_samples: Minimum number of samples to collect
    
    Returns:
        List of sampled error dicts with 'severity' added
    """
    # Filter to only errors (WER > 0)
    errors = [u for u in per_utterance if u['wer'] > 0]
    
    if not errors:
        print("[WARN] No errors found!")
        return []
    
    # Sort by WER
    errors.sort(key=lambda x: x['wer'])
    
    # Define severity buckets
    low = [e for e in errors if e['wer'] <= 0.33]
    medium = [e for e in errors if 0.33 < e['wer'] <= 0.66]
    high = [e for e in errors if e['wer'] > 0.66]
    
    print(f"  Error distribution: Low={len(low)}, Medium={len(medium)}, High={len(high)}")
    
    # Sample from each bucket
    samples_per_bucket = max(n_samples // 3, 3)  # At least 3 per bucket
    
    sampled = []
    
    for bucket, severity in [(low, 'low'), (medium, 'medium'), (high, 'high')]:
        if not bucket:
            continue
        # Every Nth element for systematic sampling
        step = max(1, len(bucket) // samples_per_bucket)
        for i in range(0, len(bucket), step):
            if len(sampled) < n_samples + 5:  # Get a few extra
                entry = bucket[i].copy()
                entry['severity'] = severity
                sampled.append(entry)
    
    # Ensure minimum samples by adding more if needed
    if len(sampled) < n_samples:
        remaining = [e for e in errors if e not in sampled]
        step = max(1, len(remaining) // (n_samples - len(sampled)))
        for i in range(0, len(remaining), step):
            if len(sampled) >= n_samples:
                break
            entry = remaining[i].copy()
            entry['severity'] = 'medium'  # default
            sampled.append(entry)
    
    return sampled[:max(n_samples, 25)]


# ─── Error Taxonomy ─────────────────────────────────────────────────────────

def build_taxonomy(sampled_errors):
    """Build error taxonomy from sampled errors.
    
    Returns:
        Dict mapping category -> list of example dicts
    """
    taxonomy = defaultdict(list)
    category_counts = Counter()
    
    for error in sampled_errors:
        categories = classify_error(error['reference'], error['prediction'])
        for cat in categories:
            category_counts[cat] += 1
            taxonomy[cat].append({
                'reference': error['reference'],
                'prediction': error['prediction'],
                'wer': error['wer'],
                'severity': error.get('severity', 'unknown'),
            })
    
    return taxonomy, category_counts


def propose_fixes(category_counts, taxonomy):
    """Propose specific, actionable fixes for top 3 error types.
    
    Returns:
        List of dicts with 'category', 'fix_title', 'fix_description', 'implementation'
    """
    fix_proposals = {
        'code_mixing': {
            'title': 'Hinglish-aware Text Normalization',
            'description': ('Add a pre/post-processing step that detects English words '
                          'transliterated in Devanagari and normalizes them. Build a '
                          'dictionary of common English-to-Devanagari mappings from the '
                          'training data. Use this during data augmentation to expose the '
                          'model to both forms.'),
            'implementation': 'Add Hinglish normalizer in fix_implementation.py',
        },
        'number_error': {
            'title': 'Number-aware Tokenization and Post-processing',
            'description': ('Hindi numbers have complex compound forms. Add a post-processing '
                          'step that normalizes number representations (both word-form and '
                          'digit-form). Also augment training data with both numerical '
                          'representations.'),
            'implementation': 'Integrate with q2_cleanup_pipeline/number_normalization.py',
        },
        'phonetic_confusion': {
            'title': 'Phonetic Similarity-aware Data Augmentation',
            'description': ('Characters with similar pronunciation (e.g., ब/व, श/स, ण/न) '
                          'are commonly confused. Add training examples with these pairs '
                          'or use a phonetic normalization layer.'),
            'implementation': 'Add confusable character augmentation in fix_implementation.py',
        },
        'disfluency_insertion': {
            'title': 'Disfluency-aware Training',
            'description': ('Filler words like "हम्म", "मतलब" are either incorrectly '
                          'inserted or deleted. Fine-tune with explicit disfluency '
                          'annotations. Add data augmentation that randomly inserts/removes '
                          'common fillers.'),
            'implementation': 'Modify training data preprocessing in fix_implementation.py',
        },
        'disfluency_deletion': {
            'title': 'Disfluency-aware Training',
            'description': 'Same fix as disfluency_insertion — bidirectional disfluency handling.',
            'implementation': 'Modify training data preprocessing in fix_implementation.py',
        },
        'repetition_error': {
            'title': 'Repetition Suppression Post-processing',
            'description': ('Add a post-processing step that detects and removes unintended '
                          'word repetitions in the output. Use n-gram analysis to detect '
                          'patterns that are unlikely to be intentional.'),
            'implementation': 'Add repetition filter in fix_implementation.py',
        },
        'insertion_error': {
            'title': 'Length-constrained Decoding',
            'description': ('Use length penalty in beam search to constrain output length '
                          'relative to input audio duration. Calibrate expected text length '
                          'per second of audio.'),
            'implementation': 'Modify generation config in fix_implementation.py',
        },
        'deletion_error': {
            'title': 'Improved Attention with Longer Context',
            'description': ('Missing words often result from attention drift. Use stricter '
                          'attention constraints or segment longer audio into smaller chunks.'),
            'implementation': 'Add chunked inference in fix_implementation.py',
        },
        'conjunct_consonant_error': {
            'title': 'Conjunct Consonant Data Augmentation',
            'description': ('Add targeted training examples rich in conjunct consonants. '
                          'Build a confusion matrix of common conjunct errors and use it '
                          'for augmentation.'),
            'implementation': 'Add targeted augmentation in fix_implementation.py',
        },
        'punctuation_only': {
            'title': 'Text Normalization Pre/Post-processing',
            'description': ('Standardize punctuation in both training data and model output. '
                          'Remove or normalize purna viram (।) vs period (.) differences.'),
            'implementation': 'Add normalizer in fix_implementation.py',
        },
        'general_substitution': {
            'title': 'More Training Data + Domain Adaptation',
            'description': ('General substitution errors require more diverse training data. '
                          'Consider adding data from similar domains or using self-training '
                          'with pseudo-labels.'),
            'implementation': 'Increase training data diversity',
        },
    }
    
    top_3 = category_counts.most_common(3)
    fixes = []
    for category, count in top_3:
        fix = fix_proposals.get(category, {
            'title': f'Fix for {category}',
            'description': f'Address {category} errors ({count} occurrences)',
            'implementation': 'Custom fix needed',
        })
        fixes.append({
            'category': category,
            'count': count,
            'fix_title': fix['title'],
            'fix_description': fix['description'],
            'implementation': fix['implementation'],
            'examples': taxonomy[category][:5],  # 3-5 examples
        })
    
    return fixes


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_report(sampled_errors, taxonomy, category_counts, fixes, output_dir):
    """Generate comprehensive error analysis report."""
    
    report_path = os.path.join(output_dir, "error_analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Error Analysis Report — Whisper-small Hindi Fine-tuned\n\n")
        
        # Sampling strategy
        f.write("## 1. Sampling Strategy\n\n")
        f.write("Errors were sampled using a **stratified systematic approach**:\n\n")
        f.write("1. All utterances with WER > 0 were sorted by WER value\n")
        f.write("2. Three severity buckets were defined:\n")
        f.write("   - **Low** (WER ≤ 33%): Minor errors, 1-2 word differences\n")
        f.write("   - **Medium** (33% < WER ≤ 66%): Moderate errors, significant word differences\n")
        f.write("   - **High** (WER > 66%): Severe errors, mostly incorrect output\n")
        f.write("3. Every Nth error was sampled from each bucket for representative coverage\n")
        f.write(f"4. Total sampled: **{len(sampled_errors)} utterances**\n\n")
        
        # Sampled examples
        f.write("## 2. Sampled Error Utterances\n\n")
        f.write("| # | Severity | WER | Reference (excerpt) | Prediction (excerpt) |\n")
        f.write("|---|----------|-----|---------------------|---------------------|\n")
        for i, err in enumerate(sampled_errors[:30], 1):
            ref_short = err['reference'][:50] + '...' if len(err['reference']) > 50 else err['reference']
            pred_short = err['prediction'][:50] + '...' if len(err['prediction']) > 50 else err['prediction']
            f.write(f"| {i} | {err['severity']} | {err['wer']:.2f} | {ref_short} | {pred_short} |\n")
        
        # Error taxonomy
        f.write("\n## 3. Error Taxonomy\n\n")
        f.write(f"**{len(category_counts)} error categories** identified:\n\n")
        
        for cat, count in category_counts.most_common():
            f.write(f"### {cat.replace('_', ' ').title()} ({count} occurrences)\n\n")
            examples = taxonomy[cat][:5]
            for j, ex in enumerate(examples, 1):
                f.write(f"**Example {j}:**\n")
                f.write(f"- **Reference**: {ex['reference']}\n")
                f.write(f"- **Model Output**: {ex['prediction']}\n")
                f.write(f"- **WER**: {ex['wer']:.2f}\n")
                # Reasoning
                cats = classify_error(ex['reference'], ex['prediction'])
                f.write(f"- **Reasoning**: Error classified as {', '.join(cats)}\n\n")
        
        # Proposed fixes
        f.write("## 4. Proposed Fixes (Top 3 Error Types)\n\n")
        for i, fix in enumerate(fixes, 1):
            f.write(f"### Fix {i}: {fix['fix_title']}\n\n")
            f.write(f"**Error Category**: {fix['category'].replace('_', ' ').title()} ({fix['count']} occurrences)\n\n")
            f.write(f"**Description**: {fix['fix_description']}\n\n")
            f.write(f"**Implementation**: {fix['implementation']}\n\n")
    
    print(f"[INFO] Report saved to: {report_path}")
    return report_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Error analysis of Whisper Hindi transcriptions")
    parser.add_argument('--results_file', type=str, default='results/detailed_results.json',
                        help='Path to detailed evaluation results JSON')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--n_samples', type=int, default=25,
                        help='Minimum number of error utterances to sample')
    parser.add_argument('--model_type', type=str, default='finetuned',
                        choices=['baseline', 'finetuned'],
                        help='Which model results to analyze')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Error Analysis — Whisper Hindi")
    print("=" * 60)
    
    # Load results
    with open(args.results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    per_utterance = results[args.model_type]['per_utterance']
    overall_wer = results[args.model_type]['overall_wer']
    print(f"  Overall WER: {overall_wer*100:.2f}%")
    print(f"  Total utterances: {len(per_utterance)}")
    
    # Step 1: Sample errors
    print("\n[STEP 1] Sampling errors...")
    sampled = sample_errors(per_utterance, args.n_samples)
    print(f"  Sampled {len(sampled)} error utterances")
    
    # Step 2: Build taxonomy
    print("\n[STEP 2] Building error taxonomy...")
    taxonomy, category_counts = build_taxonomy(sampled)
    print(f"  Found {len(category_counts)} error categories:")
    for cat, count in category_counts.most_common():
        print(f"    {cat}: {count}")
    
    # Step 3: Propose fixes
    print("\n[STEP 3] Proposing fixes for top 3 error types...")
    fixes = propose_fixes(category_counts, taxonomy)
    for fix in fixes:
        print(f"  [{fix['category']}] {fix['fix_title']}")
    
    # Step 4: Generate report
    print("\n[STEP 4] Generating report...")
    report_path = generate_report(sampled, taxonomy, category_counts, fixes, args.output)
    
    # Save sampled errors to JSON
    sampled_path = os.path.join(args.output, "sampled_errors.json")
    with open(sampled_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    
    # Save taxonomy to JSON
    taxonomy_path = os.path.join(args.output, "error_taxonomy.json")
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        json.dump({
            'category_counts': dict(category_counts),
            'categories': {k: v[:5] for k, v in taxonomy.items()},
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n[DONE] Error analysis complete!")
    print(f"  Report: {report_path}")
    print(f"  Sampled errors: {sampled_path}")
    print(f"  Taxonomy: {taxonomy_path}")


if __name__ == '__main__':
    main()
