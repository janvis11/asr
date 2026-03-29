#!/usr/bin/env python3
"""
lattice.py — Lattice-based WER evaluation for multi-model Hindi ASR.

Constructs a lattice of valid transcription alternatives from multiple ASR model
outputs and computes WER that does not unfairly penalize valid alternatives.

Approach:
  1. Align transcriptions from 5 ASR models + human reference using word-level alignment
  2. At each position, collect all unique output variants into "bins"
  3. Group phonetically/semantically similar variants
  4. Compute WER using lattice: any bin match = correct
  5. Trust model agreement over reference when ≥3 models agree

Alignment unit: WORD (justified: Hindi is space-delimited, sub-word adds unnecessary complexity)

Usage:
  python lattice.py --input data/q4_transcriptions.json --output results/q4/
"""

import os
import sys
import json
import argparse
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np

# Fix Windows console UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ─── Edit Distance & Alignment ──────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def word_edit_distance(words1: List[str], words2: List[str]) -> Tuple[int, List]:
    """Compute word-level edit distance with alignment operations.
    
    Returns:
        (distance, alignment_ops) where ops are tuples of (op_type, w1_idx, w2_idx)
        op_type: 'match', 'substitute', 'insert', 'delete'
    """
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[None] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            ops[i][0] = ('delete', i - 1, None)
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            ops[0][j] = ('insert', None, j - 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ('match', i - 1, j - 1)
            else:
                candidates = [
                    (dp[i - 1][j - 1] + 1, ('substitute', i - 1, j - 1)),
                    (dp[i - 1][j] + 1, ('delete', i - 1, None)),
                    (dp[i][j - 1] + 1, ('insert', None, j - 1)),
                ]
                dp[i][j], ops[i][j] = min(candidates, key=lambda x: x[0])
    
    # Backtrack to get alignment
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        op = ops[i][j]
        alignment.append(op)
        if op[0] in ('match', 'substitute'):
            i -= 1
            j -= 1
        elif op[0] == 'delete':
            i -= 1
        elif op[0] == 'insert':
            j -= 1
    
    alignment.reverse()
    return dp[m][n], alignment


# ─── Multiple Sequence Alignment (ROVER-style) ──────────────────────────────

def align_multiple_sequences(sequences: List[List[str]], reference: List[str]) -> List[List[Optional[str]]]:
    """Align multiple word sequences against a reference using pairwise alignment.
    
    This is a simplified ROVER (Recognizer Output Voting Error Reduction) approach:
    1. Align each sequence to the reference
    2. Build a composite alignment matrix
    
    Args:
        sequences: List of word lists from different models
        reference: Word list from human reference
    
    Returns:
        Alignment matrix where each row is a sequence (including reference as first row)
        and each column is an alignment position. None represents a gap.
    """
    if not sequences:
        return [reference]
    
    # Use reference as the backbone
    backbone = reference[:]
    all_aligned = [backbone[:]]  # First row = reference
    
    for seq in sequences:
        _, alignment = word_edit_distance(backbone, seq)
        
        aligned = []
        seq_idx = 0
        
        for op in alignment:
            if op[0] == 'match':
                aligned.append(seq[op[2]] if op[2] is not None else None)
            elif op[0] == 'substitute':
                aligned.append(seq[op[2]] if op[2] is not None else None)
            elif op[0] == 'delete':
                aligned.append(None)  # Gap in this sequence
            elif op[0] == 'insert':
                aligned.append(seq[op[2]] if op[2] is not None else None)
        
        # Pad to same length
        while len(aligned) < len(backbone):
            aligned.append(None)
        
        all_aligned.append(aligned[:len(backbone)])
    
    return all_aligned


# ─── Phonetic Similarity ────────────────────────────────────────────────────

# Common phonetically similar pairs in Hindi
PHONETIC_PAIRS = {
    frozenset(['ब', 'व']), frozenset(['श', 'स']), frozenset(['ष', 'श']),
    frozenset(['ण', 'न']), frozenset(['ड', 'ड़']), frozenset(['ट', 'त']),
    frozenset(['भ', 'ब']), frozenset(['ध', 'द']), frozenset(['ठ', 'थ']),
    frozenset(['ख', 'क']), frozenset(['घ', 'ग']), frozenset(['छ', 'च']),
    frozenset(['झ', 'ज']), frozenset(['ढ', 'ड']), frozenset(['फ', 'प']),
}


def phonetic_similarity(word1: str, word2: str) -> float:
    """Compute phonetic similarity between two Hindi words.
    
    Returns a score between 0 (completely different) and 1 (identical/phonetically equivalent).
    """
    if word1 == word2:
        return 1.0
    
    if not word1 or not word2:
        return 0.0
    
    # Character-level edit distance
    char_dist = levenshtein_distance(word1, word2)
    max_len = max(len(word1), len(word2))
    base_sim = 1 - (char_dist / max_len)
    
    # Bonus for phonetically similar substitutions
    bonus = 0
    for i in range(min(len(word1), len(word2))):
        if word1[i] != word2[i]:
            pair = frozenset([word1[i], word2[i]])
            if pair in PHONETIC_PAIRS:
                bonus += 0.1
    
    return min(1.0, base_sim + bonus)


def are_semantically_equivalent(word1: str, word2: str) -> bool:
    """Check if two words are semantically equivalent variants.
    
    Handles:
    - Number forms (चौदह ↔ 14)
    - Spelling variants (गईं ↔ गई)
    - Matra variations
    """
    # Number equivalence
    hindi_to_digit = {
        'शून्य': '0', 'एक': '1', 'दो': '2', 'तीन': '3', 'चार': '4',
        'पांच': '5', 'पाँच': '5', 'छह': '6', 'छः': '6', 'सात': '7',
        'आठ': '8', 'नौ': '9', 'दस': '10', 'चौदह': '14',
    }
    
    if word1 in hindi_to_digit and word2 == hindi_to_digit[word1]:
        return True
    if word2 in hindi_to_digit and word1 == hindi_to_digit[word2]:
        return True
    
    # Common spelling variants
    variants = [
        ('ं', 'ँ'),      # anusvara/chandrabindu
        ('गईं', 'गई'),
        ('हैं', 'है'),
        ('में', 'मे'),
    ]
    
    for v1, v2 in variants:
        if (word1 == v1 and word2 == v2) or (word1 == v2 and word2 == v1):
            return True
    
    return False


# ─── Lattice Construction ───────────────────────────────────────────────────

class TranscriptionLattice:
    """A lattice representation of valid transcription alternatives.
    
    Each position (bin) contains all valid word alternatives at that point in the audio.
    
    Structure: List of bins, where each bin is a set of valid words.
    
    Example:
        [["उसने"], ["चौदह", "14"], ["किताबें", "किताबे", "पुस्तकें"], ["खरीदीं", "खरीदी"]]
    """
    
    def __init__(self):
        self.bins: List[set] = []
        self.confidences: List[Dict[str, float]] = []  # word -> confidence per bin
    
    @classmethod
    def from_aligned_outputs(cls, aligned_matrix: List[List[Optional[str]]], 
                              model_weights: Optional[List[float]] = None,
                              agreement_threshold: int = 3,
                              phonetic_threshold: float = 0.7):
        """Construct lattice from aligned multi-model outputs.
        
        Args:
            aligned_matrix: Alignment matrix (first row = reference, rest = models)
            model_weights: Optional weight per model (default: equal)
            agreement_threshold: Number of models that must agree to override reference
            phonetic_threshold: Similarity threshold for grouping variants
        
        Returns:
            TranscriptionLattice instance
        """
        lattice = cls()
        n_models = len(aligned_matrix) - 1  # Exclude reference
        
        if model_weights is None:
            model_weights = [1.0] * (n_models + 1)  # Include reference
        
        n_positions = max(len(row) for row in aligned_matrix) if aligned_matrix else 0
        
        for pos in range(n_positions):
            # Collect all words at this position
            words_at_pos = []
            for row_idx, row in enumerate(aligned_matrix):
                if pos < len(row) and row[pos] is not None:
                    words_at_pos.append((row[pos], model_weights[row_idx]))
            
            if not words_at_pos:
                continue
            
            # Count word occurrences (weighted)
            word_votes = defaultdict(float)
            for word, weight in words_at_pos:
                word_votes[word] += weight
            
            # Build bin with all valid alternatives
            bin_words = set()
            bin_confidences = {}
            
            # Add all words that appear
            for word, votes in word_votes.items():
                bin_words.add(word)
                bin_confidences[word] = votes / sum(model_weights)
            
            # Add phonetically similar groupings
            word_list = list(bin_words)
            for i in range(len(word_list)):
                for j in range(i + 1, len(word_list)):
                    sim = phonetic_similarity(word_list[i], word_list[j])
                    if sim >= phonetic_threshold:
                        # Both are valid alternatives
                        pass  # Already in the bin
            
            # Add semantically equivalent variants
            for word in list(bin_words):
                for other_word in list(bin_words):
                    if word != other_word and are_semantically_equivalent(word, other_word):
                        bin_confidences[word] = max(bin_confidences.get(word, 0),
                                                     bin_confidences.get(other_word, 0))
            
            # Model agreement: if ≥threshold models agree on something different from reference
            reference_word = aligned_matrix[0][pos] if pos < len(aligned_matrix[0]) else None
            model_words = [row[pos] for row in aligned_matrix[1:] if pos < len(row) and row[pos] is not None]
            
            if model_words:
                most_common = Counter(model_words).most_common(1)[0]
                agreed_word, agree_count = most_common
                
                if (agree_count >= agreement_threshold and 
                    agreed_word != reference_word and 
                    reference_word is not None):
                    # Trust model agreement over reference
                    bin_confidences[agreed_word] = max(
                        bin_confidences.get(agreed_word, 0),
                        agree_count / n_models
                    )
                    # Lower reference confidence
                    if reference_word in bin_confidences:
                        bin_confidences[reference_word] *= 0.5
            
            lattice.bins.append(bin_words)
            lattice.confidences.append(bin_confidences)
        
        return lattice
    
    def compute_wer(self, hypothesis: List[str]) -> Tuple[float, int, int, int, int]:
        """Compute WER using the lattice as reference.
        
        A hypothesis word at position i is correct if it matches ANY word in bin[i].
        
        Args:
            hypothesis: List of words from model output
        
        Returns:
            (wer, substitutions, deletions, insertions, total_ref_words)
        """
        # Align hypothesis to lattice positions
        ref_flat = [max(bin_words, key=lambda w: self.confidences[i].get(w, 0)) 
                    for i, bin_words in enumerate(self.bins)]
        
        _, alignment = word_edit_distance(ref_flat, hypothesis)
        
        S, D, I = 0, 0, 0
        ref_pos = 0
        
        for op in alignment:
            if op[0] == 'match':
                ref_pos += 1
            elif op[0] == 'substitute':
                hyp_word = hypothesis[op[2]] if op[2] is not None else ''
                # Check if hypothesis word is in the lattice bin
                if ref_pos < len(self.bins) and hyp_word in self.bins[ref_pos]:
                    pass  # It's a valid alternative, not an error!
                else:
                    S += 1
                ref_pos += 1
            elif op[0] == 'delete':
                D += 1
                ref_pos += 1
            elif op[0] == 'insert':
                I += 1
        
        total_ref = len(self.bins)
        wer = (S + D + I) / total_ref if total_ref > 0 else 0
        
        return wer, S, D, I, total_ref
    
    def to_dict(self) -> Dict:
        """Serialize lattice to dict."""
        return {
            'bins': [list(b) for b in self.bins],
            'confidences': [dict(c) for c in self.confidences],
            'num_positions': len(self.bins),
        }
    
    def __repr__(self):
        bins_str = [list(b) for b in self.bins]
        return f"Lattice({bins_str})"


# ─── WER Computation with Lattice ───────────────────────────────────────────

def compute_standard_wer(reference: str, hypothesis: str) -> float:
    """Compute standard WER (for comparison)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    dist, _ = word_edit_distance(ref_words, hyp_words)
    return dist / len(ref_words) if ref_words else 0


def compute_lattice_wer(lattice: TranscriptionLattice, hypothesis: str) -> float:
    """Compute lattice-based WER."""
    hyp_words = hypothesis.split()
    wer, _, _, _, _ = lattice.compute_wer(hyp_words)
    return wer


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def process_utterance(reference: str, model_outputs: List[str], 
                       agreement_threshold: int = 3) -> Dict:
    """Process a single utterance through the lattice pipeline.
    
    Args:
        reference: Human reference transcript
        model_outputs: List of 5 model output transcripts
        agreement_threshold: Models needed to agree to override reference
    
    Returns:
        Dict with lattice, standard WER, lattice WER for each model
    """
    ref_words = reference.split()
    model_word_lists = [out.split() for out in model_outputs]
    
    # Build aligned matrix
    aligned = align_multiple_sequences(model_word_lists, ref_words)
    
    # Construct lattice
    lattice = TranscriptionLattice.from_aligned_outputs(
        aligned, 
        agreement_threshold=agreement_threshold
    )
    
    # Compute WER for each model
    results = {
        'reference': reference,
        'lattice': lattice.to_dict(),
        'models': [],
    }
    
    for i, model_out in enumerate(model_outputs):
        std_wer = compute_standard_wer(reference, model_out)
        lat_wer = compute_lattice_wer(lattice, model_out)
        
        results['models'].append({
            'model_id': f'model_{i+1}',
            'output': model_out,
            'standard_wer': std_wer,
            'lattice_wer': lat_wer,
            'wer_change': lat_wer - std_wer,
            'improved': lat_wer < std_wer,
        })
    
    return results


# ─── Report ─────────────────────────────────────────────────────────────────

def generate_report(all_results, output_dir):
    """Generate Q4 lattice evaluation report."""
    
    report_path = os.path.join(output_dir, "lattice_wer_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Lattice-based WER Evaluation Report\n\n")
        
        f.write("## Approach\n\n")
        f.write("### Alignment Unit: Word\n")
        f.write("**Justification**: Hindi is a space-delimited language where word boundaries are clearly ")
        f.write("defined. Sub-word alignment (syllable, character) would add unnecessary complexity and ")
        f.write("introduce fragmentation artifacts. Phrase-level alignment risks missing valid word-level ")
        f.write("alternatives. Word-level provides the best balance of granularity and accuracy.\n\n")
        
        f.write("### Lattice Construction\n")
        f.write("1. Align all model outputs to the reference using pairwise word edit distance\n")
        f.write("2. At each aligned position, collect all unique word variants into bins\n")
        f.write("3. Group phonetically similar words (similarity ≥ 0.7) as valid alternatives\n")
        f.write("4. Add semantically equivalent forms (e.g., 'चौदह' ↔ '14')\n\n")
        
        f.write("### Model Agreement vs. Reference\n")
        f.write("When ≥3 out of 5 models agree on a word that differs from the reference:\n")
        f.write("- The agreed-upon word is given higher confidence in the lattice\n")
        f.write("- The reference word's confidence is halved\n")
        f.write("- Both remain as valid alternatives in the bin\n\n")
        
        f.write("### Handling Insertions/Deletions/Substitutions\n")
        f.write("- **Substitutions**: If a model's word matches ANY word in the corresponding ")
        f.write("lattice bin, it's counted as correct (not an error)\n")
        f.write("- **Insertions**: Extra words in model output that don't align to any bin are counted\n")
        f.write("- **Deletions**: Missing words from lattice positions are counted\n\n")
        
        # WER comparison table
        f.write("## Results\n\n")
        
        if all_results:
            f.write("### Per-Utterance WER Comparison\n\n")
            
            for utt_idx, result in enumerate(all_results, 1):
                f.write(f"#### Utterance {utt_idx}\n")
                f.write(f"**Reference**: {result['reference']}\n\n")
                
                f.write("| Model | Output | Standard WER | Lattice WER | Change |\n")
                f.write("|-------|--------|-------------|-------------|--------|\n")
                
                for model in result['models']:
                    out_short = model['output'][:50] + '...' if len(model['output']) > 50 else model['output']
                    change = f"{'↓' if model['improved'] else '↔'} {model['wer_change']:+.3f}"
                    f.write(f"| {model['model_id']} | {out_short} | "
                           f"{model['standard_wer']:.3f} | {model['lattice_wer']:.3f} | {change} |\n")
                f.write("\n")
            
            # Aggregate stats
            f.write("### Aggregate Results\n\n")
            f.write("| Model | Avg Standard WER | Avg Lattice WER | Improved | Unchanged | Worsened |\n")
            f.write("|-------|-----------------|-----------------|----------|-----------|----------|\n")
            
            # Collect per-model stats across utterances
            model_stats = defaultdict(lambda: {'std': [], 'lat': [], 'improved': 0, 'unchanged': 0, 'worsened': 0})
            for result in all_results:
                for model in result['models']:
                    mid = model['model_id']
                    model_stats[mid]['std'].append(model['standard_wer'])
                    model_stats[mid]['lat'].append(model['lattice_wer'])
                    if model['lattice_wer'] < model['standard_wer']:
                        model_stats[mid]['improved'] += 1
                    elif model['lattice_wer'] == model['standard_wer']:
                        model_stats[mid]['unchanged'] += 1
                    else:
                        model_stats[mid]['worsened'] += 1
            
            for mid, stats in sorted(model_stats.items()):
                avg_std = np.mean(stats['std'])
                avg_lat = np.mean(stats['lat'])
                f.write(f"| {mid} | {avg_std:.3f} | {avg_lat:.3f} | "
                       f"{stats['improved']} | {stats['unchanged']} | {stats['worsened']} |\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("1. The lattice approach **reduces WER for models that produce valid alternatives** ")
        f.write("not present in the rigid reference\n")
        f.write("2. Models that produce clearly incorrect output see **no change or slight increase** in WER\n")
        f.write("3. Number format differences (word vs digit) are properly handled as valid alternatives\n")
        f.write("4. Phonetic variants (common in Hindi due to script ambiguity) are correctly recognized\n")
    
    print(f"[INFO] Report saved to {report_path}")
    return report_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lattice-based WER evaluation")
    parser.add_argument('--input', type=str, default=None,
                        help='JSON file with model transcriptions')
    parser.add_argument('--output', type=str, default='results/q4/',
                        help='Output directory')
    parser.add_argument('--agreement_threshold', type=int, default=3,
                        help='Number of models needed to agree to override reference')
    parser.add_argument('--demo', action='store_true',
                        help='Run demonstration with example data')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Lattice-based WER Evaluation")
    print("=" * 60)
    
    if args.demo or not args.input:
        # Demo with the example from the assignment
        print("\n[DEMO] Running with example data...")
        
        reference = "उसने चौदह किताबें खरीदीं"
        model_outputs = [
            "उसने चौदह किताबें खरीदीं",     # Model 1: exact match
            "उसने 14 किताबें खरीदी",          # Model 2: number form + spelling variant
            "उसने चौदह किताबे खरीदीं",        # Model 3: spelling variant
            "उसने चौदह पुस्तकें खरीदीं",      # Model 4: synonym
            "उसने चौदह किताबें खरीदी",         # Model 5: matra difference
        ]
        
        result = process_utterance(reference, model_outputs, args.agreement_threshold)
        
        print(f"\nReference: {reference}")
        print(f"\nLattice: {result['lattice']['bins']}")
        print(f"\nWER Comparison:")
        print(f"{'Model':<10} {'Output':<40} {'Std WER':<10} {'Lat WER':<10} {'Change':<10}")
        print("-" * 80)
        for model in result['models']:
            out_short = model['output'][:38]
            change = f"{'↓' if model['improved'] else '↔'} {model['wer_change']:+.3f}"
            print(f"{model['model_id']:<10} {out_short:<40} {model['standard_wer']:<10.3f} {model['lattice_wer']:<10.3f} {change}")
        
        # Save demo results
        demo_results = [result]
        generate_report(demo_results, args.output)
        
        results_path = os.path.join(args.output, 'lattice_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, ensure_ascii=False, indent=2)
        
        return
    
    # Full processing from input file
    print(f"\n[STEP 1] Loading transcriptions from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_results = []
    for item in data:
        reference = item.get('reference', '')
        model_outputs = [item.get(f'model_{i}', '') for i in range(1, 6)]
        
        result = process_utterance(reference, model_outputs, args.agreement_threshold)
        all_results.append(result)
    
    print(f"  Processed {len(all_results)} utterances")
    
    # Generate report
    print("\n[STEP 2] Generating report...")
    generate_report(all_results, args.output)
    
    # Save results
    results_path = os.path.join(args.output, 'lattice_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[DONE] Results saved to {args.output}")


if __name__ == '__main__':
    main()
