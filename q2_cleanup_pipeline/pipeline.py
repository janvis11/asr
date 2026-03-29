#!/usr/bin/env python3
"""
pipeline.py — Combined ASR output cleanup pipeline.

Takes raw ASR output (from pretrained Whisper-small) and applies:
  1. Number normalization (Hindi number words → digits)
  2. English word detection and tagging

Also generates baseline transcripts by running pretrained Whisper on audio segments,
pairs them with human references, and reports before/after WER.

Usage:
  python pipeline.py --audio_dir data/raw --transcriptions data/processed --output results/q2/
"""

import os
import sys
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm

# Fix Windows console UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q2_cleanup_pipeline.number_normalization import normalize_numbers
from q2_cleanup_pipeline.english_detection import tag_english_words, detect_english_words


# ─── Pipeline ───────────────────────────────────────────────────────────────

class ASRCleanupPipeline:
    """Combined cleanup pipeline for raw ASR output.
    
    Steps:
      1. Number normalization: Hindi number words → digits
      2. English word detection: Tag English words with [EN]...[/EN]
    """
    
    def __init__(self, normalize_numbers_flag=True, detect_english_flag=True,
                 preserve_idioms=True, english_method='all'):
        self.normalize_numbers_flag = normalize_numbers_flag
        self.detect_english_flag = detect_english_flag
        self.preserve_idioms = preserve_idioms
        self.english_method = english_method
    
    def process(self, text):
        """Run full cleanup pipeline on a single text.
        
        Args:
            text: Raw ASR output text
        
        Returns:
            Dict with cleaned text and metadata about changes
        """
        result = {
            'original': text,
            'steps': [],
        }
        
        current_text = text
        
        # Step 1: Number normalization
        if self.normalize_numbers_flag:
            normalized = normalize_numbers(current_text, preserve_idioms=self.preserve_idioms)
            if normalized != current_text:
                result['steps'].append({
                    'step': 'number_normalization',
                    'before': current_text,
                    'after': normalized,
                })
            current_text = normalized
        
        # Step 2: English word detection
        if self.detect_english_flag:
            detected = detect_english_words(current_text, method=self.english_method)
            tagged = tag_english_words(current_text, method=self.english_method)
            result['steps'].append({
                'step': 'english_detection',
                'before': current_text,
                'after': tagged,
                'english_words': [d['word'] for d in detected],
            })
            current_text = tagged
        
        result['cleaned'] = current_text
        result['num_changes'] = len(result['steps'])
        
        return result
    
    def process_batch(self, texts):
        """Process a batch of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of result dicts
        """
        return [self.process(text) for text in tqdm(texts, desc="Cleaning")]


# ─── Generate Baseline Transcripts ──────────────────────────────────────────

def generate_baseline_transcripts(audio_dir, output_path, model_name='openai/whisper-small'):
    """Run pretrained Whisper-small on audio files to generate baseline transcripts.
    
    Args:
        audio_dir: Directory with audio segment WAV files
        output_path: Path to save baseline transcripts JSON
        model_name: Whisper model to use
    
    Returns:
        List of dicts with 'audio_file', 'transcript'
    """
    from transformers import pipeline as hf_pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Find audio files
    audio_files = sorted([
        os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
        if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))
    ])
    
    if not audio_files:
        print(f"[WARN] No audio files found in {audio_dir}")
        return []
    
    print(f"[INFO] Found {len(audio_files)} audio files")
    
    results = []
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        try:
            result = pipe(
                audio_file,
                generate_kwargs={"language": "hi", "task": "transcribe"},
            )
            results.append({
                'audio_file': os.path.basename(audio_file),
                'transcript': result['text'].strip(),
            })
        except Exception as e:
            print(f"  [ERROR] {audio_file}: {e}")
            results.append({
                'audio_file': os.path.basename(audio_file),
                'transcript': '',
                'error': str(e),
            })
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Saved {len(results)} transcripts to {output_path}")
    return results


# ─── WER Comparison ─────────────────────────────────────────────────────────

def compare_wer(raw_transcripts, cleaned_transcripts, references):
    """Compare WER before and after cleanup.
    
    Args:
        raw_transcripts: List of raw ASR outputs
        cleaned_transcripts: List of cleaned outputs
        references: List of human reference transcripts
    
    Returns:
        Dict with WER comparison stats
    """
    from jiwer import wer
    
    # Remove [EN]...[/EN] tags for WER computation
    def strip_tags(text):
        return text.replace('[EN]', '').replace('[/EN]', '')
    
    cleaned_stripped = [strip_tags(t) for t in cleaned_transcripts]
    
    raw_wer = wer(references, raw_transcripts)
    cleaned_wer = wer(references, cleaned_stripped)
    
    return {
        'raw_wer': raw_wer,
        'cleaned_wer': cleaned_wer,
        'improvement': raw_wer - cleaned_wer,
        'improvement_pct': (raw_wer - cleaned_wer) / raw_wer * 100 if raw_wer > 0 else 0,
    }


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_report(pipeline_results, output_dir):
    """Generate Q2 cleanup pipeline report with examples."""
    
    report_path = os.path.join(output_dir, "cleanup_pipeline_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# ASR Output Cleanup Pipeline Report\n\n")
        
        f.write("## Pipeline Operations\n\n")
        f.write("### 1. Number Normalization\n")
        f.write("Converts spoken Hindi number words into digits.\n\n")
        
        f.write("### 2. English Word Detection\n")
        f.write("Identifies English words transliterated in Devanagari and tags them with `[EN]...[/EN]` markers.\n\n")
        
        # Find examples with changes
        number_examples = [r for r in pipeline_results if any(s['step'] == 'number_normalization' for s in r['steps'])]
        english_examples = [r for r in pipeline_results if any(s.get('english_words') for s in r['steps'])]
        
        f.write("## Number Normalization Examples\n\n")
        shown = 0
        for r in number_examples[:5]:
            for step in r['steps']:
                if step['step'] == 'number_normalization' and step['before'] != step['after']:
                    shown += 1
                    f.write(f"**Example {shown}:**\n")
                    f.write(f"- Before: {step['before']}\n")
                    f.write(f"- After: {step['after']}\n\n")
        
        f.write("## English Detection Examples\n\n")
        shown = 0
        for r in english_examples[:5]:
            for step in r['steps']:
                if step['step'] == 'english_detection' and step.get('english_words'):
                    shown += 1
                    f.write(f"**Example {shown}:**\n")
                    f.write(f"- Input: {step['before']}\n")
                    f.write(f"- Output: {step['after']}\n")
                    f.write(f"- English words: {', '.join(step['english_words'])}\n\n")
        
        # Summary stats
        total = len(pipeline_results)
        num_changed = sum(1 for r in pipeline_results if r['num_changes'] > 0)
        all_english = []
        for r in pipeline_results:
            for s in r['steps']:
                all_english.extend(s.get('english_words', []))

        f.write("## Summary Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total transcripts processed | {total} |\n")
        f.write(f"| Transcripts with changes | {num_changed} ({num_changed/max(total,1)*100:.1f}%) |\n")
        f.write(f"| Total English words detected | {len(all_english)} |\n")
        f.write(f"| Unique English words | {len(set(all_english))} |\n")

        # Tricky Edge Cases & Reasoning
        f.write("\n## Tricky Edge Cases & Reasoning\n\n")
        f.write("### Number Normalization Edge Cases\n\n")
        f.write("| Input | Output | Reasoning |\n")
        f.write("|-------|--------|----------|\n")
        f.write("| दो-चार बातें | दो-चार बातें | Idiomatic expression meaning 'a few things' — should NOT become '2-4 बातें' |\n")
        f.write("| एक दूसरे | एक दूसरे | Means 'each other' — not a cardinal number usage |\n")
        f.write("| चारों तरफ | चारों तरफ | 'चारों' is a determiner meaning 'all four sides' — not convertible |\n")
        f.write("| दोनों लोग | दोनों लोग | 'दोनों' means 'both' — pronominal usage, not cardinal |\n")
        f.write("| एक बार | एक बार | Temporal phrase 'once' — kept as-is per idiom blocklist |\n")
        f.write("| तीन सौ चौवन | 354 | Standard compound number — correctly converted |\n")
        f.write("| पच्चीस साल | 25 साल | Simple number word — correctly converted |\n\n")

        f.write("### English Detection Edge Cases\n\n")
        f.write("| Input | Detection | Reasoning |\n")
        f.write("|-------|-----------|----------|\n")
        f.write("| सन | NOT tagged | Hindi word (sun/heard) — in exception whitelist, not English 'sun' |\n")
        f.write("| मस्त | NOT tagged | Hindi adjective (happy/chill) — not English loanword |\n")
        f.write("| इंटरव्यू | [EN]tagged[/EN] | Clear English loanword with -यू suffix pattern |\n")
        f.write("| प्रॉब्लम | [EN]tagged[/EN] | English 'problem' — ऑ vowel + ब्ल consonant cluster |\n")
        f.write("| कंप्यूटर | [EN]tagged[/EN] | English 'computer' — in dictionary + -टर suffix |\n\n")

        f.write("### Where This Helps vs. Hurts\n\n")
        f.write("**Helps:**\n")
        f.write("- Downstream NLP tasks (MT, NER, TTS) need script boundary information\n")
        f.write("- Number normalizer avoids treating English number words as Hindi\n")
        f.write("- Code-switching analysis becomes possible for sociolinguistic studies\n\n")

        f.write("**Hurts / Edge Cases:**\n")
        f.write("- Devanagari words that phonetically resemble English (सन, मस्त) can produce false positives\n")
        f.write("- Mitigated by maintaining a Hindi whitelist of common false positives\n")
        f.write("- Regional/dialectal English borrowings not in dictionary may be missed\n")
        f.write("- Confidence scoring needed for borderline cases (suffix-only detection = lower confidence)\n")
    
    print(f"[INFO] Report saved to {report_path}")
    return report_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASR output cleanup pipeline")
    parser.add_argument('--input', type=str, default=None,
                        help='JSON file with raw ASR transcripts')
    parser.add_argument('--references', type=str, default=None,
                        help='JSON file with human reference transcripts')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Directory with audio files for baseline generation')
    parser.add_argument('--output', type=str, default='results/q2/',
                        help='Output directory')
    parser.add_argument('--no_numbers', action='store_true',
                        help='Disable number normalization')
    parser.add_argument('--no_english', action='store_true',
                        help='Disable English detection')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with sample data')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("ASR Output Cleanup Pipeline")
    print("=" * 60)
    
    if args.demo:
        # Run demo with sample sentences
        demo_texts = [
            "तीन सौ चौवन लोग आए थे और मेरा इंटरव्यू अच्छा गया",
            "पच्चीस साल पहले ये प्रॉब्लम नहीं थी",
            "एक हज़ार रुपये कंप्यूटर का खर्चा आया",
            "दो-चार बातें करनी हैं ऑफिस में",
            "मेरा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है उधर की एरिया में",
        ]
        
        pipeline = ASRCleanupPipeline(
            normalize_numbers_flag=True,
            detect_english_flag=True,
        )
        
        for text in demo_texts:
            result = pipeline.process(text)
            print(f"\n  Original: {result['original'].encode('utf-8').decode('utf-8')}")
            print(f"  Cleaned:  {result['cleaned'].encode('utf-8').decode('utf-8')}")
            for step in result['steps']:
                if step['step'] == 'number_normalization' and step['before'] != step['after']:
                    print(f"    [NUM] {step['before']} -> {step['after']}")
                if step.get('english_words'):
                    print(f"    [EN]  {step['english_words']}")
        return
    
    # Full pipeline
    pipeline = ASRCleanupPipeline(
        normalize_numbers_flag=not args.no_numbers,
        detect_english_flag=not args.no_english,
    )
    
    if args.audio_dir:
        print("\n[STEP 1] Generating baseline transcripts...")
        baseline_path = os.path.join(args.output, 'baseline_transcripts.json')
        generate_baseline_transcripts(args.audio_dir, baseline_path)
        args.input = baseline_path
    
    if args.input:
        print(f"\n[STEP 2] Loading transcripts from {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [d.get('transcript', d.get('text', '')) for d in data]
        
        print(f"  Loaded {len(texts)} transcripts")
        
        print("\n[STEP 3] Running cleanup pipeline...")
        results = pipeline.process_batch(texts)
        
        # Save results
        output_path = os.path.join(args.output, 'cleaned_transcripts.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Generate report
        print("\n[STEP 4] Generating report...")
        generate_report(results, args.output)
        
        # WER comparison if references available
        if args.references:
            print("\n[STEP 5] Computing WER comparison...")
            with open(args.references, 'r', encoding='utf-8') as f:
                ref_data = json.load(f)
            references = [d.get('text', d.get('transcription', '')) for d in ref_data]
            
            raw_texts = texts[:len(references)]
            cleaned_texts = [r['cleaned'] for r in results[:len(references)]]
            
            wer_stats = compare_wer(raw_texts, cleaned_texts, references)
            
            print(f"\n  WER Comparison:")
            print(f"    Raw ASR WER:     {wer_stats['raw_wer']*100:.2f}%")
            print(f"    Cleaned WER:     {wer_stats['cleaned_wer']*100:.2f}%")
            print(f"    Improvement:     {wer_stats['improvement']*100:.2f} pp ({wer_stats['improvement_pct']:.1f}%)")
        
        print(f"\n[DONE] Results saved to {args.output}")
    else:
        print("[ERROR] Please provide --input or --audio_dir")
        parser.print_help()


if __name__ == '__main__':
    main()
