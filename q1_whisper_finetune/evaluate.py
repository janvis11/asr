#!/usr/bin/env python3
"""
evaluate.py — Evaluate pretrained and fine-tuned Whisper-small on FLEURS Hindi test set.

Outputs a structured WER comparison table in both Markdown and CSV formats.

Usage:
  python evaluate.py --model_dir models/whisper-small-hi --output results/
"""

import os
import argparse
import json
import torch
import evaluate
import pandas as pd
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from datasets import load_dataset


# ─── Inference ───────────────────────────────────────────────────────────────

def transcribe_dataset(model, processor, dataset, device, batch_size=8):
    """Run Whisper inference on a dataset.
    
    Args:
        model: WhisperForConditionalGeneration
        processor: WhisperProcessor
        dataset: HuggingFace dataset with 'audio' column
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        List of predicted transcriptions
    """
    model.eval()
    predictions = []
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    
    # Set generation config for Hindi
    pipe.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="hi", task="transcribe"
    )
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Transcribing"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        audio_arrays = [sample["audio"]["array"] for sample in batch]
        
        results = pipe(
            audio_arrays,
            generate_kwargs={"language": "hi", "task": "transcribe"},
            batch_size=batch_size,
        )
        
        for result in results:
            predictions.append(result["text"].strip())
    
    return predictions


def compute_wer_detailed(predictions, references, wer_metric):
    """Compute overall WER and per-utterance WER.
    
    Returns:
        dict with 'overall_wer', 'per_utterance_wer', 'num_utterances'
    """
    overall_wer = wer_metric.compute(predictions=predictions, references=references)
    
    per_utterance = []
    for pred, ref in zip(predictions, references):
        try:
            utt_wer = wer_metric.compute(predictions=[pred], references=[ref])
        except:
            utt_wer = 1.0
        per_utterance.append({
            'reference': ref,
            'prediction': pred,
            'wer': utt_wer,
        })
    
    return {
        'overall_wer': overall_wer,
        'per_utterance': per_utterance,
        'num_utterances': len(predictions),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on FLEURS Hindi test set")
    parser.add_argument('--model_dir', type=str, default='models/whisper-small-hi',
                        help='Path to fine-tuned model directory')
    parser.add_argument('--baseline_model', type=str, default='openai/whisper-small',
                        help='Pretrained baseline model name')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ─── Load FLEURS Hindi Test ──────────────────────────────────────────
    print("\n[STEP 1] Loading FLEURS Hindi test dataset...")
    fleurs = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    print(f"  FLEURS Hindi test: {len(fleurs)} utterances")
    
    # Extract references
    references = [sample["transcription"] for sample in fleurs]
    
    wer_metric = evaluate.load("wer")
    
    # ─── Baseline Evaluation ─────────────────────────────────────────────
    print(f"\n[STEP 2] Evaluating baseline: {args.baseline_model}")
    baseline_processor = WhisperProcessor.from_pretrained(args.baseline_model, language="hi", task="transcribe")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(args.baseline_model).to(device)
    
    baseline_preds = transcribe_dataset(baseline_model, baseline_processor, fleurs, device, args.batch_size)
    baseline_results = compute_wer_detailed(baseline_preds, references, wer_metric)
    
    print(f"  Baseline WER: {baseline_results['overall_wer']*100:.2f}%")
    
    # Free memory
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ─── Fine-tuned Evaluation ───────────────────────────────────────────
    print(f"\n[STEP 3] Evaluating fine-tuned model: {args.model_dir}")
    ft_processor = WhisperProcessor.from_pretrained(args.model_dir)
    ft_model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    
    ft_preds = transcribe_dataset(ft_model, ft_processor, fleurs, device, args.batch_size)
    ft_results = compute_wer_detailed(ft_preds, references, wer_metric)
    
    print(f"  Fine-tuned WER: {ft_results['overall_wer']*100:.2f}%")
    
    # ─── Results Table ───────────────────────────────────────────────────
    print("\n[STEP 4] Generating results table...")
    
    results_table = {
        'Model': ['Whisper-small (pretrained)', 'Whisper-small (fine-tuned)'],
        'WER (%)': [
            f"{baseline_results['overall_wer']*100:.2f}",
            f"{ft_results['overall_wer']*100:.2f}",
        ],
        'Test Set': ['FLEURS Hindi', 'FLEURS Hindi'],
        'Num Utterances': [
            baseline_results['num_utterances'],
            ft_results['num_utterances'],
        ],
    }
    
    df = pd.DataFrame(results_table)
    
    # Save as CSV
    csv_path = os.path.join(args.output, "wer_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Save as Markdown
    md_path = os.path.join(args.output, "wer_results.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# WER Results — Whisper-small on FLEURS Hindi Test Set\n\n")
        f.write(df.to_markdown(index=False))
        f.write(f"\n\n**WER Improvement**: {(baseline_results['overall_wer'] - ft_results['overall_wer'])*100:.2f} percentage points\n")
    
    # Save detailed per-utterance results
    detailed_path = os.path.join(args.output, "detailed_results.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': {
                'overall_wer': baseline_results['overall_wer'],
                'per_utterance': baseline_results['per_utterance'],
            },
            'finetuned': {
                'overall_wer': ft_results['overall_wer'],
                'per_utterance': ft_results['per_utterance'],
            }
        }, f, ensure_ascii=False, indent=2)
    
    # Print table
    print("\n" + "=" * 60)
    print("WER RESULTS")
    print("=" * 60)
    print(df.to_markdown(index=False))
    print(f"\nWER Improvement: {(baseline_results['overall_wer'] - ft_results['overall_wer'])*100:.2f} pp")
    print(f"\nResults saved to: {args.output}")
    

if __name__ == '__main__':
    main()
