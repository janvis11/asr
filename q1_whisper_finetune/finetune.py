#!/usr/bin/env python3
"""
finetune.py — Fine-tune Whisper-small on Hindi ASR data.

Uses HuggingFace Seq2SeqTrainer with:
  - WhisperForConditionalGeneration (small)
  - WhisperProcessor (feature extractor + tokenizer)
  - WER evaluation metric
  - fp16 training, gradient accumulation

Usage:
  python finetune.py --data_dir data/processed --output_dir models/whisper-small-hi
"""

import os
import argparse
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk, Audio


# ─── Data Collator ───────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for Whisper fine-tuning.
    Handles padding of both input features and labels.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 so they are ignored in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if it was appended during tokenization
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─── Preprocessing Function ─────────────────────────────────────────────────

def prepare_dataset(batch, processor):
    """Preprocess a batch for Whisper training.
    
    - Extract audio features using WhisperFeatureExtractor
    - Tokenize text using WhisperTokenizer
    """
    audio = batch["audio"]

    # Compute log-Mel spectrogram features
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Tokenize target text
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(pred, tokenizer, metric):
    """Compute WER metric for evaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on Hindi ASR data")
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to preprocessed HuggingFace dataset')
    parser.add_argument('--output_dir', type=str, default='models/whisper-small-hi',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--model_name', type=str, default='openai/whisper-small',
                        help='Base Whisper model to fine-tune')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-device training batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Max training steps (-1 = use epochs)')

    args = parser.parse_args()

    print("=" * 60)
    print("Whisper-small Hindi Fine-tuning")
    print("=" * 60)

    # ─── Load Model & Processor ──────────────────────────────────────────
    print(f"\n[STEP 1] Loading model: {args.model_name}")
    
    processor = WhisperProcessor.from_pretrained(args.model_name, language="hi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Set forced decoder IDs for Hindi transcription
    model.generation_config.language = "hi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print(f"  Model parameters: {model.num_parameters():,}")

    # ─── Load Dataset ────────────────────────────────────────────────────
    print(f"\n[STEP 2] Loading dataset from {args.data_dir}")
    
    dataset = load_from_disk(args.data_dir)
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Test:  {len(dataset['test'])} samples")

    # Ensure audio is at 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # ─── Preprocess Dataset ──────────────────────────────────────────────
    print("\n[STEP 3] Preprocessing dataset (feature extraction + tokenization)")
    
    prepare_fn = partial(prepare_dataset, processor=processor)
    dataset = dataset.map(
        prepare_fn,
        remove_columns=dataset.column_names["train"],
        num_proc=1,  # Audio processing is not easily parallelizable
    )

    # ─── Data Collator ───────────────────────────────────────────────────
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ─── Metric ──────────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")
    compute_metrics_fn = partial(compute_metrics, tokenizer=processor.tokenizer, metric=wer_metric)

    # ─── Training Arguments ──────────────────────────────────────────────
    print("\n[STEP 4] Setting up training")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=25,
        report_to=["tensorboard"],
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2,
    )

    # ─── Trainer ─────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        processing_class=processor.feature_extractor,
    )

    # ─── Train ───────────────────────────────────────────────────────────
    print("\n[STEP 5] Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  FP16: {training_args.fp16}")
    print(f"  Device: {training_args.device}")

    trainer.train()

    # ─── Save ────────────────────────────────────────────────────────────
    print("\n[STEP 6] Saving fine-tuned model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print(f"\n[DONE] Fine-tuned model saved to: {args.output_dir}")
    print(f"  Final WER: {metrics.get('eval_wer', 'N/A'):.2f}%")


if __name__ == '__main__':
    main()
