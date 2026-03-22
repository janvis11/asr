# Hindi ASR Assignment — Josh Talks AI Researcher Intern (Speech & Audio)

This repository contains the complete solution for the 4-question Hindi ASR assignment.

## Project Structure

```
d:\asr\
├── data/                           # Dataset storage
│   ├── raw/                        # Downloaded audio + transcription files
│   ├── processed/                  # Preprocessed HuggingFace datasets
│   └── fleurs/                     # FLEURS Hindi test dataset cache
├── q1_whisper_finetune/            # Question 1: Whisper Fine-tuning
│   ├── preprocess.py               # Data download + preprocessing
│   ├── finetune.py                 # Whisper-small fine-tuning
│   ├── evaluate.py                 # Evaluation on FLEURS Hindi
│   ├── error_analysis.py           # Error sampling + taxonomy
│   └── fix_implementation.py       # Implement proposed fix
├── q2_cleanup_pipeline/            # Question 2: ASR Cleanup
│   ├── number_normalization.py     # Hindi numbers → digits
│   ├── english_detection.py        # English word detection
│   └── pipeline.py                 # Combined cleanup pipeline
├── q3_spelling/                    # Question 3: Spelling Detection
│   ├── spell_checker.py            # Correct vs incorrect classifier
│   └── analysis.py                 # Confidence scoring + review
├── q4_lattice/                     # Question 4: Lattice WER
│   ├── lattice.py                  # Lattice construction + WER
│   └── README.md                   # Theory + approach doc
├── doc/                            # Documentation for every file
├── results/                        # Output tables, reports, CSVs
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### Q1: Fine-tune Whisper-small
```bash
python q1_whisper_finetune/preprocess.py --dataset_csv data/dataset.csv --output_dir data/processed
python q1_whisper_finetune/finetune.py --data_dir data/processed --output_dir models/whisper-small-hi
python q1_whisper_finetune/evaluate.py --model_dir models/whisper-small-hi --output results/
python q1_whisper_finetune/error_analysis.py --model_dir models/whisper-small-hi --output results/
```

### Q2: ASR Cleanup Pipeline
```bash
python q2_cleanup_pipeline/pipeline.py --input data/raw --output results/q2/
```

### Q3: Spelling Error Detection
```bash
python q3_spelling/spell_checker.py --wordlist data/wordlist.txt --output results/q3/
python q3_spelling/analysis.py --results results/q3/classified_words.csv
```

### Q4: Lattice-based WER
```bash
python q4_lattice/lattice.py --input data/q4_transcriptions.json --output results/q4/
```
