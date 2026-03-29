# Hindi ASR System - Josh Talks AI Researcher Intern Submission

A complete Hindi Automatic Speech Recognition system built on Whisper-small with four key components: fine-tuning, output cleanup, spelling correction, and lattice-based evaluation.

**Deadline:** 29 March 2026, 10 PM

---

## Quick Start

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate      # Linux/Mac

# Run demo for each question
python q2_cleanup_pipeline/pipeline.py --demo
python q3_spelling/spell_checker.py --wordlist data/wordlist.txt --output results/q3/
python q4_lattice/lattice.py --demo --output results/q4/
```

---

## Project Structure

```
asr/
├── q1_whisper_finetune/       # Question 1: Fine-tuning
│   ├── preprocess.py          # Data download + preprocessing
│   ├── finetune.py            # Whisper-small fine-tuning
│   ├── evaluate.py            # Evaluation on FLEURS Hindi
│   ├── error_analysis.py      # Error sampling + taxonomy
│   └── fix_implementation.py  # Fix implementation + evaluation
├── q2_cleanup_pipeline/       # Question 2: Cleanup
│   ├── number_normalization.py # Hindi numbers → digits
│   ├── english_detection.py    # English word detection
│   └── pipeline.py             # Combined pipeline
├── q3_spelling/               # Question 3: Spelling
│   ├── spell_checker.py       # Classification pipeline
│   └── analysis.py            # Confidence analysis
├── q4_lattice/                # Question 4: Lattice WER
│   ├── lattice.py             # Lattice construction + WER
│   └── README.md              # Theory documentation
├── data/                      # Dataset storage
│   ├── raw/                   # Downloaded audio + transcriptions
│   ├── processed/             # Preprocessed HF datasets
│   └── dataset.csv            # Data manifest
├── results/                   # Output results
│   ├── q1/                    # Q1 WER + error analysis
│   ├── q2/                    # Q2 cleanup outputs
│   ├── q3/                    # Q3 classified words
│   └── q4/                    # Q4 lattice results
├── doc/                       # Documentation
│   └── SUBMISSION_DOCUMENTATION.md
├── models/                    # Fine-tuned models
├── requirements.txt
├── SETUP.md                   # Detailed setup guide
├── final_answers.md           # Answers summary
└── README.md                  # This file
```

---

## Question 1: Hindi ASR Fine-Tuning

### Files
- `q1_whisper_finetune/preprocess.py` - Data preprocessing
- `q1_whisper_finetune/finetune.py` - Model fine-tuning
- `q1_whisper_finetune/evaluate.py` - WER evaluation
- `q1_whisper_finetune/error_analysis.py` - Error taxonomy
- `q1_whisper_finetune/fix_implementation.py` - Fix implementation

### Usage
```bash
# Full pipeline
python q1_whisper_finetune/preprocess.py --dataset_csv data/dataset.csv --output_dir data/processed
python q1_whisper_finetune/finetune.py --data_dir data/processed --output_dir models/whisper-small-hi
python q1_whisper_finetune/evaluate.py --model_dir models/whisper-small-hi --output results/q1/
python q1_whisper_finetune/error_analysis.py --results_file results/q1/detailed_results.json --output results/q1/
```

### Key Results
- **Error Taxonomy:** 5 categories identified (Sandhi, Numerals, English loans, Insertions, Dialectal)
- **Top Fix:** Number normalization reduces WER on numeral errors by ~80%

---

## Question 2: ASR Output Cleanup

### Files
- `q2_cleanup_pipeline/number_normalization.py` - Number conversion
- `q2_cleanup_pipeline/english_detection.py` - English detection
- `q2_cleanup_pipeline/pipeline.py` - Combined pipeline

### Usage
```bash
# Demo mode
python q2_cleanup_pipeline/pipeline.py --demo

# Full pipeline
python q2_cleanup_pipeline/pipeline.py --input baseline_transcripts.json --output results/q2/
```

### Features
- **Number Normalization:** Compound numbers (तीन सौ चौवन → 354), idiom preservation
- **English Detection:** 3-stage detection (Roman, dictionary, suffix patterns)
- **Output Format:** `[EN]word[/EN]` tagging

### Demo Output
```
Original: तीन सौ चौवन लोग आए थे और मेरा इंटरव्यू अच्छा गया
Cleaned:  354 लोग आए थे और मेरा [EN]इंटरव्यू[/EN] अच्छा गया
```

---

## Question 3: Spelling Correction

### Files
- `q3_spelling/spell_checker.py` - 4-layer classification pipeline

### Usage
```bash
python q3_spelling/spell_checker.py --wordlist data/wordlist.txt --output results/q3/
```

### Classification Layers
1. Dictionary lookup (Hindi WordNet, CC-100)
2. Morphological analysis
3. Character trigram LM
4. Rule-based error detection

### Results
- **Correctly spelled:** 388 words (99.7%)
- **Incorrectly spelled:** 1 word (0.3%)
- **Confidence:** High 56.8%, Medium 4.9%, Low 38.3%

### Output Files
- `results/q3/classified_words.csv` - Full classification with confidence
- `results/q3/words_for_sheets.csv` - Google Sheets format
- `results/q3/classification_results.json` - Detailed JSON

---

## Question 4: Lattice WER

### Files
- `q4_lattice/lattice.py` - Lattice construction + WER

### Usage
```bash
# Demo mode
python q4_lattice/lattice.py --demo --output results/q4/

# Full evaluation
python q4_lattice/lattice.py --input data/q4_transcriptions.json --output results/q4/
```

### Key Features
- Word-level alignment (justified for Hindi)
- Model agreement threshold (≥3/5 models)
- Phonetic similarity grouping (≥0.7)
- Numeral equivalence handling



## Installation

### Prerequisites
- Python 3.9+
- GPU recommended (8GB+ VRAM for fine-tuning)
- FFmpeg for audio processing

### Setup
```bash
# Clone repository
git clone <repository-url>
cd asr

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Deliverables Summary

| Question | Output Files | Location |
|----------|--------------|----------|
| Q1 | WER results, error reports | `results/q1/` |
| Q2 | Cleaned transcripts, pipeline report | `results/q2/` |
| Q3 | Classified words CSV, JSON | `results/q3/` |
| Q4 | Lattice WER report, JSON | `results/q4/` |

---

## Documentation

- **SETUP.md** - Detailed installation and usage guide
- **doc/SUBMISSION_DOCUMENTATION.md** - Complete submission documentation
- **final_answers.md** - Summary of all answers

---


