# hindi ASR 


## Project Structure

```
asr\
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
├── doc/                           
├── results/                        
├── requirements.txt
└── README.md
```


