#!/usr/bin/env python3
"""
create_synthetic_dataset.py — Create a synthetic Hindi ASR dataset for demonstration.

Since the actual audio files are not accessible from GCP, this script creates
a synthetic dataset using the sample transcriptions with generated sine-wave audio.

This demonstrates the full pipeline end-to-end for the assignment submission.

Usage:
  python create_synthetic_dataset.py --output_dir data/processed
"""

import os
import json
import numpy as np
import argparse
from datasets import Dataset, DatasetDict, Audio
import soundfile as sf

SAMPLE_RATE = 16000

# Sample Hindi transcriptions from the dataset
SAMPLE_TRANSCRIPTIONS = [
    "नमस्ते दोस्तों आज मैं अपने गाँव के बारे में बात करूँगा",
    "मेरा गाँव बहुत सुंदर है और यहाँ के लोग बहुत अच्छे हैं",
    "तीन सौ चौवन लोग आए थे और मेरा इंटरव्यू अच्छा गया",
    "पच्चीस साल पहले ये प्रॉब्लम नहीं थी",
    "एक हज़ार रुपये कंप्यूटर का खर्चा आया",
    "दो-चार बातें करनी हैं ऑफिस में",
    "मेरा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है",
    "उसने चौदह किताबें खरीदीं",
    "दो लाख तीन हज़ार पाँच सौ का खर्च आया",
    "कंप्यूटर पर ऑनलाइन क्लास चल रही है",
    "लोग घूमने जाते हैं तो लाइट वगैरा लेकर जाने चाहिए",
    "एक दूसरे से बात करो समस्या हल हो जाएगी",
    "चारों तरफ लोग खड़े थे",
    "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
    "ये प्रॉब्लम सॉल्व नहीं हो रहा",
    "मैंने अपना रिज्यूमे अपडेट किया और कंपनी को ईमेल किया",
    "हम्म मतलब आपको समझ नहीं आया",
    "भारत एक बहुत पुराना देश है",
    "हिंदी भाषा बोलने वाले लोग बहुत हैं",
    "मुंबई दिल्ली और कोलकाता बड़े शहर हैं",
]


def generate_synthetic_audio(duration_sec, sample_rate=SAMPLE_RATE):
    """Generate synthetic audio (sine wave mixture) for demonstration.

    In a real scenario, this would be actual recorded speech.
    For demonstration purposes, we generate synthetic audio that:
    - Has the correct duration
    - Contains some frequency content (mimics speech spectrum)
    - Is saved as valid WAV files
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples)

    # Generate a mixture of frequencies (simulates speech-like signal)
    # Speech fundamental: 85-255 Hz, formants: 300-3500 Hz
    freqs = [150, 300, 500, 800, 1200, 2000, 2800]
    audio = np.zeros(n_samples)

    for f in freqs:
        # Add frequency component with varying amplitude (simulates prosody)
        amplitude = 0.1 / (1 + f/3000)
        audio += amplitude * np.sin(2 * np.pi * f * t)

    # Add some noise
    audio += 0.02 * np.random.randn(n_samples)

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9

    return audio, sample_rate


def create_synthetic_dataset(output_dir, test_split=0.1):
    """Create synthetic Hindi ASR dataset.

    Args:
        output_dir: Directory to save the dataset
        test_split: Fraction for test split

    Returns:
        DatasetDict with train and test splits
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    print(f"[INFO] Creating synthetic dataset with {len(SAMPLE_TRANSCRIPTIONS)} utterances")

    # Create synthetic audio files and transcripts
    data_dict = {
        'audio': [],
        'text': [],
        'duration': [],
        'recording_id': [],
        'user_id': [],
    }

    for idx, text in enumerate(SAMPLE_TRANSCRIPTIONS):
        # Estimate duration from text length (~15 chars/sec for Hindi)
        duration = max(1.0, len(text) / 12)  # ~1-4 seconds per utterance
        duration = min(duration, 5.0)  # Cap at 5 seconds for synthetic data

        # Generate synthetic audio
        audio, sr = generate_synthetic_audio(duration)

        # Save audio file
        audio_path = os.path.join(audio_dir, f"synth_{idx:04d}.wav")
        sf.write(audio_path, audio, sr)

        data_dict['audio'].append(audio_path)
        data_dict['text'].append(text)
        data_dict['duration'].append(duration)
        data_dict['recording_id'].append(f"synth_{idx:04d}")
        data_dict['user_id'].append(f"synth_user_{idx % 5:02d}")

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=SAMPLE_RATE))

    # Train/test split
    dataset = dataset.train_test_split(test_size=test_split, seed=42)

    # Save to disk
    dataset.save_to_disk(output_dir)

    # Save stats
    stats = {
        'total_utterances': len(SAMPLE_TRANSCRIPTIONS),
        'train_samples': len(dataset['train']),
        'test_samples': len(dataset['test']),
        'total_duration_hrs': sum(data_dict['duration']) / 3600,
        'avg_duration_sec': np.mean(data_dict['duration']),
        'unique_speakers': len(set(data_dict['user_id'])),
        'dataset_type': 'synthetic',
    }

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Dataset saved to {output_dir}")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Test:  {len(dataset['test'])} samples")
    print(f"  Total duration: {stats['total_duration_hrs']*60:.1f} minutes")
    print(f"  Stats saved to: {stats_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Create synthetic Hindi ASR dataset")
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save the dataset')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Fraction of data for test split')

    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Hindi ASR Dataset Creation")
    print("=" * 60)
    print("\nNOTE: Creating synthetic dataset since GCP audio files are not accessible.")
    print("In production, use real audio recordings with the preprocess.py script.\n")

    create_synthetic_dataset(args.output_dir, args.test_split)

    print("\n[DONE] Synthetic dataset created successfully!")


if __name__ == '__main__':
    main()
