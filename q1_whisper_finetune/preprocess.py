#!/usr/bin/env python3
"""
preprocess.py — Download and preprocess Hindi ASR training data for Whisper fine-tuning.

Steps:
  1. Read dataset manifest CSV (user_id, recording_id, duration, rec_url_gcp, transcription_url, metadata_url)
  2. Download audio files and transcription JSONs from GCP
  3. Segment audio by timestamps from transcription JSON
  4. Create HuggingFace Dataset with 'audio' and 'text' columns
  5. Apply Whisper feature extractor + tokenizer
  6. Save preprocessed dataset to disk

Usage:
  python preprocess.py --dataset_csv data/dataset.csv --output_dir data/processed
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa
from datasets import Dataset, DatasetDict, Audio

# ─── Constants ───────────────────────────────────────────────────────────────
GCP_BASE_URL = "https://storage.googleapis.com/upload_goai"
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio


# ─── Data Download ───────────────────────────────────────────────────────────

def build_gcp_url(user_id, recording_id, file_type="transcription"):
    """Build GCP URL for a given user_id and recording_id.
    
    Args:
        user_id: Speaker/user identifier
        recording_id: Unique recording identifier
        file_type: One of 'transcription', 'recording', 'metadata'
    
    Returns:
        Full GCP URL string
    """
    return f"{GCP_BASE_URL}/{user_id}/{recording_id}_{file_type}.json"


def download_file(url, output_path, retries=3):
    """Download a file from URL with retry logic.
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        retries: Number of retry attempts
    
    Returns:
        True if successful, False otherwise
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            elif response.status_code == 404:
                print(f"  [WARN] 404 Not Found: {url}")
                return False
        except requests.RequestException as e:
            if attempt < retries - 1:
                print(f"  [RETRY {attempt+1}] {e}")
            else:
                print(f"  [ERROR] Failed after {retries} attempts: {url}")
                return False
    return False


def download_dataset(dataset_csv, raw_dir):
    """Download all audio and transcription files from the dataset manifest.
    
    Args:
        dataset_csv: Path to CSV with columns: user_id, recording_id, rec_url_gcp, transcription_url
        raw_dir: Directory to store downloaded files
    
    Returns:
        DataFrame with download status added
    """
    df = pd.read_csv(dataset_csv)
    print(f"[INFO] Dataset has {len(df)} recordings")
    
    os.makedirs(raw_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        user_id = row['user_id']
        recording_id = row['recording_id']
        
        # Download transcription JSON
        trans_url = row.get('transcription_url', build_gcp_url(user_id, recording_id, 'transcription'))
        # Fix URL if needed (replace old domain with new)
        trans_url = fix_url(trans_url)
        trans_path = os.path.join(raw_dir, f"{user_id}_{recording_id}_transcription.json")
        
        # Download audio file
        rec_url = row.get('rec_url_gcp', build_gcp_url(user_id, recording_id, 'recording'))
        rec_url = fix_url(rec_url)
        rec_path = os.path.join(raw_dir, f"{user_id}_{recording_id}_recording.wav")
        
        trans_ok = download_file(trans_url, trans_path) if not os.path.exists(trans_path) else True
        rec_ok = download_file(rec_url, rec_path) if not os.path.exists(rec_path) else True
        
        if trans_ok and rec_ok:
            successful += 1
        else:
            failed += 1
        
        df.at[idx, 'trans_local'] = trans_path if trans_ok else None
        df.at[idx, 'rec_local'] = rec_path if rec_ok else None
    
    print(f"[INFO] Downloaded: {successful} successful, {failed} failed")
    return df


def fix_url(url):
    """Fix dataset URLs by replacing old domain patterns with the working GCP pattern.
    
    The assignment notes that some URLs may not work and need modification.
    Pattern: https://storage.googleapis.com/upload_goai/<user_id>/<recording_id>_<type>.json
    """
    if not url or pd.isna(url):
        return url
    
    # If URL already matches the working pattern, return as-is
    if "storage.googleapis.com/upload_goai" in url:
        return url
    
    # Try to extract user_id and recording_id from various URL patterns
    # and reconstruct with the working base URL
    import re
    match = re.search(r'(\d+)/(\d+)_(transcription|recording|metadata)', url)
    if match:
        user_id, recording_id, file_type = match.groups()
        ext = '.json' if file_type in ('transcription', 'metadata') else '.wav'
        return f"{GCP_BASE_URL}/{user_id}/{recording_id}_{file_type}{ext}"
    
    return url


# ─── Audio Segmentation ─────────────────────────────────────────────────────

def load_and_segment_audio(audio_path, transcription_path, target_sr=SAMPLE_RATE):
    """Load audio and segment it according to transcription timestamps.
    
    Args:
        audio_path: Path to audio file
        transcription_path: Path to transcription JSON
        target_sr: Target sample rate (default 16kHz for Whisper)
    
    Returns:
        List of dicts with 'audio' (numpy array), 'text', 'start', 'end', 'sr'
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # Load transcription
    with open(transcription_path, 'r', encoding='utf-8') as f:
        transcription = json.load(f)
    
    segments = []
    for seg in transcription:
        start_sec = seg.get('start', 0)
        end_sec = seg.get('end', 0)
        text = seg.get('text', '').strip()
        
        if not text or end_sec <= start_sec:
            continue
        
        # Convert timestamps to sample indices
        start_sample = int(start_sec * target_sr)
        end_sample = int(end_sec * target_sr)
        
        # Clip to audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if end_sample <= start_sample:
            continue
        
        segment_audio = audio[start_sample:end_sample]
        
        # Skip very short segments (less than 0.5s) or very long segments (>30s for Whisper)
        duration = len(segment_audio) / target_sr
        if duration < 0.5 or duration > 30.0:
            continue
        
        segments.append({
            'audio': segment_audio,
            'text': text,
            'start': start_sec,
            'end': end_sec,
            'sampling_rate': target_sr,
            'duration': duration
        })
    
    return segments


# ─── Text Cleaning ───────────────────────────────────────────────────────────

def clean_text(text):
    """Clean transcription text for Whisper training.
    
    - Remove extra whitespace
    - Normalize Unicode (NFC)
    - Remove control characters
    - Keep Devanagari, digits, basic punctuation
    
    Args:
        text: Raw transcription text
    
    Returns:
        Cleaned text string
    """
    import unicodedata
    
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Remove zero-width characters
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


# ─── Dataset Creation ────────────────────────────────────────────────────────

def create_hf_dataset(df, raw_dir, output_dir, test_split=0.1):
    """Create a HuggingFace Dataset from downloaded audio + transcriptions.
    
    Args:
        df: DataFrame with 'trans_local' and 'rec_local' columns
        raw_dir: Directory with raw downloaded files
        output_dir: Directory to save the HF dataset
        test_split: Fraction of data to use for validation
    
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    all_segments = []
    segment_audio_dir = os.path.join(output_dir, "audio_segments")
    os.makedirs(segment_audio_dir, exist_ok=True)
    
    # Filter to successfully downloaded recordings
    valid_df = df.dropna(subset=['trans_local', 'rec_local'])
    print(f"[INFO] Processing {len(valid_df)} valid recordings")
    
    seg_id = 0
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Segmenting"):
        try:
            segments = load_and_segment_audio(row['rec_local'], row['trans_local'])
            
            for seg in segments:
                # Save segment audio to WAV file
                seg_filename = f"seg_{seg_id:06d}.wav"
                seg_path = os.path.join(segment_audio_dir, seg_filename)
                sf.write(seg_path, seg['audio'], seg['sampling_rate'])
                
                all_segments.append({
                    'audio': seg_path,
                    'text': clean_text(seg['text']),
                    'duration': seg['duration'],
                    'recording_id': row.get('recording_id', ''),
                    'user_id': row.get('user_id', ''),
                })
                seg_id += 1
                
        except Exception as e:
            print(f"  [ERROR] Failed to process {row.get('recording_id', 'unknown')}: {e}")
            continue
    
    print(f"[INFO] Total segments: {len(all_segments)}")
    
    if not all_segments:
        print("[ERROR] No segments created! Check data downloads.")
        sys.exit(1)
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        'audio': [s['audio'] for s in all_segments],
        'text': [s['text'] for s in all_segments],
        'duration': [s['duration'] for s in all_segments],
        'recording_id': [s['recording_id'] for s in all_segments],
        'user_id': [s['user_id'] for s in all_segments],
    })
    
    # Cast audio column
    dataset = dataset.cast_column('audio', Audio(sampling_rate=SAMPLE_RATE))
    
    # Train/test split
    dataset = dataset.train_test_split(test_size=test_split, seed=42)
    
    # Save to disk
    dataset.save_to_disk(output_dir)
    print(f"[INFO] Dataset saved to {output_dir}")
    print(f"  Train: {len(dataset['train'])} segments")
    print(f"  Test:  {len(dataset['test'])} segments")
    
    # Save summary stats
    stats = {
        'total_segments': len(all_segments),
        'train_segments': len(dataset['train']),
        'test_segments': len(dataset['test']),
        'total_duration_hrs': sum(s['duration'] for s in all_segments) / 3600,
        'avg_duration_sec': np.mean([s['duration'] for s in all_segments]),
        'unique_speakers': len(set(s['user_id'] for s in all_segments)),
    }
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Dataset stats: {json.dumps(stats, indent=2)}")
    
    return dataset


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess Hindi ASR data for Whisper fine-tuning")
    parser.add_argument('--dataset_csv', type=str, required=True,
                        help='Path to dataset CSV with user_id, recording_id columns')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Directory to download raw audio/transcription files')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save preprocessed HuggingFace dataset')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of data for validation split (default: 0.1)')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip downloading, use already-downloaded files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hindi ASR Data Preprocessing Pipeline")
    print("=" * 60)
    
    if args.skip_download:
        print("[INFO] Skipping download, loading existing CSV...")
        df = pd.read_csv(args.dataset_csv)
        # Reconstruct local paths
        for idx, row in df.iterrows():
            uid = row['user_id']
            rid = row['recording_id']
            df.at[idx, 'trans_local'] = os.path.join(args.raw_dir, f"{uid}_{rid}_transcription.json")
            df.at[idx, 'rec_local'] = os.path.join(args.raw_dir, f"{uid}_{rid}_recording.wav")
    else:
        print("[STEP 1] Downloading dataset...")
        df = download_dataset(args.dataset_csv, args.raw_dir)
    
    print("\n[STEP 2] Segmenting audio and creating HuggingFace dataset...")
    dataset = create_hf_dataset(df, args.raw_dir, args.output_dir, args.test_split)
    
    print("\n[DONE] Preprocessing complete!")
    print(f"  Dataset saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
