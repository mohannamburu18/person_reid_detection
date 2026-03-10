#!/usr/bin/env python3
"""
Person Re-Identification Pipeline for Railway Surveillance
Main Entry Point

Usage:
    python main.py              # Run ReID pipeline with default settings
    python main.py --help       # Show all options
"""

import argparse
from inference.run_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description='Person Re-Identification Pipeline for Multi-Camera Surveillance'
    )
    
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Convert Market-1501 images to fake CCTV videos first'
    )
    
    args = parser.parse_args()
    
    if args.prepare_data:
        print("[INFO] Preparing dataset...")
        from data_prep.images_to_video import create_fake_videos
        create_fake_videos()
        print("\n[INFO] Dataset preparation complete!")
        print("[INFO] Now running ReID pipeline...\n")
    
    # Run the main pipeline
    run_pipeline()

if __name__ == "__main__":
    main()
