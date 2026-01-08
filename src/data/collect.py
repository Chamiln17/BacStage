"""
CLI script for collecting YouTube video metadata.

Usage:
    uv run python -m src.data.collect --channels data/raw/channels.csv --output data/raw/videos_metadata.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data.youtube_collector import YouTubeCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_collection.log')
    ]
)
logger = logging.getLogger(__name__)


def get_api_key() -> str:
    """
    Retrieve YouTube API key from environment variable.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not found in environment
    """
    load_dotenv()
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        raise ValueError(
            "YOUTUBE_API_KEY not found or invalid. "
            "Please set it in your .env file. "
            "See .env.example for template."
        )
    
    return api_key


def main() -> None:
    """Main entry point for data collection CLI."""
    parser = argparse.ArgumentParser(
        description='Extract raw video metadata from YouTube channels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run python -m src.data.collect --channels data/raw/channels.csv
  
  # Limit videos per channel
  uv run python -m src.data.collect --channels data/raw/channels.csv --max-videos 50
  
  # Set quota limit
  uv run python -m src.data.collect --channels data/raw/channels.csv --max-quota 5000
        """
    )
    
    parser.add_argument(
        '--channels',
        type=Path,
        required=True,
        help='Path to CSV file with channel data (must have channel_id column)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/raw/videos_metadata.csv'),
        help='Output CSV file path (default: data/raw/videos_metadata.csv)'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum videos to collect per channel (default: all)'
    )
    
    parser.add_argument(
        '--max-quota',
        type=int,
        default=8000,
        help='Maximum API quota units to use (default: 8000)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("YouTube Data Collection Pipeline")
    logger.info("="*60)
    
    # Get API key
    try:
        api_key = get_api_key()
        logger.info("API key loaded successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Initialize collector
    try:
        collector = YouTubeCollector(api_key)
    except Exception as e:
        logger.error(f"Failed to initialize YouTube collector: {e}")
        sys.exit(1)
    
    # Load channels
    try:
        if not args.channels.exists():
            raise FileNotFoundError(f"Channel file not found: {args.channels}")
        
        channels_df = pd.read_csv(args.channels)
        logger.info(f"Loaded {len(channels_df)} channels from {args.channels}")
        
        if 'channel_id' not in channels_df.columns:
            raise ValueError("CSV must contain 'channel_id' column")
            
    except Exception as e:
        logger.error(f"Error loading channels: {e}")
        sys.exit(1)
    
    # Collect data
    try:
        videos_df = collector.collect_from_channels(
            channels_df,
            max_videos_per_channel=args.max_videos,
            max_quota=args.max_quota
        )
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        sys.exit(1)
    
    # Save results
    if not videos_df.empty:
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        videos_df.to_csv(args.output, index=False)
        logger.info(f"✓ Saved {len(videos_df)} videos to: {args.output}")
        logger.info(f"Dataset shape: {videos_df.shape}")
        logger.info(f"Columns: {', '.join(videos_df.columns.tolist())}")
        logger.info(f"Date range: {videos_df['publish_date'].min()} to {videos_df['publish_date'].max()}")
    else:
        logger.warning("No data collected. Check logs for errors.")
        sys.exit(1)


if __name__ == '__main__':
    main()
