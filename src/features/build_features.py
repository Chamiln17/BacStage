"""
CLI script for building features from raw video metadata.

Usage:
    uv run python -m src.features.build_features --input data/raw/videos_metadata.csv --output data/processed/videos_engineered.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.features.engineer import VideoFeatureEngineer

# Setup logging (console only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Fix console encoding for Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    """Main entry point for feature engineering CLI."""
    parser = argparse.ArgumentParser(
        description="Build features from raw video metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run python -m src.features.build_features --input data/raw/videos_metadata.csv

  # Custom output path
  uv run python -m src.features.build_features \\
      --input data/raw/videos_metadata.csv \\
      --output data/processed/features.csv
        """,
    )

    parser.add_argument(
        "--input", type=Path, required=True, help="Path to raw video metadata CSV file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/videos_engineered.csv"),
        help="Output path for engineered features (default: data/processed/videos_engineered.csv)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)

    # Load raw data
    try:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")

        logger.info(f"Loading raw data from {args.input}")
        videos_df = pd.read_csv(args.input)
        logger.info(
            f"Loaded {len(videos_df)} videos with {len(videos_df.columns)} columns"
        )

    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        sys.exit(1)

    # Engineer features
    try:
        engineer = VideoFeatureEngineer()
        videos_engineered = engineer.fit_transform(videos_df)
        videos_final = engineer.select_features(videos_engineered, include_target=True)

        logger.info(f"Feature engineering complete: {videos_final.shape}")

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        sys.exit(1)

    # Save results
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        videos_final.to_csv(args.output, index=False)

        logger.info(f"✓ Saved engineered features to: {args.output}")
        logger.info(f"Final dataset shape: {videos_final.shape}")
        logger.info(f"Features: {', '.join(videos_final.columns.tolist())}")

        # Summary statistics
        logger.info("\nDataset Summary:")
        logger.info(
            f"  Date range: {videos_final['publish_date'].min()} to {videos_final['publish_date'].max()}"
        )
        logger.info(f"  Channels: {videos_final['channel_id'].nunique()}")
        logger.info(f"  Subjects: {videos_final['subject'].value_counts().to_dict()}")
        logger.info(
            f"  Engagement: {videos_final['engagement_category'].value_counts().to_dict()}"
        )

    except Exception as e:
        logger.error(f"Error saving output: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
