"""
YouTube Transcript Collector

Collects transcripts from YouTube videos using youtube-transcript-api.
Supports manual captions, auto-generated, and multiple languages (Arabic, French, English).
"""

import logging
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


import re
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from stem import Signal
from stem.control import Controller

import yt_dlp

logger = logging.getLogger(__name__)

def renew_tor_identity(control_port=9151, password=None):
    """
    Request a new identity from Tor to rotate IP.
    Default port 9151 is for Tor Browser. Standalone Tor uses 9051.
    """
    try:
        with Controller.from_port(port=control_port) as controller:
            if password:
                controller.authenticate(password=password)
            else:
                controller.authenticate()  # Try cookie/empty auth
            
            controller.signal(Signal.NEWNYM)
            logger.info("🔄 Tor Identity Rotated (New IP requested)")
            time.sleep(5)  # Wait for new circuit
            return True
            
    except Exception as e:
        logger.warning(f"Failed to rotate Tor identity: {e}")
        logger.info("Tip: Ensure 'ControlPort 9151' is enabled in torrc if using standalone Tor.")
        return False
def clean_vtt_text(vtt_content: str) -> str:
    """Clean VTT subtitle content to extract text."""
    lines = vtt_content.splitlines()
    text_lines = []
    
    # Simple state machine or regex to filter header/timestamps
    # VTT format:
    # WEBVTT
    # 
    # 00:00:01.000 --> 00:00:04.000
    # Text here
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == 'WEBVTT' or line.startswith('NOTE') or '-->' in line:
            continue
        # Remove tags like <c.color> or <b>
        clean = re.sub(r'<[^>]+>', '', line)
        # Remove simple timestamps if they appear alone (rare in strict VTT but possible)
        if re.match(r'^\d{2}:\d{2}', clean):
            continue
        # Deduplicate sequential lines (common in auto-subs)
        if text_lines and text_lines[-1] == clean:
            continue
        text_lines.append(clean)
        
    return ' '.join(text_lines)

def get_best_transcript(video_id: str, proxies: Optional[dict] = None) -> Optional[dict]:
    """
    Get the best available transcript using yt-dlp (more robust against blocking).
    
    Priority:
    1. Arabic transcript (manual)
    2. French transcript (manual)
    3. English transcript (manual)
    4. Auto-generated (any of above)
    
    Args:
        video_id: YouTube video ID
        proxies: Optional dict of proxies (e.g. {'http': 'socks5://...', 'https': '...'})
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Configure yt-dlp
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    if proxies and 'https' in proxies:
        ydl_opts['proxy'] = proxies['https']
        
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Check manual subtitles first
            subs = info.get('subtitles', {})
            auto_subs = info.get('automatic_captions', {})
            
            found_sub = None
            is_generated = False
            lang_code = None
            
            # Check priorities
            for lang in ['ar', 'fr', 'en']:
                if lang in subs:
                    found_sub = subs[lang]
                    is_generated = False
                    lang_code = lang
                    break
            
            # Fallback to auto-subs
            if not found_sub:
                for lang in ['ar', 'fr', 'en']:
                    if lang in auto_subs:
                        found_sub = auto_subs[lang]
                        is_generated = True
                        lang_code = lang
                        break
                        
            if not found_sub and (subs or auto_subs):
                # Pick any
                if subs:
                    lang_code = list(subs.keys())[0]
                    found_sub = subs[lang_code]
                    is_generated = False
                elif auto_subs:
                    lang_code = list(auto_subs.keys())[0]
                    found_sub = auto_subs[lang_code]
                    is_generated = True
            
            if found_sub:
                # Find VTT or JSON3 format
                sub_url = None
                for fmt in found_sub:
                    if fmt['ext'] == 'vtt':
                        sub_url = fmt['url']
                        break
                    if fmt['ext'] == 'json3':
                         sub_url = fmt['url'] # We'd need json parser, VTT preferred for now
                
                if not sub_url and found_sub:
                     sub_url = found_sub[0]['url'] # Fallback
                
                if sub_url:
                    # Fetch content
                    # We utilize the same proxy configuration for fetching
                    http_client = requests.Session()
                    http_client.headers.update({
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    })
                    if proxies:
                        http_client.proxies.update(proxies)
                    
                    resp = http_client.get(sub_url, timeout=10)
                    resp.raise_for_status()
                    
                    text = clean_vtt_text(resp.text)
                    
                    return {
                        'video_id': video_id,
                        'transcript_text': text,
                        'transcript_language': lang_code,
                        'transcript_language_code': lang_code,
                        'is_generated': is_generated,
                        'is_translatable': True, 
                        'segment_count': len(text.split('.')), # Rough estimate
                        'transcript_available': True,
                    }
                    
        return None

    except Exception as e:
        logger.warning(f"Error fetching transcript for {video_id}: {e}")
        return None


class TranscriptCollector:
    """Collects transcripts for multiple videos with progress tracking and rate limiting."""
    
    def __init__(self, rate_limit_delay: float = 0.2, proxies: Optional[dict] = None):
        """
        Initialize the collector.
        
        Args:
            rate_limit_delay: Seconds to wait between API calls
            proxies: Optional proxy configuration for requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.proxies = proxies
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'manual': 0,
            'generated': 0,
        }
    
    def collect_transcripts(
        self,
        videos_df: pd.DataFrame,
        video_id_column: str = 'video_id',
        checkpoint_interval: int = 5,
        checkpoint_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Collect transcripts for all videos in a DataFrame.
        
        Args:
            videos_df: DataFrame with video IDs
            video_id_column: Name of the column containing video IDs
            checkpoint_interval: Save checkpoint every N videos
            checkpoint_path: Path to save checkpoints (optional)
            
        Returns:
            DataFrame with transcript data
        """
        results = []
        total = len(videos_df)
        self.stats['total'] = total
        
        # Resumability Logic
        processed_ids = set()
        if checkpoint_path and checkpoint_path.exists():
            try:
                existing_df = pd.read_csv(checkpoint_path)
                if video_id_column in existing_df.columns:
                    processed_ids = set(existing_df[video_id_column].unique())
                    results = existing_df.to_dict('records')
                    logger.info(f"Resuming from checkpoint: {len(results)} videos already processed.")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Also check final output path for existing data
        final_output_path = Path(str(checkpoint_path).replace('_checkpoint', ''))
        if final_output_path.exists():
             try:
                final_df = pd.read_csv(final_output_path)
                if video_id_column in final_df.columns:
                    final_ids = set(final_df[video_id_column].unique())
                    new_ids = final_ids - processed_ids
                    if new_ids:
                        processed_ids.update(new_ids)
                        results.extend(final_df[final_df[video_id_column].isin(new_ids)].to_dict('records'))
                        logger.info(f"Loaded {len(new_ids)} additional records from existing output file.")
             except Exception: pass

        
        logger.info(f"Starting transcript collection for {total} videos ({len(processed_ids)} already done)")
        
        for idx, row in videos_df.iterrows():
            video_id = row[video_id_column]
            
            # Skip if already processed
            if video_id in processed_ids:
                self.stats['success'] += 1 # Count as success for stats continuity? 
                # Or just skip stats? Let's just skip processing but count for progress
                continue

            # Get transcript
            
            # Get transcript
            transcript = get_best_transcript(video_id, proxies=self.proxies)
            
            if transcript:
                results.append(transcript)
                self.stats['success'] += 1
                self.stats['consecutive_failures'] = 0  # Reset on success
                if transcript['is_generated']:
                    self.stats['generated'] += 1
                else:
                    self.stats['manual'] += 1
            else:
                results.append({
                    'video_id': video_id,
                    'transcript_text': None,
                    'transcript_language': None,
                    'transcript_language_code': None,
                    'is_generated': None,
                    'is_translatable': None,
                    'segment_count': 0,
                    'transcript_available': False,
                })
                self.stats['failed'] += 1
                self.stats.setdefault('consecutive_failures', 0)
                self.stats['consecutive_failures'] += 1
            
            # Circuit breaker & Rotation Logic
            failures = self.stats.get('consecutive_failures', 0)
            
            # Try to rotate IP every 5 failures if using Tor
            if failures > 0 and failures % 5 == 0:
                logger.warning(f"⚠️ {failures} consecutive failures. Attempting to rotate Tor identity...")
                # Try Port 9151 (Browser) first, then 9051 (System)
                if not renew_tor_identity(9151):
                    renew_tor_identity(9051)
            
            # Hard Abort after 50 failures (despite rotations)
            if failures >= 50:
                logger.error("🛑 Aborting: 50 consecutive failures detected. Likely IP blocked by YouTube.")
                logger.error("Try using a VPN or waiting for a few hours.")
                break
            
            # Progress logging
            processed = idx + 1
            if processed % 5 == 0 or processed == total:
                coverage = (self.stats['success'] / processed) * 100
                logger.info(
                    f"Progress: {processed}/{total} "
                    f"({self.stats['success']} success, {self.stats['failed']} failed, "
                    f"{coverage:.1f}% coverage)"
                )
            
            # Checkpoint
            if checkpoint_path and processed % checkpoint_interval == 0:
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(checkpoint_path, index=False)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Rate limiting (increased to reduce ban risk)
            time.sleep(1.0)
        
        # Final stats
        logger.info(f"\nTranscript Collection Complete:")
        logger.info(f"  Total videos: {self.stats['total']}")
        logger.info(f"  With transcript: {self.stats['success']}")
        logger.info(f"    - Manual: {self.stats['manual']}")
        logger.info(f"    - Auto-generated: {self.stats['generated']}")
        logger.info(f"  Without transcript: {self.stats['failed']}")
        logger.info(f"  Coverage: {(self.stats['success']/max(1, idx+1))*100:.1f}%")
        
        return pd.DataFrame(results)


def collect_transcripts_cli(input_path: Path, output_path: Path, proxy: Optional[str] = None) -> int:
    """
    CLI entry point for transcript collection.
    
    Args:
        input_path: Path to CSV with video IDs
        output_path: Path to save transcripts
        proxy: Optional proxy URL (e.g. http://user:pass@host:port)
        
    Returns:
        Exit code (0 for success)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 60)
    logger.info("YouTube Transcript Collection Pipeline")
    logger.info("=" * 60)
    
    # Load videos
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    videos_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(videos_df)} videos from {input_path}")
    
    # Parse proxy
    proxies = None
    if proxy:
        proxies = {
            "http": proxy,
            "https": proxy,
        }
        logger.info(f"Using proxy: {proxy}")
    
    # Collect transcripts
    collector = TranscriptCollector(rate_limit_delay=1.0, proxies=proxies)
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.csv"
    
    transcripts_df = collector.collect_transcripts(
        videos_df,
        video_id_column='video_id',
        checkpoint_interval=5,
        checkpoint_path=checkpoint_path,
    )
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transcripts_df.to_csv(output_path, index=False)
    logger.info(f"Saved transcripts to: {output_path}")
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect YouTube transcripts")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with video IDs")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--proxy", type=str, help="Optional proxy URL")
    
    args = parser.parse_args()
    exit(collect_transcripts_cli(args.input, args.output, args.proxy))
