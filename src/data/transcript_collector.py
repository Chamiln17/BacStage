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

def get_current_ip(proxies: Optional[dict] = None) -> Optional[str]:
    """Get current public IP address."""
    try:
        resp = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("ip")
    except Exception:
        pass
    return None

# ============================================================================
# TIER 3: SMART RETRY QUEUE
# ============================================================================

from collections import deque
from datetime import datetime
import threading

class SmartRetryQueue:
    """Intelligently retry failed videos after IP rotation."""
    
    def __init__(self, max_retries: int = 2):
        self.queue = deque()
        self.max_retries = max_retries
        self.skip_log = []
        self.lock = threading.Lock()
    
    def should_retry(self, video_id: str, error_type: str) -> bool:
        """Decide if failure should be queued for retry."""
        # Bot blocks are permanent - don't retry
        if 'bot' in error_type.lower() or 'sign in' in error_type.lower():
            with self.lock:
                self.skip_log.append({'video_id': video_id, 'reason': 'bot_blocked'})
            return False
        
        # Network/transient errors - queue for retry
        with self.lock:
            existing = next(
                (item for item in self.queue if item['video_id'] == video_id),
                None
            )
            
            if existing is None:
                self.queue.append({
                    'video_id': video_id,
                    'error_type': error_type,
                    'attempts': 1,
                    'queued_at': datetime.now()
                })
                return True
            elif existing['attempts'] < self.max_retries:
                existing['attempts'] += 1
                return True
            else:
                # Max retries reached - permanent skip
                self.queue.remove(existing)
                self.skip_log.append({'video_id': video_id, 'reason': 'max_retries'})
                return False
    
    def get_next_retry(self) -> Optional[str]:
        """Get next video to retry (if queue not empty)."""
        with self.lock:
            if not self.queue:
                return None
            
            # Get oldest item
            oldest = self.queue[0]
            # Only retry if queued for at least 10 seconds (let IP settle)
            time_since_queued = (datetime.now() - oldest['queued_at']).total_seconds()
            if time_since_queued > 10:
                return oldest['video_id']
            return None
    
    def mark_success(self, video_id: str):
        """Remove from queue on successful retry."""
        with self.lock:
            self.queue = deque(
                item for item in self.queue if item['video_id'] != video_id
            )
    
    def get_stats(self) -> dict:
        """Get retry queue statistics."""
        with self.lock:
            return {
                'in_queue': len(self.queue),
                'skipped': len(self.skip_log),
                'bot_blocked': len([s for s in self.skip_log if s['reason'] == 'bot_blocked']),
                'max_retries': len([s for s in self.skip_log if s['reason'] == 'max_retries'])
            }

def renew_tor_identity(control_port: int = 9151, password: Optional[str] = None, proxies: Optional[dict] = None) -> bool:
    """
    Request a new identity from Tor to rotate IP.
    """
    try:
        # Check old IP
        old_ip = get_current_ip(proxies)
        
        with Controller.from_port(port=control_port) as controller:
            if password:
                controller.authenticate(password=password)
            else:
                controller.authenticate()
            
            controller.signal(Signal.NEWNYM)
            
        logger.info("🔄 Signal sent to Tor Controller. Waiting for circuit...")
        time.sleep(10)  # Wait longer for circuit to build
        
        # Verify new IP
        new_ip = get_current_ip(proxies)
        
        if new_ip and old_ip and new_ip != old_ip:
            logger.info(f"✅ Tor Identity Rotated: {old_ip} -> {new_ip}")
            return True
        elif new_ip:
             logger.warning(f"⚠️ IP Verification inconclusive (IP might be same): {new_ip}")
             return True # Assume success if we got an IP back, might just be same exit node
        else:
            logger.warning("⚠️ Could not verify new IP (check connection).")
            return True # Proceed anyway
            
    except Exception as e:
        logger.warning(f"Failed to rotate Tor identity: {e}")
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

def get_best_transcript(video_id: str, proxies: Optional[dict] = None) -> tuple[Optional[dict], Optional[str]]:
    """
    Get the best available transcript using yt-dlp (more robust against blocking).
    
    Returns:
        tuple: (transcript_dict, error_message)
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
                        'failure_reason': None
                    }, None
                    
        return None, "No transcript found"

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Error fetching transcript for {video_id}: {error_msg}")
        return None, error_msg


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
            'retried': 0,  # Track retry successes
        }
        
        # Initialize Smart Retry Queue (keep this - it's useful!)
        self.retry_queue = SmartRetryQueue(max_retries=2)
    
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
                existing_df = pd.read_csv(checkpoint_path)  # <<< FIX: Load the checkpoint file
                if video_id_column in existing_df.columns:
                    # SMART RESUMABILITY:
                    # 1. Successful transcripts -> Processed (Keep)
                    # 2. Hard Failures (Bot Blocked) -> Processed (Keep, don't retry infinite loop)
                    # 3. Soft Failures (Network/Other) -> Retry (Drop from processed)
                    
                    if 'transcript_available' in existing_df.columns:
                        # Identify records to keep
                        # Keep if Success OR (Failed AND Reason is Bot/Sign-in)
                        
                        # Normalize failure reason
                        if 'failure_reason' not in existing_df.columns:
                            existing_df['failure_reason'] = None
                            
                        # Condition: Success
                        cond_success = existing_df['transcript_available'] == True
                        
                        # Condition: Bot Blocked (Soft-skip)
                        # We use a marker 'bot_blocked' or check string
                        cond_bot = existing_df['failure_reason'].astype(str).str.contains('bot|Sign in', case=False, na=False)
                        
                        # Records to keep as "Done"
                        keep_mask = cond_success | cond_bot
                        done_df = existing_df[keep_mask]
                        
                        processed_ids = set(done_df[video_id_column].unique())
                        results = done_df.to_dict('records')
                        
                        retried_count = len(existing_df) - len(done_df)
                        logger.info(f"Resuming: {len(done_df)} processed (success/skipped), retrying {retried_count} failed records.")
                    else:
                        # Fallback
                        processed_ids = set(existing_df[video_id_column].unique())
                        results = existing_df.to_dict('records')
                        logger.info(f"Resuming from checkpoint: {len(results)} videos collected.")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Also check final output path for existing data
        final_output_path = Path(str(checkpoint_path).replace('_checkpoint', ''))
        if final_output_path.exists():
             try:
                final_df = pd.read_csv(final_output_path)
                if video_id_column in final_df.columns:
                    # Logic: Only add if successful (Final output usually only has successes, but just in case)
                    if 'transcript_available' in final_df.columns:
                         final_df = final_df[final_df['transcript_available'] == True]
                    
                    final_ids = set(final_df[video_id_column].unique())
                    new_ids = final_ids - processed_ids
                    if new_ids:
                        processed_ids.update(new_ids)
                        results.extend(final_df[final_df[video_id_column].isin(new_ids)].to_dict('records'))
                        logger.info(f"Loaded {len(new_ids)} additional successful records from existing output file.")
             except Exception: pass

        
        # Index-Based Resumability (Strict Resume)
        start_index = 0
        if processed_ids:
            # Find the rows in videos_df that are in processed_ids
            # We want to restart AFTER the LAST processed video ID found in the input list
            # to strictly "start where left off" and ignore previous gaps.
            mask = videos_df[video_id_column].isin(processed_ids)
            if mask.any():
                last_match_idx = mask[mask].index[-1]
                # Assuming default RangeIndex. If not, we might need verify.
                # videos_df.index usually is RangeIndex(0, N) unless set otherwise.
                # Let's rely on row iteration order matching the input CSV order.
                
                # We simply want to skip 'N' videos where N is the index of the last processed one.
                # If indices are not 0..N, this logic needs care. 
                # Let's assume standard pd.read_csv logic where index is row number.
                start_index = last_match_idx + 1
                logger.info(f"⏭️ Strict Resumability: Skipping to index {start_index} (ignoring gaps before it).")

        logger.info(f"Starting transcript collection for {total} videos (starting at index {start_index})")
        
        for idx, row in videos_df.iterrows():
            # Strict skip
            # Note: iterrows yields independent index. If df was filtered/sorted, this might mismatch position.
            # But here we assume input_path -> read_csv -> pure RangeIndex.
            if idx < start_index:
                 # Count stats for progress tracking? 
                 # Or just ignore from stats? 
                 # Let's count them as 'skipped' implicitly or process total based on active range
                 continue
            
            video_id = row[video_id_column]
            logger.info(f"Processing video {idx}/{total}: {video_id}")
            
            # TIER 3: Check if there's a retry in queue (process after IP rotation)
            retry_vid = self.retry_queue.get_next_retry()
            if retry_vid:
                video_id = retry_vid
                self.stats['retried'] += 1
                logger.info(f"♻️ Retrying queued video: {video_id}")
            
            # Get transcript
            transcript, error_msg = get_best_transcript(video_id, proxies=self.proxies)
            
            if transcript:
                # Success
                results.append(transcript)
                self.stats['success'] += 1
                self.stats['consecutive_failures'] = 0  # Reset on success
                
                # Mark success in retry queue (if it was a retry)
                self.retry_queue.mark_success(video_id)
                
                if transcript['is_generated']:
                    self.stats['generated'] += 1
                else:
                    self.stats['manual'] += 1
            else:
                # Handle Failure - TIER 3: Smart Retry Logic
                is_bot_block = error_msg and ('Sign in' in error_msg or 'bot' in error_msg.lower())
                
                failure_reason = 'bot_blocked' if is_bot_block else 'error'
                if error_msg:
                    failure_reason += f": {error_msg[:50]}"
                
                # Decide: Queue for retry or skip permanently
                should_queue = self.retry_queue.should_retry(video_id, failure_reason)
                
                if should_queue:
                    logger.debug(f"📋 Queued for retry: {video_id}")
                else:
                    # Permanent failure - add to results as failed
                    results.append({
                        'video_id': video_id,
                        'transcript_text': None,
                        'transcript_language': None,
                        'transcript_language_code': None,
                        'is_generated': None,
                        'is_translatable': None,
                        'segment_count': 0,
                        'transcript_available': False,
                        'failure_reason': failure_reason
                    })
                
                self.stats['failed'] += 1
                
                if is_bot_block:
                    logger.warning(f"🤖 Bot detected for {video_id}. Marking as skipped and forcing rotation.")
                    self.stats['consecutive_failures'] = 5 
                else:
                    self.stats.setdefault('consecutive_failures', 0)
                    self.stats['consecutive_failures'] += 1
                    
            # Circuit breaker & Rotation Logic
            failures = self.stats.get('consecutive_failures', 0)
            
            # Try to rotate IP every 5 failures if using Tor
            if failures > 0 and failures % 5 == 0:
                logger.warning(f"⚠️ {failures} consecutive failures. Attempting to rotate Tor identity...")
                
                # Check if using Tor proxy
                is_tor = self.proxies and any('9150' in p or '9151' in p or '9050' in p for p in self.proxies.values())
                
                if is_tor:
                    # Try Port 9151 (Browser) first, then 9051 (System)
                    rotated = renew_tor_identity(9151, proxies=self.proxies)
                    if not rotated:
                        rotated = renew_tor_identity(9051, proxies=self.proxies)
                    
                    if rotated:
                        logger.info("✅ IP Rotated successfully. Resetting failure counter.")
                        self.stats['consecutive_failures'] = 0
                else:
                    logger.warning("⚠️ Not using Tor proxy, cannot rotate IP automatically.")
                    time.sleep(30) # Wait longer if we can't rotate
            
            # Hard Abort after 50 failures (despite rotations)
            # Re-check failures after potential reset above
            failures = self.stats.get('consecutive_failures', 0)
            if failures >= 50:
                logger.error("🛑 Aborting: 50 consecutive failures detected on same IP (or rotation failed).")
                logger.error("Likely IP blocked by YouTube or Proxy issues.")
                raise RuntimeError("Too many consecutive failures")
            
            # Progress logging
            processed = idx + 1
            if processed % 5 == 0 or processed == total:
                # Avoid div by zero in coverage calc if skipping
                current_session_processed = processed - start_index
                coverage = 0.0
                if current_session_processed > 0:
                   # This is tricky because stats include only this session
                   pass
                
                logger.info(
                    f"Progress: {processed}/{total} "
                    f"({self.stats['success']} success, {self.stats['failed']} failed)"
                )
            
            # Checkpoint
            if checkpoint_path and processed % checkpoint_interval == 0:
                try:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_df.to_csv(checkpoint_path, index=False)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                except OSError as e:
                     logger.warning(f"⚠️ Failed to save checkpoint (File Locked?): {e}")
            
            # Rate limiting (increased to reduce ban risk)
            time.sleep(1.0)
        
        # Final stats
        retry_stats = self.retry_queue.get_stats()
        
        logger.info(f"\nTranscript Collection Complete:")
        logger.info(f"  Total videos: {self.stats['total']}")
        logger.info(f"  With transcript: {self.stats['success']}")
        logger.info(f"    - Manual: {self.stats['manual']}")
        logger.info(f"    - Auto-generated: {self.stats['generated']}")
        logger.info(f"    - Retry successes: {self.stats['retried']}")
        logger.info(f"  Without transcript: {self.stats['failed']}")
        logger.info(f"  Retry Queue:")
        logger.info(f"    - Still queued: {retry_stats['in_queue']}")
        logger.info(f"    - Bot-blocked (skipped): {retry_stats['bot_blocked']}")
        logger.info(f"    - Max retries (skipped): {retry_stats['max_retries']}")
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
    
    try:
        transcripts_df = collector.collect_transcripts(
            videos_df,
            video_id_column='video_id',
            checkpoint_interval=5,
            checkpoint_path=checkpoint_path,
        )
    except RuntimeError as e:
        logger.error(f"Pipeline failed: {e}")
        return 1 # Return error code to trigger restart in keep_alive
    
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
