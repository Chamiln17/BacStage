"""
YouTube Transcript Collector

Collects transcripts from YouTube videos using youtube-transcript-api.
Supports manual captions, auto-generated, and multiple languages (Arabic, French, English).
"""

import logging
import time
import re
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from collections import deque
import threading

import pandas as pd
import requests
from stem import Signal
from stem.control import Controller

import yt_dlp

logger = logging.getLogger(__name__)

# ============================================================================
# SMART RETRY QUEUE
# ============================================================================

class SmartRetryQueue:
    """Intelligently retry failed videos after IP rotation."""
    
    def __init__(self, max_retries: int = 2):
        self.queue = deque()
        self.max_retries = max_retries
        self.skip_log = []
        self.lock = threading.Lock()
    
    def should_retry(self, video_id: str, error_type: str) -> bool:
        """Decide if failure should be queued for retry."""
        # Bot blocks are permanent - don't retry immediately, let rotation handle it
        # Actually, if we rotate, we CAN retry. 
        # But if it's "Sign in to view", that might be account specific? No, usually IP.
        
        # Permanent failures - Do not retry
        er_lower = error_type.lower()
        if 'no transcript' in er_lower:
            return False
            
        # If it's the HTML/JS error we added, assume it's a hard block for this IP/video combo and don't retry 
        # (or at least don't retry in immediate loop - rotation might handle it but user asked to fail directly)
        if 'html' in er_lower or 'js code' in er_lower:
             return False

        # Ensure we don't queue if it's already in queue (shouldn't happen with sequential logic but safe to check)
        existing = next((item for item in self.queue if item['video_id'] == video_id), None)
        
        if existing:
            if existing['attempts'] < self.max_retries:
                existing['attempts'] += 1
                existing['queued_at'] = datetime.now() # Reset timer to avoid immediate loop
                return True
            else:
                self.queue.remove(existing)
                self.skip_log.append({'video_id': video_id, 'reason': 'max_retries'})
                return False
        
        # New failure
        if 'bot' in error_type.lower() or 'sign in' in error_type.lower():
             # For bot errors, we might want to retry AFTER rotation.
             pass 

        self.queue.append({
            'video_id': video_id,
            'error_type': error_type,
            'attempts': 1,
            'queued_at': datetime.now()
        })
        return True

    def get_next_retry(self) -> Optional[str]:
        """Get next video to retry."""
        if not self.queue:
            return None
        
        # Look for ready item
        for item in self.queue:
            time_since_queued = (datetime.now() - item['queued_at']).total_seconds()
            if time_since_queued > 5: # 5 seconds delay
                return item['video_id']
        return None
    
    def mark_success(self, video_id: str):
        """Remove from queue on successful retry."""
        self.queue = deque(item for item in self.queue if item['video_id'] != video_id)
        
    def get_stats(self) -> dict:
         return {'in_queue': len(self.queue), 'skipped': len(self.skip_log)}


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
        # logger.info("Tip: Ensure 'ControlPort 9151' is enabled in torrc if using standalone Tor.")
        return False

def clean_vtt_text(vtt_content: str) -> str:
    """Clean VTT subtitle content to extract text."""
    lines = vtt_content.splitlines()
    text_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == 'WEBVTT' or line.startswith('NOTE') or '-->' in line:
            continue
        # Safety check for code injection
        if line.strip().startswith('window.') or 'var ' in line or 'function' in line or '{' in line:
            continue
        clean = re.sub(r'<[^>]+>', '', line)
        if re.match(r'^\d{2}:\d{2}', clean):
            continue
        if text_lines and text_lines[-1] == clean:
            continue
        text_lines.append(clean)
        
    return ' '.join(text_lines)

def get_best_transcript(video_id: str, proxies: Optional[dict] = None) -> Optional[dict]:
    """
    Get the best available transcript using yt-dlp.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
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
            # Add increased timeout
            ydl_opts['socket_timeout'] = 30
            
            info = ydl.extract_info(url, download=False)
            
            subs = info.get('subtitles', {})
            auto_subs = info.get('automatic_captions', {})
            
            found_sub = None
            is_generated = False
            lang_code = None
            
            # Check priorities: ar > fr > en
            for lang in ['ar', 'fr', 'en']:
                if lang in subs:
                    found_sub = subs[lang]
                    is_generated = False
                    lang_code = lang
                    break
            
            if not found_sub:
                for lang in ['ar', 'fr', 'en']:
                    if lang in auto_subs:
                        found_sub = auto_subs[lang]
                        is_generated = True
                        lang_code = lang
                        break
                        
            if not found_sub and (subs or auto_subs):
                if subs:
                    lang_code = list(subs.keys())[0]
                    found_sub = subs[lang_code]
                    is_generated = False
                elif auto_subs:
                    lang_code = list(auto_subs.keys())[0]
                    found_sub = auto_subs[lang_code]
                    is_generated = True
            
            if found_sub:
                sub_url = None
                for fmt in found_sub:
                    if fmt['ext'] == 'vtt':
                        sub_url = fmt['url']
                        break
                    if fmt['ext'] == 'json3':
                         sub_url = fmt['url']
                
                if not sub_url and found_sub:
                     sub_url = found_sub[0]['url']
                
                if sub_url:
                    http_client = requests.Session()
                    http_client.headers.update({
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    })
                    if proxies:
                        http_client.proxies.update(proxies)
                    
                    resp = http_client.get(sub_url, timeout=10)
                    resp.raise_for_status()
                    
                    # VALIDATION: Check Content-Type and Content
                    content_type = resp.headers.get('Content-Type', '').lower()
                    if 'html' in content_type:
                        logger.warning(f"Rejecting HTML content for {video_id} (Type: {content_type})")
                        raise ValueError("Youtube returned HTML instead of transcript (likely blocked)")
                        
                    content_snippet = resp.text[:1000].strip()
                    if content_snippet.startswith('<!DOCTYPE') or '<html' in content_snippet.lower() or 'window.WIZ_global_data' in content_snippet:
                        logger.warning(f"Rejecting HTML/JS content for {video_id}")
                        raise ValueError("Youtube returned HTML/JS instead of transcript")

                    text = clean_vtt_text(resp.text)
                    
                    # Secondary validation of result text
                    if len(text) > 50 and ('window.' in text or 'ytcfg' in text):
                         logger.warning(f"Cleaned text still looks like code for {video_id}")
                         raise ValueError("Transcript validation failed (JS code detected)")

                    return {
                        'video_id': video_id,
                        'transcript_text': text,
                        'transcript_language': lang_code,
                        'transcript_language_code': lang_code,
                        'is_generated': is_generated,
                        'is_translatable': True, 
                        'segment_count': len(text.split('.')),
                        'transcript_available': True,
                        'failure_reason': None
                    }
                    
        return None

    except Exception as e:
        logger.warning(f"Error fetching transcript for {video_id}: {e}")
        return {'error': str(e)}


class TranscriptCollector:
    """Collects transcripts for multiple videos with progress tracking and rate limiting."""
    
    def __init__(self, rate_limit_delay: float = 0.2, proxies: Optional[dict] = None):
        self.rate_limit_delay = rate_limit_delay
        self.proxies = proxies
        self.stats = {
            'total': 0, 'success': 0, 'failed': 0,
            'manual': 0, 'generated': 0, 'retried': 0,
            'new_processed': 0, 'retry_processed': 0
        }
        self.retry_queue = SmartRetryQueue(max_retries=2)
    
    def collect_transcripts(
        self,
        videos_df: pd.DataFrame,
        video_id_column: str = 'video_id',
        checkpoint_interval: int = 5,
        checkpoint_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        
        results = []
        total = len(videos_df)
        self.stats['total'] = total
        
        # Resumability
        processed_ids = set()
        if checkpoint_path and checkpoint_path.exists():
            try:
                existing_df = pd.read_csv(checkpoint_path)
                if video_id_column in existing_df.columns:
                    # Enforce string type for IDs to ensure consistent matching
                    existing_df[video_id_column] = existing_df[video_id_column].astype(str)
                    
                    # Deduplicate: keep last occurrence (or first? usually first is better if we just want to keep valid data, but last might be more recent. Let's assume generic deduplication)
                    # Actually, we want to drop duplicates based on video_id.
                    existing_df = existing_df.drop_duplicates(subset=[video_id_column])
                    
                    processed_ids = set(existing_df[video_id_column].unique())
                    results = existing_df.to_dict('records')
                    logger.info(f"Resuming from checkpoint: {len(results)} videos already processed (deduplicated).")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        logger.info(f"Starting transcript collection for {total} videos ({len(processed_ids)} already done)")
        
        # WORK LOOP
        iterator = videos_df.iterrows()
        
        for idx, row in iterator:
            video_id = str(row[video_id_column]) # Ensure string comparison
            
            # Skip if processed
            if video_id in processed_ids:
                continue

            # 1. Process Retries First (if any ready)
            retry_conf = self.retry_queue.get_next_retry()
            if retry_conf:
                 self._process_one(retry_conf, results, is_retry=True)
                 self.stats['retry_processed'] += 1
                 # Note: retries don't need to check processed_ids because they are explicitly managed
            
            # 2. Process Current
            self._process_one(video_id, results, is_retry=False)
            self.stats['new_processed'] += 1
            
            # Add to processed set immediately to prevent re-processing in this run
            # We do this regardless of success/fail because _process_one appends to results in both cases (unless it's a temp retry failure)
            # If it was queued for retry, it's NOT in results yet, but we have "processed" this iteration of the main loop.
            # However, if we mark it as processed, the main loop won't pick it up again. 
            # If it's in the retry queue, that's fine, the retry system handles it.
            processed_ids.add(video_id)
            
            # Circuit breaker & Rotation
            failures = self.stats.get('consecutive_failures', 0)
            if failures > 0 and failures % 5 == 0:
                logger.warning(f"⚠️ {failures} consecutive failures. Attempting to rotate Tor identity...")
                if self.proxies and ('9150' in str(self.proxies) or '9151' in str(self.proxies)):
                    if not renew_tor_identity(9151):
                        renew_tor_identity(9051)
                    self.stats['consecutive_failures'] = 0 # Reset after rotation try
                else:
                    time.sleep(10)

            if failures >= 50:
                logger.error("🛑 Aborting: 50 consecutive failures detected.")
                break
            
            # Progress & Checkpoint
            current_count = len(results)
            # Log every 5 videos processed/iterated
            if (idx + 1) % 5 == 0: 
                 processed_so_far = self.stats['success'] + self.stats['failed']
                 rate = (self.stats['success'] / processed_so_far * 100) if processed_so_far > 0 else 0.0
                 queue_stats = self.retry_queue.get_stats()
                 logger.info(
                    f"Progress: {idx+1}/{total} | "
                    f"New: {self.stats['new_processed']} | Retries: {self.stats['retry_processed']} | "
                    f"Results: {self.stats['success']} ✓, {self.stats['failed']} ✗, {self.stats['retried']} fixed | "
                    f"Queue: {queue_stats['in_queue']} | Success rate: {rate:.1f}%"
                 )
                 
                 if checkpoint_path and results:
                    try:
                        checkpoint_df = pd.DataFrame(results)
                        # Double check deduplication on save just in case
                        checkpoint_df = checkpoint_df.drop_duplicates(subset=[video_id_column])
                        checkpoint_df.to_csv(checkpoint_path, index=False)
                    except Exception: pass
            
            time.sleep(1.0)
            
        return pd.DataFrame(results)

    def _process_one(self, video_id, results, is_retry=False):
        # Handle both string ID or retry dict
        vid = video_id if isinstance(video_id, str) else video_id # In retry it might be just ID str from get_next_retry? 
        # Checking get_next_retry return type: returns video_id string.
        
        transcript_data = get_best_transcript(vid, proxies=self.proxies)
        
        if transcript_data and 'transcript_text' in transcript_data:
            # Success
            results.append(transcript_data)
            self.stats['success'] += 1
            self.stats['consecutive_failures'] = 0
            if transcript_data['is_generated']:
                self.stats['generated'] += 1
            else:
                self.stats['manual'] += 1
            
            if is_retry:
                self.stats['retried'] += 1
                self.retry_queue.mark_success(vid)
                logger.info(f"✅ Retry success: {vid}")
        else:
            # Failed
            error_reason = transcript_data.get('error', 'Unknown') if transcript_data else 'No transcript'
            
            # Add to retry queue?
            should_retry = self.retry_queue.should_retry(vid, error_reason)
            
            if should_retry:
                logger.info(f"♻️ Queued for retry: {vid} (Reason: {error_reason})")
                # Do NOT add to results yet
            else:
                # Permanent failure
                results.append({
                    'video_id': vid,
                    'transcript_text': None,
                    'transcript_language': None,
                    'transcript_language_code': None,
                    'is_generated': None,
                    'is_translatable': None,
                    'segment_count': 0,
                    'transcript_available': False,
                    'failure_reason': error_reason
                })
                self.stats['failed'] += 1
                if not is_retry:
                    # Only count consecutive failures for actual errors (not empty transcripts)
                    if error_reason and 'no transcript' not in str(error_reason).lower():
                        self.stats.setdefault('consecutive_failures', 0)
                        self.stats['consecutive_failures'] += 1
                    else:
                         # Reset consecutive failures if it's just a data issue (we are successfully talking to YT)
                         self.stats['consecutive_failures'] = 0

def collect_transcripts_cli(input_path: Path, output_path: Path, proxy: Optional[str] = None, workers: int = 1) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 60)
    logger.info("YouTube Transcript Collection Pipeline (Sequential)")
    logger.info("=" * 60)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    videos_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(videos_df)} videos from {input_path}")
    
    proxies = None
    if proxy:
        proxies = {"http": proxy, "https": proxy}
        logger.info(f"Using proxy: {proxy}")
    
    collector = TranscriptCollector(rate_limit_delay=1.0, proxies=proxies)
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.csv"
    
    transcripts_df = collector.collect_transcripts(
        videos_df,
        video_id_column='video_id',
        checkpoint_interval=5,
        checkpoint_path=checkpoint_path,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transcripts_df.to_csv(output_path, index=False)
    logger.info(f"Saved transcripts to: {output_path}")
    return 0

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Collect YouTube transcripts")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with video IDs")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--proxy", type=str, help="Optional proxy URL")
    parser.add_argument("--workers", type=int, default=1, help="Ignored in sequential mode")
    
    args = parser.parse_args()
    sys.exit(collect_transcripts_cli(args.input, args.output, args.proxy, args.workers))
