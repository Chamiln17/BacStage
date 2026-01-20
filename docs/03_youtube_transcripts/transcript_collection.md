# Technical Report: Robust YouTube Transcript Collection Pipeline

## 1. Overview
This document details the architecture and operational logic of the transcript collection pipeline designed for the "Bac 2025 Analysis" project. The system is engineered to collect transcripts from ~10,000 YouTube videos while mitigating aggressive IP-based rate limiting and bot detection mechanisms employed by YouTube.

## 2. Architecture & Components

### 2.1 Core Extraction (`src/data/transcript_collector.py`)
- **Library**: utilized `yt-dlp` instead of `youtube-transcript-api` for superior resistance to bot detection and better maintenance.
- **Protocol**: `SOCKS5h` proxying ensures both HTTP traffic and DNS resolution occur through the Tor network, preventing DNS leak-based blocking.
- **Prioritization**:
    1.  Manual Arabic/French/English captions (High Quality).
    2.  Auto-generated captions (Fallback).

### 2.2 Network Layer (Tor Anonymity Network)
- **Proxy**: Traffic is routed through a local Tor SOCKS proxy (`127.0.0.1:9150` for Tor Browser).
- **Identity Rotation**: The script interfaces with the Tor Control Port (`9151`) using the `stem` library.
- **Rotation Logic**:
    - Triggered automatically after **5 consecutive failures** or immediately upon **Bot Detection**.
    - Sends `NEWNYM` signal to the Tor controller.
    - Verifies IP change via external API (`api.ipify.org`) before resuming.

### 2.3 Reliability Wrappers (`keep_alive.ps1`)
- **Process Supervision**: A PowerShell wrapper monitors the Python process.
- **Auto-Restart**: If the Python script crashes (Exit Code 1), it is automatically restarted after a 60-second cooldown.
- **Unattended Operation**: Designed to run 24/7 without human intervention.

---

## 3. Operational Logic

### 3.1 Smart Resumability
To handle frequent interruptions and blocks, the system uses a robust state-management strategy:
1.  **Strict Indexing**: On startup, it identifies the *index* of the last processed video in the input CSV.
2.  **Forward-Only Processing**: It skips all videos prior to this index, ensuring it never gets stuck in a loop retrying failed videos from the beginning of the list.
3.  **Atomic Checkpoints**: Progress is saved to `transcripts_checkpoint.csv` every 5 videos.

### 3.2 Advanced Bot Handling
YouTube often serves a "Sign in to confirm you're not a bot" challenge to Tor exit nodes. The pipeline handles this without crashing:
1.  **Detection**: Intercepts `Sign in` or `bot` error messages from `yt-dlp`.
2.  **Action**:
    - **Mark as Skipped**: The video is marked as `processed` (status: `bot_blocked`) so it is not retried infinitely.
    - **Force Rotation**: Triggers an immediate IP rotation to clear the "poisoned" exit node.
3.  **Outcome**: This sacrifices a small percentage of videos (blocked) to ensure the survival of the pipeline for the remaining 95%.

---

## 4. Risks & Limitations

| Risk/Limitation            | Description                                                                                              | Mitigation                                                               |
| :------------------------- | :------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **Data Loss (Bot Blocks)** | ~5-15% of videos may be skipped if they hit "Sign in" walls repeatedly.                                  | Acceptable trade-off for scale. Can be retried manually later.           |
| **Speed**                  | Tor network latency and frequent IP rotations slow down collection significantly (~100-300 videos/hour). | Parallelization (not implemented yet) or patience.                       |
| **Tor Node Exhaustion**    | Eventually, you may cycle through all "clean" Tor exit nodes, leading to 100% failure rates.             | Pause script for 2-3 hours to allow Tor circuits to expire/refresh.      |
| **Transcript Quality**     | Auto-generated Arabic captions may have high Word Error Rate (WER).                                      | Feature engineering pipeline accounts for this using robust NLP metrics. |

## 5. Benefits

1.  **Anonymity**: Your home/university IP is completely protected from YouTube bans.
2.  **Perseverance**: The system can run for days, slowly chipping away at the dataset, where a standard script would fail in minutes.
3.  **Clean Data**: The distinction between "Manual" and "Generated" captions is preserved, allowing for quality filtering during analysis.
4.  **Resilience**: Immune to standard network flakes, file lock errors (Excel open), and temporary API blocks.

## 6. Conclusion
This pipeline prioritizes **completion** over speed. It transforms a fragile scraping task into a durable background process capable of harvesting a large-scale dataset from a hostile anti-scraping environment.
