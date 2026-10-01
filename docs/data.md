# Data and ethics

**This repository contains code only.** It ships no collected YouTube data, no transcripts, no knowledge base and no trained models. That choice is recorded in [ADR 0001](adr/0001-publish-code-only.md), and the reasons come from primary sources collected in [the publishing research](research/youtube-data-publishing.md).

The research is not legal advice. In short:

| What | Published? | Why |
| --- | --- | --- |
| Source code | Yes | Our own work; no credentials in it. |
| Video metadata and statistics | No | The YouTube API terms forbid redistributing API data, and data collected with an API key may be stored for at most 30 days. |
| Aggregated statistics, knowledge base | No | The policies also restrict creating and sharing derived metrics. |
| Transcripts | No | They reproduce the teachers' lectures, which the teachers own. |
| Trained models | No | Built from the data above; the terms do not clearly allow it, so we don't. |

## What the project collected

| | |
| --- | --- |
| Channels | 36 Algerian Bac channels, chosen by hand, one subject each |
| Videos discovered | 19,919 |
| Statistics fetched (2026-09-30) | 19,042, for 435 API quota units |
| Bac 3AS lessons after filtering | 9,801 |
| Valid caption transcripts | 4,583 collected, 4,381 for current Bac videos |
| Hand labels | 300 videos labelled by hand to check the filter |

## How to reproduce

Bring your own YouTube Data API key and rebuild everything:

```bash
cp .env.example .env                        # set YOUTUBE_API_KEY
uv run python run_pipeline.py collect --channels data/raw/channels.csv
uv run python run_pipeline.py filter_data
uv run python run_pipeline.py transcripts --input data/processed/videos_bac_only.csv --output data/processed/transcripts.csv
uv run python run_pipeline.py clean --transcripts data/processed/transcripts.csv
uv run python run_pipeline.py train
```

`data/raw/channels.csv` is the list of channels to study, with columns `channel_id,channel_name,subjects`, and `subjects` must be one of the nine [subjects](glossary.md). Statistics refreshes are cheap: one quota unit per 50 videos.

Keep the 30-day limit in mind: delete or refresh API data within 30 days of collecting it.

## Collecting responsibly

- **Quota.** Collection uses the uploads playlist and batches of 50, about 100× fewer quota units than search.
- **Transcripts.** Downloaded one video at a time with a pause between requests (`--delay`, default 2 s). After repeated refusals the collector waits; it never switches IP addresses.
- **People.** No comments, no author names and no personal data are used for modelling. Channel and video IDs identify public educational channels.
