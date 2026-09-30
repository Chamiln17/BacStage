# Can this repo publish the data it collected from YouTube?

This is a reading of primary sources, not legal advice.

Sources accessed 2026-09-30. Versions read:

- YouTube API Services Developer Policies, "Last updated 2026-09-14 UTC": https://developers.google.com/youtube/terms/developer-policies
- YouTube API Services Terms of Service, "Last updated 2026-09-14 UTC": https://developers.google.com/youtube/terms/api-services-terms-of-service
- Additional policies for derived metrics and data storage: https://developers.google.com/youtube/terms/derived-metrics-policy
- YouTube Terms of Service (English, US), "Effective as of December 15, 2023": https://www.youtube.com/t/terms?hl=en&gl=US. The version YouTube shows to a user in Algeria may be different; the Arabic page returned during this session showed a different effective date. TODO: read the version that applies to the repo owners.
- YouTube Data API `captions.download` reference, "Last updated 2026-09-15 UTC": https://developers.google.com/youtube/v3/docs/captions/download
- YouTube Researcher Program Terms, "July 11, 2022 Original publication": https://research.youtube/policies/terms/
- Berne Convention, Art. 2: https://www.wipo.int/wipolex/en/text/283698. Algeria's entry into force is listed as April 19, 1998: https://www.wipo.int/wipolex/en/treaties/ShowResults?search_what=C&treaty_id=15
- U.S. Copyright Office, Compendium (Third) ch. 300, §313.2 and §313.4(A): https://www.copyright.gov/comp3/chap300/ch300-copyrightable-authorship.pdf
- 17 U.S.C. §106: https://www.copyright.gov/title17/92chap1.html
- yt-dlp LICENSE: https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/LICENSE

## Verdict

| Item in repo | Publishable? | Why | Citation |
|---|---|---|---|
| Source code: API collector (`src/data/youtube_collector.py`, `src/data/collect.py`), feature engineering, training, RAG code, with no keys | **Yes** | Code you wrote is yours to license. The only API-policy rule on code is to keep credentials out of it. A scan of tracked files found no API key. | Developer Policies III.D: "must not ... embed your API Credentials in open source projects" |
| Source code: `src/data/transcript_collector.py` + `keep_alive.ps1` (yt-dlp over Tor, with `NEWNYM` identity rotation) | **Risky** | Publishing a working scraper that rotates IPs to get past rate limits arguably "enables" others to scrape and to circumvent usage limits. | Developer Policies III.E "Scraping"; YouTube ToS "Permissions and Restrictions" |
| Small sample of API metadata (titles, descriptions, tags, IDs, dates) | **No, as stored** | This is Non-Authorized Data, collected with an API key and no user OAuth. It may be kept for at most 30 days and then refreshed or deleted. The snapshots are dated 2026-01-10 to 2026-01-24, more than 8 months old. Redistributing API Data is prohibited, and no clause grants a right to publish it outside an API Client. A list of video IDs only, plus a rehydration script, is the lowest-risk option, but whether bare IDs count as API Data is **ambiguous**. | Developer Policies III.E.4.d, III.G.1.a; API ToS 16.1–16.2 |
| Aggregated statistics (view/like/comment counts, `engagement_score`, `like_ratio`, `channel_avg_views`, per-subject means) | **No** | The rules ban the acts themselves: aggregating API Data across channels, and creating derived metrics. Raw statistics as Non-Authorized Data may not be stored for more than 30 days. The derived-metrics exception (III.L) covers only audited developers who applied through a quota extension. | Developer Policies III.E "Data Aggregation" (a)(b), III.E.4.b, III.E.4.h, III.L |
| Trained `.pkl` models (RF, XGBoost, LightGBM, CatBoost, scaler, 30-term TF-IDF) and `data/modeling/arabert_embeddings.pkl` | **Risky** | No clause in the Developer Policies or API ToS mentions training ML models. These files hold numeric parameters and vectors, not raw text; I checked the TF-IDF vocabulary (30 terms) and the embedding matrices (6042/2014/2015 × 768). They are still built from derived metrics that the policies forbid creating, so their status is **ambiguous**. | Developer Policies III.E.4.h; API ToS 16.2 |
| `models/rag_index/` and `knowledge_base/*.json` | **No** (plain reading) | These are per-subject aggregates of API statistics, for example "Total videos: 130 … Average engagement: 3.047", means and p-values, and keyword lists. That is aggregated and derived data, published. I found no raw transcripts or descriptions in `documents.json`. | Developer Policies III.E "Data Aggregation", III.E.4.h |
| Transcripts / auto-captions (`data/processed/transcripts*.csv`) | **No** | The text reproduces the teachers' spoken lectures. Lectures are protected works under Berne Art. 2(1), and the uploader keeps ownership. An auto-caption is a mechanical copy of that lecture, not a new public-domain text. The only official way to get captions requires edit rights on the video, and this collection bypassed it with yt-dlp. | Berne Art. 2(1); YouTube ToS "Rights you Grant"; Compendium §313.2, §313.4(A); 17 U.S.C. §106; `captions.download` docs |
| Comment samples (`text`, `authorDisplayName`) | **No** (none are tracked now) | Comments are API Data subject to the same 30-day and no-redistribution rules. The text belongs to the commenter, and author names identify people. `git grep` found no tracked file with comment fields. | Developer Policies III.E.4.d; YouTube ToS "Permissions and Restrictions"; YouTube ToS "Content on the Service" |

## Findings

### 1. YouTube API Services Terms and Developer Policies

**What counts as API Data, and who owns it.**

- Developer Policies IV "Definitions" includes in "YouTube API Services" the "data, content (including audiovisual content) and information provided to API Clients … through the YouTube API services (the "API Data")".
- The same section defines "Non-Authorized Data" as "API Data accessible by an API Client without User Credentials".
- This repo calls `videos.list`, `channels.list` and `commentThreads.list` with an API key. Everything it collected is therefore Non-Authorized Data.
- API ToS §16.1 "Ownership": YouTube and its licensors "retain all rights in, title to, interest in, and ownership of … all YouTube API Services (including all API Data)".
- API ToS §16.2 "No Other Rights": "Except for the express rights contained in the Agreement, YouTube grants you no other rights or licenses".
- Developer Policies III.E, opening line: "Aside from the permissions and rights granted in this section, you and your API Clients have no further permissions or rights to API Data, including to temporarily stored API Data."

**Storage limits (III.E.4 "Refreshing, Storing, and Displaying API Data").**

- III.E.4.d: "API Clients may temporarily store limited amounts of Non-Authorized Data for as long as is necessary for the purposes of the API Client but not longer than 30 calendar days. … after 30 calendar days, the API Client must either delete or refresh the stored data."
- III.E.4.b, on statistics: "To be clear, an API Client must not store statistics retrieved as Non-Authorized Data for more than 30 days. For example, an API Client must not store the subscriber count for a YouTube channel for more than 30 days without authorization from the channel owner."
- III.E.4.e: "API Clients must use reasonable efforts to ensure that their stored API Data is consistent with the current data available through YouTube API Services."
- III.E.4.f allows displaying "historical API Data provided that it is presented accurately in context of time". That clause is about display inside an API Client. It does not lift the 30-day storage limit.
- Repo fact: the `snapshot_date` values in `data/raw/videos_metadata.csv` run from 2026-01-10 to 2026-01-24. In `data/raw/channel_statistics.csv` the value is 2026-01-24. The local copies are already past the 30-day limit, whether or not they are published.

**Redistribution (III.G "Distribution and Commercial Use").**

- III.G: "You may distribute or sell API Clients …" and "you may distribute and display YouTube audiovisual content and accompanying metadata to users through your API Clients as long as those Clients comply with the Agreement".
- The permission is to distribute through an API Client, defined in IV as "a website or software application … developed by you that accesses or uses the YouTube API Services".
- A CSV file in a git repo is not an API Client. It is a copy of the data handed to third parties.
- III.G.1.a, "Prohibited Actions": you must not "sell, purchase, lease, lend, convey, redistribute, or sublicense all or any portion of YouTube API Services". Under IV, "YouTube API Services" includes API Data.
- **Ambiguity:** no clause names a "public dataset in a repository" directly. The reading above follows from combining III.G.1.a, the IV definitions, and the "no further permissions" line in III.E.

**Aggregated and derived metrics.**

- III.E "Data Aggregation", clause (a): "Do not aggregate API Data except that you may only aggregate API Data relating to YouTube channels that are under the same content owner … Such aggregated API Data must only be viewable by that content owner."
- Clause (b): "Do not aggregate API Data or otherwise use API Data or YouTube API Services to gain insights into YouTube's usage, revenue, or any other aspects of YouTube's business."
- III.E.4.h: "Your API Clients must not (i) replace API Data with similar, independently calculated data, or (ii) access or use API Data to create new or derived data or metrics."
- The worked example in III.E.4.h says you may not calculate "a score that factors in likes, total views, or any other API Data". That describes `engagement_score`, `like_ratio` and `comment_ratio` in `data/cleaned/data_final.csv`.
- III.L "Additional policies on derived metrics and data storage": "These policies are only applicable to audited developers with analytics use cases that have explicitly applied for permission to create additional metrics and/or store statistical data through the standard quota extension request". This repo has not done that.
- **Ambiguity:** clause (b) is aimed at insights into "YouTube's business". A study of which Bac lessons get engagement is arguably not that. Clause (a) and III.E.4.h still apply on their own terms.

**ML training.**

- Neither document has a clause on training machine-learning models on API Data. I searched both pages for "machine", "artificial", "train" and "model" and found no such clause.
- The status of trained models therefore rests on the general rules above: no rights beyond those granted, and no derived data. That is **ambiguous**, not a clear yes or no.

**Captions through the official API.**

- The `captions.download` reference says: "This method requires the user to have permission to edit the video."
- The official API offers no route for a third party to download another channel's captions.
- Developer Policies III.D "Undocumented Services": "You must access data from YouTube API services only according to the means stipulated in the authorized documentation".

**Credentials.**

- III.D "API Credentials": "you must not … embed your API Credentials in open source projects."
- I scanned the tracked files outside `data/` and `models/`, and the history pickaxe for the `AIza` key prefix. The only match was inside base64 PNG output in a notebook, a false positive. `git log --all -- .env` returns nothing, and `.env.example` is tracked.

**Researcher Program (not a workaround).**

- Eligibility is limited to accredited academic institutions. Its terms, §7 "No Data Disclosure", still say: "You will not disclose, reproduce, sell, license or otherwise transfer to any third party, in part or in whole, any Program Data."

### 2. YouTube Terms of Service: automated access (yt-dlp, Tor)

These quotes come from the section "Permissions and Restrictions". It says "You are not allowed to":

- "access the Service using any automated means (such as robots, botnets or scrapers) except (a) in the case of public search engines, in accordance with YouTube's robots.txt file; or (b) with YouTube's prior written permission;"
- "access, reproduce, download, distribute, transmit, broadcast, display, sell, license, alter, modify or otherwise use any part of the Service or any Content except: (a) as expressly authorized by the Service; or (b) with prior written permission from YouTube and, if applicable, the respective rights holders;"
- "circumvent, disable, fraudulently engage with, or otherwise interfere with any part of the Service …, including security-related features or features that … (b) limit the use of the Service or Content;"
- "collect or harvest any information that might identify a person (for example, usernames or faces), unless permitted by that person or allowed under section (3) above;"

The Developer Policies add their own rule, in III.E "Scraping": "You and your API Clients must not, and must not encourage, enable, or require others to, directly or indirectly, scrape YouTube Applications or Google Applications, or obtain scraped YouTube data or content."

What this means for the repo:

- Fetching captions with yt-dlp is automated access without written permission.
- Rotating Tor identities to get around rate limits reads directly on "circumvent … features that … limit the use of the Service".
- Because the same project also uses the API, the "obtain scraped YouTube data" clause binds it as an API developer too.
- Publishing the Tor rotation loop is a further question under "enable … others to … scrape". I read that as risky, not settled.
- III.D also says "You must not mask or misrepresent your identity … when accessing or using YouTube API Services". That clause covers API access, not yt-dlp, so it is adjacent only.

### 3. Copyright in transcripts and auto-generated captions

**Who owns the spoken content.**

- YouTube ToS, "Content on the Service": Content includes "text (such as comments and scripts)".
- "Rights you Grant": "You retain ownership rights in your Content."
- "License to Other Users": other users get a license "only as enabled by a feature of the Service (such as video playback or embeds). For clarity, this license does not grant any rights or permissions for a user to make use of your Content independent of the Service."
- Berne Convention Art. 2(1) lists "lectures, addresses, sermons and other works of the same nature" as protected literary and artistic works. Algeria has been a party since April 19, 1998, according to WIPO's contracting-parties table.
- A Bac lesson is a lecture. The teacher, or whoever holds the rights, owns the words.

**Auto-generated captions.**

- Compendium §313.2 says the Office "will not register works produced by a machine or mere mechanical process that operates randomly or automatically without any creative input". That only means the caption text is not a *new* work with its own author.
- It does not remove the lecturer's copyright in the words being transcribed.
- Compendium §313.4(A) "Mere Copies": "A work that is a mere copy of another work of authorship is not copyrightable." An ASR transcript is a mechanical copy of the lecture.
- 17 U.S.C. §106 gives the copyright owner the exclusive right "(1) to reproduce the copyrighted work in copies" and "(3) to distribute copies … to the public". Publishing full transcripts does both.

**Creator-uploaded captions** are the creator's own text and are covered directly by "You retain ownership rights in your Content".

**Open points:**

- TODO: Algerian law (Ordonnance n° 03-05 of 19 July 2003) was not fetched or read in this session. The exceptions it allows, such as quotation or teaching, are unverified.
- Whether a fair-use or text-and-data-mining exception covers the *training* use is jurisdiction-specific and was not researched here. No such exception was found in the sources above that would cover *redistributing* the full text.

### 4. yt-dlp licence (code dependency only)

The yt-dlp LICENSE is the Unlicense: "This is free and unencumbered software released into the public domain. Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software … for any purpose". Depending on it and naming it in `pyproject.toml` raises no licence issue. The licence covers the tool only. It grants nothing over YouTube's data or the creators' content.

## Recommended actions for this repo

1. **Deleting files now will not unpublish them.** About 506 MB of `data/`, `models/` and `knowledge_base/` is in git history; `du` over the tracked files reports 506M. Before making the repo public, do one of two things:
   - publish from a fresh repository that holds only code, or
   - rewrite history with `git filter-repo` to remove those paths.
2. **Check whether the remote is already public.** The remote is `origin https://github.com/Chamiln17/Bac-Youtube-Analysis.git`. `gh repo view` could not resolve it in this session. TODO: confirm its visibility. If it has already been public, the data has already been distributed.
3. **Handle the local copies.** The metadata and statistics snapshots (Jan 2026) are past the 30-day limit for Non-Authorized Data. Either delete them or re-fetch them, per III.E.4.d and III.E.4.b.
4. **Keep out of the public repo:**
   - `data/raw/`, `data/processed/`, `data/cleaned/`, `data/modeling/`, `data/validation/`
   - `models/rag_index/`
   - `knowledge_base/`
   - all transcript CSVs

   Add them to `.gitignore`.
5. **For reproducibility:**
   - ship the collection code, with a README explaining how others can run it using their own API key and under their own ToS obligations;
   - optionally ship a list of video IDs. Whether bare IDs are API Data is ambiguous; this is the lowest-risk way to share the dataset definition.
6. **Trained `.pkl` models:** safest is to leave them out and document how to retrain. If you want them in the portfolio, treat them as a risk you accept on an ambiguous point, not a cleared item.
7. **Transcript collector:** either leave `transcript_collector.py`'s Tor/NEWNYM rotation and `keep_alive.ps1` out of the public repo, or remove the rotation logic. Add a README note that caption collection by scraping breaches the YouTube ToS "Permissions and Restrictions".
8. **Show results instead of data.** For the CV, show metrics, plots and write-ups of findings. Charts and summary numbers built from API Data still count as aggregated or derived data under III.E, so keep them high-level. **Ambiguous:** no clause addresses a static report.
9. **Before publishing:**
   - keep the API key out of the repo;
   - keep `.env` untracked (it is untracked now);
   - rotate any key that has ever appeared in a notebook output or log.
