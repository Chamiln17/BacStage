# Context

Glossary for **BacStage**, which predicts engagement for Algerian Bac lessons on YouTube and coaches the creators who make them.

## Terms

**Bac 3AS video**
A YouTube video aimed at Algerian students in their final secondary year (3ème année secondaire) preparing for the Baccalauréat exam. The unit the project studies; everything else is filtered out.

**Subject**
The Bac subject a video teaches. Exactly nine: Arabic, English, French, History & Geography, Islamic Sciences, Maths, Natural Sciences, Philosophy, Physics. A channel teaches one subject.

**Snapshot**
One observation of a video's public statistics (views, likes, comments) at a given date. A video accumulates snapshots across collection runs.

**Engagement score**
The number the model predicts for a video. It is derived from a video's statistics, never collected directly.

**Planned video**
A video described before it is published: title, duration, channel or subject, and optionally a description, tags, a planned publish date and a transcript. It has no statistics yet. The predictor scores planned videos.

**Caption transcript**
The text of a video's captions (creator-uploaded or YouTube auto-generated) as downloaded from YouTube.

**Whisper transcript**
Text produced by running Whisper speech-to-text on a downloaded video. Same spoken content as a caption transcript, different origin.

**Transcript**
Either a caption transcript or a Whisper transcript. Use the specific term when the origin matters.

**Valid transcript**
A transcript with real spoken text: non-empty and not a dump of YouTube page code. Only valid transcripts count toward `has_transcript` and produce transcript features.

**Private archive**
The off-repo, never-public store of transcripts, trained models, and video IDs (ADR 0001).
