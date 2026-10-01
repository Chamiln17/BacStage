# Case study: building BacStage

*How a four-person team turned 19,919 YouTube videos into an engagement model and a coach for Algerian Bac teachers, and what we fixed on the way to publishing it.*

## The question

Every year, Algerian students in their final secondary year (3AS) prepare for the Baccalauréat, and many of them learn from YouTube. Teachers publish lessons there on every subject, from complex numbers to the philosophy of science. Some of those lessons draw questions and discussion in the comments, while others with similar content get none.

We wanted to know **what separates the lessons students engage with from the ones they skip**, and to turn that into something a teacher can use *before* publishing.

## 1. Collecting the data on a free quota

The YouTube Data API gives 10,000 quota units a day, and a search costs 100 units for 50 results. Searching our way through 36 channels would have used the whole day's budget on discovery alone.

Instead, the collector reads each channel's **uploads playlist** (1 unit per 50 videos), then fetches statistics in **batches of 50** (1 unit per batch). A registry remembers every video already found, so later runs only refresh. Refreshing all 19,919 known videos on 2026-09-30 cost **435 units**, about 4% of one day's quota.

## 2. Finding the Bac lessons

The 36 channels publish more than Bac content. They also post middle-school lessons, first- and second-year material and live streams. A plain keyword match on "bac" catches too little, because many lessons never say "bac" and only name a 3AS topic like *الأعداد المركبة*. It also catches too much, because some titles mention both the Bac and a lower grade.

The filter combines three signals:

1. **Explicit grade markers** in the title, description and tags, including 3AS-only curriculum topics. All of them live in one YAML file.
2. **Channel priors.** Some channels are almost entirely Bac. A video with no marker from such a channel is probably Bac too, if it passes a soft check (long enough, a discovered term, or the channel's subject).
3. **Discovered terms.** TF-IDF over titles finds the character n-grams that separate Bac titles from others, so the vocabulary grows without anyone maintaining a list.

We checked it by hand against 300 sampled videos. On the 282 of them still online, it reaches **precision 0.942 and recall 0.821**. It keeps 9,801 of 19,042 videos.

## 3. Transcripts, and the ones that weren't

A lesson's words matter as much as its title: how fast the teacher speaks, how many questions they ask, how often they give examples. We downloaded YouTube's captions for every candidate video. Of 9,991 attempts, 5,369 returned nothing usable. **4,583 were valid transcripts**, almost all YouTube's automatic Arabic captions.

Some "transcripts" turned out to be YouTube's page code, captured when a download was blocked. That's 39 of them, each with at least two code markers such as `ytcfg` or `window.` instead of speech. We wrote one rule for spotting them (`is_valid_transcript`) and made every part of the pipeline use it. Before that, four slightly different copies of the rule disagreed with each other.

## 4. Features

Each video becomes 171 numbers:

- **55 candidate numeric features**, from which VIF selection keeps 41: title and description lengths, question marks, exam keywords, Bac markers, duration, subject, channel statistics, and 32 transcript features (readability, speaking pace, questions, examples, explanations, curriculum keywords)
- **30 TF-IDF terms** from the title and description
- **100 dimensions of AraBERT** embeddings (`aubmindlab/bert-base-arabertv2`), reduced with PCA

## 5. Modelling, and the leak

The target is an **engagement score**: `log(1 + (3 × comments + likes) / √views)`. Comments weigh triple because on lesson videos they are mostly students asking questions.

The notebook compared ten models. Linear models stalled at R² 0.49. Tree ensembles reached about 0.65–0.69, and a random forest tuned with Optuna came out on top at **R² 0.693**.

When we turned the notebook into a reusable pipeline for publishing, a review found three problems:

1. **The `train` command crashed on import**, and `predict` couldn't load what `train` saved. A test passed only because it wrote the missing file by hand.
2. **Leakage.** Channel averages, feature selection and scaling were fitted on all videos, test videos included. A test video's own views helped compute its channel's average views, and the target is built from views.
3. **A silent subject bug.** The keyword lists named maths "Mathematics", while the data says "Maths". The Maths keyword features, for 42% of all videos, quietly used every subject's keywords instead.

We rebuilt the feature step as one object that is **fitted on training videos only** and then used unchanged for prediction. A test proves it never reads views, likes or comments. Retrained on fresh statistics and scored on the notebook's own test videos, the honest pipeline reaches **R² 0.703, MAE 0.314** on 1,963 held-out lessons.

That is not "fixing the leak improved the score". The data is eight months more mature and the training set is larger, so the two numbers are not directly comparable. What we can say is that **0.70 holds up without the leak.**

## 6. The coach

A prediction alone doesn't tell a teacher what to change. The coach adds three things:

- **Best practices per subject**, computed by comparing high- and low-engagement videos and retrieved with a multilingual embedding index.
- **A thumbnail check**: brightness, contrast, colours, visual interest and on-image text read with Arabic OCR.
- **An LLM** (`gpt-oss-120b` on Groq) that combines the prediction, the practices and the thumbnail report into prioritised advice in Arabic.

Because a teacher's own channel usually isn't in our data, the coach scores every lesson as coming from a new channel and says so.

## 7. Publishing responsibly

Open-sourcing a data project is a data decision as much as a code decision. Before publishing, we read YouTube's API terms and policies:

- API data can't be redistributed, and can only be stored for 30 days.
- Derived metrics are restricted.
- Transcripts reproduce the teachers' own lectures.

So **the repository is code only**. The transcripts, models and labels live in a private archive, and anyone can rebuild the data with their own API key. We also removed an early experiment that rotated IP addresses through Tor to get around rate limits; the collector now just waits.

## What we learned

- **A pipeline is only as honest as its `fit`.** Our best notebook number came from a protocol that let test data shape the features. One object that is fit once and then transforms everything made that impossible.
- **Silent fallbacks hide bugs.** "Mathematics" vs "Maths" and "Science" vs "Natural Sciences" each fell back without an error. Making unknown names raise found both at once.
- **Tests can lie.** The predictor's test faked the very file the trainer never wrote. Testing through the real interface, train then predict, found it immediately.
- **Quota is a design constraint.** Choosing the right endpoint cut collection cost about 100× and made refreshing everything a small job.

## Team

Chamel Nadir Bouacha, Abdelkebir Achraf, Nibras Norelislam Bouzidi and lahcenbcf, as a team project in the Samsung Innovation Campus program.
