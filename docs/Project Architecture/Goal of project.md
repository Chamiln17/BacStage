
# Project Report: AI-Driven Analytics and Optimization System for Educational Content

## 1. Project Overview & Objective

The primary objective of this project is to democratize access to high-quality educational content—specifically targeting the Algerian Baccalaureate ecosystem—by empowering content creators with data-driven insights.

The system addresses the challenge of "content visibility" not by guessing, but by engineering a pipeline that:

1. **Analyzes** historical success factors from top-performing videos.
2. **Predicts** the potential engagement of a new video before it is published.
3. **Recommends** actionable improvements using Generative AI (LLMs).

## 2. System Architecture

The solution is architected as a three-stage pipeline combining statistical analysis, supervised machine learning (Regression), and Generative AI (RAG).

### Phase 1: Data Analysis & Knowledge Base Construction

The foundation of the system is a rigorous analysis of "High Engaged Videos" to establish a ground truth for what constitutes successful content in this niche.

* **Data Ingestion:** The system ingests historical video data via the YouTube Data API v3.
* **Feature Extraction:** We isolate specific metadata characteristics (Titles, Description keywords) and content features (Duration < 1 hour) that correlate with high performance.
* **Vectorization (RAG Foundation):** Instead of a static report, the "Best Practices" derived from this analysis are processed and stored in a **Vector Database**. This allows the subsequent AI agent to perform semantic searches, retrieving only the most relevant optimization rules based on the context of the user's new video.

### Phase 2: Predictive Modeling (The "Engagement Engine")

This phase allows the content creator to "test" their content in a sandbox environment before public release.

* **Input Processing (The Nibras Script):**
When a user uploads a draft video to the platform, a specialized script (developed by the team) automatically extracts the raw modalities required for analysis. This includes:
* **Metadata:** Title and Description.
* **Content Metrics:** Exact video duration.
* **Textual Content:** Full transcript extraction (Speech-to-Text).


* **The Regression Model:**
These features are fed into a trained Regression Model which predicts the video's potential performance.
* **Target Variable (Engagement Score):**
Due to API limitations (absence of Watch Time or Impressions), we engineered a custom, robust target variable to represent "Engagement." The formula prioritizes high-intent interactions (Comments) and normalizes against viral volatility (Views):
$$\text{Engagement Score} = \ln\left(1 + \frac{(3 \times \text{Comments}) + \text{Likes}}{\sqrt{\text{Views}}}\right)$$

* *Rationale:* The square root of views penalizes "empty clicks," while the log transformation handles the heavy-tailed distribution typical of social media metrics.



### Phase 3: The "Analyser Agent" (GenAI Recommendation Layer)

The final phase closes the loop by translating quantitative scores into qualitative, human-readable advice.

* **Architecture:** This module utilizes a Large Language Model (LLM) via API, symbolically represented in the architecture as a Transformer block.
* **Retrieval-Augmented Generation (RAG):**
The Agent does not hallucinate advice; it uses the specific context of the current video to query the **Vector Database** (from Phase 1). It retrieves the specific "Best Practices" that apply to the current video's topic and format.
* **Contextual Synthesis:**
The LLM receives a composite prompt containing:
1. **The Input:** Title, Description, Duration, and Transcript features.
2. **The Prediction:** The forecasted Engagement Score from Phase 2.
3. **The Knowledge:** Retrieved relevant best practices.


* **Output:** The Agent generates a tailored text report for the creator, suggesting specific changes (e.g., "Your intro is too long compared to top-performing videos in this topic," or "Include these high-value keywords in your description") to maximize the Engagement Score.

## 3. Conclusion

This system moves beyond simple analytics dashboards. By integrating a predictive regression model with a RAG-based LLM agent, we provide creators with a "virtual data scientist" that not only scores their work but actively guides them toward creating more effective, engaging educational content.