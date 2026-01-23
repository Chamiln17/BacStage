# Gap Analysis: AI-Driven Analytics System

**Date:** 2026-01-23
**Reviewer:** Senior Technical Lead / Solutions Architect

## 1. Implementation Status Audit

### A. Data Pipeline (Step 1)
*   **Status:** **Mostly Complete & Robust**
*   **Details:**
    *   **Data Ingestion:** The `src/data` module (`youtube_collector.py`, `transcript_collector.py`) is well-structured and functional for fetching videos, channels, and transcripts.
    *   **Feature Extraction ("Nibras Script"):** The core logic described as the "Nibras script" is implemented in `src/features/engineer.py`. It correctly extracts metadata, calculates the "Engagement Score" (using the log formula), and processes temporal rules (evening uploads, etc.). `scripts/integrate_transcript_features.py` handles the merging of transcript metrics.
    *   **Vectorization:** `src/features/text_embeddings.py` exists and provides a `TextEmbeddingExtractor` using AraBERT. However, it appears to be a helper utility that is currently underutilized (not connected to a vector store).

### B. Machine Learning (Step 2)
*   **Status:** **Prototype Only (In Notebooks)**
*   **Details:**
    *   **Training:** The Regression Model logic resides entirely in `notebooks/02_model_baseline_testing.ipynb`. It is **not** yet refactored into production code.
    *   **Model Storage:** `src/models` is currently empty. There is no formalized `train.py` script to serialize the trained model (e.g., as a `.pkl` or `.joblib` file) for later use.
    *   **Inference:** There is no `predict.py` or inference API. The system cannot currently accept a new video draft and output a score without running a notebook.

### C. RAG & Agent (Step 3)
*   **Status:** **Missing / Not Started**
*   **Details:**
    *   **Vector Database:** There is **no implementation** of a Vector Database (ChromaDB, FAISS, etc.). The "Best Practices" are not being stored or indexed.
    *   **Analyser Agent:** There is **no code** for the "Analyser Agent". No LLM integration (OpenAI, LangChain, etc.) was found in the codebase.
    *   **Integration:** Consequently, the connection between the Engagement Score (Step 2) and the qualitative advice (Step 3) is purely theoretical at this stage.

---

## 2. Gap Analysis (The "Left to Implement" List)

| Component              | Intended Architecture                                    | Current State                                | Gap                                                                                              |
| :--------------------- | :------------------------------------------------------- | :------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| **Model Persistence**  | Trained Regression Model saved for inference             | Notebook experiment only                     | **High**: Need to move training logic to `src/models/train_model.py` and save artifacts.         |
| **Inference Pipeline** | Script to accept new video details -> Output Score       | `engineer.py` only processes historical CSVs | **High**: Need `src/models/predict_model.py` to handle single-instance inference.                |
| **Vector Database**    | Store "Best Practices" vectors                           | `text_embeddings.py` (utility only)          | **Critical**: Need a pipeline to filter top videos, chunk text, embed, and store in a Vector DB. |
| **RAG Retrieval**      | Query Vector DB based on input context                   | Non-existent                                 | **Critical**: Implementation of retrieval logic missing.                                         |
| **Analyser Agent**     | LLM generating advice based on Score + Retrieved Context | Non-existent                                 | **Critical**: No LLM client or prompt engineering implementation.                                |

---

## 3. Technical Recommendations

### Immediate Next Steps (Prioritized)

1.  **Productionize the Regression Model:**
    *   Refactor `02_model_baseline_testing.ipynb` into `src/models/train_model.py`.
    *   Implement `src/models/predict_model.py` to accept a JSON input (simulating a video draft) and return the `engagement_score`.

2.  **Build the Knowledge Base (Phase 1 completion):**
    *   Create a script (`scripts/build_vector_db.py`) that:
        *   Selects top 10% performing videos.
        *   Chunks their transcripts/descriptions.
        *   Uses `TextEmbeddingExtractor` to create vectors.
        *   Saves them to a local Vector DB (e.g., ChromaDB).

3.  **Implement the RAG Agent (Phase 3 kick-off):**
    *   Create `src/agent/rag_engine.py` to handle retrieval.
    *   Create `src/agent/llm_client.py` to prompt the LLM (OpenAI/Gemini) with the retrieved context and predicted score.

### Dynamic Configuration
*   **Hardcoded Values:** The "Engagement Score" formula weights (3x for comments) and exam keywords list in `engineer.py` are hardcoded.
    *   *Recommendation:* Move these to a `config/model_config.yaml` to allow easy tuning without code changes.
