"""
RAG System for Best Practices Retrieval
========================================
This module implements semantic search over the knowledge base
to retrieve relevant best practices for new video content.

Components:
- Vector embeddings using multilingual sentence transformer
- FAISS vector store for fast similarity search
- Retrieval function for finding relevant best practices
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# FAISS for vector similarity search
import faiss
import numpy as np

# Sentence transformers for multilingual embeddings
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    """Represents a document chunk from the knowledge base."""
    id: str
    category: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class TranscriptMetrics:
    """
    Transcript-derived metrics for a video.
    These metrics help the RAG system provide more targeted recommendations.
    """
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    avg_words_per_sentence: float = 0.0
    lexical_diversity: float = 0.0
    unique_word_count: int = 0
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    speech_rate_wpm: float = 0.0  # Calculated: word_count / duration_minutes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "word_count": self.word_count,
            "char_count": self.char_count,
            "sentence_count": self.sentence_count,
            "avg_words_per_sentence": self.avg_words_per_sentence,
            "lexical_diversity": self.lexical_diversity,
            "unique_word_count": self.unique_word_count,
            "flesch_reading_ease": self.flesch_reading_ease,
            "flesch_kincaid_grade": self.flesch_kincaid_grade,
            "speech_rate_wpm": self.speech_rate_wpm
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptMetrics':
        """Create from dictionary."""
        return cls(
            word_count=data.get('transcript_word_count', data.get('word_count', 0)),
            char_count=data.get('transcript_char_count', data.get('char_count', 0)),
            sentence_count=data.get('transcript_sentence_count', data.get('sentence_count', 0)),
            avg_words_per_sentence=data.get('avg_words_per_sentence', 0.0),
            lexical_diversity=data.get('lexical_diversity', 0.0),
            unique_word_count=data.get('unique_word_count', 0),
            flesch_reading_ease=data.get('flesch_reading_ease', 0.0),
            flesch_kincaid_grade=data.get('flesch_kincaid_grade', 0.0),
            speech_rate_wpm=data.get('speech_rate_wpm', 0.0)
        )


# Benchmark values from high-performing videos (extracted from knowledge base)
TRANSCRIPT_BENCHMARKS = {
    "speech_rate_wpm": {"optimal": 105, "high_mean": 100.0, "low_mean": 104.2},
    "lexical_diversity": {"optimal": 0.37, "high_mean": 0.36, "low_mean": 0.27},
    "flesch_reading_ease": {"high_mean": -2112.0, "low_mean": -2301.6},  # Arabic text adaptation
    "example_density": {"high_mean": 5.6, "low_mean": 4.0},
    "avg_words_per_sentence": {"optimal_range": (10, 20)},
}


class BestPracticesRAG:
    """
    RAG system for retrieving relevant best practices.

    Uses multilingual sentence transformer for embedding Arabic/French content
    and FAISS for efficient similarity search.
    """

    def __init__(
        self,
        knowledge_base_dir: str = "knowledge_base",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize the RAG system.

        Args:
            knowledge_base_dir: Path to directory containing JSON files
            model_name: Sentence transformer model (multilingual for Arabic support)
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.model_name = model_name
        self.model = None
        self.documents: List[Document] = []
        self.index = None
        self.embedding_dim = 384  # MiniLM embedding dimension

    def load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def load_knowledge_base(self) -> List[Document]:
        """
        Load and chunk JSON documents from the knowledge base.

        Handles both subject-specific files (Maths_best_practices.json, etc.)
        and global_best_practices.json for fallback.
        """
        self.documents = []

        json_files = list(self.knowledge_base_dir.glob("*_best_practices.json"))
        # Exclude the combined file
        json_files = [f for f in json_files if f.name != "all_best_practices.json"]

        print(f"Found {len(json_files)} knowledge base files")

        # Known subjects for detection
        known_subjects = ["Maths", "Physics", "Science", "Arabic", "French", "English", "Philosophy", "History"]

        for json_path in json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Detect if this is a subject-specific file
            filename = json_path.stem.replace("_best_practices", "")
            is_subject_file = filename in known_subjects
            is_global = filename == "global"

            if is_subject_file:
                # This is a subject-specific file
                subject = filename
                chunks = self._create_subject_chunks(data, subject)
            elif is_global:
                # This is the global fallback file
                chunks = self._create_global_chunks(data)
            else:
                # Legacy format or category-based file
                category = data.get('category', json_path.stem)
                chunks = self._create_chunks(data, category)

            self.documents.extend(chunks)

        print(f"Created {len(self.documents)} document chunks")
        return self.documents

    def _create_subject_chunks(self, data: Dict, subject: str) -> List[Document]:
        """
        Create document chunks from a subject-specific best practices file.
        Each chunk is tagged with the subject for filtering.
        """
        chunks = []

        # Extract summary stats
        summary_content = f"Subject: {subject}\n"
        summary_content += f"Total videos: {data.get('total_videos', 'N/A')}\n"
        summary_content += f"High performers: {data.get('high_performers', 'N/A')}\n"
        summary_content += f"Average engagement: {data.get('avg_engagement_score', 'N/A')}\n"

        chunks.append(Document(
            id=f"{subject}_summary",
            category="subject_summary",
            content=summary_content,
            metadata={"type": "subject_summary", "subject": subject}
        ))

        # Process each category within the subject
        categories = [
            "title_optimization", "video_duration", "upload_timing",
            "content_delivery", "description_optimization", "content_strategy"
        ]

        for cat_name in categories:
            if cat_name in data:
                cat_data = data[cat_name]

                # Create recommendation chunk
                if 'recommendations' in cat_data:
                    content = f"Subject: {subject}\n"
                    content += f"Category: {cat_name}\n"
                    content += "Recommendations:\n"
                    content += "\n".join(cat_data['recommendations'])

                    chunks.append(Document(
                        id=f"{subject}_{cat_name}_recommendations",
                        category=cat_name,
                        content=content,
                        metadata={
                            "type": "recommendations",
                            "subject": subject,
                            "source_file": f"{subject}_best_practices.json"
                        }
                    ))

                # Create analysis chunks for quantitative data
                for key in ['overall_comparison', 'title_length', 'length_analysis']:
                    if key in cat_data and cat_data[key] and not cat_data[key].get('insufficient_data'):
                        comparison = cat_data[key]
                        content = f"Subject: {subject}\n"
                        content += f"Category: {cat_name}\n"
                        content += f"Metric: {comparison.get('feature', key)}\n"
                        content += f"High performers median: {comparison.get('high_median', 'N/A')}\n"
                        content += f"Low performers median: {comparison.get('low_median', 'N/A')}\n"
                        content += f"Difference: {comparison.get('difference_pct', 'N/A')}%\n"

                        chunks.append(Document(
                            id=f"{subject}_{cat_name}_{key}",
                            category=cat_name,
                            content=content,
                            metadata={
                                "type": "statistical_analysis",
                                "subject": subject,
                                "metric": key
                            }
                        ))

                # Keyword analysis
                if 'keyword_impact' in cat_data:
                    content = f"Subject: {subject}\n"
                    content += f"Category: {cat_name}\n"
                    content += "Keyword Impact:\n"
                    for kw, impact in cat_data['keyword_impact'].items():
                        lift = impact.get('lift_pct', 0)
                        if lift > 0:
                            content += f"  - '{kw}': +{lift:.1f}% lift\n"

                    chunks.append(Document(
                        id=f"{subject}_{cat_name}_keywords",
                        category=cat_name,
                        content=content,
                        metadata={"type": "keyword_analysis", "subject": subject}
                    ))

        return chunks

    def _create_global_chunks(self, data: Dict) -> List[Document]:
        """Create chunks from global best practices (fallback)."""
        chunks = []

        # Process as regular categories but tag as global
        categories = [
            "title_optimization", "video_duration", "upload_timing",
            "content_delivery", "description_optimization", "content_strategy"
        ]

        for cat_name in categories:
            if cat_name in data:
                cat_chunks = self._create_chunks(data[cat_name], cat_name)
                # Tag as global
                for chunk in cat_chunks:
                    chunk.metadata["subject"] = "global"
                chunks.extend(cat_chunks)

        return chunks

    def _create_chunks(self, data: Dict, category: str) -> List[Document]:
        """
        Create searchable document chunks from a best practices file.

        Strategy: Create multiple focused chunks per category for better retrieval.
        """
        chunks = []

        # 1. Main recommendations chunk
        if 'recommendations' in data:
            recs = data['recommendations']
            if recs:
                content = f"Category: {category}\n"
                content += "Recommendations:\n"
                content += "\n".join(recs)

                chunks.append(Document(
                    id=f"{category}_recommendations",
                    category=category,
                    content=content,
                    metadata={
                        "type": "recommendations",
                        "source_file": f"{category}_best_practices.json"
                    }
                ))

        # 2. Subject-specific chunks (for subject_specific category)
        if category == "subject_specific" and 'subjects' in data:
            for subject, subject_data in data['subjects'].items():
                content = f"Subject: {subject}\n"
                content += f"Total videos analyzed: {subject_data.get('total_videos', 'N/A')}\n"
                content += f"Optimal duration: {subject_data.get('optimal_duration_minutes', 'N/A')} minutes\n"
                content += f"Optimal title length: {subject_data.get('optimal_title_length', 'N/A')} characters\n"
                content += f"Exam-focused ratio: {subject_data.get('exam_focused_ratio', 'N/A')}%\n"

                # Add top channels if available
                if 'top_channels' in subject_data:
                    content += "Top performing channels:\n"
                    for channel, score in subject_data['top_channels'].items():
                        content += f"  - {channel}: {score:.2f}\n"

                chunks.append(Document(
                    id=f"subject_{subject}",
                    category="subject_specific",
                    content=content,
                    metadata={
                        "type": "subject_insights",
                        "subject": subject,
                        "source_file": "subject_specific_best_practices.json"
                    }
                ))

        # 3. Feature analysis chunks (for content_delivery, title_optimization, etc.)
        if 'feature_analysis' in data:
            for feature, analysis in data['feature_analysis'].items():
                content = f"Feature: {feature}\n"
                content += f"Category: {category}\n"
                content += f"High performers mean: {analysis.get('high_mean', 'N/A')}\n"
                content += f"Low performers mean: {analysis.get('low_mean', 'N/A')}\n"
                content += f"Difference: {analysis.get('difference_pct', 'N/A')}%\n"
                content += f"Statistically significant: {analysis.get('significant', 'N/A')}\n"
                content += f"Effect size: {analysis.get('effect_interpretation', 'N/A')}\n"

                chunks.append(Document(
                    id=f"{category}_{feature}",
                    category=category,
                    content=content,
                    metadata={
                        "type": "feature_analysis",
                        "feature": feature,
                        "significant": analysis.get('significant', False),
                        "source_file": f"{category}_best_practices.json"
                    }
                ))

        # 4. Keyword impact chunks (for title_optimization)
        if 'keyword_impact' in data:
            content = f"Category: {category}\n"
            content += "Keyword Impact Analysis:\n"
            for keyword, impact in data['keyword_impact'].items():
                lift = impact.get('lift_pct', 0)
                if lift > 0:  # Only include positive impact keywords
                    content += f"  - '{keyword}': +{lift:.1f}% engagement lift\n"

            chunks.append(Document(
                id=f"{category}_keywords",
                category=category,
                content=content,
                metadata={
                    "type": "keyword_analysis",
                    "source_file": f"{category}_best_practices.json"
                }
            ))

        # 5. Timing insights (for upload_timing)
        if 'by_day' in data:
            content = f"Category: {category}\n"
            content += "Best Upload Days:\n"
            for day, stats in data['by_day'].items():
                content += f"  - {day}: avg engagement {stats.get('avg_engagement', 'N/A')}\n"

            chunks.append(Document(
                id=f"{category}_days",
                category=category,
                content=content,
                metadata={
                    "type": "timing_analysis",
                    "source_file": f"{category}_best_practices.json"
                }
            ))

        if 'top_hours' in data:
            content = f"Category: {category}\n"
            content += f"Best Upload Hours: {data['top_hours']}\n"

            chunks.append(Document(
                id=f"{category}_hours",
                category=category,
                content=content,
                metadata={
                    "type": "timing_analysis",
                    "source_file": f"{category}_best_practices.json"
                }
            ))

        # 6. Duration insights
        if 'by_duration_bin' in data:
            content = f"Category: {category}\n"
            content += "Engagement by Video Duration:\n"
            for duration_bin, stats in data['by_duration_bin'].items():
                content += f"  - {duration_bin}: avg engagement {stats.get('mean', 'N/A')}, count: {stats.get('count', 'N/A')}\n"

            chunks.append(Document(
                id=f"{category}_duration_bins",
                category=category,
                content=content,
                metadata={
                    "type": "duration_analysis",
                    "source_file": f"{category}_best_practices.json"
                }
            ))

        return chunks

    def create_embeddings(self):
        """Generate embeddings for all documents."""
        if self.model is None:
            self.load_model()

        print("Generating embeddings for documents...")

        # Extract content from all documents
        contents = [doc.content for doc in self.documents]

        # Generate embeddings in batch
        embeddings = self.model.encode(
            contents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Store embeddings in documents
        for doc, emb in zip(self.documents, embeddings, strict=False):
            doc.embedding = emb

        print(f"Generated {len(embeddings)} embeddings")

    def build_index(self):
        """Build FAISS index for fast similarity search."""
        if not self.documents or self.documents[0].embedding is None:
            raise ValueError("Documents must have embeddings. Call create_embeddings() first.")

        # Stack all embeddings into a matrix
        embeddings_matrix = np.vstack([doc.embedding for doc in self.documents])

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_matrix)

        # Create FAISS index (Inner Product for cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_matrix.astype('float32'))

        print(f"Built FAISS index with {self.index.ntotal} vectors")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve most relevant documents for a query.

        Args:
            query: Search query (video title, description, etc.)
            top_k: Number of results to return
            category_filter: Optional filter by category (e.g., 'title_optimization')

        Returns:
            List of (Document, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Embed the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search (get more results if filtering)
        search_k = top_k * 3 if category_filter else top_k
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)

        results = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < len(self.documents):
                doc = self.documents[idx]

                # Apply category filter if specified
                if category_filter and doc.category != category_filter:
                    continue

                results.append((doc, float(score)))

                if len(results) >= top_k:
                    break

        return results

    def retrieve_best_practices(
        self,
        title: str,
        description: str = "",
        subject: str = "",
        transcript_metrics: Optional[TranscriptMetrics] = None,
        duration_minutes: float = 0,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Main retrieval function for finding relevant best practices.

        Args:
            title: Video title
            description: Video description (optional)
            subject: Subject/topic (e.g., 'Maths', 'Physics')
            transcript_metrics: Transcript-derived metrics (optional)
            duration_minutes: Video duration in minutes (for speech rate calculation)
            top_k: Number of results per query type

        Returns:
            Dictionary with retrieved best practices organized by relevance
        """
        results = {
            "query_info": {
                "title": title,
                "description": description[:200] if description else "",
                "subject": subject,
                "has_transcript_metrics": transcript_metrics is not None
            },
            "retrieved_practices": [],
            "subject_specific": None,
            "metric_analysis": None
        }

        # 1. Search based on title
        title_query = f"Title optimization for: {title}"
        title_results = self.retrieve(title_query, top_k=3)

        # 2. Search based on subject if provided
        subject_results = []
        if subject:
            subject_query = f"Best practices for {subject} educational videos"
            subject_results = self.retrieve(subject_query, top_k=3)

            # Find exact subject match for subject_specific field
            for doc in self.documents:
                if doc.metadata.get('subject', '').lower() == subject.lower():
                    results["subject_specific"] = {
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    break

        # 3. Search based on description if provided
        desc_results = []
        if description:
            desc_query = f"Video content optimization: {description[:500]}"
            desc_results = self.retrieve(desc_query, top_k=2)

        # 4. Analyze transcript metrics and retrieve relevant content delivery practices
        content_results = []
        if transcript_metrics:
            # Calculate speech rate if not provided
            if transcript_metrics.speech_rate_wpm == 0 and duration_minutes > 0:
                transcript_metrics.speech_rate_wpm = transcript_metrics.word_count / duration_minutes

            # Perform metric analysis against benchmarks
            metric_analysis = self._analyze_transcript_metrics(transcript_metrics)
            results["metric_analysis"] = metric_analysis

            # Search for content delivery best practices
            content_query = "Speaking pace speech rate lexical diversity content delivery"
            content_results = self.retrieve(content_query, top_k=2)

        # 5. Always retrieve timing and duration practices
        timing_query = "Best upload days hours timing"
        timing_results = self.retrieve(timing_query, top_k=2)

        duration_query = "Video duration optimal length minutes"
        duration_results = self.retrieve(duration_query, top_k=1)

        # 6. Combine and deduplicate results WITH SUBJECT FILTERING
        seen_ids = set()
        subject_results_list = []
        global_results_list = []
        generic_results_list = []

        for doc, score in title_results + subject_results + desc_results + content_results + timing_results + duration_results:
            if doc.id not in seen_ids:
                doc_subject = doc.metadata.get('subject', '')
                seen_ids.add(doc.id)

                result_item = {
                    "id": doc.id,
                    "category": doc.category,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": round(score, 4)
                }

                if doc_subject:
                    # This is a subject-tagged document
                    if doc_subject.lower() == 'global':
                        global_results_list.append(result_item)
                    elif subject and doc_subject.lower() == subject.lower():
                        # Matching subject - high priority
                        subject_results_list.append(result_item)
                    # Skip other subjects
                else:
                    # Generic document without subject tag
                    generic_results_list.append(result_item)

        # Prioritize: subject-specific > global > generic
        # Use global as fallback if subject-specific results are < 2
        all_results = subject_results_list.copy()

        # Add global fallback if insufficient subject-specific results
        if len(all_results) < 3:
            for item in global_results_list:
                if item['id'] not in {r['id'] for r in all_results}:
                    all_results.append(item)

        # Add generic results
        for item in generic_results_list:
            if item['id'] not in {r['id'] for r in all_results}:
                all_results.append(item)

        # Sort by relevance score
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        results["retrieved_practices"] = all_results[:top_k]

        # Add metadata about retrieval sources
        results["retrieval_info"] = {
            "subject_specific_count": len(subject_results_list),
            "global_fallback_used": len(subject_results_list) < 3 and len(global_results_list) > 0,
            "total_candidates": len(all_results)
        }

        return results

    def _analyze_transcript_metrics(self, metrics: TranscriptMetrics) -> Dict[str, Any]:
        """
        Analyze transcript metrics against benchmarks from high-performing videos.

        Returns insights about where the video stands relative to top performers.
        """
        analysis = {
            "metrics_provided": metrics.to_dict(),
            "comparisons": [],
            "recommendations": []
        }

        # 1. Speech Rate Analysis
        if metrics.speech_rate_wpm > 0:
            optimal = TRANSCRIPT_BENCHMARKS["speech_rate_wpm"]["optimal"]
            diff = metrics.speech_rate_wpm - optimal

            if abs(diff) <= 10:
                status = "✓ optimal"
            elif metrics.speech_rate_wpm < optimal - 10:
                status = "⚠️ too slow"
                analysis["recommendations"].append(
                    f"Increase speaking pace from {metrics.speech_rate_wpm:.0f} to ~{optimal} words/min"
                )
            else:
                status = "⚠️ too fast"
                analysis["recommendations"].append(
                    f"Slow down speaking pace from {metrics.speech_rate_wpm:.0f} to ~{optimal} words/min"
                )

            analysis["comparisons"].append({
                "metric": "speech_rate_wpm",
                "value": round(metrics.speech_rate_wpm, 1),
                "benchmark": optimal,
                "status": status
            })

        # 2. Lexical Diversity Analysis
        if metrics.lexical_diversity > 0:
            high_mean = TRANSCRIPT_BENCHMARKS["lexical_diversity"]["high_mean"]
            low_mean = TRANSCRIPT_BENCHMARKS["lexical_diversity"]["low_mean"]

            if metrics.lexical_diversity >= high_mean:
                status = "✓ excellent vocabulary diversity"
            elif metrics.lexical_diversity >= (high_mean + low_mean) / 2:
                status = "○ moderate vocabulary diversity"
                analysis["recommendations"].append(
                    f"Improve vocabulary diversity from {metrics.lexical_diversity:.2f} to ~{high_mean:.2f}"
                )
            else:
                status = "⚠️ low vocabulary diversity"
                analysis["recommendations"].append(
                    f"Use more varied vocabulary (current: {metrics.lexical_diversity:.2f}, target: {high_mean:.2f})"
                )

            analysis["comparisons"].append({
                "metric": "lexical_diversity",
                "value": round(metrics.lexical_diversity, 3),
                "benchmark": high_mean,
                "status": status
            })

        # 3. Sentence Length Analysis
        if metrics.avg_words_per_sentence > 0:
            optimal_range = TRANSCRIPT_BENCHMARKS["avg_words_per_sentence"]["optimal_range"]

            if optimal_range[0] <= metrics.avg_words_per_sentence <= optimal_range[1]:
                status = "✓ good sentence length"
            elif metrics.avg_words_per_sentence < optimal_range[0]:
                status = "⚠️ sentences too short"
                analysis["recommendations"].append(
                    "Use slightly longer, more complete sentences for clarity"
                )
            else:
                status = "⚠️ sentences too long"
                analysis["recommendations"].append(
                    "Break down long sentences for easier student comprehension"
                )

            analysis["comparisons"].append({
                "metric": "avg_words_per_sentence",
                "value": round(metrics.avg_words_per_sentence, 1),
                "benchmark": f"{optimal_range[0]}-{optimal_range[1]} words",
                "status": status
            })

        # 4. Reading Ease (Flesch)
        if metrics.flesch_reading_ease != 0:
            high_mean = TRANSCRIPT_BENCHMARKS["flesch_reading_ease"]["high_mean"]

            # For Arabic, higher (less negative) is generally better
            if metrics.flesch_reading_ease > high_mean:
                status = "✓ accessible language"
            else:
                status = "○ complex language"
                analysis["recommendations"].append(
                    "Consider simplifying language for better student comprehension"
                )

            analysis["comparisons"].append({
                "metric": "flesch_reading_ease",
                "value": round(metrics.flesch_reading_ease, 1),
                "benchmark": round(high_mean, 1),
                "status": status
            })

        return analysis

    def save(self, path: str = "models/rag_index"):
        """Save the index and documents for later use."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "faiss.index"))

        # Save documents (without embeddings to save space)
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                "id": doc.id,
                "category": doc.category,
                "content": doc.content,
                "metadata": doc.metadata
            })

        with open(save_path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        # Save embeddings separately
        embeddings = np.vstack([doc.embedding for doc in self.documents])
        np.save(save_path / "embeddings.npy", embeddings)

        print(f"Saved RAG index to {save_path}")

    def load(self, path: str = "models/rag_index"):
        """Load a previously saved index."""
        load_path = Path(path)

        # Load model first
        if self.model is None:
            self.load_model()

        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "faiss.index"))

        # Load documents
        with open(load_path / "documents.json", 'r', encoding='utf-8') as f:
            docs_data = json.load(f)

        # Load embeddings
        embeddings = np.load(load_path / "embeddings.npy")

        # Reconstruct documents
        self.documents = []
        for doc_data, emb in zip(docs_data, embeddings, strict=False):
            self.documents.append(Document(
                id=doc_data['id'],
                category=doc_data['category'],
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                embedding=emb
            ))

        print(f"Loaded RAG index with {len(self.documents)} documents")


def build_rag_system(
    knowledge_base_dir: str = "knowledge_base",
    save_path: str = "models/rag_index"
) -> BestPracticesRAG:
    """
    Build and save the complete RAG system.

    Args:
        knowledge_base_dir: Path to knowledge base JSON files
        save_path: Where to save the index

    Returns:
        Initialized BestPracticesRAG system
    """
    rag = BestPracticesRAG(knowledge_base_dir=knowledge_base_dir)

    # Load and process knowledge base
    rag.load_knowledge_base()

    # Create embeddings
    rag.create_embeddings()

    # Build search index
    rag.build_index()

    # Save for later use
    rag.save(save_path)

    return rag


# ============================================
# USAGE EXAMPLE
# ============================================

def main():
    """Demo the RAG system."""

    # Build the RAG system
    print("=" * 50)
    print("Building RAG System")
    print("=" * 50)

    rag = build_rag_system()

    # Test retrieval
    print("\n" + "=" * 50)
    print("Testing Retrieval")
    print("=" * 50)

    # Example query
    test_title = "حل تمرين في النهايات باك 2024"
    test_description = "شرح مفصل لحل تمارين النهايات للسنة الثالثة ثانوي"
    test_subject = "Maths"

    print("\nQuery:")
    print(f"  Title: {test_title}")
    print(f"  Subject: {test_subject}")
    print(f"  Description: {test_description[:50]}...")

    results = rag.retrieve_best_practices(
        title=test_title,
        description=test_description,
        subject=test_subject,
        top_k=5
    )

    print(f"\nRetrieved {len(results['retrieved_practices'])} best practices:")
    print("-" * 40)

    for i, practice in enumerate(results['retrieved_practices'], 1):
        print(f"\n{i}. [{practice['category']}] (score: {practice['relevance_score']:.3f})")
        print(f"   {practice['content'][:200]}...")

    if results['subject_specific']:
        print(f"\n📚 Subject-Specific Insights for {test_subject}:")
        print(results['subject_specific']['content'][:300])

    # Test with transcript metrics
    print("\n" + "=" * 50)
    print("Testing Retrieval WITH Transcript Metrics")
    print("=" * 50)

    # Create sample transcript metrics
    test_metrics = TranscriptMetrics(
        word_count=3500,
        char_count=18000,
        sentence_count=180,
        avg_words_per_sentence=19.4,
        lexical_diversity=0.32,
        unique_word_count=1200,
        flesch_reading_ease=-1500.0,
        flesch_kincaid_grade=8.5,
        speech_rate_wpm=0  # Will be calculated from duration
    )

    print("\nQuery with transcript metrics:")
    print(f"  Title: {test_title}")
    print(f"  Subject: {test_subject}")
    print("  Duration: 30 minutes")
    print(f"  Word count: {test_metrics.word_count}")
    print(f"  Lexical diversity: {test_metrics.lexical_diversity}")

    results_with_metrics = rag.retrieve_best_practices(
        title=test_title,
        description=test_description,
        subject=test_subject,
        transcript_metrics=test_metrics,
        duration_minutes=30,
        top_k=5
    )

    print("\n📊 Metric Analysis:")
    if results_with_metrics.get('metric_analysis'):
        analysis = results_with_metrics['metric_analysis']

        print("\nComparisons against benchmarks:")
        for comp in analysis['comparisons']:
            print(f"  - {comp['metric']}: {comp['value']} (benchmark: {comp['benchmark']}) {comp['status']}")

        if analysis['recommendations']:
            print("\n📌 Transcript-based Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")

    print(f"\nRetrieved {len(results_with_metrics['retrieved_practices'])} best practices")

    return rag


if __name__ == "__main__":
    main()

