"""
LLM Recommendation Agent for Educational Video Content
=======================================================
This module uses an LLM (Groq) to generate personalized, actionable
recommendations for educational video creators based on:
- Retrieved best practices from RAG
- Video characteristics (title, description, transcript)
- Predicted engagement score (when available)

The agent synthesizes all inputs into clear, prioritized recommendations.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Groq for LLM inference
from groq import Groq

# Local imports
from rag_system import BestPracticesRAG, TranscriptMetrics
from thumbnail_analyzer import analyze_thumbnail

# Load environment variables
load_dotenv()


@dataclass
class VideoInput:
    """Input data for a video to analyze."""
    title: str
    description: str = ""
    transcript: str = ""
    subject: str = ""
    duration_minutes: float = 0
    is_exam_focused: bool = False
    transcript_metrics: Optional['TranscriptMetrics'] = None
    thumbnail_path: Optional[str] = None  # Path to thumbnail image


@dataclass
class RecommendationOutput:
    """Structured output from the recommendation agent."""
    predicted_score: float
    score_interpretation: str
    strengths: List[str]
    critical_improvements: List[str]
    detailed_recommendations: List[Dict[str, str]]
    benchmarks: Dict[str, Any]
    raw_response: str


class RecommendationPromptTemplate:
    """
    Structured prompt templates for the LLM recommendation agent.

    Provides clear context and expected output format for generating
    actionable video improvement recommendations.
    """

    SYSTEM_PROMPT = """أنت مستشار خبير في تحسين الفيديوهات التعليمية للبكالوريا الجزائرية.

إرشادات:
1. استخدم البيانات المقدمة لتقديم توصيات محددة
2. عند ذكر نسب التحسين، استخدم قيم difference_pct من البيانات
3. قدم توصيات عملية وقابلة للتنفيذ
4. اذكر الأرقام من البيانات المقدمة (المدة المثالية، طول العنوان، الكلمات المفتاحية)
5. استخدم جداول للمقارنات

ملاحظة: البيانات مستخرجة من تحليل أكثر من 10,000 فيديو تعليمي جزائري."""

    USER_PROMPT_TEMPLATE = """# تحليل الفيديو

## 1. معلومات الفيديو

| الخاصية | القيمة |
|---------|--------|
| العنوان | {title} |
| طول العنوان | {title_length} حرف |
| المادة | {subject} |
| المدة | {duration_minutes} دقيقة |
| محتوى امتحاني | {exam_focus_text} |

**الوصف:** {description}

{prediction_section}

{transcript_metrics_section}

{thumbnail_section}

---

## 2. البيانات المرجعية من قاعدة المعرفة

### معايير المادة ({subject})
{subject_benchmarks}

### توصيات العنوان والكلمات المفتاحية
{title_practices}

### توصيات التوقيت
{timing_practices}

### توصيات المدة
{duration_practices}

---

## 3. المطلوب

قدم تحليلاً مفصلاً:

### أ. جدول المقارنة
| الخاصية | قيمة الفيديو | المعيار المثالي | الفجوة |
|---------|-------------|-----------------|--------|
| طول العنوان | {title_length} | (من البيانات) | |
| المدة | {duration_minutes} دقيقة | (من البيانات) | |
| (أكمل باقي الخصائص) |

### ب. توصيات العنوان
- تحليل الكلمات المفتاحية الموجودة
- كلمات مفتاحية مقترحة مع نسبة التأثير (difference_pct)
- عنوان بديل مقترح

### ج. توصيات المدة
- هل المدة مناسبة للمادة؟
- المدة المثالية حسب البيانات

### د. توصيات المحتوى والإلقاء
{content_recommendations}

### هـ. توصيات الوصف
**البيانات المستخرجة من التحليل:**
{description_recommendations}

**ملاحظة:** يجب ذكر الأرقام المحددة من البيانات أعلاه (عدد الأحرف المثالي، النسب المئوية) في توصياتك.

### و. توصيات التوقيت
- أفضل أيام النشر (من البيانات)
- أفضل ساعات النشر (من البيانات)

### ز. توصيات خاصة بالمادة ({subject})
- ما يميز الفيديوهات الناجحة في هذه المادة

### ح. توصيات الصورة المصغرة (Thumbnail)
{thumbnail_recommendations}"""

    TRANSCRIPT_METRICS_SECTION = """
### مقاييس الإلقاء

| المقياس | القيمة | المعيار | الحالة |
|---------|--------|---------|--------|
| سرعة الكلام | {speech_rate} ك/د | {speech_benchmark} | {speech_status} |
| تنوع المفردات | {lexical_diversity} | {lexical_benchmark} | {lexical_status} |
| طول الجملة | {avg_sentence_length} | {sentence_benchmark} | {sentence_status} |

**توصيات الإلقاء:**
{transcript_recommendations}
"""

    NO_TRANSCRIPT_SECTION = ""

    THUMBNAIL_SECTION = """
### تحليل الصورة المصغرة (Thumbnail)

**الميزات البصرية:**
| الخاصية | القيمة | المعيار المثالي |
|---------|--------|-----------------|
| السطوع | {brightness:.0f} | 80-180 |
| التباين | {contrast:.2f} | 0.15-0.35 |
| التشبع | {saturation:.0f} | 80-160 |
| الألوان السائدة | {dominant_colors} | - |

**النص المستخرج من الصورة:**
{extracted_text}

**ملاحظة للمحلل:** قيّم النص المستخرج واقترح تحسينات لعدد الكلمات، حجم الخط، وموقع النص.
"""

    NO_THUMBNAIL_SECTION = ""

    @classmethod
    def build_prompt(
        cls,
        video: 'VideoInput',
        retrieved_practices: List[Dict],
        subject_benchmarks: Optional[Dict] = None,
        metric_analysis: Optional[Dict] = None,
        thumbnail_analysis: Optional[Dict] = None,
        total_videos: int = 10000,
        prediction: Optional[Dict] = None,
    ) -> str:
        """Build the complete prompt for the LLM."""

        # Separate practices by category
        title_practices = []
        timing_practices = []
        duration_practices = []
        description_practices = []
        other_practices = []

        for practice in retrieved_practices:
            category = practice.get('category', '')
            content = practice.get('content', '')

            if 'title' in category.lower():
                title_practices.append(content)
            elif 'timing' in category.lower() or 'upload' in category.lower():
                timing_practices.append(content)
            elif 'duration' in category.lower():
                duration_practices.append(content)
            elif 'description' in category.lower():
                description_practices.append(content)
            else:
                other_practices.append(content)

        # Extract description recommendations from subject benchmarks if available
        if subject_benchmarks and 'description_optimization' in subject_benchmarks:
            desc_opt = subject_benchmarks['description_optimization']
            if 'recommendations' in desc_opt:
                description_practices.extend(desc_opt['recommendations'])

        # Format subject benchmarks
        benchmarks_text = cls._format_benchmarks(subject_benchmarks, video.subject)

        # Handle transcript metrics section
        if metric_analysis and metric_analysis.get('comparisons'):
            transcript_section = cls._format_transcript_metrics(metric_analysis, subject_benchmarks)
            content_recs = "استخدم بيانات الإلقاء أعلاه لتقديم توصيات"
        else:
            transcript_section = cls.NO_TRANSCRIPT_SECTION
            content_recs = "لم يتم توفير بيانات الإلقاء"

        # Handle thumbnail section
        if thumbnail_analysis and not thumbnail_analysis.get('error'):
            thumbnail_section = cls._format_thumbnail_analysis(thumbnail_analysis)
            thumbnail_recs = "قدم توصيات لتحسين الصورة المصغرة بناءً على البيانات أعلاه"
        else:
            thumbnail_section = cls.NO_THUMBNAIL_SECTION
            thumbnail_recs = "لم يتم توفير صورة مصغرة للتحليل"

        # Compute exam focus text
        exam_focus_text = "نعم ✓" if video.is_exam_focused else "لا"

        # Format description recommendations
        desc_recs = '\n'.join(description_practices) if description_practices else "راجع التوصيات العامة للوصف"

        # Build the prompt
        prompt = cls.USER_PROMPT_TEMPLATE.format(
            title=video.title,
            title_length=len(video.title),
            subject=video.subject or "غير محدد",
            duration_minutes=video.duration_minutes or "غير محدد",
            exam_focus_text=exam_focus_text,
            description=video.description or "لا يوجد وصف",
            prediction_section=cls._format_prediction(prediction),
            transcript_metrics_section=transcript_section,
            thumbnail_section=thumbnail_section,
            subject_benchmarks=benchmarks_text,
            title_practices='\n'.join(title_practices) if title_practices else "راجع التوصيات العامة",
            timing_practices='\n'.join(timing_practices) if timing_practices else "أفضل الأيام: الخميس، الأربعاء. أفضل الساعات: 16-20",
            duration_practices='\n'.join(duration_practices) if duration_practices else "المدة المثالية: 20-35 دقيقة",
            description_recommendations=desc_recs,
            content_recommendations=content_recs,
            thumbnail_recommendations=thumbnail_recs,
            total_videos=total_videos
        )


        return prompt

    @classmethod
    def _format_transcript_metrics(cls, metric_analysis: Dict, subject_benchmarks: Optional[Dict] = None) -> str:
        """Format transcript metrics with comparison data and subject-specific benchmarks."""
        comparisons = metric_analysis.get('comparisons', [])
        recommendations = metric_analysis.get('recommendations', [])

        def get_metric(name):
            for comp in comparisons:
                if comp['metric'] == name:
                    return comp
            return None

        speech = get_metric('speech_rate_wpm')
        lexical = get_metric('lexical_diversity')
        sentence = get_metric('avg_words_per_sentence')

        # Get subject-specific benchmarks or use defaults
        speech_benchmark = "105"
        lexical_benchmark = "0.36"
        sentence_benchmark = "10-20"

        if subject_benchmarks and 'content_delivery' in subject_benchmarks:
            cd = subject_benchmarks['content_delivery']
            if 'feature_analysis' in cd:
                fa = cd['feature_analysis']
                if 'speech_rate_wpm' in fa and fa['speech_rate_wpm'].get('high_median'):
                    speech_benchmark = f"{fa['speech_rate_wpm']['high_median']:.0f}"
                if 'lexical_diversity' in fa and fa['lexical_diversity'].get('high_median'):
                    lexical_benchmark = f"{fa['lexical_diversity']['high_median']:.2f}"
                if 'avg_words_per_sentence' in fa and fa['avg_words_per_sentence'].get('high_median'):
                    val = fa['avg_words_per_sentence']['high_median']
                    sentence_benchmark = f"{val:.0f}-{val*1.5:.0f}"

        return cls.TRANSCRIPT_METRICS_SECTION.format(
            speech_rate=speech['value'] if speech else 'N/A',
            speech_status=speech['status'] if speech else '',
            speech_benchmark=speech_benchmark,
            lexical_diversity=lexical['value'] if lexical else 'N/A',
            lexical_status=lexical['status'] if lexical else '',
            lexical_benchmark=lexical_benchmark,
            avg_sentence_length=sentence['value'] if sentence else 'N/A',
            sentence_status=sentence['status'] if sentence else '',
            sentence_benchmark=sentence_benchmark,
            transcript_recommendations='\n'.join(f'- {r}' for r in recommendations) if recommendations else 'لا توجد مشاكل'
        )

    @classmethod
    def _format_thumbnail_analysis(cls, thumbnail_analysis: Dict) -> str:
        """Format thumbnail analysis for the LLM prompt."""
        visual = thumbnail_analysis.get('visual', {})
        text_data = thumbnail_analysis.get('text', {})

        # Extract visual features
        brightness = visual.get('brightness', 0)
        contrast = visual.get('contrast', 0)
        saturation = visual.get('saturation', 0)

        # Format dominant colors
        colors = visual.get('dominant_colors', [])
        if colors:
            colors_str = ", ".join([c.get('label', 'Unknown') for c in colors[:3]])
        else:
            colors_str = "غير محدد"

        # Get extracted text for LLM to analyze
        extracted_text = text_data.get('content', '')
        if extracted_text:
            extracted_text = f'"{extracted_text}"'
        else:
            extracted_text = "لم يتم اكتشاف نص في الصورة"

        return cls.THUMBNAIL_SECTION.format(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            dominant_colors=colors_str,
            extracted_text=extracted_text
        )

    @staticmethod
    def _format_practices(practices: List[Dict]) -> str:
        """Format retrieved practices for the prompt."""
        if not practices:
            return "لا تتوفر ممارسات مسترجعة"

        formatted = []
        for i, practice in enumerate(practices, 1):
            category = practice.get('category', 'عام')
            content = practice.get('content', '')
            score = practice.get('relevance_score', 0)

            formatted.append(f"**{i}. [{category}]** (ملاءمة: {score:.2f})")
            formatted.append(content)
            formatted.append("")

        return "\n".join(formatted)

    CATEGORY_AR = {"Low": "منخفض", "Medium": "متوسط", "High": "مرتفع"}
    THIRD_AR = {"Low": "الثلث الأدنى", "Medium": "الثلث الأوسط", "High": "الثلث الأعلى"}

    @classmethod
    def _format_prediction(cls, prediction: Optional[Dict]) -> str:
        """One section with the engagement model's estimate for this planned video."""
        if not prediction:
            return ""
        category = prediction["engagement_category"]
        text = (
            f"**التفاعل المتوقع (نموذج التعلم الآلي):** {cls.CATEGORY_AR[category]} "
            f"({cls.THIRD_AR[category]} مقارنة بفيديوهات التدريب، درجة {prediction['engagement_score']:.2f})"
        )
        if not prediction.get("known_channel", True):
            text += "\nالقناة غير موجودة في بيانات التدريب، لذلك التقدير مبني على متوسطات القنوات."
        return text + "\nاربط توصياتك بهذا التقدير: ما الذي يرفع التفاعل المتوقع؟"

    @staticmethod
    def _format_benchmarks(benchmarks: Optional[Dict], subject: str) -> str:
        """Format subject benchmarks for the prompt."""
        if not benchmarks:
            return f"لا تتوفر معايير مرجعية محددة للمادة: {subject}"

        lines = []
        lines.append(f"- عدد الفيديوهات المحللة: {benchmarks.get('total_videos', 'N/A')}")
        lines.append(f"- المدة المثالية: {benchmarks.get('optimal_duration_minutes', 'N/A')} دقيقة")
        lines.append(f"- طول العنوان المثالي: {benchmarks.get('optimal_title_length', 'N/A')} حرف")
        lines.append(f"- نسبة المحتوى المركز على الامتحان: {benchmarks.get('exam_focused_ratio', 'N/A')}%")

        if 'top_channels' in benchmarks:
            lines.append("\nأفضل القنوات في هذه المادة:")
            for channel, score in benchmarks['top_channels'].items():
                lines.append(f"  - {channel}: {score:.2f}")

        return "\n".join(lines)


class RecommendationAgent:
    """
    LLM-powered recommendation agent for educational video optimization.

    Integrates:
    - RAG system for best practices retrieval
    - Groq LLM for recommendation generation
    - Structured prompts for consistent output
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        rag_index_path: str = "models/rag_index"
    ):
        """
        Initialize the recommendation agent.

        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model_name: Groq model to use (or set GROQ_MODEL env var; default openai/gpt-oss-120b)
            rag_index_path: Path to saved RAG index
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.rag_index_path = rag_index_path

        # Initialize components
        self.rag = None
        self.client = None

    def initialize(self):
        """Initialize RAG and LLM components."""
        # Configure Groq
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable.")

        self.client = Groq(api_key=self.api_key)
        print(f"Initialized Groq client with model: {self.model_name}")

        # Load RAG system
        self.rag = BestPracticesRAG()
        self.rag.load(self.rag_index_path)
        print(f"Loaded RAG index from: {self.rag_index_path}")

    def analyze_video(
        self,
        video: VideoInput,
        prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a video and generate recommendations.

        Args:
            video: Video characteristics
            prediction: Optional EngagementPredictor.predict() output
                (engagement_score, engagement_category, known_channel)

        Returns:
            Dictionary with analysis and recommendations
        """
        if self.rag is None or self.client is None:
            self.initialize()

        # 1. Retrieve relevant best practices WITH transcript metrics
        print(f"Retrieving best practices for: {video.title[:50]}...")
        rag_results = self.rag.retrieve_best_practices(
            title=video.title,
            description=video.description,
            subject=video.subject,
            transcript_metrics=video.transcript_metrics,
            duration_minutes=video.duration_minutes,
            top_k=5
        )

        # 2. Get subject-specific benchmarks - try loading full JSON first
        subject_benchmarks = self._load_subject_benchmarks(video.subject)

        # Fall back to parsing from RAG content if JSON not found
        if not subject_benchmarks and rag_results.get('subject_specific'):
            subject_content = rag_results['subject_specific'].get('content', '')
            subject_benchmarks = self._parse_subject_benchmarks(subject_content)

        # 3. Get metric analysis from RAG
        metric_analysis = rag_results.get('metric_analysis')

        # 4. Analyze thumbnail if provided (BEFORE LLM call to include in prompt)
        thumbnail_analysis = None
        if video.thumbnail_path:
            try:
                print(f"Analyzing thumbnail: {video.thumbnail_path}")
                thumbnail_analysis = analyze_thumbnail(video.thumbnail_path)
            except Exception as e:
                print(f"Warning: Thumbnail analysis failed: {e}")
                thumbnail_analysis = {"error": str(e)}

        # 5. Build the prompt with all context INCLUDING thumbnail
        prompt = RecommendationPromptTemplate.build_prompt(
            video=video,
            retrieved_practices=rag_results['retrieved_practices'],
            subject_benchmarks=subject_benchmarks,
            metric_analysis=metric_analysis,
            thumbnail_analysis=thumbnail_analysis,
            total_videos=10030,
            prediction=prediction,
        )

        # 6. Generate recommendations with Groq (low temperature for factual output)
        print("Generating recommendations with LLM...")
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": RecommendationPromptTemplate.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.model_name,
            temperature=0.3,
            max_tokens=4096
        )

        response_text = chat_completion.choices[0].message.content
        # The LLM uses <br> for line breaks inside Markdown tables; Markdown shows them as text.
        response_text = re.sub(r"\s*<br\s*/?>\s*", " ", response_text or "", flags=re.IGNORECASE)

        # 7. Structure the output
        result = {
            "video_info": {
                "title": video.title,
                "subject": video.subject,
                "duration_minutes": video.duration_minutes,
                "is_exam_focused": video.is_exam_focused,
                "has_thumbnail": video.thumbnail_path is not None
            },
            "prediction": prediction,
            "retrieved_practices_count": len(rag_results['retrieved_practices']),
            "subject_benchmarks": subject_benchmarks,
            "metric_analysis": metric_analysis,
            "thumbnail_analysis": thumbnail_analysis,
            "recommendations": response_text,
            "rag_context": rag_results['retrieved_practices'][:3]
        }

        return result

    def _parse_subject_benchmarks(self, content: str) -> Dict[str, Any]:
        """Parse subject benchmarks from RAG content string."""
        import re

        benchmarks = {}

        # Extract total videos
        match = re.search(r'Total videos analyzed:\s*(\d+)', content)
        if match:
            benchmarks['total_videos'] = int(match.group(1))

        # Extract optimal duration
        match = re.search(r'Optimal duration:\s*([\d.]+)', content)
        if match:
            benchmarks['optimal_duration_minutes'] = float(match.group(1))

        # Extract optimal title length
        match = re.search(r'Optimal title length:\s*([\d.]+)', content)
        if match:
            benchmarks['optimal_title_length'] = float(match.group(1))

        # Extract exam-focused ratio
        match = re.search(r'Exam-focused ratio:\s*([\d.]+)', content)
        if match:
            benchmarks['exam_focused_ratio'] = float(match.group(1))

        # Extract top channels
        channels_match = re.findall(r'-\s*(.+?):\s*([\d.]+)', content)
        if channels_match:
            benchmarks['top_channels'] = {name.strip(): float(score) for name, score in channels_match[:3]}

        return benchmarks

    def _load_subject_benchmarks(self, subject: str) -> Optional[Dict[str, Any]]:
        """Load full subject benchmarks from JSON file for dynamic metrics."""
        if not subject:
            return None

        # Try to load subject-specific JSON file
        kb_dir = Path(__file__).parent.parent / "knowledge_base"
        subject_file = kb_dir / f"{subject}_best_practices.json"

        if subject_file.exists():
            try:
                with open(subject_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load subject benchmarks: {e}")

        return None

    def generate_quick_tips(self, video: VideoInput) -> List[str]:
        """
        Generate quick, actionable tips without full analysis.

        Useful for real-time feedback during video creation.
        """
        tips = []

        # Title analysis
        title_len = len(video.title)
        if title_len < 30:
            tips.append(f"⚠️ العنوان قصير جداً ({title_len} حرف). الطول المثالي: 45-60 حرف")
        elif title_len > 80:
            tips.append(f"⚠️ العنوان طويل جداً ({title_len} حرف). حاول اختصاره")
        else:
            tips.append(f"✓ طول العنوان مناسب ({title_len} حرف)")

        # Keywords check
        high_impact_keywords = ['باك', 'bac', 'حل', 'تمرين', 'ملخص', 'شرح', 'امتحان']
        found_keywords = [kw for kw in high_impact_keywords if kw in video.title.lower()]

        if found_keywords:
            tips.append(f"✓ يحتوي العنوان على كلمات مفتاحية فعالة: {', '.join(found_keywords)}")
        else:
            tips.append("💡 أضف كلمات مفتاحية مثل: باك، حل، تمرين، ملخص")

        # Duration check
        if video.duration_minutes:
            if video.duration_minutes < 5:
                tips.append("⚠️ الفيديو قصير جداً. المدة المثالية: 15-40 دقيقة")
            elif video.duration_minutes > 60:
                tips.append("💡 فكر في تقسيم الفيديو لأجزاء أقصر")
            else:
                tips.append(f"✓ مدة الفيديو مناسبة ({video.duration_minutes} دقيقة)")

        # Description check
        if not video.description:
            tips.append("⚠️ أضف وصفاً للفيديو يتضمن ملخص المحتوى وروابط مفيدة")
        elif len(video.description) < 100:
            tips.append("💡 الوصف قصير. أضف تفاصيل أكثر وروابط للمصادر")

        # Exam focus
        if video.is_exam_focused:
            tips.append("✓ المحتوى مركز على الامتحان - هذا يزيد التفاعل بنسبة ~25%")

        return tips


def create_recommendation_report(result: Dict[str, Any]) -> str:
    """
    Create a formatted markdown report from recommendations.

    Args:
        result: Output from RecommendationAgent.analyze_video()

    Returns:
        Formatted markdown string
    """
    report = []

    # Header
    report.append("# 📊 تقرير توصيات تحسين الفيديو")
    report.append("")

    # Video info
    info = result['video_info']
    report.append("## معلومات الفيديو")
    report.append(f"- **العنوان:** {info['title']}")
    report.append(f"- **المادة:** {info['subject'] or 'غير محدد'}")
    report.append(f"- **المدة:** {info['duration_minutes'] or 'غير محدد'} دقيقة")
    report.append("")

    # Predicted score
    if result.get('prediction'):
        report.append("## التفاعل المتوقع")
        report.append(RecommendationPromptTemplate._format_prediction(result['prediction']).split("\n")[0])
        report.append("")

    # Thumbnail analysis
    if result.get('thumbnail_analysis') and not result['thumbnail_analysis'].get('error'):
        thumb = result['thumbnail_analysis']
        report.append("## 🖼️ تحليل الصورة المصغرة")
        report.append("")

        # Visual recommendations
        if thumb.get('recommendations', {}).get('visual'):
            report.append("### التحسينات البصرية")
            for rec in thumb['recommendations']['visual']:
                report.append(f"- {rec}")
            report.append("")

        # Composition recommendations only (text recs handled by LLM)
        if thumb.get('recommendations', {}).get('composition'):
            report.append("### تحسينات التركيب")
            for rec in thumb['recommendations']['composition']:
                report.append(f"- {rec}")
            report.append("")

    # Main recommendations
    report.append("## التوصيات")
    report.append("")
    report.append(result['recommendations'])

    return "\n".join(report)


# ============================================
# USAGE EXAMPLE
# ============================================

def main():
    """Demonstrate the recommendation agent."""

    print("=" * 60)
    print("LLM Recommendation Agent Demo")
    print("=" * 60)

    # Create sample transcript metrics
    sample_metrics = TranscriptMetrics(
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

    # Create a sample video input with transcript metrics
    sample_video = VideoInput(
        title="حل تمرين في النهايات باك 2024",
        description="شرح مفصل لحل تمارين النهايات للسنة الثالثة ثانوي",
        subject="Maths",
        duration_minutes=25,
        is_exam_focused=True,
        transcript="",
        transcript_metrics=sample_metrics,
        thumbnail_path="test-thumbnails/nordine.jpeg",
    )

    # Initialize agent
    agent = RecommendationAgent()

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️ GROQ_API_KEY not set. Showing quick tips only.\n")

        # Generate quick tips (doesn't require LLM)
        tips = agent.generate_quick_tips(sample_video)
        print("Quick Tips:")
        print("-" * 40)
        for tip in tips:
            print(f"  {tip}")

        print("\n💡 To use full LLM recommendations, set GROQ_API_KEY in .env file")
        return

    # Initialize and run full analysis
    agent.initialize()

    print(f"\nAnalyzing video: {sample_video.title}")
    print("-" * 40)

    result = agent.analyze_video(sample_video)

    # Create report
    report = create_recommendation_report(result)
    print("\n" + report)

    # Save report
    output_path = Path("docs/sample_recommendation.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {output_path}")

    return result


if __name__ == "__main__":
    main()
