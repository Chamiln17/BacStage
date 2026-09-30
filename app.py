"""
Bac YouTube Video Optimizer - Streamlit Web Interface
======================================================
A user-friendly interface for Algerian Baccalaureate educational video creators
to get AI-powered recommendations for improving their content.
"""

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path

# Add agent directory to path
sys.path.insert(0, str(Path(__file__).parent / "agent"))

from src.features.bac_keywords import SUBJECTS  # noqa: E402

MODEL_DIR = Path(__file__).parent / "models"
SUBJECT_LABELS = {
    "Maths": "📐 الرياضيات",
    "Physics": "⚛️ الفيزياء",
    "Natural Sciences": "🔬 العلوم الطبيعية",
    "Arabic": "📖 اللغة العربية",
    "French": "🇫🇷 اللغة الفرنسية",
    "English": "🇬🇧 اللغة الإنجليزية",
    "Philosophy": "🤔 الفلسفة",
    "History & Geography": "🏛️ التاريخ والجغرافيا",
    "Islamic Sciences": "🕌 العلوم الإسلامية",
}

# Import agent modules after path setup
try:
    from recommendation_agent import RecommendationAgent, VideoInput, create_recommendation_report
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page config
st.set_page_config(
    page_title="محسّن فيديوهات البكالوريا",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL support and styling
st.markdown("""
<style>
    /* RTL Support for Arabic */
    .rtl {
        direction: rtl;
        text-align: right;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Card styling */
    .recommendation-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    /* Success metric */
    .metric-good {
        color: #28a745;
        font-weight: bold;
    }

    /* Warning metric */
    .metric-warning {
        color: #ffc107;
        font-weight: bold;
    }

    /* Error metric */
    .metric-bad {
        color: #dc3545;
        font-weight: bold;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False


@st.cache_resource
def load_agent():
    """Load the recommendation agent (cached)."""
    try:
        from recommendation_agent import RecommendationAgent
        agent = RecommendationAgent()
        agent.initialize()
        return agent, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_predictor():
    """Load the engagement model (cached). Returns (predictor, error)."""
    if not (MODEL_DIR / "model.joblib").exists():
        return None, "no model"
    try:
        from src.models.predict_model import EngagementPredictor
        return EngagementPredictor(MODEL_DIR), None
    except Exception as e:
        return None, str(e)


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 محسّن فيديوهات البكالوريا</h1>
        <p>أداة ذكاء اصطناعي لتحسين محتوى الفيديوهات التعليمية للطلاب الجزائريين</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with info."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/youtube-play.png", width=80)
        st.title("حول الأداة")

        st.markdown("""
        ### 📊 ما تفعله هذه الأداة؟

        تحلل هذه الأداة فيديوهاتك التعليمية وتقدم:

        - ✅ توصيات لتحسين العنوان
        - ✅ اقتراحات للمدة المثالية
        - ✅ تحليل الصورة المصغرة
        - ✅ أفضل أوقات النشر
        - ✅ نصائح خاصة بالمادة

        ---

        ### 📚 المواد المدعومة
        - الرياضيات
        - الفيزياء
        - العلوم الطبيعية
        - اللغة العربية
        - اللغة الفرنسية
        - اللغة الإنجليزية
        - الفلسفة
        - التاريخ والجغرافيا
        - العلوم الإسلامية

        ---

        ### 📈 مصدر البيانات
        تم تدريب النظام على تحليل أكثر من **10,000 فيديو** تعليمي جزائري.
        """)


def render_input_form():
    """Render the input form for video details."""
    st.header("📝 معلومات الفيديو")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input(
            "عنوان الفيديو *",
            placeholder="مثال: حل تمارين النهايات باك 2024",
            help="أدخل عنوان الفيديو كما سيظهر على يوتيوب"
        )

        subject = st.selectbox(
            "المادة *",
            options=list(SUBJECTS),
            index=list(SUBJECTS).index("Maths"),
            format_func=lambda x: SUBJECT_LABELS.get(x, x),
        )

        duration = st.number_input(
            "مدة الفيديو (بالدقائق) *",
            min_value=1,
            max_value=180,
            value=25,
            help="أدخل المدة التقريبية للفيديو"
        )

    with col2:
        description = st.text_area(
            "وصف الفيديو",
            placeholder="أضف وصفاً مختصراً لمحتوى الفيديو...",
            height=100
        )

        is_exam_focused = st.checkbox(
            "🎯 محتوى مركز على الامتحان",
            value=True,
            help="حدد هذا الخيار إذا كان الفيديو يركز على تمارين الامتحانات"
        )

        thumbnail_file = st.file_uploader(
            "الصورة المصغرة (Thumbnail)",
            type=["jpg", "jpeg", "png"],
            help="ارفع الصورة المصغرة للفيديو للحصول على توصيات بصرية"
        )

    # Video upload section
    st.subheader("📹 رفع الفيديو (اختياري)")
    video_file = st.file_uploader(
        "ملف الفيديو",
        type=["mp4", "mov", "avi", "mkv"],
        help="ارفع الفيديو للحصول على تحليل محتوى متقدم (اختياري)"
    )
    if video_file:
        st.video(video_file)
        st.info("ℹ️ تم رفع الفيديو بنجاح! سيتم استخدامه للتحليل المتقدم.")

    return {
        "title": title,
        "subject": subject,
        "duration": duration,
        "description": description,
        "is_exam_focused": is_exam_focused,
        "thumbnail_file": thumbnail_file,
        "video_file": video_file
    }


def analyze_video(agent, predictor, form_data):
    """Analyze the video and return recommendations."""
    # A planned video: no channel, so the model uses its new-channel estimate.
    prediction = None
    if predictor is not None:
        prediction = predictor.predict({
            "title": form_data["title"],
            "description": form_data["description"],
            "duration_sec": form_data["duration"] * 60,
            "subject": form_data["subject"],
        })

    # Handle thumbnail upload
    thumbnail_path = None
    if form_data["thumbnail_file"]:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(form_data["thumbnail_file"].getvalue())
            thumbnail_path = tmp.name

    # Create VideoInput
    video = VideoInput(
        title=form_data["title"],
        description=form_data["description"],
        subject=form_data["subject"],
        duration_minutes=form_data["duration"],
        is_exam_focused=form_data["is_exam_focused"],
        thumbnail_path=thumbnail_path
    )

    # Analyze
    result = agent.analyze_video(video, prediction=prediction)

    # Cleanup temp file
    if thumbnail_path and os.path.exists(thumbnail_path):
        os.unlink(thumbnail_path)

    return result


def render_results(result):
    """Render the analysis results."""
    st.header("📊 نتائج التحليل")

    # Video info summary
    info = result['video_info']
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("المادة", info['subject'])
    with col2:
        st.metric("المدة", f"{info['duration_minutes']} دقيقة")
    with col3:
        st.metric("محتوى امتحاني", "✅ نعم" if info['is_exam_focused'] else "❌ لا")
    with col4:
        st.metric("ممارسات مسترجعة", result['retrieved_practices_count'])

    st.divider()

    # Engagement model estimate
    prediction = result.get('prediction')
    if prediction:
        labels = {"Low": "📉 منخفض", "Medium": "📊 متوسط", "High": "📈 مرتفع"}
        thirds = {"Low": "الثلث الأدنى", "Medium": "الثلث الأوسط", "High": "الثلث الأعلى"}
        category = prediction['engagement_category']
        st.metric("التفاعل المتوقع", labels[category], help=f"درجة {prediction['engagement_score']:.2f}")
        st.caption(
            f"مقارنة بـ {thirds[category]} من فيديوهات التدريب. "
            "تقدير لقناة جديدة: لا تُستخدم إحصائيات قناتك."
        )
        st.divider()

    # Thumbnail analysis
    if result.get('thumbnail_analysis') and not result['thumbnail_analysis'].get('error'):
        thumb = result['thumbnail_analysis']

        with st.expander("🖼️ تحليل الصورة المصغرة", expanded=True):
            visual = thumb.get('visual', {})

            col1, col2, col3 = st.columns(3)
            with col1:
                brightness = visual.get('brightness', 0)
                status = "✅" if 80 <= brightness <= 180 else "⚠️"
                st.metric("السطوع", f"{status} {brightness:.0f}")
            with col2:
                contrast = visual.get('contrast', 0)
                status = "✅" if 0.15 <= contrast <= 0.35 else "⚠️"
                st.metric("التباين", f"{status} {contrast:.2f}")
            with col3:
                saturation = visual.get('saturation', 0)
                status = "✅" if 80 <= saturation <= 160 else "⚠️"
                st.metric("التشبع", f"{status} {saturation:.0f}")

            # Dominant colors
            colors = visual.get('dominant_colors', [])
            if colors:
                st.write("**الألوان السائدة:**")
                color_cols = st.columns(len(colors))
                for i, color in enumerate(colors):
                    with color_cols[i]:
                        rgb = color.get('rgb', [128, 128, 128])
                        st.markdown(
                            f'<div style="background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); '
                            f'padding: 20px; border-radius: 5px; text-align: center; color: white;">'
                            f'{color.get("label", "Unknown")}<br>{color.get("percentage", 0):.1f}%</div>',
                            unsafe_allow_html=True
                        )

    st.divider()

    # Main recommendations
    st.subheader("📋 التوصيات التفصيلية")
    st.markdown(result['recommendations'])


def main():
    """Main application."""
    init_session_state()
    render_header()
    render_sidebar()

    # Check if agent is available
    if not AGENT_AVAILABLE:
        st.error(f"❌ خطأ في استيراد المكتبات: {IMPORT_ERROR}")
        st.stop()

    # Load agent
    with st.spinner("جاري تحميل نظام التوصيات..."):
        agent, error = load_agent()

    if error:
        st.error(f"❌ خطأ في تحميل النظام: {error}")
        st.stop()

    st.success("✅ تم تحميل نظام التوصيات بنجاح!")

    predictor, predictor_error = load_predictor()
    if predictor_error == "no model":
        st.info("ℹ️ لا يوجد نموذج تفاعل مدرَّب. لتفعيل التقدير: `uv run python run_pipeline.py train`")
    elif predictor_error:
        st.warning(f"⚠️ تعذّر تحميل نموذج التفاعل: {predictor_error}")

    # Input form
    form_data = render_input_form()

    # Analyze button
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button(
            "🚀 تحليل الفيديو وإنشاء التوصيات",
            type="primary",
            use_container_width=True,
            disabled=not form_data["title"]
        )

    if analyze_clicked:
        if not form_data["title"]:
            st.warning("⚠️ يرجى إدخال عنوان الفيديو")
        else:
            with st.spinner("🔄 جاري التحليل... قد يستغرق هذا بضع ثوانٍ"):
                try:
                    result = analyze_video(agent, predictor, form_data)
                    st.session_state.analysis_result = result
                except Exception as e:
                    st.error(f"❌ حدث خطأ: {str(e)}")

    # Display results
    if st.session_state.analysis_result:
        render_results(st.session_state.analysis_result)

        st.divider()

        report = create_recommendation_report(st.session_state.analysis_result)

        st.download_button(
            label="📥 تحميل التقرير (Markdown)",
            data=report,
            file_name="video_recommendations.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()
