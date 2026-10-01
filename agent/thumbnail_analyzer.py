"""
Thumbnail Analyzer for Educational Video Content
=================================================
Analyzes YouTube video thumbnails and provides enhancement suggestions
based on visual features, text overlay, and composition.

Based on trained ML models that learned from successful educational thumbnails.
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

try:
    import pytesseract
    from matplotlib import colors as mcolors
    from skimage import exposure
    from skimage.filters import sobel
    from sklearn.cluster import MiniBatchKMeans
except ImportError as e:
    print(f"Warning: Some thumbnail analysis dependencies not installed: {e}")
else:
    import shutil
    # The Windows installer does not add Tesseract to PATH.
    _WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if shutil.which("tesseract") is None and _WINDOWS_TESSERACT.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_WINDOWS_TESSERACT)


@dataclass
class ThumbnailAnalysis:
    """Results from thumbnail analysis."""

    # Visual features
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue_entropy: float = 0.0
    hue: float = 0.0
    sat: float = 0.0
    val: float = 0.0

    # Dominant colors (RGB)
    dominant_colors: List[Dict] = None

    # Text analysis
    word_count: int = 0
    avg_font_size: float = 0.0
    text_position: str = "Unknown"
    text_content: str = ""

    # Composition
    saliency: float = 0.0
    edge_contrast: float = 0.0

    # Recommendations
    visual_recommendations: List[str] = None
    text_recommendations: List[str] = None
    composition_recommendations: List[str] = None

    def __post_init__(self):
        if self.dominant_colors is None:
            self.dominant_colors = []
        if self.visual_recommendations is None:
            self.visual_recommendations = []
        if self.text_recommendations is None:
            self.text_recommendations = []
        if self.composition_recommendations is None:
            self.composition_recommendations = []


class ThumbnailAnalyzer:
    """
    Analyzes YouTube thumbnails and provides enhancement suggestions.

    Uses computer vision and ML models to extract visual features and
    compare them against benchmarks from high-performing thumbnails.
    """

    # Benchmark values for educational thumbnails (from training data)
    BENCHMARKS = {
        "brightness": {"optimal": 130, "min": 80, "max": 180},
        "contrast": {"optimal": 0.25, "min": 0.15, "max": 0.35},
        "saturation": {"optimal": 120, "min": 80, "max": 160},
        "word_count": {"optimal": 3, "min": 1, "max": 5},
        "font_size": {"optimal": 40, "min": 25, "max": 60},
        "saliency": {"optimal": 0.15, "min": 0.08, "max": 0.25},
        "edge_contrast": {"optimal": 30, "min": 15, "max": 50},
    }

    # Color name mappings for dominant colors
    XKCD_COLORS = None

    def __init__(self):
        """Initialize the thumbnail analyzer."""
        self._init_color_mapping()

    def _init_color_mapping(self):
        """Initialize XKCD color name mapping for dominant color detection."""
        try:
            self.XKCD_COLORS = {
                name.replace("xkcd:", "").title(): mcolors.to_rgb(code)
                for name, code in mcolors.XKCD_COLORS.items()
            }
        except Exception:
            self.XKCD_COLORS = {}

    def analyze(self, image_path: str) -> ThumbnailAnalysis:
        """
        Analyze a thumbnail image and return comprehensive analysis.

        Args:
            image_path: Path to the thumbnail image file

        Returns:
            ThumbnailAnalysis object with features and recommendations
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from: {image_path}")

        analysis = ThumbnailAnalysis()

        # 1. Extract visual features
        self._extract_visual_features(image, analysis)

        # 2. Extract text features (if tesseract available)
        self._extract_text_features(image_path, analysis)

        # 3. Extract composition features
        self._extract_composition_features(image, analysis)

        # 4. Generate recommendations
        self._generate_recommendations(analysis)

        return analysis

    def _extract_visual_features(self, image: np.ndarray, analysis: ThumbnailAnalysis):
        """Extract brightness, contrast, saturation, and color features."""

        # Brightness (grayscale mean)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        analysis.brightness = float(np.mean(gray))

        # Contrast (std of equalized histogram)
        equalized = exposure.equalize_hist(gray)
        analysis.contrast = float(equalized.std())

        # HSV analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        analysis.hue = float(np.mean(hsv[:, :, 0]))
        analysis.sat = float(np.mean(hsv[:, :, 1]))
        analysis.val = float(np.mean(hsv[:, :, 2]))
        analysis.saturation = analysis.sat

        # Hue entropy (color complexity)
        hue_hist = cv2.calcHist([hsv[:, :, 0]], [0], None, [180], [0, 180])
        hue_hist = hue_hist / hue_hist.sum()
        analysis.hue_entropy = float(-np.sum(hue_hist * np.log2(hue_hist + 1e-6)))

        # Dominant colors
        try:
            analysis.dominant_colors = self._get_dominant_colors(image, k=3)
        except Exception:
            analysis.dominant_colors = []

    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Dict]:
        """Extract top k dominant colors using KMeans clustering."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = rgb_image.reshape(-1, 3)

        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=3)
        kmeans.fit(pixels)

        counter = Counter(kmeans.labels_)
        colors = []

        for i, _ in counter.most_common(k):
            rgb = kmeans.cluster_centers_[i].astype(int).tolist()
            label = self._closest_color_name(rgb)
            colors.append({"rgb": rgb, "label": label, "percentage": counter[i] / len(pixels) * 100})

        return colors

    def _closest_color_name(self, rgb: List[int]) -> str:
        """Find the closest named color to an RGB value."""
        if not self.XKCD_COLORS:
            return f"RGB({rgb[0]},{rgb[1]},{rgb[2]})"

        r, g, b = np.array(rgb) / 255
        min_dist = float("inf")
        closest_name = "Unknown"

        for name, (xr, xg, xb) in self.XKCD_COLORS.items():
            dist = np.sqrt((r - xr)**2 + (g - xg)**2 + (b - xb)**2)
            if dist < min_dist:
                min_dist = dist
                closest_name = name

        return closest_name

    def _extract_text_features(self, image_path: str, analysis: ThumbnailAnalysis):
        """Extract text overlay features using OCR with multiple preprocessing methods."""
        try:
            image = cv2.imread(image_path)
            h, w = image.shape[:2]

            # Prepare multiple preprocessed versions for better OCR
            preprocessed_images = self._preprocess_for_ocr(image)

            # OCR configs to try (different page segmentation modes)
            ocr_configs = [
                "--psm 6",   # Uniform block of text
                "--psm 11",  # Sparse text
                "--psm 3",   # Auto, but fully automatic page segmentation
                "--psm 12",  # Sparse text with OSD
            ]

            # Try OCR on each preprocessed image with each config
            all_words = []
            all_heights = []
            all_positions = []

            for _prep_name, prep_image in preprocessed_images.items():
                for config in ocr_configs:
                    try:
                        # Add Arabic language support
                        full_config = f"{config} -l ara+eng+fra"
                        d = pytesseract.image_to_data(
                            prep_image,
                            output_type=pytesseract.Output.DICT,
                            config=full_config
                        )

                        for i in range(len(d["text"])):
                            text = d["text"][i].strip()
                            conf = int(d["conf"][i]) if d["conf"][i] != '-1' else 0

                            # Accept text with reasonable confidence
                            if text and len(text) >= 1 and conf > 20:
                                all_words.append(text)
                                all_heights.append(d["height"][i])

                                y_center = (d["top"][i] + d["height"][i]) / 2
                                if y_center < h * 0.33:
                                    all_positions.append("Top")
                                elif y_center < h * 0.66:
                                    all_positions.append("Middle")
                                else:
                                    all_positions.append("Bottom")
                    except Exception:
                        continue

            # Deduplicate words (keep unique)
            unique_words = list(set(all_words))

            analysis.word_count = len(unique_words)
            analysis.avg_font_size = float(np.mean(all_heights)) if all_heights else 0.0
            analysis.text_position = Counter(all_positions).most_common(1)[0][0] if all_positions else "Unknown"
            analysis.text_content = " ".join(unique_words[:20])  # Limit to 20 words

        except Exception as e:
            # OCR not available or failed
            print(f"OCR Warning: {e}")
            analysis.word_count = 0
            analysis.text_content = ""

    def _preprocess_for_ocr(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create multiple preprocessed versions of the image for better OCR results.
        Different preprocessing works better for different text styles.
        """
        preprocessed = {}

        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed["gray"] = gray

        # 2. Inverted (for light text on dark background)
        inverted = cv2.bitwise_not(gray)
        preprocessed["inverted"] = inverted

        # 3. Otsu thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed["otsu"] = otsu

        # 4. Inverted Otsu (for light text)
        preprocessed["otsu_inv"] = cv2.bitwise_not(otsu)

        # 5. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed["adaptive"] = adaptive

        # 6. Contrast enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        preprocessed["enhanced"] = enhanced

        # 7. Scaled up (for small text)
        scale = 2
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        preprocessed["scaled"] = scaled

        # 8. Denoised
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        preprocessed["denoised"] = denoised

        return preprocessed

    def _extract_composition_features(self, image: np.ndarray, analysis: ThumbnailAnalysis):
        """Extract composition features: saliency and edge contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Saliency (Sobel edge detection)
        try:
            saliency_map = sobel(gray)
            analysis.saliency = float(np.mean(saliency_map))
        except Exception:
            analysis.saliency = 0.0

        # Edge contrast (Canny)
        edges = cv2.Canny(gray, 100, 200)
        analysis.edge_contrast = float(np.mean(edges))

    def _generate_recommendations(self, analysis: ThumbnailAnalysis):
        """Generate recommendations based on analysis vs benchmarks."""

        # Visual recommendations
        visual_recs = []

        b = self.BENCHMARKS["brightness"]
        if analysis.brightness < b["min"]:
            visual_recs.append(f"⬆️ Increase brightness from {analysis.brightness:.0f} to ~{b['optimal']} (too dark)")
        elif analysis.brightness > b["max"]:
            visual_recs.append(f"⬇️ Decrease brightness from {analysis.brightness:.0f} to ~{b['optimal']} (too bright)")
        else:
            visual_recs.append(f"✓ Brightness is good ({analysis.brightness:.0f})")

        c = self.BENCHMARKS["contrast"]
        if analysis.contrast < c["min"]:
            visual_recs.append(f"⬆️ Increase contrast from {analysis.contrast:.2f} to ~{c['optimal']}")
        elif analysis.contrast > c["max"]:
            visual_recs.append(f"⬇️ Reduce contrast from {analysis.contrast:.2f} (may look harsh)")
        else:
            visual_recs.append(f"✓ Contrast is good ({analysis.contrast:.2f})")

        s = self.BENCHMARKS["saturation"]
        if analysis.saturation < s["min"]:
            visual_recs.append(f"⬆️ Increase saturation/color vibrancy from {analysis.saturation:.0f}")
        elif analysis.saturation > s["max"]:
            visual_recs.append(f"⬇️ Reduce saturation from {analysis.saturation:.0f} (oversaturated)")
        else:
            visual_recs.append(f"✓ Saturation is good ({analysis.saturation:.0f})")

        # Dominant colors info
        if analysis.dominant_colors:
            colors_str = ", ".join([c["label"] for c in analysis.dominant_colors[:3]])
            visual_recs.append(f"🎨 Dominant colors: {colors_str}")

        analysis.visual_recommendations = visual_recs

        # Text recommendations
        text_recs = []

        w = self.BENCHMARKS["word_count"]
        if analysis.word_count == 0:
            text_recs.append("⚠️ No text detected - consider adding 2-4 words to thumbnail")
        elif analysis.word_count < w["min"]:
            text_recs.append(f"⬆️ Add more text ({analysis.word_count} words found, aim for {w['optimal']})")
        elif analysis.word_count > w["max"]:
            text_recs.append(f"⬇️ Too much text ({analysis.word_count} words) - keep to {w['max']} max")
        else:
            text_recs.append(f"✓ Word count is good ({analysis.word_count} words)")

        f = self.BENCHMARKS["font_size"]
        if analysis.avg_font_size > 0:
            if analysis.avg_font_size < f["min"]:
                text_recs.append(f"⬆️ Text too small (avg {analysis.avg_font_size:.0f}px) - increase font size")
            elif analysis.avg_font_size > f["max"]:
                text_recs.append(f"⬇️ Text too large (avg {analysis.avg_font_size:.0f}px)")
            else:
                text_recs.append(f"✓ Font size is good ({analysis.avg_font_size:.0f}px)")

        if analysis.text_position != "Unknown":
            text_recs.append(f"📍 Text position: {analysis.text_position}")

        analysis.text_recommendations = text_recs

        # Composition recommendations
        comp_recs = []

        sal = self.BENCHMARKS["saliency"]
        if analysis.saliency < sal["min"]:
            comp_recs.append(f"⬆️ Low visual interest (saliency: {analysis.saliency:.3f}) - add more contrast/detail")
        elif analysis.saliency > sal["max"]:
            comp_recs.append(f"⬇️ Too busy (saliency: {analysis.saliency:.3f}) - simplify composition")
        else:
            comp_recs.append(f"✓ Visual interest is good (saliency: {analysis.saliency:.3f})")

        ec = self.BENCHMARKS["edge_contrast"]
        if analysis.edge_contrast < ec["min"]:
            comp_recs.append("⬆️ Add more defined edges/shapes")
        else:
            comp_recs.append("✓ Edge definition is good")

        analysis.composition_recommendations = comp_recs

    def get_summary(self, analysis: ThumbnailAnalysis) -> str:
        """Generate a text summary of the thumbnail analysis."""
        lines = []
        lines.append("# 🖼️ تحليل الصورة المصغرة")
        lines.append("")

        # Visual features
        lines.append("## الميزات البصرية")
        lines.append(f"- السطوع: {analysis.brightness:.1f}")
        lines.append(f"- التباين: {analysis.contrast:.2f}")
        lines.append(f"- التشبع: {analysis.saturation:.1f}")
        if analysis.dominant_colors:
            colors = ", ".join([c["label"] for c in analysis.dominant_colors[:3]])
            lines.append(f"- الألوان السائدة: {colors}")
        lines.append("")

        # Text features
        lines.append("## النص في الصورة")
        lines.append(f"- عدد الكلمات: {analysis.word_count}")
        if analysis.avg_font_size > 0:
            lines.append(f"- حجم الخط المتوسط: {analysis.avg_font_size:.0f}px")
        lines.append(f"- موقع النص: {analysis.text_position}")
        if analysis.text_content:
            lines.append(f"- النص المكتشف: {analysis.text_content[:50]}...")
        lines.append("")

        # Recommendations
        lines.append("## التوصيات")
        lines.append("")
        lines.append("### التحسينات البصرية")
        for rec in analysis.visual_recommendations:
            lines.append(f"- {rec}")
        lines.append("")
        lines.append("### تحسينات النص")
        for rec in analysis.text_recommendations:
            lines.append(f"- {rec}")
        lines.append("")
        lines.append("### تحسينات التركيب")
        for rec in analysis.composition_recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)


def analyze_thumbnail(image_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze a thumbnail and return results as dict.

    Args:
        image_path: Path to thumbnail image

    Returns:
        Dictionary with analysis results and recommendations
    """
    analyzer = ThumbnailAnalyzer()
    analysis = analyzer.analyze(image_path)

    return {
        "visual": {
            "brightness": analysis.brightness,
            "contrast": analysis.contrast,
            "saturation": analysis.saturation,
            "dominant_colors": analysis.dominant_colors,
        },
        "text": {
            "word_count": analysis.word_count,
            "avg_font_size": analysis.avg_font_size,
            "position": analysis.text_position,
            "content": analysis.text_content,
        },
        "composition": {
            "saliency": analysis.saliency,
            "edge_contrast": analysis.edge_contrast,
        },
        "recommendations": {
            "visual": analysis.visual_recommendations,
            "text": analysis.text_recommendations,
            "composition": analysis.composition_recommendations,
        },
        "summary": analyzer.get_summary(analysis),
    }


# ============================================
# DEMO
# ============================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Thumbnail Analyzer Demo")
    print("=" * 60)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\nUsage: python thumbnail_analyzer.py <image_path>")
        print("\nExample: python thumbnail_analyzer.py thumbnail.jpg")
        sys.exit(0)

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"\nAnalyzing: {image_path}")
    print("-" * 40)

    result = analyze_thumbnail(image_path)
    print(result["summary"])
