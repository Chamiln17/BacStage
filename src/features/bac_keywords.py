"""
Bac Keywords - Domain-Specific Vocabulary for Algerian Bac Curriculum

This module provides subject-specific keyword dictionaries for:
1. Feature extraction (keyword density, domain alignment)
2. Content classification (exam focus, pedagogical markers)

Sources:
- config/filter_config.yaml (Bac markers)
- docs/00_data_filtering_manual/random_keywordspersubject.md (3AS curriculum)
"""

from typing import Dict, List, Set

# =============================================================================
# SUBJECT-SPECIFIC CURRICULUM KEYWORDS
# =============================================================================

MATHEMATICS_KEYWORDS: List[str] = [
    # Arabic
    "الأعداد المركبة", "التكامل", "الدوال الأصلية", "النهايات",
    "الدالة الأسية", "الدالة اللوغاريتمية", "التحويلات النقطية",
    "المتتاليات العددية", "الهندسة في الفضاء", "الاشتقاقية",
    "الاستمرارية", "حساب التكامل", "التزايد المقارن",
    # French
    "fonction", "dérivée", "intégrale", "limite", "équation",
    "complexe", "logarithme", "exponentielle", "suite", "géométrie",
]

PHYSICS_KEYWORDS: List[str] = [
    # Arabic
    "التحولات النووية", "ثنائي القطب RC", "ثنائي القطب RL",
    "التطورات المهتزة", "مفهوم الموجة", "تطور جملة ميكانيكية",
    "ظواهر كهربائية", "تحول كيميائي", "المتابعة الزمنية",
    # French
    "force", "énergie", "momentum", "vitesse", "onde",
    "nucléaire", "électrique", "mécanique", "chimique",
]

NATURAL_SCIENCES_KEYWORDS: List[str] = [
    # Arabic
    "الاتصال العصبي", "المناعة", "التركيب الضوئي", "التحولات الطاقوية",
    "بيكرت البروتين", "التحفيز الأنزيمي", "التكتونية",
    "التخصص الوظيفي للبروتينات", "الدور في الدفاع عن الذات",
    "البنية الداخلية للأرض", "النشاط التكتوني",
    # French
    "molécule", "cellule", "organisme", "réaction", "évolution",
    "photosynthèse", "protéine", "enzyme", "immunité", "neuronal",
]

HISTORY_KEYWORDS: List[str] = [
    # Arabic
    "القطبية الثنائية", "الصراع بين الشرق والغرب", "الجزائر 1919",
    "الجزائر 1989", "العالم الثالث", "الأزمات الدولية",
    # French
    "bipolaire", "guerre froide", "décolonisation", "indépendance",
]

GEOGRAPHY_KEYWORDS: List[str] = [
    # Arabic
    "الاقتصاد العالمي", "القوى الاقتصادية الكبرى",
    "الاقتصاد والتنمية", "دول الجنوب",
    # French
    "économie mondiale", "puissances économiques", "développement",
]

PHILOSOPHY_KEYWORDS: List[str] = [
    # Arabic
    "فلسفة العلوم", "الإشكالية", "السؤال والإشكالية",
    "العلاقات بين الناس", "انطباق الفكر",
    # French
    "philosophie", "problématique", "éthique", "logique",
]

ISLAMIC_SCIENCES_KEYWORDS: List[str] = [
    # Arabic
    "القرآن الكريم", "الحديث الشريف", "العقيدة والفكر",
    "الفقه وأصوله", "السيرة والهداية",
]

# =============================================================================
# BAC MARKERS (Exam-Related Terms)
# =============================================================================

BAC_EXAM_MARKERS: List[str] = [
    # Core markers
    "bac", "3as", "بكالوريا", "باك", "ثالثة ثانوي",
    "السنة الثالثة ثانوي", "terminale", "3 ثانوي",
    "السنة 3 ثانوي", "جميع الشعب", "شعب علمية", "شعب أدبية",
    # Strong intent phrases
    "مراجعة بكالوريا", "تحضير بكالوريا", "تصحيح بكالوريا",
    "موضوع بكالوريا", "حل موضوع بكالوريا",
    "bac blanc", "révision bac", "corrigé bac", "sujet bac",
]

# =============================================================================
# PEDAGOGICAL MARKERS
# =============================================================================

EXPLANATION_MARKERS: List[str] = [
    # Arabic
    "شرح", "درس", "ملخص", "تمرين", "حل", "فهم",
    # French
    "cours", "explication", "résumé", "exercice", "solution",
    # English
    "how to", "tutorial", "lesson", "summary", "explained",
]

QUESTION_MARKERS: List[str] = [
    "?", "؟", "كيف", "لماذا", "ما هو", "comment", "pourquoi",
]

# =============================================================================
# AGGREGATED DICTIONARIES
# =============================================================================

SUBJECT_KEYWORDS: Dict[str, List[str]] = {
    "Mathematics": MATHEMATICS_KEYWORDS,
    "Physics": PHYSICS_KEYWORDS,
    "Natural Sciences": NATURAL_SCIENCES_KEYWORDS,
    "History & Geography": HISTORY_KEYWORDS + GEOGRAPHY_KEYWORDS,
    "Philosophy": PHILOSOPHY_KEYWORDS,
    "Islamic Sciences": ISLAMIC_SCIENCES_KEYWORDS,
}

# Flattened set for fast lookup
ALL_DOMAIN_KEYWORDS: Set[str] = set()
for keywords in SUBJECT_KEYWORDS.values():
    ALL_DOMAIN_KEYWORDS.update(kw.lower() for kw in keywords)

ALL_BAC_MARKERS: Set[str] = {m.lower() for m in BAC_EXAM_MARKERS}
ALL_PEDAGOGICAL_MARKERS: Set[str] = {m.lower() for m in EXPLANATION_MARKERS}


def count_domain_keywords(text: str, subject: str = None) -> int:
    """
    Count domain-specific keywords in text.
    
    Args:
        text: Input text (title, description, or transcript)
        subject: Optional subject to filter keywords
        
    Returns:
        Count of matching keywords
    """
    if not text:
        return 0
    
    text_lower = text.lower()
    
    if subject and subject in SUBJECT_KEYWORDS:
        keywords = [kw.lower() for kw in SUBJECT_KEYWORDS[subject]]
    else:
        keywords = ALL_DOMAIN_KEYWORDS
    
    return sum(1 for kw in keywords if kw in text_lower)


def count_bac_markers(text: str) -> int:
    """Count Bac/exam-related markers in text."""
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for marker in ALL_BAC_MARKERS if marker in text_lower)


def count_pedagogical_markers(text: str) -> int:
    """Count pedagogical/explanation markers in text."""
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for marker in ALL_PEDAGOGICAL_MARKERS if marker in text_lower)


def get_exam_keyword_intensity(text: str, word_count: int = None) -> float:
    """
    Calculate normalized intensity of exam-related keywords.
    
    Args:
        text: Input text
        word_count: Optional pre-calculated word count
        
    Returns:
        Intensity score (0.0 to 1.0+)
    """
    if not text:
        return 0.0
    
    if word_count is None:
        word_count = len(text.split())
    
    if word_count == 0:
        return 0.0
    
    bac_count = count_bac_markers(text)
    return bac_count / word_count
