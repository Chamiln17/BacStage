"""
Best Practices Extractor for Educational Video Recommendation System
====================================================================
This module analyzes historical video performance data to identify
patterns that distinguish high-performing videos from low-performing ones.
Output: Structured JSON documents for use in RAG knowledge base.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BestPracticesExtractor:
    """Extract actionable best practices from historical video data."""

    def __init__(self, data_path: str):
        """
        Initialize the extractor with the dataset path.

        Args:
            data_path: Path to data_final.csv
        """
        self.data_path = data_path
        self.df = None
        self.best_practices = {}

    def load_data(self) -> pd.DataFrame:
        """Load and prepare the dataset."""
        self.df = pd.read_csv(self.data_path)

        # Convert publish_date to datetime
        self.df['publish_date'] = pd.to_datetime(self.df['publish_date'])

        # Extract time features
        self.df['publish_hour'] = self.df['publish_date'].dt.hour
        self.df['publish_day'] = self.df['publish_date'].dt.day_name()
        self.df['publish_month'] = self.df['publish_date'].dt.month

        # Duration in minutes for easier interpretation
        self.df['duration_minutes'] = self.df['duration_sec'] / 60

        print(f"Loaded {len(self.df):,} videos")
        print(f"Engagement distribution:\n{self.df['engagement_category'].value_counts()}")

        return self.df

    def get_data_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of the dataset."""
        if self.df is None:
            self.load_data()

        summary = {
            "total_videos": len(self.df),
            "date_range": {
                "start": str(self.df['publish_date'].min()),
                "end": str(self.df['publish_date'].max())
            },
            "engagement_distribution": self.df['engagement_category'].value_counts().to_dict(),
            "subjects": self.df['subject'].value_counts().to_dict(),
            "channels_count": self.df['channel_id'].nunique(),
            "avg_engagement_score": round(self.df['engagement_score'].mean(), 2),
            "missing_transcripts": int((self.df['has_transcript'] == 0).sum())
        }
        return summary

    def compare_engagement_groups(self, feature: str, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Compare a numerical feature between High and Low engagement groups.

        Args:
            feature: Column name to analyze
            df: Optional dataframe (uses self.df if not provided)

        Returns statistics and significance test results.
        """
        data = df if df is not None else self.df
        high = data[data['engagement_category'] == 'High'][feature].dropna()
        low = data[data['engagement_category'] == 'Low'][feature].dropna()

        # Need minimum samples
        if len(high) < 5 or len(low) < 5:
            return {
                "feature": feature,
                "insufficient_data": True,
                "high_count": len(high),
                "low_count": len(low)
            }

        # Mann-Whitney U test (non-parametric, better for non-normal data)
        stat, p_value = stats.mannwhitneyu(high, low, alternative='two-sided')

        # Effect size (Cohen's d approximation)
        pooled_std = np.sqrt((high.std()**2 + low.std()**2) / 2)
        effect_size = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0

        return {
            "feature": feature,
            "high_mean": round(high.mean(), 4),
            "high_median": round(high.median(), 4),
            "low_mean": round(low.mean(), 4),
            "low_median": round(low.median(), 4),
            "difference_pct": round((high.mean() - low.mean()) / low.mean() * 100, 2) if low.mean() != 0 else None,
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "effect_size": round(effect_size, 4),
            "effect_interpretation": self._interpret_effect_size(effect_size)
        }

    def extract_for_subject(self, subject: str) -> Dict[str, Any]:
        """
        Extract all best practices for a specific subject.

        Args:
            subject: Subject name (e.g., 'Maths', 'Physics')

        Returns:
            Complete best practices dict for that subject
        """
        subject_df = self.df[self.df['subject'] == subject].copy()

        if len(subject_df) < 30:
            return {"insufficient_data": True, "count": len(subject_df)}

        high_count = len(subject_df[subject_df['engagement_category'] == 'High'])
        if high_count < 10:
            return {"insufficient_data": True, "high_count": high_count}

        return {
            "subject": subject,
            "total_videos": len(subject_df),
            "high_performers": high_count,
            "avg_engagement_score": round(subject_df['engagement_score'].mean(), 3),
            "title_optimization": self.extract_title_best_practices(subject_df),
            "video_duration": self.extract_duration_best_practices(subject_df),
            "upload_timing": self.extract_timing_best_practices(subject_df),
            "content_delivery": self.extract_transcript_best_practices(subject_df),
            "description_optimization": self.extract_description_best_practices(subject_df),
            "content_strategy": self.extract_exam_focus_impact(subject_df),
            "top_channels": subject_df.groupby('channel_title')['engagement_score'].mean().nlargest(5).to_dict()
        }

    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def extract_title_best_practices(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze title characteristics of high-performing videos.

        Args:
            df: Optional dataframe (uses self.df if not provided)
        """
        data = df if df is not None else self.df
        high_df = data[data['engagement_category'] == 'High']
        low_df = data[data['engagement_category'] == 'Low']

        # Title length analysis
        title_length_comparison = self.compare_engagement_groups('title_length', data)

        # Check for patterns in high-performing titles
        high_titles = high_df['title'].str.lower()
        low_titles = low_df['title'].str.lower()

        # Keywords that appear more in high-performing videos
        keywords_to_check = [
            'باك', 'bac', 'حل', 'تمرين', 'ملخص', 'شرح', 'درس',
            'سهل', 'بسيط', 'مفصل', 'شامل', 'مراجعة', 'امتحان',
            '2024', '2023', '2025', 'رقم', 'الجزء'
        ]

        keyword_impact = {}
        for kw in keywords_to_check:
            high_ratio = high_titles.str.contains(kw, na=False).mean() if len(high_titles) > 0 else 0
            low_ratio = low_titles.str.contains(kw, na=False).mean() if len(low_titles) > 0 else 0
            if low_ratio > 0:
                lift = (high_ratio - low_ratio) / low_ratio * 100
            else:
                lift = 100 if high_ratio > 0 else 0
            keyword_impact[kw] = {
                "high_usage_pct": round(high_ratio * 100, 2),
                "low_usage_pct": round(low_ratio * 100, 2),
                "lift_pct": round(lift, 2)
            }

        # Check for question marks and numbers
        has_question = data['title'].str.contains(r'\?', regex=True, na=False)
        has_number = data['title'].str.contains(r'\d', regex=True, na=False)

        question_impact = {
            "high_with_question_pct": round(has_question[data['engagement_category'] == 'High'].mean() * 100, 2),
            "low_with_question_pct": round(has_question[data['engagement_category'] == 'Low'].mean() * 100, 2)
        }

        number_impact = {
            "high_with_number_pct": round(has_number[data['engagement_category'] == 'High'].mean() * 100, 2),
            "low_with_number_pct": round(has_number[data['engagement_category'] == 'Low'].mean() * 100, 2)
        }

        return {
            "category": "title_optimization",
            "title_length": title_length_comparison,
            "keyword_impact": keyword_impact,
            "question_in_title": question_impact,
            "numbers_in_title": number_impact,
            "recommendations": self._generate_title_recommendations(
                title_length_comparison, keyword_impact, question_impact, number_impact
            )
        }

    def _generate_title_recommendations(self, length_stats, keywords, questions, numbers) -> List[str]:
        """Generate actionable title recommendations."""
        recommendations = []

        # Title length recommendation - with defensive check
        if not length_stats.get('insufficient_data'):
            optimal_length = length_stats.get('high_median', 60)  # Default to 60 if missing
            recommendations.append(
                f"✓ Optimal title length: ~{optimal_length:.0f} characters"
            )
        else:
            recommendations.append("✓ Optimal title length: ~60 characters (default)")

        # Top performing keywords
        top_keywords = sorted(
            keywords.items(),
            key=lambda x: x[1].get('lift_pct', 0),
            reverse=True
        )[:5]

        if top_keywords:
            kw_list = [kw for kw, _ in top_keywords if keywords[kw].get('lift_pct', 0) > 0]
            if kw_list:
                recommendations.append(
                    f"✓ High-impact keywords: {', '.join(kw_list)}"
                )

        # Question marks
        if questions.get('high_with_question_pct', 0) > questions.get('low_with_question_pct', 0):
            recommendations.append(
                "✓ Consider using questions in titles to increase curiosity"
            )

        # Numbers
        if numbers.get('high_with_number_pct', 0) > numbers.get('low_with_number_pct', 0):
            recommendations.append(
                "✓ Include specific numbers (e.g., رقم 1, الجزء 2) in titles"
            )

        return recommendations

    def extract_duration_best_practices(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze optimal video duration patterns.

        Args:
            df: Optional dataframe (uses self.df if not provided)
        """
        data = df if df is not None else self.df
        duration_comparison = self.compare_engagement_groups('duration_minutes', data)

        # Skip subject breakdown when analyzing a single subject
        subject_duration = {}
        if df is None:  # Only do this for global analysis
            for subject in data['subject'].unique():
                subject_df = data[data['subject'] == subject]
                high_df = subject_df[subject_df['engagement_category'] == 'High']

                if len(high_df) >= 10:
                    subject_duration[subject] = {
                        "optimal_minutes_median": round(high_df['duration_minutes'].median(), 1),
                        "optimal_minutes_q1": round(high_df['duration_minutes'].quantile(0.25), 1),
                        "optimal_minutes_q3": round(high_df['duration_minutes'].quantile(0.75), 1),
                        "sample_size": len(high_df)
                    }

        # Duration bins analysis
        bins = [0, 5, 10, 15, 20, 30, 60, float('inf')]
        labels = ['0-5min', '5-10min', '10-15min', '15-20min', '20-30min', '30-60min', '60+min']
        data_copy = data.copy()
        data_copy['duration_bin'] = pd.cut(data_copy['duration_minutes'], bins=bins, labels=labels)

        duration_engagement = data_copy.groupby('duration_bin')['engagement_score'].agg(['mean', 'count']).round(3)
        duration_by_bin = duration_engagement.to_dict('index')

        # Generate recommendations
        recommendations = []
        if duration_comparison.get('high_median'):
            recommendations.append(f"✓ Optimal duration: {duration_comparison['high_median']:.0f} minutes")
        recommendations.append("✓ Very short (<5min) and very long (>60min) videos generally underperform")

        return {
            "category": "video_duration",
            "overall_comparison": duration_comparison,
            "by_subject": subject_duration if subject_duration else None,
            "by_duration_bin": duration_by_bin,
            "recommendations": recommendations
        }

    def extract_timing_best_practices(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze optimal upload timing patterns.

        Args:
            df: Optional dataframe (uses self.df if not provided)
        """
        data = df if df is not None else self.df

        # Day of week analysis
        day_engagement = data.groupby('publish_day').agg({
            'engagement_score': ['mean', 'count'],
            'view_count': 'mean'
        }).round(3)
        day_engagement.columns = ['avg_engagement', 'video_count', 'avg_views']

        best_days = day_engagement.sort_values('avg_engagement', ascending=False)

        # Hour analysis
        hour_engagement = data.groupby('publish_hour').agg({
            'engagement_score': ['mean', 'count']
        }).round(3)
        hour_engagement.columns = ['avg_engagement', 'video_count']

        best_hours = hour_engagement.sort_values('avg_engagement', ascending=False).head(5)

        # Is weekday vs weekend
        weekday_comparison = self.compare_engagement_groups('is_weekday', data)

        return {
            "category": "upload_timing",
            "by_day": best_days.to_dict('index'),
            "by_hour": hour_engagement.to_dict('index'),
            "top_hours": best_hours.index.tolist(),
            "weekday_vs_weekend": weekday_comparison,
            "recommendations": [
                f"✓ Best upload days: {', '.join(best_days.head(3).index.tolist())}",
                f"✓ Best upload hours: {best_hours.index.tolist()[:3]} (local time)",
                "✓ Consider your target audience's schedule when uploading"
            ]
        }

    def extract_transcript_best_practices(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze transcript-related patterns in high-performing videos.

        Args:
            df: Optional dataframe (uses self.df if not provided)
        """
        data = df if df is not None else self.df

        # Only analyze videos with valid transcripts (non-zero values)
        transcript_df = data[data['has_transcript'] == 1].copy()

        features = [
            'speech_rate_wpm',
            'question_density',
            'example_density',
            'explanation_density',
            'lexical_diversity',
            'flesch_reading_ease',
            'avg_words_per_sentence'  # Added for dynamic sentence length benchmarks
        ]

        feature_analysis = {}
        for feature in features:
            if feature in transcript_df.columns:
                # Filter for valid non-zero values for rate-based features
                if feature in ['speech_rate_wpm', 'lexical_diversity']:
                    valid_df = transcript_df[transcript_df[feature] > 0]
                else:
                    valid_df = transcript_df

                high = valid_df[valid_df['engagement_category'] == 'High'][feature].dropna()
                low = valid_df[valid_df['engagement_category'] == 'Low'][feature].dropna()

                if len(high) > 5 and len(low) > 5:
                    # Compute statistics on filtered data
                    stat, p_value = stats.mannwhitneyu(high, low, alternative='two-sided')
                    pooled_std = np.sqrt((high.std()**2 + low.std()**2) / 2)
                    effect_size = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0

                    feature_analysis[feature] = {
                        "feature": feature,
                        "high_mean": round(float(high.mean()), 4),
                        "high_median": round(float(high.median()), 4),
                        "low_mean": round(float(low.mean()), 4),
                        "low_median": round(float(low.median()), 4),
                        "difference_pct": round((high.mean() - low.mean()) / low.mean() * 100, 2) if low.mean() != 0 else None,
                        "p_value": round(float(p_value), 6),
                        "significant": bool(p_value < 0.05),
                        "effect_size": round(float(effect_size), 4),
                        "effect_interpretation": self._interpret_effect_size(effect_size),
                        "sample_size_high": len(high),
                        "sample_size_low": len(low)
                    }

        return {
            "category": "content_delivery",
            "feature_analysis": feature_analysis,
            "recommendations": self._generate_transcript_recommendations(feature_analysis)
        }

    def _generate_transcript_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations from transcript analysis."""
        recommendations = []

        if 'speech_rate_wpm' in analysis:
            optimal_rate = analysis['speech_rate_wpm']['high_median']
            recommendations.append(
                f"✓ Optimal speaking pace: ~{optimal_rate:.0f} words per minute"
            )

        if 'question_density' in analysis:
            if analysis['question_density']['high_mean'] > analysis['question_density']['low_mean']:
                recommendations.append(
                    "✓ Include rhetorical questions to engage students"
                )

        if 'example_density' in analysis:
            if analysis['example_density']['high_mean'] > analysis['example_density']['low_mean']:
                recommendations.append(
                    "✓ Include more worked examples (high performers average more examples)"
                )

        if 'lexical_diversity' in analysis:
            optimal_diversity = analysis['lexical_diversity']['high_median']
            recommendations.append(
                f"✓ Vocabulary diversity target: {optimal_diversity:.2f}"
            )

        return recommendations

    def extract_subject_specific_practices(self) -> Dict[str, Any]:
        """Extract best practices specific to each subject."""
        subject_practices = {}

        for subject in self.df['subject'].unique():
            subject_df = self.df[self.df['subject'] == subject]

            if len(subject_df) < 50:  # Skip subjects with too few videos
                continue

            high_df = subject_df[subject_df['engagement_category'] == 'High']

            if len(high_df) < 10:
                continue

            subject_practices[subject] = {
                "total_videos": len(subject_df),
                "high_performers": len(high_df),
                "avg_engagement_score": round(subject_df['engagement_score'].mean(), 3),
                "optimal_duration_minutes": round(high_df['duration_minutes'].median(), 1),
                "optimal_title_length": round(high_df['title_length'].median(), 0),
                "exam_focused_ratio": round(high_df['is_exam_focused'].mean() * 100, 1),
                "top_channels": subject_df.groupby('channel_title')['engagement_score'].mean().nlargest(3).to_dict()
            }

        return {
            "category": "subject_specific",
            "subjects": subject_practices
        }

    def extract_description_best_practices(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze description characteristics.

        Args:
            df: Optional dataframe (uses self.df if not provided)
        """
        data = df if df is not None else self.df
        high_df = data[data['engagement_category'] == 'High']
        low_df = data[data['engagement_category'] == 'Low']

        description_length = self.compare_engagement_groups('description_length', data)

        # Check for various patterns in descriptions
        has_link = data['description'].str.contains(r'http', regex=True, na=False)
        has_timestamp = data['description'].str.contains(r'\d{1,2}:\d{2}', regex=True, na=False)
        has_hashtag = data['description'].str.contains(r'#\w+', regex=True, na=False)
        has_emoji = data['description'].str.contains(r'[😀-🙏🌀-🗿]', regex=True, na=False)
        has_social = data['description'].str.contains(r'(instagram|facebook|twitter|telegram|tiktok)', regex=True, case=False, na=False)

        link_impact = {
            "high_with_links_pct": round(has_link[data['engagement_category'] == 'High'].mean() * 100, 2),
            "low_with_links_pct": round(has_link[data['engagement_category'] == 'Low'].mean() * 100, 2)
        }

        timestamp_impact = {
            "high_with_timestamps_pct": round(has_timestamp[data['engagement_category'] == 'High'].mean() * 100, 2),
            "low_with_timestamps_pct": round(has_timestamp[data['engagement_category'] == 'Low'].mean() * 100, 2)
        }

        hashtag_impact = {
            "high_with_hashtags_pct": round(has_hashtag[data['engagement_category'] == 'High'].mean() * 100, 2),
            "low_with_hashtags_pct": round(has_hashtag[data['engagement_category'] == 'Low'].mean() * 100, 2)
        }

        emoji_impact = {
            "high_with_emojis_pct": round(has_emoji[data['engagement_category'] == 'High'].mean() * 100, 2),
            "low_with_emojis_pct": round(has_emoji[data['engagement_category'] == 'Low'].mean() * 100, 2)
        }

        social_impact = {
            "high_with_social_pct": round(has_social[data['engagement_category'] == 'High'].mean() * 100, 2),
            "low_with_social_pct": round(has_social[data['engagement_category'] == 'Low'].mean() * 100, 2)
        }

        # Keyword analysis in descriptions
        desc_keywords = ['باك', 'bac', 'امتحان', 'رابط', 'تمارين', 'حلول', 'ملخص', 'شرح', 'تابعوني', 'اشتركوا']
        keyword_impact = {}
        high_descs = high_df['description'].str.lower() if len(high_df) > 0 else pd.Series([])
        low_descs = low_df['description'].str.lower() if len(low_df) > 0 else pd.Series([])

        for kw in desc_keywords:
            high_ratio = high_descs.str.contains(kw, na=False).mean() if len(high_descs) > 0 else 0
            low_ratio = low_descs.str.contains(kw, na=False).mean() if len(low_descs) > 0 else 0
            lift = ((high_ratio - low_ratio) / low_ratio * 100) if low_ratio > 0 else (100 if high_ratio > 0 else 0)
            keyword_impact[kw] = {
                "high_usage_pct": round(high_ratio * 100, 2),
                "low_usage_pct": round(low_ratio * 100, 2),
                "lift_pct": round(lift, 2)
            }

        # Generate comprehensive recommendations
        recommendations = self._generate_description_recommendations(
            description_length, link_impact, timestamp_impact, hashtag_impact,
            emoji_impact, social_impact, keyword_impact
        )

        return {
            "category": "description_optimization",
            "length_analysis": description_length,
            "links_impact": link_impact,
            "timestamp_impact": timestamp_impact,
            "hashtag_impact": hashtag_impact,
            "emoji_impact": emoji_impact,
            "social_links_impact": social_impact,
            "keyword_impact": keyword_impact,
            "recommendations": recommendations
        }

    def _generate_description_recommendations(self, length_stats, links, timestamps,
                                               hashtags, emojis, social, keywords) -> List[str]:
        """Generate comprehensive description recommendations based on data analysis."""
        recommendations = []

        # 1. Length recommendations
        if not length_stats.get('insufficient_data'):
            # Use mean if median is 0 (many empty descriptions skew median)
            high_val = length_stats.get('high_median', 0) or length_stats.get('high_mean', 0)
            low_val = length_stats.get('low_median', 0) or length_stats.get('low_mean', 0)
            diff_pct = length_stats.get('difference_pct') or 0  # Handle None

            if high_val > 0 and diff_pct < 0:  # High performers have shorter descriptions
                recommendations.append(
                    f"✓ Keep descriptions concise - high performers average {high_val:.0f} chars vs {low_val:.0f} for low performers"
                )
                recommendations.append("✓ Focus on quality over quantity in descriptions")
            elif high_val > 0:
                recommendations.append(
                    f"✓ Include detailed descriptions of ~{high_val:.0f} characters"
                )

        # 2. Links recommendations
        high_links = links.get('high_with_links_pct', 0)
        low_links = links.get('low_with_links_pct', 0)
        if high_links > low_links:
            recommendations.append(f"✓ Include links in description ({high_links:.0f}% of top videos have links)")
        elif low_links > high_links + 10:
            recommendations.append(f"⚠️ Avoid excessive links - low performers use more ({low_links:.0f}% vs {high_links:.0f}%)")

        # 3. Timestamp recommendations
        high_ts = timestamps.get('high_with_timestamps_pct', 0)
        low_ts = timestamps.get('low_with_timestamps_pct', 0)
        if high_ts > low_ts:
            recommendations.append(f"✓ Add timestamps for navigation ({high_ts:.0f}% of top videos use them)")
        elif high_ts < low_ts - 5:
            recommendations.append("✓ Timestamps are optional - top performers don't rely on them heavily")

        # 4. Hashtag recommendations
        high_hash = hashtags.get('high_with_hashtags_pct', 0)
        low_hash = hashtags.get('low_with_hashtags_pct', 0)
        if high_hash > low_hash:
            recommendations.append("✓ Use relevant hashtags (#bac, #exam, subject tags)")
        elif low_hash > high_hash + 10:
            recommendations.append("⚠️ Don't overuse hashtags - top performers use them sparingly")

        # 5. Emoji recommendations
        high_emoji = emojis.get('high_with_emojis_pct', 0)
        low_emoji = emojis.get('low_with_emojis_pct', 0)
        if high_emoji > low_emoji:
            recommendations.append("✓ Use emojis to make descriptions visually appealing 📚✨")
        elif low_emoji > high_emoji + 10:
            recommendations.append("⚠️ Minimal emoji usage preferred by top performers")

        # 6. Social media recommendations
        high_social = social.get('high_with_social_pct', 0)
        low_social = social.get('low_with_social_pct', 0)
        if high_social > low_social:
            recommendations.append(f"✓ Include social media links ({high_social:.0f}% of top videos)")

        # 7. Keyword recommendations
        top_keywords = sorted(
            [(kw, data) for kw, data in keywords.items() if data.get('lift_pct', 0) > 10],
            key=lambda x: x[1].get('lift_pct', 0),
            reverse=True
        )[:5]

        if top_keywords:
            kw_list = [kw for kw, _ in top_keywords]
            recommendations.append(f"✓ High-impact description keywords: {', '.join(kw_list)}")

        # 8. General best practices
        recommendations.append("✓ Include call-to-action (اشتركوا، فعلوا الجرس)")
        recommendations.append("✓ Mention the subject and topic clearly at the start")
        recommendations.append("✓ Add chapter/unit information for easy searchability")

        # 9. SEO recommendations
        recommendations.append("✓ Use searchable terms students would look for")
        recommendations.append("✓ Include exam year if relevant (باكالوريا 2024)")

        return recommendations

    def extract_exam_focus_impact(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze the impact of exam-focused content.

        Args:
            df: Optional dataframe (uses self.df if not provided)
        """
        data = df if df is not None else self.df
        exam_focused = data[data['is_exam_focused'] == 1]['engagement_score']
        not_exam_focused = data[data['is_exam_focused'] == 0]['engagement_score']

        # Need minimum samples
        if len(exam_focused) < 5 or len(not_exam_focused) < 5:
            return {
                "category": "content_strategy",
                "insufficient_data": True,
                "recommendations": ["✓ Focus on exam-related content for better engagement"]
            }

        stat, p_value = stats.mannwhitneyu(exam_focused, not_exam_focused, alternative='two-sided')

        return {
            "category": "content_strategy",
            "exam_focused_avg_engagement": round(exam_focused.mean(), 3),
            "general_content_avg_engagement": round(not_exam_focused.mean(), 3),
            "lift_pct": round((exam_focused.mean() - not_exam_focused.mean()) / not_exam_focused.mean() * 100, 2) if not_exam_focused.mean() != 0 else 0,
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "recommendations": [
                "✓ Exam-focused content (باك, امتحان) generally performs better",
                "✓ Reference specific exam years for higher relevance",
                "✓ Focus on commonly tested topics"
            ]
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete best practices extraction.

        Now returns subject-specific recommendations for each subject,
        plus global recommendations as fallback.
        """
        if self.df is None:
            self.load_data()

        # Get list of subjects with enough data
        subjects = self.df['subject'].unique()

        # Extract per-subject best practices
        subject_practices = {}
        for subject in subjects:
            print(f"Extracting best practices for: {subject}")
            subject_data = self.extract_for_subject(subject)
            if not subject_data.get('insufficient_data'):
                subject_practices[subject] = subject_data

        # Extract global best practices as fallback
        global_practices = {
            "subject": "global",
            "total_videos": len(self.df),
            "title_optimization": self.extract_title_best_practices(),
            "video_duration": self.extract_duration_best_practices(),
            "upload_timing": self.extract_timing_best_practices(),
            "content_delivery": self.extract_transcript_best_practices(),
            "description_optimization": self.extract_description_best_practices(),
            "content_strategy": self.extract_exam_focus_impact()
        }

        self.best_practices = {
            "generated_at": datetime.now().isoformat(),
            "data_summary": self.get_data_summary(),
            "by_subject": subject_practices,
            "global": global_practices
        }

        return self.best_practices

    def save_to_json(self, output_dir: str) -> List[str]:
        """
        Save best practices as separate JSON files for RAG.

        Creates:
        - Per-subject files: Maths_best_practices.json, Physics_best_practices.json, etc.
        - Global file: global_best_practices.json
        - Combined file: all_best_practices.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # Save per-subject files
        if 'by_subject' in self.best_practices:
            for subject, data in self.best_practices['by_subject'].items():
                filename = f"{subject}_best_practices.json"
                filepath = output_path / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

                saved_files.append(str(filepath))
                print(f"Saved: {filepath}")

        # Save global best practices
        if 'global' in self.best_practices:
            filepath = output_path / "global_best_practices.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.best_practices['global'], f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            saved_files.append(str(filepath))
            print(f"Saved: {filepath}")

        # Save combined summary
        summary_path = output_path / "all_best_practices.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.best_practices, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        saved_files.append(str(summary_path))
        print(f"Saved: {summary_path}")

        return saved_files

    def generate_markdown_report(self) -> str:
        """Generate a human-readable markdown report."""
        if not self.best_practices:
            self.run_full_analysis()

        report = []
        report.append("# Best Practices Report for Algerian Bac Educational Videos\n")
        report.append(f"*Generated: {self.best_practices['generated_at']}*\n")

        # Data Summary
        summary = self.best_practices['data_summary']
        report.append("## Dataset Overview\n")
        report.append(f"- **Total Videos Analyzed**: {summary['total_videos']:,}")
        report.append(f"- **Unique Channels**: {summary['channels_count']:,}")
        report.append(f"- **Date Range**: {summary['date_range']['start'][:10]} to {summary['date_range']['end'][:10]}")
        report.append(f"- **Average Engagement Score**: {summary['avg_engagement_score']}\n")

        # Recommendations sections
        sections = [
            ('title_optimization', 'Title Optimization', '📝'),
            ('video_duration', 'Video Duration', '⏱️'),
            ('upload_timing', 'Upload Timing', '📅'),
            ('content_delivery', 'Content Delivery', '🎤'),
            ('description_optimization', 'Description Optimization', '📋'),
            ('content_strategy', 'Content Strategy', '🎯')
        ]

        for key, title, emoji in sections:
            if key in self.best_practices and 'recommendations' in self.best_practices[key]:
                report.append(f"\n## {emoji} {title}\n")
                for rec in self.best_practices[key]['recommendations']:
                    report.append(f"- {rec}")

        return '\n'.join(report)
# ============================================
# USAGE EXAMPLE
# ============================================
def main():
    """Main execution function."""
    # Initialize extractor
    extractor = BestPracticesExtractor(
        data_path='data/processed/videos_engineered.csv'  # run_pipeline.py engineer
    )

    # Load data
    extractor.load_data()

    # Run full analysis
    best_practices = extractor.run_full_analysis()

    # Save to knowledge base
    saved_files = extractor.save_to_json('knowledge_base/')
    print(f"\nSaved {len(saved_files)} knowledge base files")

    # Generate report
    report = extractor.generate_markdown_report()
    print("\n" + "="*50)
    print(report)

    return best_practices
if __name__ == "__main__":
    main()
