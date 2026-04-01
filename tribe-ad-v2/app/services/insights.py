"""
Insight Engine
==============
Generates actionable recommendations based on cognitive signal analysis.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Threshold definitions: (low_threshold, high_threshold, low_insight, high_insight)
SIGNAL_RULES: Dict[str, Tuple[float, float, str, str]] = {
    "saliency": (
        4.5, 8.5,
        "Weak visual attention -> improve contrast, framing, or movement in key scenes.",
        "Excellent visual saliency -> maintain current visual impact strategy."
    ),
    "motion": (
        4.0, 8.0,
        "Low motion signal -> add dynamic camera cuts or object movement for attention retention.",
        "Strong motion dynamics -> ensure motion doesn't overwhelm the core message."
    ),
    "novelty": (
        4.5, 8.0,
        "Low novelty -> content feels predictable; introduce surprising visual or narrative elements.",
        "High novelty -> ensure audience can still follow the core message."
    ),
    "emotion_intensity": (
        4.5, 8.5,
        "Low emotional intensity -> introduce stronger emotional hooks in opening seconds.",
        "Strong emotional engagement -> leverage this for brand recall moments."
    ),
    "emotion_valence": (
        3.5, 7.5,
        "Negative emotional tone -> rebalance with more positive or reassuring cues.",
        "Positive emotional tone -> amplify feel-good moments for sharing potential."
    ),
    "relatability": (
        4.5, 8.0,
        "Low relatability -> add recognizable human context and audience-aligned scenarios.",
        "High relatability -> use for testimonial or UGC-style content extensions."
    ),
    "distinctiveness": (
        4.5, 8.0,
        "Low distinctiveness -> creative appears generic; introduce a unique visual motif.",
        "High distinctiveness -> ensure brand elements are consistently present."
    ),
    "story_coherence": (
        5.0, 8.5,
        "Weak story coherence -> tighten narrative sequence and improve transition logic.",
        "Strong narrative flow -> consider longer-form content to leverage storytelling."
    ),
    "cognitive_load": (
        3.0, 7.0,
        "Very low cognitive load -> add complexity or depth to maintain engagement.",
        "High cognitive load -> simplify concurrent visual elements and reduce simultaneous claims."
    ),
    "clarity": (
        5.0, 8.5,
        "Low clarity -> sharpen value proposition and make the core message explicit earlier.",
        "Excellent clarity -> use as template for future campaigns."
    ),
    "info_density": (
        3.5, 7.0,
        "Low information density -> add more value-packed content per second.",
        "High information density -> reduce clutter and prioritize one key takeaway per scene."
    ),
    "repetition": (
        4.0, 7.0,
        "Low repetition -> key elements aren't reinforced; increase visual/audio repetition of brand.",
        "High repetition -> add variation to prevent audience fatigue."
    ),
}


def generate_insights(signals: Dict[str, float]) -> List[str]:
    """
    Generate actionable insights based on cognitive signal values.
    
    Each signal is evaluated against thresholds to produce recommendations.
    Returns a list of actionable insights prioritized by importance.
    """
    insights: List[str] = []
    strengths: List[str] = []
    
    for signal, (low_thresh, high_thresh, low_insight, high_insight) in SIGNAL_RULES.items():
        value = signals.get(signal, 5.0)
        
        if value < low_thresh:
            insights.append(low_insight)
        elif value > high_thresh:
            strengths.append(high_insight)
    
    # Prioritize: issues first, then top 2 strengths
    if strengths and not insights:
        insights.append(strengths[0])
    
    # Default insight if everything is balanced
    if not insights:
        insights.append(
            "Balanced cognitive-emotional profile -> iterate with A/B creative "
            "variants for incremental lift."
        )
    
    # Add overall assessment
    avg_score = sum(signals.values()) / len(signals) if signals else 5.0
    if avg_score >= 7.0:
        insights.insert(0, f"⭐ Strong overall performance (score: {avg_score:.1f}/10)")
    elif avg_score <= 4.5:
        insights.insert(0, f"⚠️ Below-average performance (score: {avg_score:.1f}/10) - prioritize improvements")
    
    return insights
