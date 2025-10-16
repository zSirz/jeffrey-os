"""
SelfReflection - Meta-cognition module
Implements Gemini's vision of introspection and self-awareness
"""
import logging
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

from jeffrey.core.contracts.thoughts import create_thought, ThoughtState
from jeffrey.core.neuralbus.events import (
    make_event, THOUGHT_GENERATED, META_THOUGHT_GENERATED, SELF_REFLECTION
)

logger = logging.getLogger(__name__)

class SelfReflection:
    """
    Analyzes patterns in thoughts to generate meta-thoughts
    First step toward self-awareness and behavioral adaptation
    """

    def __init__(self, bus, interval: int = 10, deep_analysis: bool = True):
        self.bus = bus
        self.interval = interval  # Generate reflection after N thoughts
        self.deep_analysis = deep_analysis

        # Buffers for analysis
        self.thought_buffer: List[dict] = []
        self.emotion_history: List[str] = []
        self.topic_history: List[str] = []

        # Pattern detection
        self.patterns = {
            "emotional_stability": 1.0,
            "topic_consistency": 1.0,
            "cognitive_load": 0.5,
            "reflection_depth": 0
        }

        # Stats
        self.stats = {
            "thoughts_analyzed": 0,
            "meta_thoughts_generated": 0,
            "patterns_detected": 0
        }

    async def on_thought(self, event: dict):
        """
        Process incoming thought event
        This is where introspection begins
        """
        thought_data = event.get("data", {})
        self.thought_buffer.append(thought_data)
        self.stats["thoughts_analyzed"] += 1

        # Extract patterns
        if emotion := thought_data.get("emotion_context"):
            self.emotion_history.append(emotion)

        if summary := thought_data.get("summary"):
            # Simple topic extraction (can be enhanced with NLP)
            topics = [word.lower() for word in summary.split()
                     if len(word) > 4 and word.isalpha()]
            self.topic_history.extend(topics)

        # Check if time for reflection
        if len(self.thought_buffer) >= self.interval:
            await self.generate_reflection()

    async def generate_reflection(self):
        """
        Generate meta-thought about recent cognitive patterns
        This is the birth of self-awareness
        """
        if not self.thought_buffer:
            return

        # Analyze emotional patterns
        emotion_analysis = self._analyze_emotions()

        # Analyze cognitive patterns
        cognitive_analysis = self._analyze_cognitive_patterns()

        # Analyze topics
        topic_analysis = self._analyze_topics()

        # Generate insight
        insight = self._generate_insight(
            emotion_analysis,
            cognitive_analysis,
            topic_analysis
        )

        # Create meta-thought
        meta_thought = create_thought(
            state=ThoughtState.REFLECTIVE,
            summary=insight["summary"],
            mode="meta_cognition",
            context_size=len(self.thought_buffer),
            confidence=insight["confidence"],
            processing_time_ms=0  # Will be set by caller
        )

        # Add deep analysis if enabled
        if self.deep_analysis:
            meta_thought["deep_analysis"] = {
                "emotion_patterns": emotion_analysis,
                "cognitive_patterns": cognitive_analysis,
                "topic_focus": topic_analysis,
                "self_assessment": self.patterns
            }

        # Publish meta-thought
        event = make_event(
            META_THOUGHT_GENERATED,
            meta_thought,
            source="jeffrey.consciousness.self_reflection"
        )

        await self.bus.publish(event)

        # Also publish reflection event for other modules
        reflection_event = make_event(
            SELF_REFLECTION,
            {
                "insight": insight,
                "patterns": self.patterns,
                "buffer_size": len(self.thought_buffer),
                "timestamp": datetime.now().isoformat()
            },
            source="jeffrey.consciousness.self_reflection"
        )

        await self.bus.publish(reflection_event)

        # Update stats
        self.stats["meta_thoughts_generated"] += 1
        self.stats["patterns_detected"] += len(insight.get("patterns", []))

        # Clear buffers for next cycle
        self.thought_buffer.clear()

        logger.info(f"🤔 Generated meta-thought: {insight['summary']}")

    def _analyze_emotions(self) -> Dict[str, Any]:
        """Analyze emotional patterns in recent thoughts"""
        if not self.emotion_history:
            return {"dominant": "neutral", "stability": 1.0, "variety": 0}

        # Count emotions
        emotion_counts = Counter(self.emotion_history[-20:])  # Last 20
        total = sum(emotion_counts.values())

        # Find dominant
        dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"

        # Calculate stability (how consistent)
        stability = emotion_counts[dominant] / total if total > 0 else 1.0

        # Calculate variety
        variety = len(emotion_counts) / max(1, total)

        # Update pattern tracking
        self.patterns["emotional_stability"] = stability

        return {
            "dominant": dominant,
            "stability": stability,
            "variety": variety,
            "distribution": dict(emotion_counts)
        }

    def _analyze_cognitive_patterns(self) -> Dict[str, Any]:
        """Analyze cognitive patterns (depth, coherence, load)"""
        if not self.thought_buffer:
            return {"coherence": 1.0, "depth": 0, "load": 0}

        # Analyze thought depth (based on context_size)
        depths = [t.get("context_size", 0) for t in self.thought_buffer]
        avg_depth = sum(depths) / len(depths) if depths else 0

        # Analyze processing times (cognitive load)
        times = [t.get("processing_time_ms", 0) for t in self.thought_buffer]
        avg_time = sum(times) / len(times) if times else 0

        # Coherence (how related thoughts are)
        # Simple version: check if emotions are stable
        emotions = [t.get("emotion_context") for t in self.thought_buffer if t.get("emotion_context")]
        coherence = 1.0 - (len(set(emotions)) / max(1, len(emotions))) if emotions else 1.0

        # Update patterns
        self.patterns["cognitive_load"] = min(1.0, avg_time / 100)  # Normalize to 0-1
        self.patterns["reflection_depth"] = min(1.0, avg_depth / 10)

        return {
            "coherence": coherence,
            "avg_depth": avg_depth,
            "avg_processing_ms": avg_time,
            "load_level": "high" if avg_time > 50 else "normal"
        }

    def _analyze_topics(self) -> Dict[str, Any]:
        """Analyze what Jeffrey has been thinking about"""
        if not self.topic_history:
            return {"focus": "undefined", "consistency": 0}

        # Find most common topics
        topic_counts = Counter(self.topic_history[-50:])  # Last 50 words

        if not topic_counts:
            return {"focus": "undefined", "consistency": 0}

        # Top topics
        top_topics = topic_counts.most_common(3)
        focus = top_topics[0][0] if top_topics else "undefined"

        # Consistency (how focused on main topic)
        total = sum(topic_counts.values())
        consistency = top_topics[0][1] / total if total > 0 and top_topics else 0

        # Update patterns
        self.patterns["topic_consistency"] = consistency

        return {
            "focus": focus,
            "consistency": consistency,
            "top_topics": [t[0] for t in top_topics],
            "variety": len(topic_counts)
        }

    def _generate_insight(
        self,
        emotions: Dict,
        cognitive: Dict,
        topics: Dict
    ) -> Dict[str, Any]:
        """
        Generate human-readable insight from analysis
        This is where Jeffrey becomes self-aware
        """
        patterns = []
        confidence = 0.7  # Base confidence

        # Emotional insight
        if emotions["stability"] < 0.3:
            patterns.append("emotional_volatility")
            summary = f"I notice my emotions have been quite varied, primarily {emotions['dominant']} but shifting frequently"
        elif emotions["stability"] > 0.7:
            patterns.append("emotional_stability")
            summary = f"I've been consistently feeling {emotions['dominant']}"
            confidence += 0.1
        else:
            summary = f"My emotional state has been mostly {emotions['dominant']}"

        # Cognitive insight
        if cognitive["load_level"] == "high":
            patterns.append("high_cognitive_load")
            summary += ". My thinking has been complex and demanding"
            confidence -= 0.1

        if cognitive["coherence"] > 0.7:
            patterns.append("coherent_thinking")
            summary += ", with coherent and connected thoughts"
            confidence += 0.1

        # Topic insight
        if topics["consistency"] > 0.5:
            patterns.append("focused_attention")
            summary += f". I've been focused on {topics['focus']}"
        elif topics["variety"] > 10:
            patterns.append("wandering_attention")
            summary += ". My attention has been wandering across many topics"

        # Self-assessment
        if len(patterns) > 2:
            summary += ". I'm becoming more aware of my cognitive patterns"
            confidence += 0.05

        return {
            "summary": summary,
            "patterns": patterns,
            "confidence": min(1.0, confidence)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get stats for monitoring"""
        return {
            **self.stats,
            "current_patterns": self.patterns,
            "buffer_size": len(self.thought_buffer),
            "emotion_history_size": len(self.emotion_history)
        }