"""Meta-learning integration with advanced pattern recognition and vectorization"""

import re
import time
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from jeffrey.utils.logger import get_logger, log_method

logger = get_logger("MetaLearningIntegration")

# Type checking only - avoid circular imports
if TYPE_CHECKING:
    from jeffrey.core.memory.unified_memory import UnifiedMemory


class PatternExtractor:
    """Advanced pattern extraction with vectorization"""

    @staticmethod
    def extract_ngrams(text: str, n_range=(2, 4)) -> list[str]:
        """Extract n-grams efficiently"""
        words = text.lower().split()
        ngrams = []

        for n in range(n_range[0], min(n_range[1] + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i : i + n]))

        return ngrams

    @staticmethod
    def compute_tfidf(ngrams: list[str], document_freq: dict) -> dict[str, float]:
        """Compute TF-IDF scores"""
        tf = Counter(ngrams)
        total = len(ngrams)

        tfidf = {}
        for term, count in tf.items():
            # Term frequency
            tf_score = count / total if total > 0 else 0

            # Inverse document frequency (with smoothing)
            docs_with_term = document_freq.get(term, 0) + 1
            idf_score = np.log(1000 / docs_with_term)  # Assume 1000 docs

            tfidf[term] = tf_score * idf_score

        return tfidf


class MetaLearningIntegration:
    """Advanced meta-learning with real pattern extraction and learning"""

    def __init__(self, memory: Optional["UnifiedMemory"] = None):
        self.memory = memory  # Injected dependency

        # Pattern storage with statistics
        self.patterns = defaultdict(
            lambda: {
                "count": 0,
                "confidence": 0.5,
                "last_seen": 0,
                "success_rate": 0.5,
                "embeddings": [],
            }
        )

        # Concept graph with weights
        self.concept_graph = defaultdict(lambda: defaultdict(float))

        # Document frequencies for TF-IDF
        self.document_freq = defaultdict(int)

        # Learning parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95  # Confidence decay over time
        self.min_confidence = 0.1
        self.max_confidence = 0.95

        # Pattern templates for extraction
        self.templates = {
            "question": ["what", "how", "why", "when", "where", "who", "which"],
            "action": ["create", "make", "build", "fix", "repair", "solve", "implement"],
            "emotion": ["happy", "sad", "excited", "worried", "curious", "confused"],
            "technical": ["code", "function", "class", "module", "system", "architecture"],
        }

        # Statistics
        self.stats = {
            "patterns_seen": 0,
            "patterns_learned": 0,
            "concepts_discovered": 0,
            "accuracy": 0.0,
            "avg_confidence": 0.5,
        }

        # Pattern extractor
        self.extractor = PatternExtractor()

    @log_method
    async def initialize(self):
        """Initialize and load historical patterns"""
        logger.info("🧠 MetaLearningIntegration initializing...")

        # Load patterns from memory if available
        if self.memory:
            try:
                patterns = await self.memory.query({"type": "pattern", "limit": 500})

                for p in patterns:
                    pattern_value = p.get("value", "")
                    if pattern_value:
                        self.patterns[pattern_value] = {
                            "count": p.get("count", 1),
                            "confidence": p.get("confidence", 0.5),
                            "last_seen": p.get("_timestamp", time.time()),
                            "success_rate": p.get("success_rate", 0.5),
                            "embeddings": p.get("embeddings", []),
                        }

                        # Update document frequency
                        self.document_freq[pattern_value] += 1

                self.stats["patterns_learned"] = len(self.patterns)
                logger.info(f"✅ Loaded {len(self.patterns)} historical patterns")

            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")

    async def extract_patterns(self, payload: dict) -> list[dict[str, Any]]:
        """Extract real patterns using advanced NLP techniques"""
        text = payload.get("input", "") or payload.get("text", "")
        if not text:
            return []

        patterns = []

        # 1. Extract n-grams with TF-IDF scoring
        ngrams = self.extractor.extract_ngrams(text, n_range=(2, 4))
        tfidf_scores = self.extractor.compute_tfidf(ngrams, self.document_freq)

        # Get top patterns by TF-IDF
        top_ngrams = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        for ngram, score in top_ngrams:
            # Calculate importance with decay
            time_factor = 1.0
            if ngram in self.patterns:
                time_elapsed = time.time() - self.patterns[ngram]["last_seen"]
                time_factor = self.decay_factor ** (time_elapsed / 3600)  # Decay per hour

            importance = score * time_factor

            if importance > 0.1:  # Threshold
                patterns.append(
                    {
                        "type": "ngram",
                        "value": ngram,
                        "importance": float(importance),
                        "tfidf": float(score),
                        "context": text[:100],
                    }
                )

        # 2. Template matching with confidence
        words = text.lower().split()
        for template_type, keywords in self.templates.items():
            matches = sum(1 for kw in keywords if kw in words)
            if matches > 0:
                confidence = min(0.9, matches / len(keywords))
                patterns.append(
                    {
                        "type": f"template_{template_type}",
                        "value": template_type,
                        "importance": confidence,
                        "matches": matches,
                        "context": text[:100],
                    }
                )

        # 3. Extract key concepts (named entities simulation) - FIXED
        concepts = self._extract_concepts(text)
        for concept in concepts[:5]:
            patterns.append(
                {
                    "type": "concept",
                    "value": concept["value"],
                    "importance": concept["score"],
                    "category": concept["category"],
                    "context": text[:100],
                }
            )

            # Update concept graph
            for other in concepts:
                if other["value"] != concept["value"]:
                    # Increase edge weight
                    self.concept_graph[concept["value"]][other["value"]] += 0.1

        # Update statistics
        self.stats["patterns_seen"] += len(patterns)

        # Update document frequencies
        for p in patterns:
            self.document_freq[p["value"]] += 1

        # Store top patterns if memory available
        if self.memory and patterns:
            for p in patterns[:5]:
                await self.memory.store({"type": "pattern", **p, "timestamp": time.time()})

        return patterns

    def _extract_concepts(self, text: str) -> list[dict[str, Any]]:
        """Extract key concepts with categorization (FIXED for capital detection)"""
        concepts = []

        # FIX: Extract capitalized words from original text before lowercasing
        caps = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text)
        for w in caps:
            concepts.append({"value": w, "score": 0.8, "category": "entity"})

        # Now work with lowercase words for other patterns
        words = text.lower().split()

        for word in words:
            # Technical terms (contains numbers or underscores)
            if any(c.isdigit() or c == "_" for c in word):
                concepts.append({"value": word, "score": 0.7, "category": "technical"})

            # Long words (likely important)
            elif len(word) > 7 and word.isalpha():
                concepts.append({"value": word, "score": 0.6, "category": "keyword"})

        # Sort by score and deduplicate
        seen = set()
        unique_concepts = []
        for c in sorted(concepts, key=lambda x: x["score"], reverse=True):
            if c["value"] not in seen:
                seen.add(c["value"])
                unique_concepts.append(c)

        return unique_concepts[:10]

    async def learn(self, example: dict) -> dict:
        """Learn from example with reinforcement and decay"""
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        success = example.get("success", False)

        # Extract patterns from both input and output
        all_patterns = await self.extract_patterns({"text": f"{input_text} {output_text}"})

        # Reinforcement learning with temporal difference
        for pattern in all_patterns:
            key = pattern["value"]

            # Initialize if new pattern
            if key not in self.patterns:
                self.patterns[key] = {
                    "count": 0,
                    "confidence": 0.5,
                    "last_seen": time.time(),
                    "success_rate": 0.5,
                    "embeddings": [],
                }

            # Update pattern statistics
            pattern_data = self.patterns[key]
            pattern_data["count"] += 1
            pattern_data["last_seen"] = time.time()

            # Update success rate (moving average)
            alpha = 0.1  # Learning rate for success rate
            pattern_data["success_rate"] = (1 - alpha) * pattern_data["success_rate"] + alpha * (1 if success else 0)

            # Update confidence using temporal difference learning
            old_confidence = pattern_data["confidence"]

            # TD target
            reward = 1 if success else -0.5
            td_target = reward + self.decay_factor * pattern_data["success_rate"]

            # TD error
            td_error = td_target - old_confidence

            # Update confidence
            new_confidence = old_confidence + self.learning_rate * td_error
            pattern_data["confidence"] = max(self.min_confidence, min(self.max_confidence, new_confidence))

            # Store embedding if significant change
            if abs(pattern_data["confidence"] - old_confidence) > 0.1:
                pattern_data["embeddings"].append(
                    {
                        "confidence": pattern_data["confidence"],
                        "timestamp": time.time(),
                        "context": input_text[:50],
                    }
                )
                # Keep only last 10 embeddings
                pattern_data["embeddings"] = pattern_data["embeddings"][-10:]

        # Update global statistics
        self.stats["patterns_learned"] = len(self.patterns)
        self.stats["concepts_discovered"] = len(self.concept_graph)

        # Calculate average confidence
        if self.patterns:
            total_confidence = sum(p["confidence"] for p in self.patterns.values())
            self.stats["avg_confidence"] = total_confidence / len(self.patterns)

        # Calculate accuracy (weighted by recency)
        recent_patterns = [p for p in self.patterns.values() if time.time() - p["last_seen"] < 3600]
        if recent_patterns:
            self.stats["accuracy"] = np.mean([p["success_rate"] for p in recent_patterns])

        logger.debug(f"📊 Learned from example: {self.stats}")

        return {
            "patterns_updated": len(all_patterns),
            "total_patterns": len(self.patterns),
            "avg_confidence": self.stats["avg_confidence"],
            "accuracy": self.stats["accuracy"],
        }

    async def estimate_confidence(self, query_type: str) -> float:
        """Estimate confidence for query type using pattern statistics"""
        # Find relevant patterns
        relevant_patterns = []

        for pattern_key, pattern_data in self.patterns.items():
            if query_type.lower() in pattern_key.lower():
                # Weight by recency
                time_weight = self.decay_factor ** ((time.time() - pattern_data["last_seen"]) / 3600)
                weighted_confidence = pattern_data["confidence"] * time_weight
                relevant_patterns.append(weighted_confidence)

        if not relevant_patterns:
            # No relevant patterns, use base confidence
            return 0.3

        # Weighted average confidence
        avg_confidence = np.mean(relevant_patterns)

        # Boost based on total experience
        experience_factor = min(0.3, self.stats["patterns_learned"] / 1000)

        return min(self.max_confidence, avg_confidence + experience_factor)

    async def get_similar_patterns(self, text: str, k: int = 5) -> list[dict]:
        """Find k most similar patterns using vectorized similarity"""
        text_words = set(text.lower().split())

        similarities = []

        for pattern_key, pattern_data in self.patterns.items():
            pattern_words = set(pattern_key.split())

            if not pattern_words:
                continue

            # Jaccard similarity
            intersection = len(text_words & pattern_words)
            union = len(text_words | pattern_words)
            jaccard = intersection / union if union > 0 else 0

            # Weight by confidence and recency
            time_weight = self.decay_factor ** ((time.time() - pattern_data["last_seen"]) / 3600)
            weighted_score = jaccard * pattern_data["confidence"] * time_weight

            if weighted_score > 0:
                similarities.append(
                    {
                        "pattern": pattern_key,
                        "similarity": float(jaccard),
                        "confidence": pattern_data["confidence"],
                        "weighted_score": float(weighted_score),
                        "count": pattern_data["count"],
                        "success_rate": pattern_data["success_rate"],
                    }
                )

        # Sort by weighted score
        similarities.sort(key=lambda x: x["weighted_score"], reverse=True)

        return similarities[:k]

    def get_concept_neighbors(self, concept: str, k: int = 5) -> list[tuple]:
        """Get k nearest neighbors in concept graph"""
        if concept not in self.concept_graph:
            return []

        neighbors = []
        for neighbor, weight in self.concept_graph[concept].items():
            neighbors.append((neighbor, weight))

        # Sort by weight
        neighbors.sort(key=lambda x: x[1], reverse=True)

        return neighbors[:k]
