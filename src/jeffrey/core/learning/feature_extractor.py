"""
Jeffrey OS Phase 0.8 - Advanced Feature Extractor
Multi-language semantic extraction with embeddings and causal markers
"""

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependencies with graceful fallbacks
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Import Jeffrey OS components
from feedback.models import Proposal, ProposalType, RiskLevel


@dataclass
class FeatureVector:
    """Container for extracted features"""

    basic_features: dict[str, float]
    semantic_features: np.ndarray
    causal_features: dict[str, float]
    temporal_features: dict[str, float]
    emotional_features: dict[str, float]
    linguistic_features: dict[str, float]
    contextual_features: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticEmbedding:
    """Semantic embedding with metadata"""

    vector: np.ndarray
    language: str
    confidence: float
    model_name: str
    extraction_time: datetime = field(default_factory=datetime.now)


class MultiLanguageEmbedder:
    """Handles multi-language semantic embeddings"""

    def __init__(self, cache_dir: str = "data/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.models = {}
        self.embedding_cache = {}

        # Load available models
        self._load_embedding_models()

        # Language detection patterns
        self.language_patterns = {
            "fr": ["le", "la", "les", "de", "du", "des", "et", "est", "être", "avoir"],
            "es": ["el", "la", "los", "las", "de", "del", "y", "es", "ser", "estar"],
            "de": ["der", "die", "das", "und", "ist", "sein", "haben", "mit", "für"],
            "it": ["il", "la", "lo", "gli", "di", "del", "e", "è", "essere", "avere"],
            "pt": ["o", "a", "os", "as", "de", "do", "e", "é", "ser", "ter"],
        }

        self.logger = logging.getLogger(__name__)

    def _load_embedding_models(self):
        """Load available embedding models"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Multi-language model (best quality)
                self.models["multilingual"] = SentenceTransformer("distiluse-base-multilingual-cased")
                self.logger.info("Loaded multilingual sentence transformer")
            except Exception as e:
                self.logger.warning(f"Failed to load multilingual model: {e}")

            try:
                # English model (high quality for English)
                self.models["english"] = SentenceTransformer("all-mpnet-base-v2")
                self.logger.info("Loaded English sentence transformer")
            except Exception as e:
                self.logger.warning(f"Failed to load English model: {e}")

        if not self.models:
            self.logger.warning("No sentence transformer models available, using fallback")

    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not text.strip():
            return "en"

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        if len(words) < 3:
            return "en"  # Default for short texts

        # Score languages based on common words
        language_scores = {}
        for lang, patterns in self.language_patterns.items():
            score = sum(1 for word in words if word in patterns)
            if score > 0:
                language_scores[lang] = score / len(words)

        if language_scores:
            return max(language_scores, key=language_scores.get)

        return "en"  # Default

    def get_embedding(self, text: str, language: str = None) -> SemanticEmbedding:
        """Get semantic embedding for text"""
        if not text.strip():
            return SemanticEmbedding(
                vector=np.zeros(384),  # Default size
                language="en",
                confidence=0.0,
                model_name="empty",
            )

        # Detect language if not provided
        if language is None:
            language = self.detect_language(text)

        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"{text_hash}_{language}"

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Get embedding
        embedding = self._compute_embedding(text, language)

        # Cache result
        self.embedding_cache[cache_key] = embedding

        return embedding

    def _compute_embedding(self, text: str, language: str) -> SemanticEmbedding:
        """Compute embedding using best available model"""
        # Choose best model for language
        if language == "en" and "english" in self.models:
            model = self.models["english"]
            model_name = "all-mpnet-base-v2"
        elif "multilingual" in self.models:
            model = self.models["multilingual"]
            model_name = "distiluse-base-multilingual-cased"
        else:
            # Fallback to simple word embeddings
            return self._compute_simple_embedding(text, language)

        try:
            # Compute embedding
            vector = model.encode(text)

            # Calculate confidence based on text length and model quality
            confidence = min(1.0, len(text.split()) / 10.0)
            if model_name == "all-mpnet-base-v2":
                confidence *= 1.1  # Boost for better model

            return SemanticEmbedding(vector=vector, language=language, confidence=confidence, model_name=model_name)

        except Exception as e:
            self.logger.warning(f"Failed to compute embedding: {e}")
            return self._compute_simple_embedding(text, language)

    def _compute_simple_embedding(self, text: str, language: str) -> SemanticEmbedding:
        """Simple fallback embedding"""
        # Create basic feature vector
        features = []

        # Length features
        features.append(len(text) / 1000.0)  # Normalized length
        features.append(len(text.split()) / 100.0)  # Normalized word count

        # Character features
        features.append(sum(1 for c in text if c.isupper()) / len(text))
        features.append(sum(1 for c in text if c.isdigit()) / len(text))
        features.append(text.count("!") / len(text))
        features.append(text.count("?") / len(text))

        # Pad to standard size
        while len(features) < 384:
            features.append(0.0)

        return SemanticEmbedding(
            vector=np.array(features[:384]),
            language=language,
            confidence=0.3,
            model_name="simple_fallback",
        )

    def get_similarity(self, embedding1: SemanticEmbedding, embedding2: SemanticEmbedding) -> float:
        """Calculate similarity between embeddings"""
        if TORCH_AVAILABLE:
            tensor1 = torch.tensor(embedding1.vector)
            tensor2 = torch.tensor(embedding2.vector)
            return F.cosine_similarity(tensor1, tensor2, dim=0).item()
        else:
            # Fallback cosine similarity
            dot_product = np.dot(embedding1.vector, embedding2.vector)
            norm1 = np.linalg.norm(embedding1.vector)
            norm2 = np.linalg.norm(embedding2.vector)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)


class CausalMarkerExtractor:
    """Extracts causal markers from text"""

    def __init__(self):
        # Causal markers by language
        self.causal_markers = {
            "en": {
                "cause": ["because", "due to", "caused by", "since", "as", "owing to"],
                "effect": ["therefore", "thus", "consequently", "as a result", "so", "hence"],
                "reason": ["reason", "rationale", "justification", "explanation", "why"],
                "condition": ["if", "when", "unless", "provided that", "assuming"],
                "contrast": ["but", "however", "although", "despite", "whereas", "while"],
            },
            "fr": {
                "cause": [
                    "parce que",
                    "à cause de",
                    "causé par",
                    "puisque",
                    "comme",
                    "en raison de",
                ],
                "effect": [
                    "donc",
                    "ainsi",
                    "par conséquent",
                    "en conséquence",
                    "alors",
                    "c'est pourquoi",
                ],
                "reason": ["raison", "justification", "explication", "pourquoi"],
                "condition": ["si", "quand", "à moins que", "à condition que", "en supposant"],
                "contrast": ["mais", "cependant", "bien que", "malgré", "tandis que", "alors que"],
            },
            "es": {
                "cause": ["porque", "debido a", "causado por", "ya que", "como", "a causa de"],
                "effect": [
                    "por lo tanto",
                    "así",
                    "por consiguiente",
                    "como resultado",
                    "entonces",
                    "por eso",
                ],
                "reason": ["razón", "justificación", "explicación", "por qué"],
                "condition": ["si", "cuando", "a menos que", "siempre que", "suponiendo"],
                "contrast": [
                    "pero",
                    "sin embargo",
                    "aunque",
                    "a pesar de",
                    "mientras que",
                    "en cambio",
                ],
            },
            "de": {
                "cause": ["weil", "wegen", "verursacht durch", "da", "aufgrund", "infolge"],
                "effect": ["deshalb", "daher", "folglich", "infolgedessen", "also", "somit"],
                "reason": ["Grund", "Begründung", "Erklärung", "warum"],
                "condition": ["wenn", "falls", "es sei denn", "vorausgesetzt", "angenommen"],
                "contrast": ["aber", "jedoch", "obwohl", "trotz", "während", "wohingegen"],
            },
            "it": {
                "cause": ["perché", "a causa di", "causato da", "poiché", "siccome", "per via di"],
                "effect": ["quindi", "così", "di conseguenza", "pertanto", "dunque", "per questo"],
                "reason": ["ragione", "giustificazione", "spiegazione", "perché"],
                "condition": ["se", "quando", "a meno che", "purché", "supponendo"],
                "contrast": ["ma", "tuttavia", "sebbene", "nonostante", "mentre", "invece"],
            },
            "pt": {
                "cause": ["porque", "devido a", "causado por", "já que", "como", "por causa de"],
                "effect": ["portanto", "assim", "consequentemente", "por isso", "então", "logo"],
                "reason": ["razão", "justificativa", "explicação", "por que"],
                "condition": ["se", "quando", "a menos que", "desde que", "supondo"],
                "contrast": ["mas", "porém", "embora", "apesar de", "enquanto", "no entanto"],
            },
        }

        # Intensity markers
        self.intensity_markers = {
            "en": {
                "strong": ["very", "extremely", "highly", "completely", "totally", "absolutely"],
                "weak": ["somewhat", "slightly", "a bit", "rather", "quite", "fairly"],
                "negative": ["not", "never", "no", "none", "neither", "without"],
            },
            "fr": {
                "strong": [
                    "très",
                    "extrêmement",
                    "hautement",
                    "complètement",
                    "totalement",
                    "absolument",
                ],
                "weak": ["quelque peu", "légèrement", "un peu", "plutôt", "assez", "relativement"],
                "negative": ["ne", "jamais", "non", "aucun", "ni", "sans"],
            },
            "es": {
                "strong": [
                    "muy",
                    "extremadamente",
                    "altamente",
                    "completamente",
                    "totalmente",
                    "absolutamente",
                ],
                "weak": [
                    "algo",
                    "ligeramente",
                    "un poco",
                    "bastante",
                    "relativamente",
                    "más o menos",
                ],
                "negative": ["no", "nunca", "ningún", "ni", "sin", "jamás"],
            },
        }

    def extract_causal_markers(self, text: str, language: str = "en") -> dict[str, float]:
        """Extract causal markers from text"""
        markers = {}
        text_lower = text.lower()

        # Get markers for language
        lang_markers = self.causal_markers.get(language, self.causal_markers["en"])

        # Count causal markers
        for marker_type, marker_list in lang_markers.items():
            count = sum(1 for marker in marker_list if marker in text_lower)
            markers[f"causal_{marker_type}"] = count / len(marker_list) if marker_list else 0

        # Add intensity markers
        intensity_markers = self.intensity_markers.get(language, self.intensity_markers["en"])
        for intensity_type, intensity_list in intensity_markers.items():
            count = sum(1 for marker in intensity_list if marker in text_lower)
            markers[f"intensity_{intensity_type}"] = count / len(intensity_list) if intensity_list else 0

        return markers

    def analyze_causal_structure(self, text: str, language: str = "en") -> dict[str, Any]:
        """Analyze causal structure of text"""
        sentences = re.split(r"[.!?]+", text)

        structure = {
            "causal_chains": [],
            "reasoning_depth": 0,
            "logical_flow": 0.0,
            "argument_strength": 0.0,
        }

        causal_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue

            markers = self.extract_causal_markers(sentence, language)
            total_causal = sum(markers.get(f"causal_{t}", 0) for t in ["cause", "effect", "reason"])

            if total_causal > 0:
                causal_sentences.append({"text": sentence.strip(), "causal_strength": total_causal, "markers": markers})

        # Build causal chains
        structure["causal_chains"] = self._build_causal_chains(causal_sentences)
        structure["reasoning_depth"] = len(causal_sentences)
        structure["logical_flow"] = self._calculate_logical_flow(causal_sentences)
        structure["argument_strength"] = self._calculate_argument_strength(causal_sentences)

        return structure

    def _build_causal_chains(self, causal_sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build causal chains from sentences"""
        chains = []

        for i, sentence in enumerate(causal_sentences):
            chain = {
                "position": i,
                "text": sentence["text"],
                "causal_strength": sentence["causal_strength"],
                "chain_length": 1,
            }

            # Look for connected sentences
            for j in range(i + 1, len(causal_sentences)):
                if causal_sentences[j]["causal_strength"] > 0.1:
                    chain["chain_length"] += 1
                else:
                    break

            chains.append(chain)

        return chains

    def _calculate_logical_flow(self, causal_sentences: list[dict[str, Any]]) -> float:
        """Calculate logical flow score"""
        if len(causal_sentences) < 2:
            return 0.5

        # Check for logical progression
        flow_score = 0.0

        for i in range(len(causal_sentences) - 1):
            curr_markers = causal_sentences[i]["markers"]
            next_markers = causal_sentences[i + 1]["markers"]

            # Check if cause is followed by effect
            if curr_markers.get("causal_cause", 0) > 0 and next_markers.get("causal_effect", 0) > 0:
                flow_score += 1.0

            # Check for reasoning progression
            if curr_markers.get("causal_reason", 0) > 0 and next_markers.get("causal_condition", 0) > 0:
                flow_score += 0.5

        return flow_score / (len(causal_sentences) - 1)

    def _calculate_argument_strength(self, causal_sentences: list[dict[str, Any]]) -> float:
        """Calculate argument strength"""
        if not causal_sentences:
            return 0.0

        # Base strength from causal markers
        base_strength = np.mean([s["causal_strength"] for s in causal_sentences])

        # Boost for intensity markers
        intensity_boost = 0.0
        for sentence in causal_sentences:
            markers = sentence["markers"]
            strong_intensity = markers.get("intensity_strong", 0)
            weak_intensity = markers.get("intensity_weak", 0)

            intensity_boost += strong_intensity - weak_intensity * 0.5

        intensity_boost /= len(causal_sentences)

        return min(1.0, base_strength + intensity_boost * 0.3)


class TemporalFeatureExtractor:
    """Extracts temporal features from context"""

    def __init__(self):
        self.temporal_markers = {
            "en": {
                "urgency": ["urgent", "asap", "immediately", "quickly", "rush", "critical"],
                "delay": ["later", "eventually", "someday", "when possible", "no rush"],
                "frequency": ["always", "often", "sometimes", "rarely", "never"],
                "duration": ["minutes", "hours", "days", "weeks", "months", "years"],
            },
            "fr": {
                "urgency": [
                    "urgent",
                    "dès que possible",
                    "immédiatement",
                    "rapidement",
                    "critique",
                ],
                "delay": ["plus tard", "éventuellement", "un jour", "quand possible", "pas pressé"],
                "frequency": ["toujours", "souvent", "parfois", "rarement", "jamais"],
                "duration": ["minutes", "heures", "jours", "semaines", "mois", "années"],
            },
        }

    def extract_temporal_features(self, text: str, timestamp: datetime, language: str = "en") -> dict[str, float]:
        """Extract temporal features"""
        features = {}
        text_lower = text.lower()

        # Time-based features
        features["hour_of_day"] = timestamp.hour / 24.0
        features["day_of_week"] = timestamp.weekday() / 7.0
        features["day_of_month"] = timestamp.day / 31.0
        features["month_of_year"] = timestamp.month / 12.0

        # Temporal markers
        markers = self.temporal_markers.get(language, self.temporal_markers["en"])

        for marker_type, marker_list in markers.items():
            count = sum(1 for marker in marker_list if marker in text_lower)
            features[f"temporal_{marker_type}"] = count / len(marker_list) if marker_list else 0

        # Time pressure indicators
        features["time_pressure"] = (
            features.get("temporal_urgency", 0) * 2 + features.get("temporal_delay", 0) * -1 + 0.5  # Baseline
        )

        features["time_pressure"] = max(0, min(1, features["time_pressure"]))

        return features


class EmotionalFeatureExtractor:
    """Extracts emotional features from text"""

    def __init__(self):
        self.emotion_lexicons = {
            "en": {
                "joy": ["happy", "excited", "thrilled", "delighted", "pleased", "glad"],
                "fear": ["afraid", "scared", "worried", "anxious", "nervous", "concerned"],
                "anger": ["angry", "furious", "upset", "frustrated", "annoyed", "irritated"],
                "sadness": ["sad", "disappointed", "depressed", "unhappy", "down", "dejected"],
                "surprise": [
                    "surprised",
                    "amazed",
                    "shocked",
                    "stunned",
                    "astonished",
                    "bewildered",
                ],
                "disgust": [
                    "disgusted",
                    "revolted",
                    "repulsed",
                    "nauseated",
                    "sickened",
                    "appalled",
                ],
                "trust": ["trust", "confident", "sure", "certain", "reliable", "dependable"],
                "anticipation": ["eager", "hopeful", "optimistic", "expectant", "looking forward"],
            },
            "fr": {
                "joy": ["heureux", "excité", "ravi", "enchanté", "content", "joyeux"],
                "fear": ["peur", "effrayé", "inquiet", "anxieux", "nerveux", "préoccupé"],
                "anger": ["en colère", "furieux", "fâché", "frustré", "agacé", "irrité"],
                "sadness": ["triste", "déçu", "déprimé", "malheureux", "abattu", "découragé"],
                "surprise": ["surpris", "étonné", "choqué", "stupéfait", "ébahi", "perplexe"],
                "disgust": ["dégoûté", "révolté", "répugné", "nauséeux", "écœuré", "indigné"],
                "trust": [
                    "confiance",
                    "confiant",
                    "sûr",
                    "certain",
                    "fiable",
                    "digne de confiance",
                ],
                "anticipation": ["impatient", "plein d'espoir", "optimiste", "dans l'attente"],
            },
        }

        self.intensity_modifiers = {
            "en": {
                "amplifiers": ["very", "extremely", "incredibly", "totally", "completely"],
                "diminishers": ["slightly", "somewhat", "a bit", "rather", "quite"],
            },
            "fr": {
                "amplifiers": [
                    "très",
                    "extrêmement",
                    "incroyablement",
                    "totalement",
                    "complètement",
                ],
                "diminishers": ["légèrement", "quelque peu", "un peu", "plutôt", "assez"],
            },
        }

    def extract_emotional_features(self, text: str, language: str = "en") -> dict[str, float]:
        """Extract emotional features from text"""
        features = {}
        text_lower = text.lower()
        words = text_lower.split()

        # Get emotion lexicons for language
        emotions = self.emotion_lexicons.get(language, self.emotion_lexicons["en"])
        modifiers = self.intensity_modifiers.get(language, self.intensity_modifiers["en"])

        # Extract basic emotions
        for emotion, emotion_words in emotions.items():
            score = 0.0

            for word in emotion_words:
                if word in text_lower:
                    base_score = 1.0

                    # Check for intensity modifiers
                    word_index = text_lower.find(word)
                    if word_index > 0:
                        preceding_text = text_lower[:word_index].split()[-3:]  # Look at 3 words before

                        for amplifier in modifiers.get("amplifiers", []):
                            if amplifier in preceding_text:
                                base_score *= 1.5
                                break

                        for diminisher in modifiers.get("diminishers", []):
                            if diminisher in preceding_text:
                                base_score *= 0.7
                                break

                    score += base_score

            features[f"emotion_{emotion}"] = score / len(emotion_words) if emotion_words else 0

        # Calculate emotional valence (positive/negative)
        positive_emotions = ["joy", "trust", "anticipation", "surprise"]
        negative_emotions = ["fear", "anger", "sadness", "disgust"]

        positive_score = sum(features.get(f"emotion_{e}", 0) for e in positive_emotions)
        negative_score = sum(features.get(f"emotion_{e}", 0) for e in negative_emotions)

        features["emotional_valence"] = (positive_score - negative_score) / max(1, positive_score + negative_score)
        features["emotional_intensity"] = positive_score + negative_score

        # Emotional stability (consistency)
        emotion_scores = [features.get(f"emotion_{e}", 0) for e in emotions.keys()]
        features["emotional_stability"] = 1 - np.std(emotion_scores) if emotion_scores else 0.5

        return features


class LinguisticFeatureExtractor:
    """Extracts linguistic features from text"""

    def __init__(self):
        self.spacy_models = {}
        self._load_spacy_models()

    def _load_spacy_models(self):
        """Load spaCy models if available"""
        if not SPACY_AVAILABLE:
            return

        models = {
            "en": "en_core_web_sm",
            "fr": "fr_core_news_sm",
            "es": "es_core_news_sm",
            "de": "de_core_news_sm",
            "it": "it_core_news_sm",
            "pt": "pt_core_news_sm",
        }

        for lang, model_name in models.items():
            try:
                self.spacy_models[lang] = spacy.load(model_name)
            except OSError:
                continue  # Model not installed

    def extract_linguistic_features(self, text: str, language: str = "en") -> dict[str, float]:
        """Extract linguistic features"""
        features = {}

        # Basic text statistics
        features["text_length"] = len(text) / 1000.0  # Normalized
        features["word_count"] = len(text.split()) / 100.0  # Normalized
        features["sentence_count"] = len(re.split(r"[.!?]+", text)) / 10.0  # Normalized

        # Character-level features
        features["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features["digit_ratio"] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features["punctuation_ratio"] = sum(1 for c in text if c in ".,;:!?") / len(text) if text else 0

        # Sentence-level features
        sentences = re.split(r"[.!?]+", text)
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            features["avg_sentence_length"] = np.mean(sentence_lengths) / 20.0 if sentence_lengths else 0
            features["sentence_length_var"] = np.var(sentence_lengths) / 100.0 if sentence_lengths else 0

        # Advanced features with spaCy
        if language in self.spacy_models:
            features.update(self._extract_spacy_features(text, language))
        else:
            features.update(self._extract_simple_linguistic_features(text))

        return features

    def _extract_spacy_features(self, text: str, language: str) -> dict[str, float]:
        """Extract features using spaCy"""
        features = {}

        try:
            nlp = self.spacy_models[language]
            doc = nlp(text)

            # Part-of-speech features
            pos_counts = defaultdict(int)
            for token in doc:
                pos_counts[token.pos_] += 1

            total_tokens = len(doc)
            for pos, count in pos_counts.items():
                features[f"pos_{pos.lower()}"] = count / total_tokens if total_tokens > 0 else 0

            # Named entity features
            entity_counts = defaultdict(int)
            for ent in doc.ents:
                entity_counts[ent.label_] += 1

            for label, count in entity_counts.items():
                features[f"entity_{label.lower()}"] = count / total_tokens if total_tokens > 0 else 0

            # Syntactic complexity
            features["syntactic_complexity"] = self._calculate_syntactic_complexity(doc)

            # Readability
            features["readability"] = self._calculate_readability(doc)

        except Exception as e:
            logging.warning(f"spaCy feature extraction failed: {e}")
            features.update(self._extract_simple_linguistic_features(text))

        return features

    def _extract_simple_linguistic_features(self, text: str) -> dict[str, float]:
        """Simple linguistic features fallback"""
        features = {}

        # Word complexity
        words = text.split()
        if words:
            avg_word_length = np.mean([len(word) for word in words])
            features["avg_word_length"] = avg_word_length / 10.0

            # Count complex words (> 6 characters)
            complex_words = sum(1 for word in words if len(word) > 6)
            features["complex_word_ratio"] = complex_words / len(words)

        # Repetition
        word_counts = Counter(words)
        if word_counts:
            max_repetition = max(word_counts.values())
            features["max_word_repetition"] = max_repetition / len(words)

        return features

    def _calculate_syntactic_complexity(self, doc) -> float:
        """Calculate syntactic complexity"""
        if not doc:
            return 0.0

        # Count dependency relations
        dep_counts = defaultdict(int)
        for token in doc:
            dep_counts[token.dep_] += 1

        # Complex relations indicate higher complexity
        complex_deps = ["advcl", "ccomp", "xcomp", "acl", "relcl"]
        complex_count = sum(dep_counts.get(dep, 0) for dep in complex_deps)

        return complex_count / len(doc)

    def _calculate_readability(self, doc) -> float:
        """Calculate readability score"""
        if not doc:
            return 0.0

        # Simplified readability based on sentence and word length
        sentences = list(doc.sents)
        if not sentences:
            return 0.0

        avg_sentence_length = np.mean([len(sent) for sent in sentences])
        avg_word_length = np.mean([len(token.text) for token in doc if token.is_alpha])

        # Flesch-like score (simplified)
        readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / 4.7)

        return max(0, min(1, readability / 100))


class AdvancedFeatureExtractor:
    """
    Advanced feature extractor with multi-language semantic understanding
    """

    def __init__(self, data_dir: str = "data/features"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        self.embedder = MultiLanguageEmbedder()
        self.causal_extractor = CausalMarkerExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.emotional_extractor = EmotionalFeatureExtractor()
        self.linguistic_extractor = LinguisticFeatureExtractor()

        # Feature cache
        self.feature_cache = {}

        # Feature normalization
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_means = {}
        self.feature_stds = {}

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_features(self, proposal: Proposal, rationale: str = "", language: str = None) -> FeatureVector:
        """Extract comprehensive features from proposal and rationale"""
        if language is None:
            language = self.embedder.detect_language(rationale)

        # Generate cache key
        cache_key = self._generate_cache_key(proposal, rationale, language)

        # Check cache
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Extract all feature types
        basic_features = self._extract_basic_features(proposal)
        semantic_features = self._extract_semantic_features(rationale, language)
        causal_features = self._extract_causal_features(proposal, rationale, language)
        temporal_features = self._extract_temporal_features(proposal, rationale, language)
        emotional_features = self._extract_emotional_features(rationale, language)
        linguistic_features = self._extract_linguistic_features(rationale, language)
        contextual_features = self._extract_contextual_features(proposal, rationale)

        # Combine features
        feature_vector = FeatureVector(
            basic_features=basic_features,
            semantic_features=semantic_features,
            causal_features=causal_features,
            temporal_features=temporal_features,
            emotional_features=emotional_features,
            linguistic_features=linguistic_features,
            contextual_features=contextual_features,
            metadata={
                "language": language,
                "extraction_time": datetime.now(),
                "proposal_id": proposal.id,
            },
        )

        # Cache result
        self.feature_cache[cache_key] = feature_vector

        return feature_vector

    def _generate_cache_key(self, proposal: Proposal, rationale: str, language: str) -> str:
        """Generate cache key for features"""
        content = f"{proposal.id}_{rationale}_{language}"
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_basic_features(self, proposal: Proposal) -> dict[str, float]:
        """Extract basic proposal features"""
        features = {}

        # Proposal characteristics
        features["impact_score"] = proposal.impact_score
        features["risk_level"] = self._encode_risk_level(proposal.risk_level)
        features["proposal_type"] = self._encode_proposal_type(proposal.type)
        features["sources_count"] = len(proposal.sources) / 10.0  # Normalized

        # Description features
        features["description_length"] = len(proposal.description) / 1000.0
        features["description_word_count"] = len(proposal.description.split()) / 100.0

        # Plan features
        if proposal.detailed_plan:
            features["plan_length"] = len(proposal.detailed_plan) / 1000.0
            features["plan_word_count"] = len(proposal.detailed_plan.split()) / 100.0
        else:
            features["plan_length"] = 0.0
            features["plan_word_count"] = 0.0

        # Temporal features
        features["created_hour"] = proposal.created_at.hour / 24.0
        features["created_weekday"] = proposal.created_at.weekday() / 7.0
        features["created_month"] = proposal.created_at.month / 12.0

        # Source analysis
        if proposal.sources:
            event_types = [s.event_type for s in proposal.sources]
            features["source_diversity"] = len(set(event_types)) / len(event_types)

            # Time span of sources
            timestamps = [s.timestamp for s in proposal.sources]
            if len(timestamps) > 1:
                time_span = (max(timestamps) - min(timestamps)).total_seconds()
                features["source_time_span"] = min(1.0, time_span / 86400.0)  # Days
            else:
                features["source_time_span"] = 0.0

        return features

    def _extract_semantic_features(self, rationale: str, language: str) -> np.ndarray:
        """Extract semantic features using embeddings"""
        if not rationale.strip():
            return np.zeros(384)  # Default embedding size

        # Get semantic embedding
        embedding = self.embedder.get_embedding(rationale, language)

        return embedding.vector

    def _extract_causal_features(self, proposal: Proposal, rationale: str, language: str) -> dict[str, float]:
        """Extract causal features"""
        features = {}

        # Causal markers in rationale
        causal_markers = self.causal_extractor.extract_causal_markers(rationale, language)
        features.update(causal_markers)

        # Causal structure analysis
        causal_structure = self.causal_extractor.analyze_causal_structure(rationale, language)
        features["causal_reasoning_depth"] = causal_structure["reasoning_depth"] / 10.0
        features["causal_logical_flow"] = causal_structure["logical_flow"]
        features["causal_argument_strength"] = causal_structure["argument_strength"]
        features["causal_chain_count"] = len(causal_structure["causal_chains"]) / 5.0

        # Causal markers in proposal description
        desc_markers = self.causal_extractor.extract_causal_markers(proposal.description, language)
        for key, value in desc_markers.items():
            features[f"desc_{key}"] = value

        return features

    def _extract_temporal_features(self, proposal: Proposal, rationale: str, language: str) -> dict[str, float]:
        """Extract temporal features"""
        features = {}

        # Temporal markers in rationale
        temporal_features = self.temporal_extractor.extract_temporal_features(rationale, proposal.created_at, language)
        features.update(temporal_features)

        # Proposal timing context
        now = datetime.now()
        age_hours = (now - proposal.created_at).total_seconds() / 3600.0
        features["proposal_age_hours"] = min(1.0, age_hours / 168.0)  # Weeks

        return features

    def _extract_emotional_features(self, rationale: str, language: str) -> dict[str, float]:
        """Extract emotional features"""
        return self.emotional_extractor.extract_emotional_features(rationale, language)

    def _extract_linguistic_features(self, rationale: str, language: str) -> dict[str, float]:
        """Extract linguistic features"""
        return self.linguistic_extractor.extract_linguistic_features(rationale, language)

    def _extract_contextual_features(self, proposal: Proposal, rationale: str) -> dict[str, float]:
        """Extract contextual features"""
        features = {}

        # Consistency features
        desc_words = set(proposal.description.lower().split())
        rationale_words = set(rationale.lower().split())

        if desc_words and rationale_words:
            overlap = len(desc_words & rationale_words)
            features["desc_rationale_overlap"] = overlap / len(desc_words | rationale_words)
        else:
            features["desc_rationale_overlap"] = 0.0

        # Specificity features
        specific_words = ["specific", "exactly", "precisely", "particular", "detailed"]
        general_words = ["general", "overall", "broadly", "typically", "usually"]

        rationale_lower = rationale.lower()
        features["specificity_score"] = (
            sum(1 for word in specific_words if word in rationale_lower)
            - sum(1 for word in general_words if word in rationale_lower)
        ) / max(1, len(rationale.split()))

        # Confidence indicators
        confidence_words = ["sure", "certain", "confident", "definitely", "clearly"]
        uncertainty_words = ["maybe", "perhaps", "possibly", "might", "uncertain"]

        features["confidence_indicators"] = (
            sum(1 for word in confidence_words if word in rationale_lower)
            - sum(1 for word in uncertainty_words if word in rationale_lower)
        ) / max(1, len(rationale.split()))

        return features

    def _encode_risk_level(self, risk_level: RiskLevel) -> float:
        """Encode risk level as numeric value"""
        encoding = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0,
        }
        return encoding.get(risk_level, 0.5)

    def _encode_proposal_type(self, proposal_type: ProposalType) -> float:
        """Encode proposal type as numeric value"""
        encoding = {
            ProposalType.SECURITY: 0.9,
            ProposalType.BUGFIX: 0.7,
            ProposalType.OPTIMIZATION: 0.5,
            ProposalType.FEATURE: 0.3,
        }
        return encoding.get(proposal_type, 0.5)

    def combine_features(self, feature_vector: FeatureVector) -> np.ndarray:
        """Combine all features into a single vector"""
        combined = []

        # Add basic features
        combined.extend(feature_vector.basic_features.values())

        # Add semantic features
        combined.extend(feature_vector.semantic_features.flatten())

        # Add other feature types
        for feature_dict in [
            feature_vector.causal_features,
            feature_vector.temporal_features,
            feature_vector.emotional_features,
            feature_vector.linguistic_features,
            feature_vector.contextual_features,
        ]:
            combined.extend(feature_dict.values())

        return np.array(combined)

    def normalize_features(self, feature_vectors: list[FeatureVector]) -> list[np.ndarray]:
        """Normalize feature vectors"""
        if not feature_vectors:
            return []

        # Combine all features
        combined_features = [self.combine_features(fv) for fv in feature_vectors]

        # Normalize if sklearn available
        if self.scaler and len(combined_features) > 1:
            try:
                normalized = self.scaler.fit_transform(combined_features)
                return [norm for norm in normalized]
            except Exception as e:
                self.logger.warning(f"Normalization failed: {e}")

        # Simple normalization fallback
        feature_array = np.array(combined_features)
        if feature_array.size > 0:
            means = np.mean(feature_array, axis=0)
            stds = np.std(feature_array, axis=0)
            stds[stds == 0] = 1  # Avoid division by zero

            normalized = (feature_array - means) / stds
            return [norm for norm in normalized]

        return combined_features

    def get_feature_importance(self, feature_vector: FeatureVector) -> dict[str, float]:
        """Get feature importance analysis"""
        importance = {}

        # Basic feature importance
        for name, value in feature_vector.basic_features.items():
            importance[name] = abs(value)

        # Semantic feature importance (top components)
        semantic_importance = np.abs(feature_vector.semantic_features)
        if len(semantic_importance) > 0:
            importance["semantic_top"] = np.max(semantic_importance)
            importance["semantic_avg"] = np.mean(semantic_importance)

        # Other feature importances
        for feature_type, features in [
            ("causal", feature_vector.causal_features),
            ("temporal", feature_vector.temporal_features),
            ("emotional", feature_vector.emotional_features),
            ("linguistic", feature_vector.linguistic_features),
            ("contextual", feature_vector.contextual_features),
        ]:
            if features:
                importance[f"{feature_type}_max"] = max(abs(v) for v in features.values())
                importance[f"{feature_type}_avg"] = np.mean([abs(v) for v in features.values()])

        return importance

    def export_features(self, feature_vectors: list[FeatureVector], filepath: str):
        """Export features to file"""
        export_data = {
            "features": [
                {
                    "basic": fv.basic_features,
                    "semantic": fv.semantic_features.tolist(),
                    "causal": fv.causal_features,
                    "temporal": fv.temporal_features,
                    "emotional": fv.emotional_features,
                    "linguistic": fv.linguistic_features,
                    "contextual": fv.contextual_features,
                    "metadata": fv.metadata,
                }
                for fv in feature_vectors
            ],
            "export_time": datetime.now().isoformat(),
            "version": "0.8.0",
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Features exported to {filepath}")

    def get_extraction_stats(self) -> dict[str, Any]:
        """Get feature extraction statistics"""
        stats = {
            "cached_features": len(self.feature_cache),
            "embedder_models": len(self.embedder.models),
            "supported_languages": list(self.embedder.language_patterns.keys()),
            "feature_types": [
                "basic",
                "semantic",
                "causal",
                "temporal",
                "emotional",
                "linguistic",
                "contextual",
            ],
        }

        if self.feature_cache:
            # Analyze cached features
            sample_features = next(iter(self.feature_cache.values()))
            stats["feature_dimensions"] = {
                "basic": len(sample_features.basic_features),
                "semantic": len(sample_features.semantic_features),
                "causal": len(sample_features.causal_features),
                "temporal": len(sample_features.temporal_features),
                "emotional": len(sample_features.emotional_features),
                "linguistic": len(sample_features.linguistic_features),
                "contextual": len(sample_features.contextual_features),
            }

        return stats
