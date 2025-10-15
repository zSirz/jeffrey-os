"""
jeffrey/ml/proto.py
Classifieur prototypique avec apprentissage en ligne pour Jeffrey OS Phase 1.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmotionPrediction:
    """Résultat d'une prédiction émotionnelle."""

    primary: str
    confidence: float
    all_scores: dict[str, float]
    margin: float  # Écart entre top1 et top2
    abstention: bool
    reason: str | None = None
    secondary: list[tuple[str, float]] | None = None


class ProtoEmotion:
    """Prototype d'une émotion (centroïde + statistiques).

    Utilise EMA (Exponential Moving Average) pour update online.
    """

    def __init__(self, label: str, dimension: int, alpha: float = 0.05, k: int = 1):
        """Initialise un prototype.

        Args:
            label: Nom de l'émotion
            dimension: Dimensionnalité des embeddings
            alpha: Coefficient EMA (0 = jamais update, 1 = oublie tout)
            k: Nombre de centroïdes (1 = single, 3 = multi-modal)
        """
        self.label = label
        self.dimension = dimension
        self.alpha = alpha
        self.k = k

        # Centroïdes (k=1 → liste de 1, k=3 → liste de 3)
        self.centroids: list[np.ndarray | None] = [None] * k

        # Statistiques par centroïde
        self.n_samples = [0] * k
        self.n_updates_today = [0] * k
        self.last_update = [None] * k
        self.running_variance = [None] * k

        # Compatibilité ancienne API (premier centroïde)
        self.centroid = None  # Will point to centroids[0]
        self.n_samples_total = 0
        self.n_updates_today_total = 0
        self.last_update_any = None
        self.running_variance_first = None

    def initialize(self, embeddings: np.ndarray):
        """Initialise les centroïdes avec k-means.

        Args:
            embeddings: Array (n_samples, dimension)
        """
        if embeddings.shape[0] == 0:
            logger.warning(f"Cannot initialize {self.label}: no embeddings")
            return

        if self.k == 1:
            # Single centroïde (moyenne)
            self.centroids[0] = embeddings.mean(axis=0)
            self.centroids[0] = self.centroids[0] / (np.linalg.norm(self.centroids[0]) + 1e-9)
            self.n_samples[0] = len(embeddings)
            self.running_variance[0] = embeddings.var(axis=0) + 1e-6
        else:
            # k-means pour initialiser k centroïdes
            try:
                from sklearn.cluster import KMeans

                n_clusters = min(self.k, len(embeddings))
                if n_clusters < self.k:
                    logger.warning(f"{self.label}: only {n_clusters} samples, padding centroids")

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(embeddings)

                for i in range(n_clusters):
                    self.centroids[i] = kmeans.cluster_centers_[i]
                    self.centroids[i] = self.centroids[i] / (np.linalg.norm(self.centroids[i]) + 1e-9)

                    # Assigner samples
                    cluster_mask = kmeans.labels_ == i
                    self.n_samples[i] = cluster_mask.sum()

                    if self.n_samples[i] > 0:
                        cluster_embeddings = embeddings[cluster_mask]
                        self.running_variance[i] = cluster_embeddings.var(axis=0) + 1e-6
                    else:
                        self.running_variance[i] = np.ones(self.dimension) * 1e-6
            except ImportError:
                logger.warning(f"sklearn not available, using single centroid for {self.label}")
                # Fallback to single centroid
                self.centroids[0] = embeddings.mean(axis=0)
                self.centroids[0] = self.centroids[0] / (np.linalg.norm(self.centroids[0]) + 1e-9)
                self.n_samples[0] = len(embeddings)
                self.running_variance[0] = embeddings.var(axis=0) + 1e-6

        # Compatibility: update old API
        self.centroid = self.centroids[0]
        self.n_samples_total = sum(self.n_samples)
        self.running_variance_first = self.running_variance[0]

        logger.info(f"Initialized {self.label} with k={self.k} centroïdes")

    def update(
        self, embedding: np.ndarray, weight: float = 1.0, check_outlier: bool = True, max_updates_per_day: int = 100
    ) -> bool:
        """Met à jour le centroïde le plus proche.

        Args:
            embedding: Nouvel embedding
            weight: Poids de l'update
            check_outlier: Vérifier outliers
            max_updates_per_day: Quota quotidien

        Returns:
            True si update effectué
        """
        # Normaliser
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-9)

        # Trouver centroïde le plus proche
        best_idx = 0
        best_sim = -1
        for i, centroid in enumerate(self.centroids):
            if centroid is not None:
                sim = np.dot(centroid, embedding_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

        # Check quota
        today = datetime.now().date()
        if self.last_update[best_idx] and self.last_update[best_idx].date() == today:
            if self.n_updates_today[best_idx] >= max_updates_per_day:
                return False
        else:
            self.n_updates_today[best_idx] = 0

        # Check outlier
        if check_outlier and self.centroids[best_idx] is not None and self.running_variance[best_idx] is not None:
            try:
                diff = embedding_norm - self.centroids[best_idx]
                maha_dist = np.sqrt(np.sum((diff**2) / self.running_variance[best_idx]))
                if maha_dist > 2.5:
                    return False
            except Exception:
                pass

        # Update EMA
        if self.centroids[best_idx] is None:
            self.centroids[best_idx] = embedding_norm
            self.running_variance[best_idx] = np.ones(self.dimension) * 1e-6
        else:
            effective_alpha = self.alpha * weight
            self.centroids[best_idx] = (1 - effective_alpha) * self.centroids[
                best_idx
            ] + effective_alpha * embedding_norm
            if self.centroids[best_idx] is not None:
                self.centroids[best_idx] = self.centroids[best_idx] / (np.linalg.norm(self.centroids[best_idx]) + 1e-9)

            if self.centroids[best_idx] is not None and self.running_variance[best_idx] is not None:
                diff = embedding_norm - self.centroids[best_idx]
                self.running_variance[best_idx] = (1 - effective_alpha) * self.running_variance[
                    best_idx
                ] + effective_alpha * (diff**2)

        self.n_samples[best_idx] += 1
        self.n_updates_today[best_idx] += 1
        self.last_update[best_idx] = datetime.now()

        # Compatibility: update old API
        self.centroid = self.centroids[0]
        self.n_samples_total = sum(self.n_samples)
        self.n_updates_today_total = sum(self.n_updates_today)
        self.last_update_any = max([lu for lu in self.last_update if lu is not None], default=None)
        self.running_variance_first = self.running_variance[0]

        return True

    def get_max_similarity(self, embedding: np.ndarray) -> float:
        """Calcule similarité max avec tous les centroïdes.

        Args:
            embedding: Embedding normalisé

        Returns:
            Similarité maximale
        """
        max_sim = -1
        for centroid in self.centroids:
            if centroid is not None:
                sim = np.dot(centroid, embedding)
                max_sim = max(max_sim, sim)
        return max_sim

    def to_dict(self) -> dict:
        """Sérialisation pour sauvegarde."""
        return {
            "label": self.label,
            "dimension": self.dimension,
            "alpha": self.alpha,
            "k": self.k,
            "centroids": [c.tolist() if c is not None else None for c in self.centroids],
            "running_variance": [rv.tolist() if rv is not None else None for rv in self.running_variance],
            "n_samples": self.n_samples,
            "n_updates_today": self.n_updates_today,
            "last_update": [lu.isoformat() if lu else None for lu in self.last_update],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ProtoEmotion':
        """Désérialisation."""
        k = data.get("k", 1)
        proto = cls(label=data["label"], dimension=data["dimension"], alpha=data["alpha"], k=k)

        # Load new format (multi-centroids)
        if "centroids" in data:
            for i, c_data in enumerate(data["centroids"]):
                if c_data and i < k:
                    proto.centroids[i] = np.array(c_data)
            for i, rv_data in enumerate(data["running_variance"]):
                if rv_data and i < k:
                    proto.running_variance[i] = np.array(rv_data)
            proto.n_samples = data["n_samples"]
            proto.n_updates_today = data["n_updates_today"]
            for i, lu_data in enumerate(data["last_update"]):
                if lu_data and i < k:
                    proto.last_update[i] = datetime.fromisoformat(lu_data)
        else:
            # Backward compatibility (old format)
            if data.get("centroid"):
                proto.centroids[0] = np.array(data["centroid"])
            if data.get("running_variance"):
                proto.running_variance[0] = np.array(data["running_variance"])
            proto.n_samples[0] = data.get("n_samples", 0)
            proto.n_updates_today[0] = data.get("n_updates_today", 0)
            if data.get("last_update"):
                proto.last_update[0] = datetime.fromisoformat(data["last_update"])

        # Update compatibility fields
        proto.centroid = proto.centroids[0]
        proto.n_samples_total = sum(proto.n_samples)
        proto.running_variance_first = proto.running_variance[0]

        return proto


class ProtoClassifier:
    """Classifieur prototypique avec apprentissage en ligne.

    Architecture:
    - 26 prototypes (un par émotion)
    - Cosine similarity + softmax pour scoring
    - Abstention dual-seuils
    - Micro-boosts conservatifs (emojis, outrage moral)
    - Learning via EMA

    Performance attendue:
    - Inference: <0.5ms (26 cosine sims)
    - Learning: <1ms (EMA update)
    """

    # Émotions cibles (26 labels)
    EMOTION_LABELS = [
        "anger",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "disgust",
        "neutral",
        "frustration",
        "determination",
        "relief",
        "exhaustion",
        "better",
        "vulnerability",
        "amusement",
        "betrayal",
        "clarification",
        "confusion",
        "contentment",
        "despair",
        "discomfort",
        "motivation",
        "negative",
        "panic",
        "pride",
        "reflective",
        "tired",
    ]

    # Micro-boosts conservatifs (soft rules)
    EMOJI_BOOSTS = {
        "😡": ("anger", 0.05),
        "🤬": ("anger", 0.05),
        "😨": ("fear", 0.05),
        "😰": ("fear", 0.05),
        "😊": ("joy", 0.05),
        "😄": ("joy", 0.05),
        "😢": ("sadness", 0.05),
        "😭": ("sadness", 0.05),
        "😤": ("frustration", 0.05),
        "🤢": ("disgust", 0.05),
    }

    OUTRAGE_KEYWORDS = [
        "inadmissible",
        "inacceptable",
        "scandaleux",
        "révoltant",
        "honteux",
        "abus",
        "arnaque",
        "scandale",
    ]

    def __init__(
        self,
        dimension: int = 384,
        alpha: float = 0.05,
        temperature: float = 0.07,
        min_confidence: float = 0.25,
        min_margin: float = 0.08,
        allowed_labels: list[str] | None = None,
        k_prototypes: int = 1,
    ):
        """Initialise le classifieur.

        Args:
            dimension: Dimensionnalité des embeddings
            alpha: Coefficient EMA pour updates
            temperature: Température softmax (plus bas = plus confiant)
            min_confidence: Seuil minimum de confiance (abstention si <)
            min_margin: Marge minimum entre top1 et top2 (abstention si <)
            allowed_labels: Liste restreinte d'émotions (ou None = toutes)
            k_prototypes: Nombre de prototypes par label (1 ou 3)
        """
        self.dimension = dimension
        self.alpha = alpha
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.k_prototypes = k_prototypes

        # Labels autorisés (restriction 26→8 ou autre)
        self.ALL_LABELS = self.EMOTION_LABELS if allowed_labels is None else allowed_labels

        # Seuils par label (per-class thresholds pour meilleure calibration)
        self.min_conf_per_label = defaultdict(lambda: self.min_confidence)
        self.min_margin_per_label = defaultdict(lambda: self.min_margin)

        # Créer les prototypes avec k centroïdes
        self.prototypes = {label: ProtoEmotion(label, dimension, alpha, k=k_prototypes) for label in self.ALL_LABELS}

        logger.info(f"ProtoClassifier initialized with {len(self.prototypes)} emotions")

    def bootstrap(self, labeled_data: dict[str, np.ndarray]):
        """Bootstrap les prototypes avec des données initiales.

        Args:
            labeled_data: Dict {emotion: embeddings_array}
        """
        for label, embeddings in labeled_data.items():
            if label in self.prototypes:
                self.prototypes[label].initialize(embeddings)
            else:
                logger.warning(f"Unknown emotion: {label}")

        logger.info(f"Bootstrapped {len(labeled_data)} emotion prototypes")

    def _raw_similarities(self, embedding: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Calcule les similarités brutes (avant softmax).

        Args:
            embedding: Embedding normalisé

        Returns:
            (labels, similarities) - similarités brutes pour calibration
        """
        labels, sims = [], []
        for label in self.ALL_LABELS:
            proto = self.prototypes.get(label)
            if proto is None:
                continue
            sim = proto.get_max_similarity(embedding)
            if sim > -1:
                labels.append(label)
                sims.append(sim)
        return labels, np.array(sims, dtype=np.float32)

    def _softmax_from_similarities(self, sims: np.ndarray, T: float) -> np.ndarray:
        """Applique softmax avec température aux similarités.

        Args:
            sims: Similarités brutes
            T: Température

        Returns:
            Probabilités après softmax
        """
        logits = sims / max(1e-6, T)
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / (exp.sum() + 1e-12)

    def _compute_scores(self, embedding: np.ndarray) -> dict[str, float]:
        """Calcule les scores de similarité pour toutes les émotions.

        Args:
            embedding: Embedding normalisé (shape: dimension,)

        Returns:
            Dict {emotion: score} avec scores entre 0 et 1
        """
        labels, sims = self._raw_similarities(embedding)
        if not labels:
            return {}

        probs = self._softmax_from_similarities(sims, self.temperature)
        return dict(zip(labels, probs.tolist()))

    def _compute_logits_array(self, embedding: np.ndarray, labels_order: list[str]) -> np.ndarray:
        """Calcule les similarités brutes (logits) pour calibration.

        Args:
            embedding: Embedding normalisé
            labels_order: Ordre des labels pour alignement

        Returns:
            Array de similarités brutes (pas de softmax)
        """
        logits = []
        for label in labels_order:
            if label in self.prototypes:
                proto = self.prototypes[label]
                max_sim = proto.get_max_similarity(embedding)
                logits.append(max_sim if max_sim > -1 else -100.0)
            else:
                logits.append(-100.0)

        return np.array(logits, dtype=np.float32)

    def _apply_boosts(self, scores: dict[str, float], text: str) -> dict[str, float]:
        """Applique des micro-boosts conservatifs.

        Args:
            scores: Scores actuels
            text: Texte original

        Returns:
            Scores ajustés (léger boost, pas de remplacement)
        """
        text_lower = text.lower()

        # Boost emojis
        for emoji, (emotion, boost) in self.EMOJI_BOOSTS.items():
            if emoji in text and emotion in scores:
                scores[emotion] = min(1.0, scores[emotion] + boost)

        # Boost outrage moral → anger
        has_outrage = any(kw in text_lower for kw in self.OUTRAGE_KEYWORDS)
        if has_outrage and "anger" in scores:
            # Boost anger, reduce sadness/neutral légèrement
            scores["anger"] = min(1.0, scores["anger"] + 0.05)
            if "sadness" in scores:
                scores["sadness"] *= 0.95
            if "neutral" in scores:
                scores["neutral"] *= 0.95

        # Re-normaliser
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def predict(self, embedding: np.ndarray, text: str | None = None, return_all: bool = False) -> EmotionPrediction:
        """Prédit l'émotion à partir d'un embedding.

        Args:
            embedding: Embedding du texte (shape: dimension,)
            text: Texte original (pour micro-boosts)
            return_all: Retourner toutes les émotions secondaires

        Returns:
            EmotionPrediction avec prédiction et métadonnées
        """
        # Normaliser embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-9)

        # Calculer scores avec nouvelle méthode séparée
        labels, sims = self._raw_similarities(embedding_norm)
        if not labels:
            return EmotionPrediction(
                primary="neutral", confidence=0.0, all_scores={}, margin=0.0, abstention=True, reason="no_prototypes"
            )

        probs = self._softmax_from_similarities(sims, self.temperature)
        scores = dict(zip(labels, probs.tolist()))

        # Appliquer micro-boosts si texte fourni
        if text:
            scores = self._apply_boosts(scores, text)

        # Trier par score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top1_label, top1_score = sorted_scores[0]
        top2_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        margin = top1_score - top2_score

        # Seuils par label
        min_conf = self.min_conf_per_label[top1_label]
        min_margin = self.min_margin_per_label[top1_label]

        # Décision d'abstention (seuils par label)
        abstention = False
        reason = None

        if top1_score < min_conf:
            abstention = True
            reason = f"low_confidence_p1={top1_score:.3f}<{min_conf:.3f}"
        elif margin < min_margin:
            abstention = True
            reason = f"low_margin={margin:.3f}<{min_margin:.3f}"

        # Si abstention → neutral
        if abstention:
            primary = "neutral"
            confidence = 0.5  # Confiance moyenne pour abstention
        else:
            primary = top1_label
            confidence = top1_score

        # Émotions secondaires (top 3)
        secondary = None
        if return_all:
            secondary = sorted_scores[1:4]

        return EmotionPrediction(
            primary=primary,
            confidence=confidence,
            all_scores=scores,
            margin=margin,
            abstention=abstention,
            reason=reason,
            secondary=secondary,
        )

    def learn(self, embedding: np.ndarray, label: str, confidence: float = 1.0) -> bool:
        """Apprend d'un nouvel exemple (update prototype).

        Args:
            embedding: Embedding du texte
            label: Émotion correcte
            confidence: Confiance dans la correction (0-1)

        Returns:
            True si update effectué, False sinon
        """
        if label not in self.prototypes:
            logger.warning(f"Unknown emotion for learning: {label}")
            return False

        # Normaliser embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-9)

        # Update prototype
        success = self.prototypes[label].update(
            embedding_norm, weight=confidence, check_outlier=True, max_updates_per_day=100
        )

        if success:
            logger.debug(f"✅ Learned: {label} (confidence={confidence:.2f})")

        return success

    def set_per_label_thresholds(self, thresholds: dict[str, tuple[float, float]]):
        """Définit les seuils min_confidence et min_margin par label.

        Args:
            thresholds: Dict {label: (min_conf, min_margin)}
        """
        for label, (min_conf, min_margin) in thresholds.items():
            self.min_conf_per_label[label] = min_conf
            self.min_margin_per_label[label] = min_margin

        logger.info(f"Per-label thresholds set for {len(thresholds)} labels")

    def set_temperature(self, temperature: float):
        """Définit la température avec clipping [0.3, 3.0]."""
        self.temperature = float(max(0.3, min(3.0, temperature)))
        logger.info(f"Temperature set to {self.temperature:.4f}")

    def calibrate_temperature_nll(self, embeddings: np.ndarray, labels: list[str]) -> float:
        """Calibre la température pour minimiser le NLL.

        Args:
            embeddings: Embeddings de validation (shape: n, dim)
            labels: Labels corrects (shape: n)

        Returns:
            Temperature optimale
        """
        # Calculer les logits bruts (similarités avant softmax) une seule fois
        labels_order = self.ALL_LABELS
        all_logits = []

        for emb in embeddings:
            # ✅ Utiliser logits bruts, pas scores après softmax
            logits = self._compute_logits_array(emb, labels_order)
            all_logits.append(logits)

        sims_matrix = np.stack(all_logits, axis=0)
        label_to_idx = {label: i for i, label in enumerate(labels_order)}

        y_idx = np.array([label_to_idx.get(y, -1) for y in labels], dtype=np.int32)
        mask = y_idx >= 0
        sims_matrix = sims_matrix[mask]
        y_idx = y_idx[mask]
        if len(y_idx) == 0:
            return 1.0  # fallback

        best_T, best_nll = 1.0, float("inf")
        # Grid search température baseline large (0.8-2.0)
        for T in np.arange(0.8, 2.1, 0.05):
            logits = sims_matrix / T
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
            # NLL
            p = probs[np.arange(len(y_idx)), y_idx]
            nll = -np.log(np.clip(p, 1e-12, 1.0)).mean()
            if nll < best_nll:
                best_nll, best_T = nll, float(T)

        # Fallback si best_nll n'a pas vraiment battu T=1.0
        # (sims_matrix et y_idx sont déjà filtrés par le mask à ce stade)
        baseline_logits = sims_matrix / 1.0
        baseline_logits = baseline_logits - baseline_logits.max(axis=1, keepdims=True)
        baseline_probs = np.exp(baseline_logits)
        baseline_probs = baseline_probs / baseline_probs.sum(axis=1, keepdims=True)
        baseline_probs_y = baseline_probs[np.arange(len(y_idx)), y_idx]
        baseline_probs_y = np.clip(baseline_probs_y, 1e-8, 1.0)

        # Pas de fallback forcé, laisser calibration choisir
        if not np.isfinite(best_nll):
            best_T = 1.0  # Fallback neutre seulement si échec total
            logger.warning("NLL optimization failed, using neutral T=1.0")

        self.set_temperature(best_T)
        logger.info(f"Calibrated temperature: {best_T:.4f} (NLL: {best_nll:.4f})")
        logger.info("  Temperature grid tested: 0.8-2.0 (step 0.05)")
        return best_T

    def save(self, filepath: Path):
        """Sauvegarde les prototypes."""
        data = {
            "dimension": self.dimension,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
            "k_prototypes": self.k_prototypes,
            "min_conf_per_label": dict(self.min_conf_per_label),
            "min_margin_per_label": dict(self.min_margin_per_label),
            "prototypes": {label: proto.to_dict() for label, proto in self.prototypes.items()},
            "saved_at": datetime.now().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Prototypes saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'ProtoClassifier':
        """Charge les prototypes."""
        with open(filepath) as f:
            data = json.load(f)

        classifier = cls(
            dimension=data["dimension"],
            alpha=data["alpha"],
            temperature=data["temperature"],
            min_confidence=data["min_confidence"],
            min_margin=data["min_margin"],
            k_prototypes=data.get("k_prototypes", 1),
        )

        # Charger seuils par label
        if "min_conf_per_label" in data:
            classifier.min_conf_per_label.update(data["min_conf_per_label"])
        if "min_margin_per_label" in data:
            classifier.min_margin_per_label.update(data["min_margin_per_label"])

        # Charger prototypes
        for label, proto_data in data["prototypes"].items():
            classifier.prototypes[label] = ProtoEmotion.from_dict(proto_data)

        logger.info(f"Prototypes loaded from {filepath}")
        return classifier


if __name__ == "__main__":
    # Test rapide
    logging.basicConfig(level=logging.INFO)

    # Créer classifier
    classifier = ProtoClassifier(dimension=384)

    # Bootstrap avec données synthétiques
    np.random.seed(42)
    labeled_data = {
        "anger": np.random.randn(10, 384),
        "joy": np.random.randn(10, 384),
        "sadness": np.random.randn(10, 384),
    }
    classifier.bootstrap(labeled_data)

    # Test prédiction
    test_embedding = np.random.randn(384)
    prediction = classifier.predict(test_embedding, return_all=True)

    print(f"✅ Prediction: {prediction.primary} (conf={prediction.confidence:.2f})")
    print(f"✅ Margin: {prediction.margin:.3f}")
    print(f"✅ Abstention: {prediction.abstention}")

    # Test learning
    success = classifier.learn(test_embedding, "anger", confidence=0.9)
    print(f"✅ Learning: {success}")
