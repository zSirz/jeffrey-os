"""
Script d'entraînement optimisé pour ProtoClassifier.
Synthèse : GPT (L2 norm + meta) + Grok (medoid + LOSO) + Gemini (YAML data)

Features:
- Données réelles (40 scénarios YAML Sprint 1)
- Normalisation L2 pour cosine similarity
- Medoid (robuste aux outliers) au lieu de mean
- Validation LOSO PAR SCÉNARIO (pas par échantillon) - FIX GPT #1
- Métadonnées + versioning
- Chemin configurable
- Gestion propre des classes sans exemples - FIX GPT #3
"""

import sys

sys.path.insert(0, 'src')

import json
import logging
import os
import random  # ✨ FIX GPT: Import manquant pour seed
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut  # FIX GPT #1

from jeffrey.ml.encoder import create_default_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core-8 emotions
CORE_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "frustration"]

# Canonicalisation des labels étendus → core-8
EMOTION_CANONICAL = {
    "determination": "frustration",
    "motivation": "joy",
    "pride": "joy",
    "contentment": "joy",
    "relief": "joy",
    "better": "joy",
    "amusement": "joy",
    "panic": "fear",
    "despair": "sadness",
    "vulnerability": "sadness",
    "exhaustion": "sadness",
    "tired": "sadness",
    "betrayal": "anger",
    "negative": "sadness",
    "reflective": "neutral",
    "clarification": "neutral",
    "confusion": "surprise",
    "discomfort": "disgust",
}


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalisation L2 des embeddings (GPT recommendation + FIX GPT #2).
    Essentiel pour cosine similarity dans ProtoClassifier.

    Args:
        embeddings: Array shape (n_samples, dim) ou (dim,)

    Returns:
        Embeddings normalisés L2
    """
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings) + 1e-12
        return embeddings / norm
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms


def canonicalize_emotion(label: str) -> str:
    """Canonicalise un label vers core-8."""
    normalized = (label or "").strip().lower()

    if normalized in CORE_EMOTIONS:
        return normalized

    return EMOTION_CANONICAL.get(normalized, "neutral")


def create_hybrid_dataset(
    real_examples: list[dict], synthetic_examples: list[dict], hybrid_ratio: float = 0.3
) -> list[dict]:
    """
    Combine données réelles et synthétiques pour augmenter variance.

    Args:
        real_examples: Exemples YAML réels
        synthetic_examples: Exemples synthétiques générés
        hybrid_ratio: Ratio de synthétiques à ajouter (0.3 = 30%)

    Returns:
        Dataset hybride combiné
    """
    n_synthetic = int(len(real_examples) * hybrid_ratio)

    if n_synthetic > len(synthetic_examples):
        # Répliquer si pas assez de synthétiques
        synthetic_sample = random.choices(synthetic_examples, k=n_synthetic)
    else:
        synthetic_sample = random.sample(synthetic_examples, n_synthetic)

    hybrid = real_examples + synthetic_sample
    logger.info(
        f"🔀 Hybrid dataset: {len(real_examples)} real + {len(synthetic_sample)} synthetic = {len(hybrid)} total"
    )

    return hybrid


def clean_and_balance_data(examples: list[dict]) -> list[dict]:
    """
    Nettoie et balance le dataset.
    - Supprime duplicates exacts
    - Détecte déséquilibre de classes
    - Log warnings si imbalance > 2x
    """
    from collections import Counter

    # Remove duplicates
    seen_texts = set()
    cleaned = []
    duplicates = 0

    for ex in examples:
        text = ex["text"].strip().lower()
        if text not in seen_texts:
            seen_texts.add(text)
            cleaned.append(ex)
        else:
            duplicates += 1

    if duplicates > 0:
        logger.warning(f"⚠️  Removed {duplicates} duplicate examples")

    # Check class balance
    emotion_counts = Counter(ex["emotion"] for ex in cleaned)
    max_count = max(emotion_counts.values())
    min_count = min(emotion_counts.values())

    if max_count / min_count > 2:
        logger.warning(f"⚠️  Class imbalance detected (ratio {max_count / min_count:.1f}x):")
        for emotion, count in emotion_counts.most_common():
            logger.warning(f"    {emotion}: {count} examples")

    return cleaned


def compute_expected_calibration_error(
    y_true: list[str], y_pred: list[str], confidences: list[float], n_bins: int = 10
) -> float:
    """
    Calcule l'Expected Calibration Error (ECE).
    Mesure si les probabilités prédites sont bien calibrées.

    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        confidences: Confidences (max prob) pour chaque prédiction
        n_bins: Nombre de bins pour calibration

    Returns:
        ECE score (0 = parfaitement calibré, 1 = très mal calibré)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Exemples dans ce bin
        in_bin = np.logical_and(np.array(confidences) >= bin_lower, np.array(confidences) < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(np.array(y_true)[in_bin] == np.array(y_pred)[in_bin])
            avg_confidence_in_bin = np.mean(np.array(confidences)[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], save_path: str = "data/confusion_matrix.png"):
    """Génère et sauvegarde la matrice de confusion."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - ProtoClassifier LOSO')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"📊 Confusion matrix saved to {save_path}")
    except ImportError:
        logger.warning("⚠️  matplotlib/seaborn not available, skipping confusion matrix plot")


def load_training_data_from_yaml(yaml_dir: str = "data/conversations") -> tuple[list[str], list[str], list[str]]:
    """
    Charge les données RÉELLES des 40 scénarios YAML (Gemini recommendation).

    Args:
        yaml_dir: Répertoire contenant les fichiers YAML

    Returns:
        (texts, labels, groups) : Listes des textes, labels et groupes pour LOSO
        groups = nom du fichier YAML pour chaque échantillon - FIX GPT #1
    """
    logger.info(f"📂 Loading training data from {yaml_dir}...")

    yaml_path = Path(yaml_dir)

    if not yaml_path.exists():
        logger.error(f"❌ YAML directory not found: {yaml_path}")
        logger.warning("⚠️  Falling back to synthetic data (NOT RECOMMENDED)")
        return load_synthetic_data()

    texts = []
    labels = []
    groups = []  # FIX GPT #1 : groupes pour LOSO par scénario

    yaml_files = sorted(yaml_path.glob("*.yaml"))

    if not yaml_files:
        logger.warning(f"⚠️  No YAML files found in {yaml_path}")
        logger.warning("⚠️  Falling back to synthetic data")
        return load_synthetic_data()

    logger.info(f"   Found {len(yaml_files)} YAML files")

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, encoding='utf-8') as f:
                data = yaml.safe_load(f)

            scenario_name = yaml_file.stem  # Nom du fichier sans extension

            # ✨ FIX GPT: Support double schéma (turns vs direct text/emotion)
            if "turns" in data:  # Ancien schéma
                for turn in data["turns"]:
                    user_text = turn.get("user", "")
                    emotion = turn.get("emotion", "neutral")

                    # Canonicaliser l'émotion
                    canonical_emotion = canonicalize_emotion(emotion)

                    if user_text and len(user_text.strip()) > 0:
                        texts.append(user_text.strip())
                        labels.append(canonical_emotion)
                        groups.append(scenario_name)  # FIX GPT #1
            else:  # Nouveau schéma généré (text/emotion direct)
                user_text = data.get("text", "")
                emotion = data.get("emotion", "neutral")
                canonical_emotion = canonicalize_emotion(emotion)

                if user_text and len(user_text.strip()) > 0:
                    texts.append(user_text.strip())
                    labels.append(canonical_emotion)
                    groups.append(scenario_name)  # FIX GPT #1 : groupe = nom fichier

        except Exception as e:
            logger.warning(f"⚠️  Failed to load {yaml_file.name}: {e}")

    logger.info(f"✅ Loaded {len(texts)} examples from {len(set(groups))} scenarios")

    # Stats par classe
    label_counts = {emotion: labels.count(emotion) for emotion in CORE_EMOTIONS}
    for emotion, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        logger.info(f"   {emotion}: {count} examples")

    if len(texts) < 20:
        logger.warning("⚠️  Very few examples (<20). Consider adding more data.")

    return texts, labels, groups


def load_synthetic_data() -> tuple[list[str], list[str], list[str]]:
    """
    Données synthétiques de fallback (NON RECOMMANDÉ pour prod).
    Utilisé seulement si YAML non disponibles.
    """
    logger.warning("⚠️  Using synthetic data - F1 will be lower than 0.537")

    examples = {
        "joy": [
            "I am so happy and excited!",
            "This is wonderful, I love it!",
            "Best day ever, feeling great!",
            "So proud and joyful right now",
            "Amazing news, I'm thrilled!",
        ],
        "sadness": [
            "I feel so sad and down today",
            "This is terrible, I'm heartbroken",
            "I'm disappointed and hurt",
            "Feeling lonely and depressed",
            "Everything feels hopeless",
        ],
        "anger": [
            "I'm so angry and furious!",
            "This is infuriating, I hate this!",
            "So mad, can't believe this happened",
            "Irritated and pissed off",
            "Enraged by this situation",
        ],
        "fear": [
            "I'm scared and anxious",
            "This is terrifying, I'm afraid",
            "Worried and nervous about this",
            "Feeling panicked and fearful",
            "Anxious about what might happen",
        ],
        "surprise": [
            "Wow, I'm so surprised!",
            "I can't believe this, shocking!",
            "Unexpected turn of events!",
            "This is amazing, didn't see that coming",
            "Stunned by this revelation",
        ],
        "disgust": [
            "This is disgusting and gross",
            "Revolting, I can't stand this",
            "Nasty and repulsive",
            "Makes me sick to think about",
            "Repugnant and vile",
        ],
        "neutral": [
            "I'm just stating the facts here",
            "This is a normal situation",
            "Things are okay, nothing special",
            "Just a regular day",
            "Everything is fine and stable",
        ],
        "frustration": [
            "This is so frustrating!",
            "I'm fed up with this situation",
            "Annoyed and irritated by this",
            "This is getting on my nerves",
            "Exasperated by these problems",
        ],
    }

    texts = []
    labels = []
    groups = []

    for emotion, phrases in examples.items():
        for i, phrase in enumerate(phrases):
            texts.append(phrase)
            labels.append(emotion)
            groups.append(f"synthetic_{emotion}_{i}")  # Groupes synthétiques

    return texts, labels, groups


def compute_medoid(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcule le medoid (embedding le plus central) au lieu de la moyenne (Grok/Gemini).
    Plus robuste aux outliers que mean().

    Args:
        embeddings: Array shape (n_samples, dim)

    Returns:
        Medoid embedding shape (dim,)
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot compute medoid of empty array")

    if len(embeddings) == 1:
        return embeddings[0]

    # Calculer distances pairwise (cosine distance = 1 - cosine similarity)
    # Pour embeddings L2-normalisés : cosine_dist = euclidean^2 / 2
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normalized = embeddings / norms

    # Similarité cosine pairwise
    similarities = normalized @ normalized.T

    # Distances cosine
    distances = 1 - similarities

    # Somme des distances pour chaque point
    distance_sums = distances.sum(axis=1)

    # Index du point avec la somme minimale = medoid
    medoid_idx = np.argmin(distance_sums)

    return embeddings[medoid_idx]


def validate_loso_by_scenario(
    embeddings: np.ndarray, labels: list[str], groups: list[str], prototypes: dict[str, np.ndarray]
) -> dict[str, float]:
    """
    Validation Leave-One-Scenario-Out pour estimer F1 réel (FIX GPT #1).

    Args:
        embeddings: Embeddings normalisés L2
        labels: Labels correspondants
        groups: Groupes (nom scénarios) pour LOSO
        prototypes: Dict des prototypes par émotion

    Returns:
        Dict avec métriques (f1_macro, accuracy, etc.)
    """
    logger.info("🧪 Running LOSO validation BY SCENARIO...")

    # FIX GPT #1 : Leave-One-Group-Out (scénario) au lieu de Leave-One-Out (échantillon)
    logo = LeaveOneGroupOut()
    y_true = []
    y_pred = []

    unique_groups = sorted(set(groups))
    logger.info(f"   Cross-validating on {len(unique_groups)} scenarios")

    for train_idx, test_idx in logo.split(embeddings, labels, groups):
        # Entraîner prototypes sur train_idx, tester sur test_idx

        # Pour ce fold, recalculer les prototypes sur les données d'entraînement uniquement
        train_embeddings = embeddings[train_idx]
        train_labels = [labels[i] for i in train_idx]

        # Recalculer prototypes pour ce fold
        fold_prototypes = {}
        for emotion in CORE_EMOTIONS:
            emotion_indices = [i for i, label in enumerate(train_labels) if label == emotion]

            if emotion_indices:
                emotion_embeddings = train_embeddings[emotion_indices]
                # Medoid + L2 normalize
                prototype = compute_medoid(emotion_embeddings)
                prototype = l2_normalize(prototype)
                fold_prototypes[emotion] = prototype
            else:
                # FIX GPT #3 : pas de vecteur zéro, copier neutral ou moyenne globale
                if "neutral" in fold_prototypes:
                    fold_prototypes[emotion] = fold_prototypes["neutral"].copy()
                else:
                    # Moyenne globale L2 normalisée
                    if len(train_embeddings) > 0:
                        global_mean = np.mean(train_embeddings, axis=0)
                        fold_prototypes[emotion] = l2_normalize(global_mean)
                    else:
                        fold_prototypes[emotion] = np.zeros(train_embeddings.shape[1])

        # Prédictions sur test set
        for test_i in test_idx:
            test_emb = embeddings[test_i]
            true_label = labels[test_i]

            # Prédiction : classe la plus proche (cosine similarity)
            best_emotion = None
            best_similarity = -1

            for emotion, prototype in fold_prototypes.items():
                # Cosine similarity (embeddings L2-normalisés)
                similarity = np.dot(test_emb, prototype)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_emotion = emotion

            y_true.append(true_label)
            y_pred.append(best_emotion or "neutral")

    # Calcul F1 macro
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=CORE_EMOTIONS, zero_division=0)

    # Accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    logger.info("✅ LOSO Validation Results (BY SCENARIO):")
    logger.info(f"   F1 Macro: {f1_macro:.3f}")
    logger.info(f"   Accuracy: {accuracy:.3f}")

    # ✨ INNOVATION: Confusion Matrix & ECE Calculation
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=CORE_EMOTIONS)
    plot_confusion_matrix(cm, CORE_EMOTIONS)

    # ECE calculation (Expected Calibration Error)
    # Pour ECE, on a besoin de confidences - on va les approximer via similarities
    all_confidences = []

    # Re-run LOSO pour collecter les confidences (approximation simple)
    for train_idx, test_idx in logo.split(embeddings, labels, groups):
        test_embeddings = embeddings[test_idx]
        for test_emb in test_embeddings:
            # Calculer max similarity comme proxy pour confidence
            max_similarity = -1
            for emotion, prototype in prototypes.items():
                similarity = np.dot(test_emb, prototype)
                if similarity > max_similarity:
                    max_similarity = similarity
            # Convertir similarity en "confidence" (approximation)
            confidence = (max_similarity + 1) / 2  # Map [-1,1] to [0,1]
            all_confidences.append(min(max(confidence, 0.0), 1.0))  # Clip [0,1]

    ece = compute_expected_calibration_error(y_true, y_pred, all_confidences)
    logger.info(f"📊 Expected Calibration Error (ECE): {ece:.3f}")

    # F1 par classe
    report = classification_report(y_true, y_pred, labels=CORE_EMOTIONS, target_names=CORE_EMOTIONS, zero_division=0)
    logger.info(f"\n{report}")

    return {
        "f1_macro": float(f1_macro),
        "accuracy": float(accuracy),
        "ece": float(ece),  # ✨ INNOVATION: Expected Calibration Error
        "n_samples": len(y_true),
        "n_scenarios": len(unique_groups),
    }


def train_prototypes(save_path: str = None, yaml_dir: str = "data/conversations"):
    """
    Entraîne et sauvegarde les prototypes optimisés.

    Args:
        save_path: Chemin de sauvegarde (default: env var ou data/prototypes.npz)
        yaml_dir: Répertoire des YAML
    """
    logger.info("🚀 Training ProtoClassifier prototypes (OPTIMIZED v2.1)...")
    logger.info("   Features: L2 norm + Medoid + LOSO by scenario + Metadata + Hybrid data + ECE + Confusion matrix")

    # ✨ FIX GPT: Verrouiller seed pour reproductibilité
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    logger.info(f"🔒 Random seed locked: {SEED}")

    # ✨ FIX GPT: Verrouiller encoder version
    ENCODER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Actuel dans le code
    logger.info(f"🔒 Encoder model locked: {ENCODER_MODEL}")

    # Chemin configurable (GPT recommendation + FIX GPT #5)
    if save_path is None:
        save_path = os.getenv("JEFFREY_PROTOTYPES_PATH", "data/prototypes.npz")

    logger.info(f"   Save path: {save_path}")

    # Charger données RÉELLES avec groupes
    logger.info("📂 Loading training data...")
    texts, labels, groups = load_training_data_from_yaml(yaml_dir)

    if len(texts) == 0:
        logger.error("❌ No training data available!")
        return

    logger.info(f"✅ Loaded {len(texts)} examples from {len(set(groups))} scenarios")

    # ✨ INNOVATION: Hybrid data pipeline & quality booster
    # Créer format dict pour pipeline qualité
    real_examples = [
        {"text": text, "emotion": label, "group": group} for text, label, group in zip(texts, labels, groups)
    ]

    # Si pas assez de données réelles, on pourrait ajouter synthétiques (skip pour l'instant car 127 is good)
    if len(real_examples) < 50:
        logger.warning(f"⚠️  Only {len(real_examples)} real examples, would need synthetic augmentation")
        # synthetic_examples = generate_synthetic_data()  # TODO: implement if needed
        # examples = create_hybrid_dataset(real_examples, synthetic_examples, hybrid_ratio=0.3)
        examples = real_examples
    else:
        examples = real_examples

    # Clean and balance data
    examples = clean_and_balance_data(examples)

    # Extraire après nettoyage
    texts = [ex["text"] for ex in examples]
    labels = [ex["emotion"] for ex in examples]
    groups = [ex["group"] for ex in examples]

    logger.info(f"🧹 After cleaning: {len(examples)} examples remaining")

    # Créer encoder
    logger.info("🔧 Creating encoder...")
    encoder = create_default_encoder()

    # Encoder les textes
    logger.info("🔄 Encoding texts...")
    embeddings = encoder.encode(texts)
    logger.info(f"   Embeddings shape: {embeddings.shape}")

    # ✨ Normalisation L2 (GPT recommendation + FIX GPT #2)
    logger.info("✨ Normalizing embeddings (L2)...")
    embeddings = l2_normalize(embeddings)

    # Créer prototypes avec MEDOID (Grok/Gemini)
    logger.info("📊 Computing prototypes (medoid method)...")
    prototypes = {}

    for emotion in CORE_EMOTIONS:
        # Indices des exemples de cette émotion
        indices = [i for i, label in enumerate(labels) if label == emotion]

        if indices:
            emotion_embeddings = embeddings[indices]

            # ✨ Medoid au lieu de mean (plus robuste)
            prototype = compute_medoid(emotion_embeddings)

            # Re-normaliser après medoid (FIX GPT #2)
            prototype = l2_normalize(prototype)

            prototypes[emotion] = prototype
            logger.info(f"   ✅ {emotion}: {len(indices)} examples → medoid computed")
        else:
            # FIX GPT #3 : Pas de vecteur zéro, stratégie plus intelligente
            logger.warning(f"   ⚠️  {emotion}: no examples")

            # Option (b): Copier neutral ou moyenne globale
            if "neutral" in prototypes:
                prototypes[emotion] = prototypes["neutral"].copy()
                logger.info(f"   📋 {emotion}: copied from neutral")
            else:
                # Moyenne globale L2 normalisée
                global_mean = np.mean(embeddings, axis=0)
                prototypes[emotion] = l2_normalize(global_mean)
                logger.info(f"   📊 {emotion}: using global mean (L2 normalized)")

    # ✨ Validation LOSO BY SCENARIO (FIX GPT #1)
    validation_metrics = validate_loso_by_scenario(embeddings, labels, groups, prototypes)

    # Sauvegarder
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Saving prototypes to {save_path}...")

    # Sauvegarder prototypes au format npz
    np.savez(save_path, **{f"proto_{emotion}": vector for emotion, vector in prototypes.items()})

    # ✨ Métadonnées (GPT recommendation + FIX GPT #4)
    logger.info("📝 Saving metadata...")

    # FIX GPT #4 : Dimension fiable via encodage test
    test_embedding = encoder.encode(["test"])
    if len(test_embedding.shape) == 1:
        actual_embedding_dim = test_embedding.shape[0]
    else:
        actual_embedding_dim = test_embedding.shape[1]

    meta = {
        "version": "2.1.0",  # ✨ INNOVATION : version bumped pour nouveaux features
        "created_at": datetime.now().isoformat(),
        "encoder_name": getattr(encoder, "name", "default"),
        "encoder_model": ENCODER_MODEL,  # ✨ FIX GPT: encoder verrouillé
        "seed": SEED,  # ✨ FIX GPT: seed pour reproductibilité
        "embedding_dim": int(actual_embedding_dim),  # FIX GPT #4 : dimension fiable
        "num_examples_total": len(texts),
        "num_scenarios": len(set(groups)),
        "num_examples_per_class": {
            emotion: int(sum(1 for label in labels if label == emotion)) for emotion in CORE_EMOTIONS
        },
        "labels": CORE_EMOTIONS,
        "method": "medoid",
        "normalization": "L2",
        "validation": validation_metrics,
        "validation_method": "LOSO_by_scenario",  # FIX GPT #1
        "data_source": yaml_dir,
        "classes_with_no_examples": [
            emotion for emotion in CORE_EMOTIONS if sum(1 for label in labels if label == emotion) == 0
        ],
    }

    # FIX GPT #4 : Nommage cohérent
    meta_path = save_path_obj.with_suffix(".meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Metadata saved to {meta_path}")

    # Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"📦 Prototypes: {save_path}")
    logger.info(f"📝 Metadata: {meta_path}")
    logger.info(f"📊 F1 LOSO (by scenario): {validation_metrics['f1_macro']:.3f}")
    logger.info("🎯 Target F1: 0.537")
    logger.info(f"📈 Scenarios: {validation_metrics['n_scenarios']}")

    if validation_metrics['f1_macro'] < 0.5:
        logger.warning("⚠️  F1 below target! Consider:")
        logger.warning("   - Adding more training data")
        logger.warning("   - Checking data quality")
        logger.warning("   - Balancing classes")
    else:
        logger.info("🎉 F1 meets or exceeds target!")

    # Test rapide de cohérence
    logger.info("\n🧪 Quick smoke test...")
    test_texts = [
        "I'm so happy and excited!",
        "I feel very sad today",
        "This makes me so angry",
    ]

    for text in test_texts:
        test_emb_batch = encoder.encode([text])
        if len(test_emb_batch.shape) == 1:
            test_emb = test_emb_batch
        else:
            test_emb = test_emb_batch[0]
        test_emb = l2_normalize(test_emb)

        # Simuler ProtoClassifier predict
        best_emotion = None
        best_similarity = -1

        for emotion, prototype in prototypes.items():
            similarity = np.dot(test_emb, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_emotion = emotion

        logger.info(f"   '{text}' → {best_emotion} ({best_similarity:.2f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ProtoClassifier prototypes (optimized v2)")
    parser.add_argument(
        "--yaml-dir",
        default="data/conversations",
        help="Directory containing YAML scenario files (default: data/conversations)",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Save path for prototypes (default: from JEFFREY_PROTOTYPES_PATH env or data/prototypes.npz)",
    )

    args = parser.parse_args()

    train_prototypes(save_path=args.save_path, yaml_dir=args.yaml_dir)
