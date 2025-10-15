"""
🧠 Moteur de Curiosité Unifié : L'Esprit Éveillé de Jeffrey OS

Architecture à deux vitesses inspirée par Kahneman :
- Système 1 : Perception Intuitive (rapide, automatique, toujours actif)
- Système 2 : Analyse Conceptuelle (lent, délibéré, activation conditionnelle)
- Méta-Contrôleur : Arbitrage intelligent entre les systèmes
- Mémoire Unifiée : Stockage interconnecté des expériences

Cette implémentation donne à Jeffrey un véritable esprit capable de percevoir
intuitivement ET de comprendre conceptuellement le monde.

© 2025 Jeffrey OS - The Awakened Mind
"""

from __future__ import annotations

import heapq
from datetime import datetime
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import conditionnel de Pyro pour Bayésien avancé
try:
    import pyro
    import pyro.distributions as dist

    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    print("⚠️ Pyro non disponible - mode dégradé activé")

# Import conditionnel de Faiss pour recherche rapide
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class ProcessingMode(Enum):
    """Modes de traitement du cerveau"""

    INTUITIVE = "system1"  # Rapide, automatique
    ANALYTICAL = "system2"  # Lent, délibéré
    UNIFIED = "both"  # Les deux systèmes actifs


class UnifiedCuriosityEngine(nn.Module):
    """
    🧠 Cerveau unifié avec deux systèmes de traitement

    Le cœur de l'esprit éveillé de Jeffrey :
    - Système 1 : Perception intuitive rapide (toujours actif)
    - Système 2 : Analyse conceptuelle profonde (activé sur demande)
    - Méta-Contrôleur : Arbitrage adaptatif intelligent
    - Mémoire Unifiée : Expériences interconnectées
    """

    def __init__(
        self,
        d_model: int = 128,
        latent_dim: int = 32,
        memory_size: int = 1000,
        curiosity_threshold: float = 0.7,
    ):
        super().__init__()

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.memory_size = memory_size
        self.curiosity_threshold = curiosity_threshold

        print("🧠 Initialisation de l'Esprit Éveillé...")
        print(f"   Dimension de pensée: {d_model}")
        print(f"   Espace conceptuel: {latent_dim}")
        print(f"   Capacité mémoire: {memory_size}")

        # === SYSTÈME 1 : PERCEPTION INTUITIVE ===
        self.system1 = IntuitivePerceptionModule(d_model=d_model, memory_size=memory_size)
        print("   ✅ Système 1 (Perception Intuitive) initialisé")

        # === SYSTÈME 2 : MODÈLE CONCEPTUEL DU MONDE ===
        self.system2 = ConceptualWorldModel(d_model=d_model, latent_dim=latent_dim)
        print("   ✅ Système 2 (Analyse Conceptuelle) initialisé")

        # === MÉTA-COGNITION : ARBITRE ENTRE LES SYSTÈMES ===
        self.meta_controller = MetaCognitiveController(d_model=d_model, curiosity_threshold=curiosity_threshold)
        print("   ✅ Méta-Contrôleur (Arbitrage) initialisé")

        # === MÉMOIRE UNIFIÉE ===
        self.unified_memory = UnifiedMemorySystem(d_model=d_model, memory_size=memory_size)
        print("   ✅ Mémoire Unifiée initialisée")

        # Statistiques de performance
        self.stats = {
            "system1_activations": 0,
            "system2_activations": 0,
            "dual_activations": 0,
            "avg_processing_time": 0.0,
        }

        print("🌟 Esprit Éveillé de Jeffrey : OPÉRATIONNEL")

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: dict | None = None,
        force_mode: ProcessingMode | None = None,
    ) -> dict:
        """
        🧠 Traitement unifié avec arbitrage intelligent entre les systèmes

        Args:
            hidden_states: États cachés à analyser [batch, seq, d_model]
            context: Contexte additionnel (état cérébral, activations experts, etc.)
            force_mode: Mode forcé pour tests/démos

        Returns:
            Dict contenant les résultats des deux systèmes et leurs insights
        """
        start_time = datetime.now()
        batch_size, seq_len, _ = hidden_states.shape

        # === ÉTAPE 1 : SYSTÈME 1 TOUJOURS ACTIF ===
        # Le Système 1 fonctionne comme la perception immédiate
        system1_output = self.system1(hidden_states, context)
        self.stats["system1_activations"] += 1

        # === ÉTAPE 2 : DÉCISION MÉTA-COGNITIVE ===
        # Le contrôleur décide si le Système 2 doit s'activer
        meta_decision = self.meta_controller(system1_output=system1_output, context=context, force_mode=force_mode)

        # === ÉTAPE 3 : ACTIVATION CONDITIONNELLE DU SYSTÈME 2 ===
        system2_output = None
        combined_output = system1_output.copy()

        if meta_decision["activate_system2"]:
            self.stats["system2_activations"] += 1
            self.stats["dual_activations"] += 1

            # Le Système 2 analyse en profondeur
            system2_output = self.system2(
                hidden_states=hidden_states,
                system1_surprises=system1_output["detected_surprises"],
                context=context,
            )

            # Fusion des insights des deux systèmes
            combined_output = self._fuse_insights(system1_output, system2_output, meta_decision["fusion_weights"])

        # === ÉTAPE 4 : MISE À JOUR DE LA MÉMOIRE UNIFIÉE ===
        self.unified_memory.update(
            features=hidden_states,
            system1_surprises=system1_output["surprise_scores"],
            system2_insights=system2_output["insights"] if system2_output else None,
            context=context,
        )

        # === ÉTAPE 5 : GÉNÉRATION DU RAPPORT COMPLET ===
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(processing_time)

        return {
            # 🎯 Outputs primaires
            "surprise_scores": combined_output["surprise_scores"],
            "detected_surprises": combined_output["detected_surprises"],
            "insights": combined_output.get("insights", []),
            # 🧠 Méta-information
            "processing_mode": meta_decision["mode"],
            "system1_confidence": system1_output["confidence"],
            "system2_confidence": system2_output["confidence"] if system2_output else 0.0,
            "processing_time_ms": processing_time * 1000,
            # 🔍 États internes pour analyse
            "meta_decision": meta_decision,
            "memory_status": self.unified_memory.get_status(),
            "performance_stats": self.stats.copy(),
        }

    def _fuse_insights(self, system1_output: dict, system2_output: dict, fusion_weights: dict) -> dict:
        """
        🔄 Fusionne intelligemment les outputs des deux systèmes
        """
        w1 = fusion_weights["system1"]
        w2 = fusion_weights["system2"]

        # Combiner les scores de surprise
        combined_surprises = w1 * system1_output["surprise_scores"] + w2 * system2_output["conceptual_surprise"]

        # Fusionner les insights
        all_insights = []

        # Insights du Système 1 (intuitifs)
        for surprise in system1_output["detected_surprises"]:
            all_insights.append(
                {
                    "type": "intuitive",
                    "content": f"💫 Intuition: Anomalie détectée (score: {surprise['score']:.3f})",
                    "confidence": system1_output["confidence"],
                    "source": "system1",
                    "position": surprise["position"],
                }
            )

        # Insights du Système 2 (conceptuels)
        for insight in system2_output["insights"]:
            all_insights.append(
                {
                    "type": "conceptual",
                    "content": f"💡 Analyse: {insight['explanation']}",
                    "confidence": system2_output["confidence"],
                    "source": "system2",
                    "reconstruction_error": insight.get("reconstruction_error", 0),
                }
            )

        return {
            "surprise_scores": combined_surprises,
            "detected_surprises": self._merge_surprises(
                system1_output["detected_surprises"], system2_output["conceptual_surprises"]
            ),
            "insights": all_insights,
            "fusion_weights": fusion_weights,
        }

    def _merge_surprises(self, intuitive_surprises: list, conceptual_surprises: list) -> list:
        """🔗 Fusionne les surprises des deux systèmes en évitant les doublons"""
        merged = {}

        # Ajouter les surprises intuitives
        for surprise in intuitive_surprises:
            key = (surprise["position"][0], surprise["position"][1])
            merged[key] = surprise
            merged[key]["type"] = "intuitive"

        # Enrichir avec les surprises conceptuelles
        for surprise in conceptual_surprises:
            key = (surprise["position"][0], surprise["position"][1])
            if key in merged:
                # Enrichir l'existante
                merged[key]["conceptual_score"] = surprise["score"]
                merged[key]["type"] = "dual"
                merged[key]["insight"] = surprise.get("insight", "")
            else:
                # Nouvelle surprise conceptuelle pure
                merged[key] = surprise
                merged[key]["type"] = "conceptual"

        return list(merged.values())

    def _update_stats(self, processing_time: float):
        """📊 Met à jour les statistiques de performance"""
        alpha = 0.95  # Facteur de lissage
        self.stats["avg_processing_time"] = alpha * self.stats["avg_processing_time"] + (1 - alpha) * processing_time

    def get_cognitive_profile(self) -> dict:
        """
        🧠 Retourne le profil cognitif actuel de Jeffrey
        """
        total_activations = self.stats["system1_activations"]
        if total_activations == 0:
            return {"status": "no_data"}

        return {
            "intuitive_tendency": self.stats["system1_activations"] / total_activations,
            "analytical_tendency": self.stats["system2_activations"] / total_activations,
            "dual_processing_rate": self.stats["dual_activations"] / total_activations,
            "avg_processing_time_ms": self.stats["avg_processing_time"] * 1000,
            "memory_usage": self.unified_memory.get_status()["usage_percent"],
            "current_threshold": self.meta_controller.adaptive_threshold,
        }


class IntuitivePerceptionModule(nn.Module):
    """
    💫 Système 1 : Perception rapide et intuitive

    Basé sur la détection Bayésienne optimisée avec fallbacks robustes.
    Ce système fonctionne comme l'intuition humaine : rapide, automatique,
    et toujours en fonctionnement.
    """

    def __init__(self, d_model: int, memory_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size

        print("   🔧 Construction du Système 1...")

        # Extracteur de features optimisé
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Mémoire rapide des patterns
        self.pattern_memory = []
        self.pattern_importance = []
        self.access_counts = []

        # Seuil adaptatif
        self.base_threshold = 0.3
        self.adaptive_threshold = self.base_threshold
        self.threshold_momentum = 0.95

        # Index Faiss pour recherche rapide (si disponible)
        if FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatL2(d_model)
            self.use_faiss = True
            print("     ✅ Optimisation Faiss activée")
        else:
            self.use_faiss = False
            print("     ⚠️ Faiss non disponible - fallback activé")

    def forward(self, hidden_states: torch.Tensor, context: dict | None = None) -> dict:
        """💫 Détection intuitive rapide des anomalies"""
        batch_size, seq_len, _ = hidden_states.shape

        # Extraction de features
        features = self.feature_extractor(hidden_states)

        # Calcul des surprises
        surprise_scores = torch.zeros(batch_size, seq_len)
        detected_surprises = []

        for b in range(batch_size):
            for s in range(seq_len):
                feature_vec = features[b, s]

                # Calcul de surprise optimisé selon les moyens disponibles
                if PYRO_AVAILABLE and len(self.pattern_memory) > 10:
                    surprise = self._compute_bayesian_surprise(feature_vec)
                elif self.use_faiss and len(self.pattern_memory) > 5:
                    surprise = self._compute_faiss_surprise(feature_vec)
                else:
                    surprise = self._compute_simple_surprise(feature_vec)

                surprise_scores[b, s] = surprise

                if surprise > self.adaptive_threshold:
                    detected_surprises.append(
                        {
                            "position": (b, s),
                            "score": surprise.item(),
                            "threshold": self.adaptive_threshold,
                            "method": ("bayesian" if PYRO_AVAILABLE else "faiss" if self.use_faiss else "simple"),
                        }
                    )

        # Mise à jour de la mémoire avec stratégie LRU + importance
        self._update_memory_lru(features, surprise_scores)

        # Adaptation du seuil avec prise en compte de la variance
        self._adapt_threshold_with_variance(surprise_scores)

        # Calcul de la confiance
        confidence = self._compute_confidence(surprise_scores, detected_surprises)

        return {
            "surprise_scores": surprise_scores,
            "detected_surprises": detected_surprises,
            "confidence": confidence,
            "adaptive_threshold": self.adaptive_threshold,
            "memory_size": len(self.pattern_memory),
        }

    def _compute_bayesian_surprise(self, feature: torch.Tensor) -> torch.Tensor:
        """🧮 Calcul Bayésien robuste avec protection adversariale"""
        if not self._is_adversarial(feature):
            # Prendre un échantillon récent pour la prior
            memory_tensor = torch.stack(self.pattern_memory[-100:])
            memory_mean = memory_tensor.mean(dim=0)
            memory_cov = torch.cov(memory_tensor.T) + torch.eye(self.d_model) * 1e-4

            try:
                prior = dist.MultivariateNormal(memory_mean, memory_cov)
                posterior_mean = feature
                posterior_cov = torch.eye(self.d_model) * 0.1
                posterior = dist.MultivariateNormal(posterior_mean, posterior_cov)

                kl_div = torch.distributions.kl.kl_divergence(posterior, prior)
                surprise = torch.sigmoid(kl_div / 10.0)
            except:
                # Fallback en cas d'erreur numérique
                surprise = self._compute_simple_surprise(feature)
        else:
            surprise = torch.tensor(0.0)  # Suppression adversariale

        return surprise

    def _compute_faiss_surprise(self, feature: torch.Tensor) -> torch.Tensor:
        """⚡ Calcul optimisé avec Faiss pour fallback rapide"""
        if len(self.pattern_memory) < 5:
            return torch.tensor(0.5)

        # Mise à jour périodique de l'index Faiss
        if len(self.pattern_memory) % 10 == 0:
            memory_array = torch.stack(self.pattern_memory).numpy()
            self.faiss_index = faiss.IndexFlatL2(self.d_model)
            self.faiss_index.add(memory_array)

        # Recherche k-NN
        k = min(5, len(self.pattern_memory))
        distances, _ = self.faiss_index.search(feature.unsqueeze(0).numpy(), k)

        avg_distance = distances.mean()
        surprise = torch.sigmoid(torch.tensor(avg_distance) - 1.0)

        return surprise

    def _compute_simple_surprise(self, feature: torch.Tensor) -> torch.Tensor:
        """🎯 Fallback simple basé sur distance L2"""
        if len(self.pattern_memory) == 0:
            return torch.tensor(0.5)

        # Utiliser les patterns récents pour éviter le surcoût
        memory_tensor = torch.stack(self.pattern_memory[-50:])
        distances = torch.norm(memory_tensor - feature.unsqueeze(0), dim=1)

        k = min(5, len(distances))
        k_nearest = torch.topk(distances, k, largest=False).values
        avg_distance = k_nearest.mean()

        surprise = torch.sigmoid(avg_distance - 1.0)
        return surprise

    def _is_adversarial(self, feature: torch.Tensor, epsilon: float = 0.1) -> bool:
        """🛡️ Détection simple d'exemples adversariaux"""
        # Vérifier les valeurs extrêmes
        if torch.any(torch.abs(feature) > 10.0):
            return True

        # Vérifier la variance anormale
        if feature.std() < 0.01 or feature.std() > 5.0:
            return True

        return False

    def _update_memory_lru(self, features: torch.Tensor, surprise_scores: torch.Tensor):
        """💾 Mise à jour LRU avec importance pondérée"""
        # Sélectionner les patterns à mémoriser
        high_surprise_mask = surprise_scores > self.adaptive_threshold * 0.8

        for b in range(features.shape[0]):
            for s in range(features.shape[1]):
                if high_surprise_mask[b, s] or np.random.random() < 0.1:
                    self.pattern_memory.append(features[b, s].detach())
                    self.pattern_importance.append(surprise_scores[b, s].item())
                    self.access_counts.append(1)

        # Pruning LRU + importance
        if len(self.pattern_memory) > self.memory_size:
            # Calculer les scores combinés
            scores_heap = []
            for i in range(len(self.pattern_memory)):
                # Score = importance * access_count * recency
                recency = 1.0 / (len(self.pattern_memory) - i + 1)
                score = self.pattern_importance[i] * self.access_counts[i] * recency
                heapq.heappush(scores_heap, (score, i))

            # Retirer les patterns avec les scores les plus bas
            n_remove = len(self.pattern_memory) - self.memory_size
            indices_to_remove = sorted([heapq.heappop(scores_heap)[1] for _ in range(n_remove)], reverse=True)

            for idx in indices_to_remove:
                del self.pattern_memory[idx]
                del self.pattern_importance[idx]
                del self.access_counts[idx]

    def _adapt_threshold_with_variance(self, surprise_scores: torch.Tensor):
        """📈 Adaptation du seuil avec prise en compte de la variance"""
        current_mean = surprise_scores.mean().item()
        current_var = surprise_scores.var().item()

        # Taux de surprise actuel
        surprise_rate = (surprise_scores > self.adaptive_threshold).float().mean().item()
        target_rate = 0.05  # Objectif : 5% de surprises

        # Ajustement basé sur le taux ET la variance
        if surprise_rate > target_rate * 1.5:
            # Trop de surprises - augmenter le seuil
            adjustment = 1.02 + current_var * 0.05
            self.adaptive_threshold *= adjustment
        elif surprise_rate < target_rate * 0.5:
            # Pas assez de surprises - diminuer le seuil
            adjustment = 0.98 - current_var * 0.05
            self.adaptive_threshold *= adjustment

        # Momentum decay adaptatif
        momentum = 0.99 if current_var < 0.01 else 0.95
        self.adaptive_threshold = momentum * self.adaptive_threshold + (1 - momentum) * self.base_threshold

        # Limites de sécurité
        self.adaptive_threshold = max(0.2, min(0.9, self.adaptive_threshold))

    def _compute_confidence(self, surprise_scores: torch.Tensor, detected_surprises: list) -> float:
        """🎯 Calcule la confiance du Système 1 dans ses détections"""
        if len(detected_surprises) == 0:
            return 0.5  # Confiance neutre si aucune détection

        # Facteurs de confiance
        detection_rate = len(detected_surprises) / surprise_scores.numel()
        avg_surprise_strength = np.mean([s["score"] for s in detected_surprises])
        variance = surprise_scores.var().item()

        # Confiance basée sur :
        # - Taux de détection raisonnable (ni trop, ni trop peu)
        # - Force des surprises détectées
        # - Stabilité (faible variance)
        rate_confidence = 1.0 - abs(detection_rate - 0.05) / 0.05  # Optimal à 5%
        strength_confidence = min(1.0, avg_surprise_strength / 0.8)
        stability_confidence = 1.0 / (1.0 + variance)

        confidence = (rate_confidence + strength_confidence + stability_confidence) / 3

        return float(np.clip(confidence, 0.0, 1.0))


class ConceptualWorldModel(nn.Module):
    """
    💡 Système 2 : Modèle conceptuel profond du monde

    Utilise un VAE pour comprendre la structure sous-jacente des patterns.
    Ce système fonctionne comme la réflexion humaine : lent, délibéré,
    mais capable de générer des insights profonds.
    """

    def __init__(self, d_model: int, latent_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        print("   🔧 Construction du Système 2...")

        # === ENCODER (Comprendre le monde) ===
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(d_model // 4, latent_dim)
        self.fc_logvar = nn.Linear(d_model // 4, latent_dim)

        # === DECODER (Reconstruire le monde) ===
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model // 4),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 4),
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # === CLUSTERING CONCEPTUEL ===
        self.concept_clusters = []
        self.cluster_centroids = []
        self.max_clusters = 20

        print("     ✅ VAE conceptuel initialisé")
        print("     ✅ Clustering adaptatif activé")

    def forward(
        self,
        hidden_states: torch.Tensor,
        system1_surprises: list[dict],
        context: dict | None = None,
    ) -> dict:
        """
        💡 Analyse conceptuelle profonde des patterns surprenants
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Reshape pour traitement par batch
        x = hidden_states.view(-1, self.d_model)

        # === ENCODING ===
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # === DECODING ===
        reconstruction = self.decoder(z)

        # === CALCUL DES SURPRISES CONCEPTUELLES ===
        reconstruction = reconstruction.view(batch_size, seq_len, self.d_model)
        reconstruction_error = F.mse_loss(reconstruction, hidden_states, reduction="none").mean(dim=-1)

        # KL divergence pour mesurer la nouveauté conceptuelle
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.view(batch_size, seq_len)

        # Surprise conceptuelle combinée
        conceptual_surprise = torch.sigmoid(reconstruction_error + 0.1 * kl_loss)

        # === ANALYSE DES INSIGHTS ===
        insights = []
        conceptual_surprises = []

        # Analyser uniquement les positions signalées par le Système 1
        for surprise in system1_surprises:
            b, s = surprise["position"]

            if conceptual_surprise[b, s] > 0.5:  # Seuil conceptuel
                # Analyser pourquoi c'est conceptuellement surprenant
                insight = self._generate_insight(
                    pattern=hidden_states[b, s],
                    latent=z[b * seq_len + s],
                    reconstruction_error=reconstruction_error[b, s].item(),
                    context=context,
                )

                insights.append(insight)
                conceptual_surprises.append(
                    {
                        "position": (b, s),
                        "score": conceptual_surprise[b, s].item(),
                        "reconstruction_error": reconstruction_error[b, s].item(),
                        "insight": insight["explanation"],
                    }
                )

        # === MISE À JOUR DES CONCEPTS ===
        self._update_concept_clusters(z.detach(), conceptual_surprise.view(-1))

        # Calcul de la confiance
        confidence = self._compute_confidence(reconstruction_error, insights)

        return {
            "conceptual_surprise": conceptual_surprise,
            "conceptual_surprises": conceptual_surprises,
            "reconstruction": reconstruction,
            "latent_representation": z.view(batch_size, seq_len, self.latent_dim),
            "insights": insights,
            "confidence": confidence,
            "concept_clusters": len(self.concept_clusters),
        }

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """🎲 Reparameterization trick pour VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _generate_insight(
        self,
        pattern: torch.Tensor,
        latent: torch.Tensor,
        reconstruction_error: float,
        context: dict | None = None,
    ) -> dict:
        """
        💡 Génère une explication conceptuelle de la surprise
        """
        # Trouver le cluster conceptuel le plus proche
        if len(self.cluster_centroids) > 0:
            distances = [
                F.cosine_similarity(latent.unsqueeze(0), centroid.unsqueeze(0)).item()
                for centroid in self.cluster_centroids
            ]
            closest_cluster = np.argmax(distances)
            similarity = distances[closest_cluster]
        else:
            closest_cluster = -1
            similarity = 0.0

        # Générer l'explication basée sur l'analyse
        if reconstruction_error > 0.7:
            if similarity < 0.5:
                explanation = "Concept entièrement nouveau qui défie mon modèle actuel du monde"
                insight_type = "revolutionary"
            else:
                explanation = f"Variation extrême du concept #{closest_cluster} (similarité: {similarity:.2f})"
                insight_type = "extreme_variation"
        elif reconstruction_error > 0.4:
            if similarity < 0.7:
                explanation = "Extension inhabituelle d'un concept familier"
                insight_type = "conceptual_extension"
            else:
                explanation = f"Anomalie dans le concept #{closest_cluster}"
                insight_type = "conceptual_anomaly"
        else:
            explanation = "Pattern cohérent avec ma compréhension actuelle"
            insight_type = "confirmation"

        # Enrichir avec le contexte des experts
        if context and "expert_activations" in context:
            active_experts = [k for k, v in context["expert_activations"].items() if v > 0]
            if len(active_experts) > 2:
                explanation += f" (impliquant {', '.join(active_experts[:3])})"

        return {
            "explanation": explanation,
            "type": insight_type,
            "reconstruction_error": reconstruction_error,
            "closest_cluster": closest_cluster,
            "cluster_similarity": similarity,
            "confidence": 1.0 - reconstruction_error,  # Confiance inversement proportionnelle à l'erreur
        }

    def _update_concept_clusters(self, latent_vectors: torch.Tensor, importance_scores: torch.Tensor):
        """
        🧠 Met à jour les clusters conceptuels dans l'espace latent
        """
        # Sélectionner les vecteurs importants
        important_mask = importance_scores > importance_scores.mean()
        important_latents = latent_vectors[important_mask]

        if len(important_latents) == 0:
            return

        # Clustering simple par distance
        for latent in important_latents:
            if len(self.cluster_centroids) == 0:
                # Premier cluster
                self.cluster_centroids.append(latent)
                self.concept_clusters.append([latent])
            else:
                # Trouver le cluster le plus proche
                distances = [torch.norm(latent - centroid) for centroid in self.cluster_centroids]
                min_dist = min(distances)

                if min_dist > 0.5 and len(self.cluster_centroids) < self.max_clusters:
                    # Nouveau cluster
                    self.cluster_centroids.append(latent)
                    self.concept_clusters.append([latent])
                else:
                    # Ajouter au cluster existant
                    closest_idx = np.argmin(distances)
                    self.concept_clusters[closest_idx].append(latent)

                    # Mettre à jour le centroïde (moyenne mobile)
                    self.cluster_centroids[closest_idx] = torch.stack(self.concept_clusters[closest_idx]).mean(dim=0)

    def _compute_confidence(self, reconstruction_errors: torch.Tensor, insights: list[dict]) -> float:
        """
        🎯 Calcule la confiance du Système 2 dans sa compréhension
        """
        if len(insights) == 0:
            return 0.8  # Confiance de base si pas d'insights

        # Facteurs de confiance
        avg_reconstruction_quality = 1.0 - reconstruction_errors.mean().item()
        insight_confidence = np.mean([i["confidence"] for i in insights])
        cluster_maturity = min(1.0, len(self.cluster_centroids) / 10)  # Plus de clusters = meilleure compréhension

        confidence = 0.4 * avg_reconstruction_quality + 0.4 * insight_confidence + 0.2 * cluster_maturity

        return float(np.clip(confidence, 0.0, 1.0))


class MetaCognitiveController(nn.Module):
    """
    🎛️ Contrôleur méta-cognitif qui arbitre entre les deux systèmes

    Ce module décide intelligemment quand activer le Système 2
    basé sur l'output du Système 1 et le contexte.
    """

    def __init__(self, d_model: int, curiosity_threshold: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.curiosity_threshold = curiosity_threshold
        self.adaptive_threshold = curiosity_threshold

        print("   🔧 Construction du Méta-Contrôleur...")

        # Réseau de décision
        self.decision_network = nn.Sequential(
            nn.Linear(10, 20),  # 10 features méta
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3),  # 3 décisions possibles (S1, S2, Both)
        )

        # Historique pour apprentissage adaptatif
        self.decision_history = []
        self.outcome_history = []

        print("     ✅ Réseau de décision initialisé")

    def forward(
        self,
        system1_output: dict,
        context: dict | None = None,
        force_mode: ProcessingMode | None = None,
    ) -> dict:
        """
        🎛️ Décide quel système utiliser basé sur les méta-features
        """
        if force_mode is not None:
            # Mode forcé pour tests/démos
            return {
                "activate_system2": force_mode in [ProcessingMode.ANALYTICAL, ProcessingMode.UNIFIED],
                "mode": force_mode,
                "fusion_weights": (
                    {"system1": 0.5, "system2": 0.5}
                    if force_mode == ProcessingMode.UNIFIED
                    else (
                        {"system1": 1.0, "system2": 0.0}
                        if force_mode == ProcessingMode.INTUITIVE
                        else {"system1": 0.0, "system2": 1.0}
                    )
                ),
            }

        # Extraire les méta-features
        meta_features = self._extract_meta_features(system1_output, context)

        # Décision du réseau
        decision_logits = self.decision_network(meta_features)
        decision_probs = F.softmax(decision_logits, dim=-1)

        # Interprétation de la décision
        decision_idx = decision_probs.argmax().item()

        if decision_idx == 0:  # Système 1 seul
            mode = ProcessingMode.INTUITIVE
            activate_system2 = False
            fusion_weights = {"system1": 1.0, "system2": 0.0}
        elif decision_idx == 1:  # Système 2 seul
            mode = ProcessingMode.ANALYTICAL
            activate_system2 = True
            fusion_weights = {"system1": 0.0, "system2": 1.0}
        else:  # Les deux systèmes
            mode = ProcessingMode.UNIFIED
            activate_system2 = True
            # Pondération dynamique basée sur les probabilités
            s1_weight = decision_probs[0].item()
            s2_weight = decision_probs[1].item()
            total = s1_weight + s2_weight + 0.001
            fusion_weights = {"system1": s1_weight / total, "system2": s2_weight / total}

        # Adaptation du seuil
        self._adapt_curiosity_threshold(system1_output)

        # Enregistrer la décision
        decision = {
            "activate_system2": activate_system2,
            "mode": mode,
            "fusion_weights": fusion_weights,
            "decision_confidence": decision_probs.max().item(),
            "meta_features": meta_features.tolist(),
            "adaptive_threshold": self.adaptive_threshold,
        }

        self.decision_history.append(decision)

        return decision

    def _extract_meta_features(self, system1_output: dict, context: dict | None) -> torch.Tensor:
        """
        🧠 Extrait les features méta-cognitives pour la décision
        """
        features = []

        # Features du Système 1
        features.append(system1_output["surprise_scores"].max().item())  # Max surprise
        features.append(system1_output["surprise_scores"].mean().item())  # Avg surprise
        features.append(system1_output["surprise_scores"].std().item())  # Variance
        features.append(len(system1_output["detected_surprises"]))  # Nombre de détections
        features.append(system1_output["confidence"])  # Confiance S1
        features.append(float(system1_output["surprise_scores"].max() > self.adaptive_threshold))

        # Features contextuelles
        if context:
            features.append(context.get("arousal", 0.5))
            features.append(context.get("creativity", 0.5))
            features.append(context.get("previous_system2_activation", 0.0))
        else:
            features.extend([0.5, 0.5, 0.0])

        # Feature temporelle (rythme circadien cognitif)
        features.append(len(self.decision_history) % 10 / 10.0)

        return torch.tensor(features, dtype=torch.float32)

    def _adapt_curiosity_threshold(self, system1_output: dict):
        """
        📈 Adapte le seuil de curiosité basé sur les patterns récents
        """
        if len(self.decision_history) < 10:
            return

        # Analyser les décisions récentes
        recent_s2_activations = sum(1 for d in self.decision_history[-20:] if d["activate_system2"])

        s2_rate = recent_s2_activations / 20.0

        # Objectif : ~20% d'activation du Système 2
        target_rate = 0.2

        if s2_rate > target_rate * 1.5:
            # Trop d'activations S2 - augmenter le seuil
            self.adaptive_threshold *= 1.05
        elif s2_rate < target_rate * 0.5:
            # Pas assez d'activations S2 - diminuer le seuil
            self.adaptive_threshold *= 0.95

        # Limites de sécurité
        self.adaptive_threshold = max(0.5, min(0.9, self.adaptive_threshold))


class UnifiedMemorySystem:
    """
    💾 Système de mémoire unifié pour les deux systèmes

    Stocke les expériences des deux systèmes de manière interconnectée,
    permettant un apprentissage et une récupération optimaux.
    """

    def __init__(self, d_model: int, memory_size: int) -> None:
        self.d_model = d_model
        self.memory_size = memory_size

        print("   🔧 Construction de la Mémoire Unifiée...")

        # Mémoires séparées mais interconnectées
        self.intuitive_memory = []  # Patterns rapides
        self.conceptual_memory = []  # Insights profonds
        self.episodic_memory = []  # Épisodes complets

        # Métadonnées
        self.timestamps = []
        self.importance_scores = []
        self.access_patterns = []

        print("     ✅ Mémoire tripartite initialisée")

    def update(
        self,
        features: torch.Tensor,
        system1_surprises: torch.Tensor,
        system2_insights: list[dict] | None,
        context: dict | None,
    ):
        """
        💾 Met à jour la mémoire unifiée avec les nouvelles expériences
        """
        current_time = datetime.now()

        # Mise à jour de la mémoire intuitive
        high_surprise_mask = system1_surprises > system1_surprises.mean() + system1_surprises.std()

        for b in range(features.shape[0]):
            for s in range(features.shape[1]):
                if high_surprise_mask[b, s]:
                    self.intuitive_memory.append(
                        {
                            "feature": features[b, s].detach(),
                            "surprise": system1_surprises[b, s].item(),
                            "timestamp": current_time,
                            "context": context,
                        }
                    )

        # Mise à jour de la mémoire conceptuelle
        if system2_insights:
            for insight in system2_insights:
                self.conceptual_memory.append({"insight": insight, "timestamp": current_time, "context": context})

        # Création d'épisodes (quand les deux systèmes sont actifs)
        if len(self.intuitive_memory) > 0 and system2_insights:
            episode = {
                "intuitive_surprises": len([m for m in self.intuitive_memory if m["timestamp"] == current_time]),
                "conceptual_insights": len(system2_insights),
                "timestamp": current_time,
                "context": context,
                "significance": self._compute_episode_significance(system2_insights),
            }
            self.episodic_memory.append(episode)

        # Gestion de la capacité avec stratégie intelligente
        self._manage_capacity()

    def _compute_episode_significance(self, insights: list[dict]) -> float:
        """⭐ Calcule la significance d'un épisode"""
        if not insights:
            return 0.0

        # Basé sur le type et la confiance des insights
        type_weights = {
            "revolutionary": 1.0,
            "extreme_variation": 0.8,
            "conceptual_extension": 0.6,
            "conceptual_anomaly": 0.7,
            "confirmation": 0.2,
        }

        total_significance = 0.0
        for insight in insights:
            type_weight = type_weights.get(insight.get("type", "confirmation"), 0.5)
            confidence = insight.get("confidence", 0.5)
            total_significance += type_weight * confidence

        return total_significance / len(insights)

    def _manage_capacity(self):
        """
        🧹 Gère la capacité mémoire avec stratégie intelligente
        """
        # Intuitive memory : garder les plus surprenants et récents
        if len(self.intuitive_memory) > self.memory_size:
            # Score = surprise * recency
            scores = []
            current_time = datetime.now()

            for mem in self.intuitive_memory:
                age_minutes = (current_time - mem["timestamp"]).total_seconds() / 60
                recency = 1.0 / (1.0 + age_minutes)
                score = mem["surprise"] * recency
                scores.append(score)

            # Garder les top scores
            indices_to_keep = np.argsort(scores)[-self.memory_size :]
            self.intuitive_memory = [self.intuitive_memory[i] for i in indices_to_keep]

        # Conceptual memory : garder tous les insights importants
        if len(self.conceptual_memory) > self.memory_size // 2:
            # Garder les plus récents et les plus "révolutionnaires"
            self.conceptual_memory = sorted(
                self.conceptual_memory,
                key=lambda x: (x["insight"]["type"] == "revolutionary", x["timestamp"]),
                reverse=True,
            )[: self.memory_size // 2]

        # Episodic memory : compression temporelle basée sur la significance
        if len(self.episodic_memory) > 100:
            # Garder les épisodes les plus significatifs
            self.episodic_memory = sorted(self.episodic_memory, key=lambda x: x.get("significance", 0.0), reverse=True)[
                :100
            ]

    def get_status(self) -> dict:
        """
        📊 Retourne l'état actuel de la mémoire
        """
        return {
            "intuitive_patterns": len(self.intuitive_memory),
            "conceptual_insights": len(self.conceptual_memory),
            "episodes": len(self.episodic_memory),
            "usage_percent": (len(self.intuitive_memory) + len(self.conceptual_memory)) / self.memory_size * 100,
            "oldest_memory": (
                min(
                    [m["timestamp"] for m in self.intuitive_memory + self.episodic_memory],
                    default=datetime.now(),
                ).isoformat()
                if self.intuitive_memory or self.episodic_memory
                else None
            ),
            "most_significant_episode": max([e.get("significance", 0.0) for e in self.episodic_memory], default=0.0),
        }

    def retrieve_similar_patterns(self, query_pattern: torch.Tensor, top_k: int = 5) -> list[dict]:
        """
        🔍 Récupère les patterns similaires de la mémoire intuitive
        """
        if not self.intuitive_memory:
            return []

        # Calculer les similarités
        similarities = []
        for i, mem in enumerate(self.intuitive_memory):
            similarity = F.cosine_similarity(query_pattern, mem["feature"], dim=0).item()
            similarities.append((similarity, i))

        # Retourner les top-k plus similaires
        similarities.sort(reverse=True)
        return [self.intuitive_memory[i] for _, i in similarities[:top_k]]

    def get_conceptual_insights_by_type(self, insight_type: str) -> list[dict]:
        """
        💡 Récupère les insights conceptuels par type
        """
        return [mem["insight"] for mem in self.conceptual_memory if mem["insight"].get("type") == insight_type]
