import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class ModuleStats:
    """Statistiques pour UCB1 (Upper Confidence Bound)"""

    module_id: str
    phase: str

    # Pour UCB1
    n_calls: int = 0
    cumulative_reward: float = 0.0

    # Métriques additionnelles
    successes: int = 0
    failures: int = 0
    cumulative_latency_ms: float = 0.0

    # Embedding pour vector routing
    embedding: np.ndarray | None = None

    @property
    def avg_reward(self) -> float:
        return self.cumulative_reward / max(self.n_calls, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.cumulative_latency_ms / max(self.n_calls, 1)

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / max(total, 1)

    def calculate_ucb1(self, total_calls: int, c: float = 2.0) -> float:
        """Calcule le score UCB1 pour exploration/exploitation"""
        if self.n_calls == 0:
            return float("inf")  # Non exploré = priorité max

        exploitation = self.avg_reward
        exploration = c * math.sqrt(math.log(total_calls) / self.n_calls)

        return exploitation + exploration


class ContextualBanditScheduler:
    """
    Scheduler avancé avec UCB1 contextuel et apprentissage
    """

    def __init__(self, exploration_factor: float = 2.0):
        self.exploration_factor = exploration_factor

        # Stats par module
        self.module_stats = {}  # module_id -> ModuleStats

        # Stats par contexte (pour contextual bandits)
        self.context_stats = defaultdict(dict)  # intent -> module_id -> stats

        # Historique pour ML
        self.decision_history = []

        # Embeddings pour vector routing
        self.module_embeddings = {}  # module_id -> embedding

        # Total des appels
        self.total_calls = 0

        # Mode de sélection
        self.selection_mode = "ucb1"  # 'ucb1', 'thompson', 'vector'

    def register_module(self, module_info: dict):
        """Enregistre un module avec ses métadonnées"""

        module_id = module_info["module_id"]
        phase = module_info["phase"]

        # Créer stats si nouveau
        if module_id not in self.module_stats:
            self.module_stats[module_id] = ModuleStats(module_id=module_id, phase=phase)

        # Stocker embedding si fourni
        if "embedding" in module_info:
            self.module_embeddings[module_id] = np.array(module_info["embedding"])
            self.module_stats[module_id].embedding = self.module_embeddings[module_id]

    def select_modules(self, phase: str, budget_ms: float, context: dict, strategy: str = "ucb1") -> list[str]:
        """
        Sélectionne les modules selon UCB1 contextuel

        Strategies:
        - ucb1: Upper Confidence Bound classique
        - thompson: Thompson Sampling (bayésien)
        - vector: Similarité vectorielle
        - quorum: Sélection par quorum
        """

        self.total_calls += 1

        # Récupérer modules de la phase
        phase_modules = [stats for stats in self.module_stats.values() if stats.phase == phase]

        if not phase_modules:
            return []

        # Appliquer stratégie
        if strategy == "vector" and "embedding" in context:
            selected = self._select_by_vector_similarity(phase_modules, context["embedding"], budget_ms)
        elif strategy == "thompson":
            selected = self._select_by_thompson_sampling(phase_modules, budget_ms, context)
        elif strategy == "quorum":
            selected = self._select_quorum(phase_modules, budget_ms, context.get("quorum_size", 2))
        else:  # ucb1 par défaut
            selected = self._select_by_ucb1(phase_modules, budget_ms, context)

        # Logger décision pour apprentissage
        self._log_decision(phase, selected, context)

        return [m.module_id for m in selected]

    def _select_by_ucb1(self, modules: list[ModuleStats], budget_ms: float, context: dict) -> list[ModuleStats]:
        """Sélection UCB1 contextuelle"""

        intent = context.get("intent", "unknown")

        # Calculer scores UCB1
        scores = []
        for module in modules:
            # UCB1 contextuel si on a des stats pour cet intent
            if intent in self.context_stats and module.module_id in self.context_stats[intent]:
                context_stats = self.context_stats[intent][module.module_id]
                score = context_stats.calculate_ucb1(self.total_calls, self.exploration_factor)
            else:
                # UCB1 global sinon
                score = module.calculate_ucb1(self.total_calls, self.exploration_factor)

            # Pénaliser si trop lent pour le budget
            if module.avg_latency_ms > budget_ms * 0.5:
                score *= 0.5

            scores.append((score, module))

        # Trier par score décroissant
        scores.sort(key=lambda x: x[0], reverse=True)

        # Sélectionner en respectant le budget
        selected = []
        spent_ms = 0

        for score, module in scores:
            expected_cost = max(module.avg_latency_ms, 50)  # Min 50ms

            if spent_ms + expected_cost <= budget_ms * 0.7:  # 70% du budget max
                selected.append(module)
                spent_ms += expected_cost
            elif len(selected) == 0:
                # Au moins un module
                selected.append(module)
                break

        return selected

    def _select_by_vector_similarity(
        self, modules: list[ModuleStats], query_embedding: np.ndarray, budget_ms: float
    ) -> list[ModuleStats]:
        """Sélection par similarité vectorielle"""

        similarities = []

        for module in modules:
            if module.embedding is not None:
                # Cosine similarity
                similarity = np.dot(query_embedding, module.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(module.embedding)
                )
                similarities.append((similarity, module))

        # Trier par similarité
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Prendre les plus similaires dans le budget
        selected = []
        spent_ms = 0

        for sim, module in similarities:
            if spent_ms + module.avg_latency_ms <= budget_ms * 0.7:
                selected.append(module)
                spent_ms += module.avg_latency_ms

        return selected

    def _select_by_thompson_sampling(
        self, modules: list[ModuleStats], budget_ms: float, context: dict
    ) -> list[ModuleStats]:
        """Thompson Sampling (échantillonnage bayésien)"""

        samples = []

        for module in modules:
            # Beta distribution pour le taux de succès
            alpha = module.successes + 1
            beta = module.failures + 1

            # Échantillonner
            sample = np.random.beta(alpha, beta)

            # Ajuster par latence
            if module.avg_latency_ms > 0:
                sample *= 100 / module.avg_latency_ms  # Favoriser les rapides

            samples.append((sample, module))

        # Trier par échantillon
        samples.sort(key=lambda x: x[0], reverse=True)

        # Sélectionner
        selected = []
        spent_ms = 0

        for sample, module in samples:
            if spent_ms + module.avg_latency_ms <= budget_ms * 0.7:
                selected.append(module)
                spent_ms += module.avg_latency_ms

        return selected

    def _select_quorum(self, modules: list[ModuleStats], budget_ms: float, quorum_size: int) -> list[ModuleStats]:
        """Sélection par quorum (N modules les plus fiables)"""

        # Trier par taux de succès
        reliable = sorted(modules, key=lambda m: m.success_rate, reverse=True)

        # Prendre le quorum
        selected = []
        spent_ms = 0

        for module in reliable[:quorum_size]:
            if spent_ms + module.avg_latency_ms <= budget_ms:
                selected.append(module)
                spent_ms += module.avg_latency_ms

        return selected

    def update_module_performance(
        self,
        module_id: str,
        latency_ms: float,
        success: bool,
        reward: float,
        context: dict | None = None,
    ):
        """Met à jour les statistiques après exécution"""

        if module_id not in self.module_stats:
            return

        # Stats globales
        stats = self.module_stats[module_id]
        stats.n_calls += 1
        stats.cumulative_reward += reward
        stats.cumulative_latency_ms += latency_ms

        if success:
            stats.successes += 1
        else:
            stats.failures += 1

        # Stats contextuelles
        if context and "intent" in context:
            intent = context["intent"]
            if module_id not in self.context_stats[intent]:
                self.context_stats[intent][module_id] = ModuleStats(module_id=module_id, phase=stats.phase)

            ctx_stats = self.context_stats[intent][module_id]
            ctx_stats.n_calls += 1
            ctx_stats.cumulative_reward += reward
            ctx_stats.cumulative_latency_ms += latency_ms

            if success:
                ctx_stats.successes += 1
            else:
                ctx_stats.failures += 1

    def _log_decision(self, phase: str, selected: list[ModuleStats], context: dict):
        """Enregistre la décision pour apprentissage futur"""

        self.decision_history.append(
            {
                "timestamp": time.time(),
                "phase": phase,
                "selected": [m.module_id for m in selected],
                "context": context,
                "total_calls": self.total_calls,
            }
        )

        # Limiter historique
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]

    def calculate_reward(self, quality_score: float, latency_ms: float, user_satisfaction: float = 0.5) -> float:
        """Calcule la récompense pour une exécution"""

        # Reward = qualité - pénalité latence + satisfaction
        latency_penalty = min(1.0, latency_ms / 1000)  # Normaliser à [0,1]

        reward = quality_score * 0.5 + (1 - latency_penalty) * 0.3 + user_satisfaction * 0.2

        return max(0, min(1, reward))  # Clamp [0,1]

    def get_insights(self) -> dict:
        """Retourne des insights pour monitoring"""

        # Top performers par phase
        top_by_phase = {}
        phases = set(s.phase for s in self.module_stats.values())

        for phase in phases:
            phase_modules = [s for s in self.module_stats.values() if s.phase == phase and s.n_calls > 0]
            if phase_modules:
                top = max(phase_modules, key=lambda s: s.avg_reward)
                top_by_phase[phase] = {
                    "module_id": top.module_id,
                    "avg_reward": top.avg_reward,
                    "success_rate": top.success_rate,
                }

        return {
            "total_calls": self.total_calls,
            "num_modules": len(self.module_stats),
            "top_performers": top_by_phase,
            "exploration_rate": self._calculate_exploration_rate(),
        }

    def _calculate_exploration_rate(self) -> float:
        """Calcule le taux d'exploration actuel"""

        if not self.decision_history:
            return 1.0

        # Regarder les 100 dernières décisions
        recent = self.decision_history[-100:]

        # Compter les modules peu explorés
        exploration_count = 0
        for decision in recent:
            for module_id in decision["selected"]:
                if module_id in self.module_stats:
                    if self.module_stats[module_id].n_calls < 10:
                        exploration_count += 1

        total = sum(len(d["selected"]) for d in recent)
        return exploration_count / max(total, 1)

    def update_reward(self, module_id: str, reward: float, intent: str | None = None):
        """Met à jour les rewards d'un module"""
        if module_id not in self.module_stats:
            self.module_stats[module_id] = ModuleStats(module_id=module_id, phase="unknown")

        stats = self.module_stats[module_id]
        stats.n_calls += 1
        stats.cumulative_reward += reward
        if reward > 0:
            stats.successes += 1
        else:
            stats.failures += 1

        # Si un intent est fourni, mettre à jour aussi les stats contextuelles
        if intent:
            if intent not in self.context_stats:
                self.context_stats[intent] = {}
            if module_id not in self.context_stats[intent]:
                self.context_stats[intent][module_id] = ModuleStats(module_id=module_id, phase=stats.phase)
            ctx_stats = self.context_stats[intent][module_id]
            ctx_stats.n_calls += 1
            ctx_stats.cumulative_reward += reward
            if reward > 0:
                ctx_stats.successes += 1
            else:
                ctx_stats.failures += 1

    def update_latency(self, module_id: str, latency_ms: float):
        """Met à jour la latence d'un module"""
        if module_id not in self.module_stats:
            self.module_stats[module_id] = ModuleStats(module_id=module_id, phase="unknown")

        self.module_stats[module_id].cumulative_latency_ms += latency_ms

    def get_module_stats(self, module_id: str) -> ModuleStats | None:
        """Helper pour récupérer les stats d'un module (utile pour timeouts)"""
        return self.module_stats.get(module_id)

    def save_state(self, filepath: str):
        """Sauvegarde l'état complet incluant phase et contextes"""

        # Préparer les stats globales
        module_stats_dict = {}
        for module_id, stats in self.module_stats.items():
            module_stats_dict[module_id] = {
                "phase": stats.phase,  # IMPORTANT: sauver la phase
                "n_calls": stats.n_calls,
                "cumulative_reward": stats.cumulative_reward,
                "successes": stats.successes,
                "failures": stats.failures,
                "cumulative_latency_ms": stats.cumulative_latency_ms,
            }

        # Préparer les stats contextuelles
        context_stats_dict = {}
        for intent, modules in self.context_stats.items():
            context_stats_dict[intent] = {}
            for module_id, stats in modules.items():
                if isinstance(stats, ModuleStats):
                    context_stats_dict[intent][module_id] = {
                        "phase": stats.phase,  # Sauver phase contextuelle
                        "n_calls": stats.n_calls,
                        "cumulative_reward": stats.cumulative_reward,
                        "successes": stats.successes,
                        "failures": stats.failures,
                        "cumulative_latency_ms": stats.cumulative_latency_ms,
                    }

        # Créer l'état complet
        state = {
            "module_stats": module_stats_dict,
            "context_stats": context_stats_dict,
            "total_calls": self.total_calls,
            "version": "2.0",  # Pour compatibilité future
        }

        # Sauvegarder
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        print(
            f"✅ État sauvegardé: {len(self.module_stats)} modules, "
            f"{len(self.context_stats)} contextes, {self.total_calls} appels"
        )

    def load_state(self, filepath: str):
        """Charge et reconstruit l'état complet depuis un fichier"""
        try:
            with open(filepath) as f:
                state = json.load(f)

            # Reconstruction stats globales
            self.module_stats = {}
            for module_id, data in state.get("module_stats", {}).items():
                ms = ModuleStats(module_id=module_id, phase=data.get("phase", "unknown"))
                ms.n_calls = data.get("n_calls", 0)
                ms.cumulative_reward = data.get("cumulative_reward", 0.0)
                ms.successes = data.get("successes", 0)
                ms.failures = data.get("failures", 0)
                ms.cumulative_latency_ms = data.get("cumulative_latency_ms", 0.0)
                self.module_stats[module_id] = ms

            # Reconstruction stats contextuelles
            self.context_stats = defaultdict(dict)
            for intent, modules in state.get("context_stats", {}).items():
                for module_id, data in modules.items():
                    cs = ModuleStats(module_id=module_id, phase=data.get("phase", "unknown"))
                    cs.n_calls = data.get("n_calls", 0)
                    cs.cumulative_reward = data.get("cumulative_reward", 0.0)
                    cs.successes = data.get("successes", 0)
                    cs.failures = data.get("failures", 0)
                    cs.cumulative_latency_ms = data.get("cumulative_latency_ms", 0.0)
                    self.context_stats[intent][module_id] = cs

            self.total_calls = state.get("total_calls", 0)

            print(
                f"✅ État restauré: {len(self.module_stats)} modules, "
                f"{len(self.context_stats)} contextes, {self.total_calls} appels"
            )

        except FileNotFoundError:
            print(f"⚠️  Fichier {filepath} non trouvé - démarrage état vide")
        except json.JSONDecodeError as e:
            print(f"❌ Erreur parsing JSON: {e}")
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")


# Alias pour compatibilité avec les tests
BasalGangliaScheduler = ContextualBanditScheduler
