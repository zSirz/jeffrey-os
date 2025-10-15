# src/jeffrey/core/symbiosis/symbiosis_engine.py

import json
import time
from pathlib import Path


class SymbiosisEngine:
    """
    Module système pour Jeffrey OS.

    Ce module implémente les fonctionnalités essentielles pour module système pour jeffrey os.
    Il fournit une architecture robuste et évolutive intégrant les composants
    nécessaires au fonctionnement optimal du système. L'implémentation suit
    les principes de modularité et d'extensibilité pour faciliter l'évolution
    future du système.

    Le module gère l'initialisation, la configuration, le traitement des données,
    la communication inter-composants, et la persistance des états. Il s'intègre
    harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
    une séparation claire des responsabilités.

    L'architecture interne permet une évolution adaptative basée sur les interactions
    et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
    cohérente et authentique.
    """

    def __init__(self) -> None:
        self.symbiosis_graph = {}
        self.strength_matrix = {}
        self.evolution_history = []
        self.config = {
            "evolution_rate": 0.1,
            "pruning_threshold": 0.1,
            "max_failures": 3,
            "reward_signals": ["latency_down", "success_up", "errors_down"],
        }

    def create_auto_link(self, source: str, target: str, strength: float = 1.0):
        """Créer un lien symbiotique auto-évolutif"""

        # Validation sécurité (GPT/Gemini)
        if not self._validate_link_security(source, target):
            return False

        link_id = f"{source}<->{target}"
        self.symbiosis_graph[link_id] = {
            "source": source,
            "target": target,
            "strength": strength,
            "success_count": 0,
            "fail_count": 0,
            "created_at": time.time(),
            "last_evolution": time.time(),
        }

        # Auto-link spécial core-bridge (critique)
        if "core" in source and "bridge" in target:
            self.symbiosis_graph[link_id]["critical"] = True
            self.symbiosis_graph[link_id]["strength"] = 2.0  # Plus fort

        return True

    def evolve_link_strength(self, link_id: str, success: bool, context: dict = None):
        """Evolution hebbienne avec ML feedback"""

        if link_id not in self.symbiosis_graph:
            return

        link = self.symbiosis_graph[link_id]

        # ML prediction (simple pour commencer)
        adaptation_factor = self._ml_predict_adaptation(link, context)

        if success:
            # Renforcement hebbien avec ML
            link["strength"] = min(10.0, link["strength"] * (1 + self.config["evolution_rate"] * adaptation_factor))
            link["success_count"] += 1
        else:
            # Affaiblissement
            link["strength"] *= 1 - self.config["evolution_rate"]
            link["fail_count"] += 1

            # Pruning si trop faible SAUF liens critiques
            if not link.get("critical", False):
                if (
                    link["strength"] < self.config["pruning_threshold"]
                    or link["fail_count"] > self.config["max_failures"]
                ):
                    self._prune_link(link_id)

        link["last_evolution"] = time.time()
        self._log_evolution(link_id, success, link["strength"])

    def _ml_predict_adaptation(self, link: dict, context: dict = None) -> float:
        """Prédit le facteur d'adaptation ML"""

        # Simple heuristique pour démarrer
        # Plus tard : réseau de neurones sur l'historique

        base_factor = 1.0

        # Si beaucoup de succès récents, accélérer
        if link["success_count"] > link["fail_count"] * 2:
            base_factor *= 1.5

        # Si lien critique, évoluer plus vite
        if link.get("critical", False):
            base_factor *= 2.0

        # Si contexte de charge élevée
        if context and context.get("load", 0) > 80:
            base_factor *= 0.5  # Ralentir l'évolution sous charge

        return base_factor

    def _validate_link_security(self, source: str, target: str) -> bool:
        """Validation sécurité des liens (Gemini/GPT)"""

        # Règles de sécurité
        forbidden = [
            ("avatar", "core"),  # Avatar ne peut pas accéder directement au core
            ("skill", "guardian"),  # Skill ne peut pas modifier guardian
            ("external", "memory"),  # External ne peut pas écrire directement en mémoire
        ]

        for forbidden_source, forbidden_target in forbidden:
            if forbidden_source in source and forbidden_target in target:
                # Sauf si passe par bridge
                if "bridge" not in source and "bridge" not in target:
                    return False

        return True

    def _prune_link(self, link_id: str):
        """Supprime un lien faible (méthode manquante ajoutée)"""
        if link_id in self.symbiosis_graph:
            # Log avant suppression
            self._log_evolution(link_id, False, 0, action="pruned")
            # Supprimer
            del self.symbiosis_graph[link_id]
            # Nettoyer la matrice si présente
            if link_id in self.strength_matrix:
                del self.strength_matrix[link_id]

    def _log_evolution(self, link_id: str, success: bool, strength: float, action: str = "evolved"):
        """Log l'évolution d'un lien (méthode manquante ajoutée)"""
        log_entry = {
            "timestamp": time.time(),
            "link_id": link_id,
            "success": success,
            "strength": strength,
            "action": action,
        }
        self.evolution_history.append(log_entry)

        # Garder seulement les 1000 derniers logs pour éviter explosion mémoire
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]

    def get_strong_links(self, threshold: float = 2.0) -> list[str]:
        """Retourne les liens forts (pour optimisation)"""
        return [link_id for link_id, link in self.symbiosis_graph.items() if link["strength"] > threshold]

    def save_state(self, filepath: str = "symbiosis_state.json"):
        """Persiste l'état pour reprise après crash"""
        state = {
            "graph": self.symbiosis_graph,
            "matrix": self.strength_matrix,
            "config": self.config,
            "timestamp": time.time(),
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str = "symbiosis_state.json"):
        """Charge l'état précédent"""
        if Path(filepath).exists():
            with open(filepath) as f:
                state = json.load(f)
                self.symbiosis_graph = state["graph"]
                self.strength_matrix = state.get("matrix", {})
                self.config.update(state.get("config", {}))
