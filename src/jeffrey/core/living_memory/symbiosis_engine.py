#!/usr/bin/env python3
"""
SymbiosisEngine - Moteur d'adaptation synaptique
Stub avec fonctionnalités de base (correction GPT)
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SymbiosisEngine:
    """
    Moteur de symbiose pour connexions adaptatives
    Implémente la plasticité synaptique du cerveau
    """

    def __init__(self):
        # Graphe des connexions (poids entre modules)
        self.connections: dict[tuple[str, str], float] = defaultdict(float)

        # Historique des interactions
        self.interaction_history: list[dict[str, Any]] = []

        # Seuils d'adaptation
        self.strengthen_threshold = 0.7
        self.weaken_threshold = 0.3
        self.prune_threshold = 0.1

        # État
        self._initialized = False

    async def initialize(self, bus: Any) -> None:
        """Initialise le moteur avec le bus neuronal"""
        self.bus = bus

        # S'abonner aux événements pour apprendre
        if bus:
            bus.register_handler("*", self._learn_from_event)

        # Charger l'état précédent si disponible
        await self._load_state()

        self._initialized = True
        logger.info("SymbiosisEngine initialized")

    async def strengthen_connection(self, source: str, target: str, weight: float = 0.1) -> None:
        """Renforce une connexion entre deux modules"""
        key = (source, target)
        self.connections[key] = min(1.0, self.connections[key] + weight)

        logger.debug(f"Strengthened {source}  {target}: {self.connections[key]:.2f}")

        # Enregistrer l'interaction
        self.interaction_history.append(
            {
                "source": source,
                "target": target,
                "weight": weight,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Nettoyer l'historique si trop grand
        if len(self.interaction_history) > 10000:
            self.interaction_history = self.interaction_history[-5000:]

    async def weaken_connection(self, source: str, target: str, weight: float = 0.05) -> None:
        """Affaiblit une connexion"""
        key = (source, target)
        self.connections[key] = max(0.0, self.connections[key] - weight)

        # Prune si trop faible
        if self.connections[key] < self.prune_threshold:
            del self.connections[key]
            logger.debug(f"Pruned weak connection: {source}  {target}")

    async def get_strongest_connections(self, module: str, limit: int = 5) -> list[tuple[str, float]]:
        """Retourne les connexions les plus fortes d'un module"""
        connections = []

        for (source, target), weight in self.connections.items():
            if source == module:
                connections.append((target, weight))
            elif target == module:
                connections.append((source, weight))

        # Trier par poids décroissant
        connections.sort(key=lambda x: x[1], reverse=True)

        return connections[:limit]

    async def analyze_history(self, session_id: str) -> dict[str, Any]:
        """
        Analyse l'historique pour suggestions proactives
        (Amélioration Grok : ML-based suggestions)
        """
        # Analyser les patterns d'interaction
        patterns = defaultdict(int)

        for interaction in self.interaction_history[-100:]:  # Dernières 100 interactions
            patterns[interaction["source"]] += 1
            patterns[interaction["target"]] += 1

        # Trouver les modules les plus actifs
        most_active = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]

        # Générer des suggestions basées sur les patterns
        suggestions = []

        for module, count in most_active:
            if "memory" in module.lower() and count > 10:
                suggestions.append("Consolidation mémoire recommandée")
            elif "emotion" in module.lower() and count > 15:
                suggestions.append("État émotionnel intense détecté")
            elif "external" in module.lower() and count > 20:
                suggestions.append("Activité externe élevée - vérifier sécurité")

        return {
            "patterns": dict(patterns),
            "most_active_modules": most_active,
            "suggestions": suggestions,
            "total_interactions": len(self.interaction_history),
        }

    async def _learn_from_event(self, envelope: Any) -> None:
        """Apprend des événements qui passent dans le bus"""
        # Extraire source et topic
        source = envelope.source or "unknown"
        topic = envelope.topic

        # Renforcer les connexions observées
        if source != "unknown":
            await self.strengthen_connection(source, topic, 0.01)

    async def _load_state(self) -> None:
        """Charge l'état précédent depuis le disque"""
        state_file = Path("data/symbiosis_state.json")

        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)

                # Restaurer les connexions
                for key_str, weight in state.get("connections", {}).items():
                    source, target = key_str.split("")
                    self.connections[(source.strip(), target.strip())] = weight

                logger.info(f"Loaded {len(self.connections)} connections from state")

            except Exception as e:
                logger.error(f"Failed to load symbiosis state: {e}")

    async def save_state(self) -> None:
        """Sauvegarde l'état sur le disque"""
        state_file = Path("data/symbiosis_state.json")
        state_file.parent.mkdir(exist_ok=True)

        try:
            # Convertir les connexions en format sérialisable
            connections_dict = {f"{source}  {target}": weight for (source, target), weight in self.connections.items()}

            state = {
                "connections": connections_dict,
                "timestamp": datetime.now().isoformat(),
                "stats": {
                    "total_connections": len(self.connections),
                    "avg_weight": (sum(self.connections.values()) / len(self.connections) if self.connections else 0),
                },
            }

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            logger.info("Symbiosis state saved")

        except Exception as e:
            logger.error(f"Failed to save symbiosis state: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques du moteur"""
        return {
            "total_connections": len(self.connections),
            "avg_connection_weight": (
                sum(self.connections.values()) / len(self.connections) if self.connections else 0
            ),
            "interaction_history_size": len(self.interaction_history),
            "strongest_overall": (
                sorted(self.connections.items(), key=lambda x: x[1], reverse=True)[:5] if self.connections else []
            ),
        }
