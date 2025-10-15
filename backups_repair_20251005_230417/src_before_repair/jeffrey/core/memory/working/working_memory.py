"""
Mémoire de travail pour traitement temps réel.

Ce module implémente les fonctionnalités essentielles pour mémoire de travail pour traitement temps réel.
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

from __future__ import annotations

from collections import deque


class WorkingMemory:
    """
    Mémoire de travail - contexte court terme
    Garde les échanges récents pour cohérence
    """

    def __init__(self, max_exchanges: int = 10) -> None:
        self.max_exchanges = max_exchanges
        self.exchanges = deque(maxlen=max_exchanges)
        self.current_topic = None
        self.entities = {}  # Entités mentionnées

    def add_exchange(self, user_input: str, assistant_response: str):
        """Ajoute un échange à la mémoire"""

        exchange = {"user": user_input, "assistant": assistant_response}

        self.exchanges.append(exchange)

        # Extraction basique d'entités (noms propres)
        self._extract_entities(user_input)
        self._extract_entities(assistant_response)

    def _extract_entities(self, text: str):
        """Extraction simple d'entités (à améliorer)"""

        # Détection basique de noms propres (mots capitalisés)
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                # Évite les débuts de phrase
                if word not in ["Je", "Tu", "Il", "Elle", "Nous", "Vous"]:
                    self.entities[word] = self.entities.get(word, 0) + 1

    def get_context(self, limit: int = 3) -> list[dict]:
        """
        Récupère le contexte récent pour le LLM
        """

        if not self.exchanges:
            return []

        # Retourne les N derniers échanges
        recent = list(self.exchanges)[-limit:]

        return recent

    def get_full_context(self) -> list[dict]:
        """Récupère tout le contexte"""

        return list(self.exchanges)

    def search_keyword(self, keyword: str) -> list[dict]:
        """Recherche un mot-clé dans la mémoire"""

        keyword_lower = keyword.lower()
        results = []

        for exchange in self.exchanges:
            if keyword_lower in exchange["user"].lower() or keyword_lower in exchange["assistant"].lower():
                results.append(exchange)

        return results

    def get_entities(self) -> dict[str, int]:
        """Retourne les entités mentionnées"""

        # Tri par fréquence
        return dict(sorted(self.entities.items(), key=lambda x: x[1], reverse=True))

    def set_topic(self, topic: str) -> None:
        """Définit le sujet actuel"""

        self.current_topic = topic

    def get_topic(self) -> str | None:
        """Récupère le sujet actuel"""

        return self.current_topic

    def clear(self):
        """Efface la mémoire de travail"""

        self.exchanges.clear()
        self.entities.clear()
        self.current_topic = None

    def export(self) -> dict:
        """Exporte pour sauvegarde"""

        return {
            "exchanges": list(self.exchanges),
            "entities": self.entities,
            "current_topic": self.current_topic,
        }

    def import_state(self, state: dict):
        """Importe un état sauvegardé"""

        if "exchanges" in state:
            self.exchanges = deque(state["exchanges"], maxlen=self.max_exchanges)

        if "entities" in state:
            self.entities = state["entities"]

        if "current_topic" in state:
            self.current_topic = state["current_topic"]
