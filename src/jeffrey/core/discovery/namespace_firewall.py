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

from __future__ import annotations

import logging
import re

from .policy_bus import Domain  # Une seule source de vérité pour Domain


class NamespaceFirewall:
    """Pare-feu simple pour les domaines"""

    def __init__(self) -> None:
        self.logger = logging.getLogger("firewall")
        self.load_rules()

    def load_rules(self):
        """Charger les règles de base"""
        self.rules = {
            Domain.BRAIN: {
                "allowed_topics": [
                    "percept.*",
                    "plan.*",
                    "mem.*",
                    "consciousness.*",
                    "emotion.*",
                    "system.*",
                ],
                "forbidden_topics": ["bridge.*", "avatar.*", "skill.*"],
            },
            Domain.BRIDGE: {
                "allowed_topics": ["bridge.*", "system.*"],
                "forbidden_topics": ["consciousness.*", "mem.store"],
            },
            Domain.AVATAR: {
                "allowed_topics": ["avatar.*", "skill.*"],
                "forbidden_topics": ["consciousness.*", "mem.*", "bridge.admin"],
            },
            Domain.SKILL: {
                "allowed_topics": ["skill.*", "system.*"],
                "forbidden_topics": ["consciousness.*", "mem.*"],
            },
        }

    def validate_subscription(self, domain: Domain, topic: str) -> tuple[bool, str | None]:
        """Valider qu'un domaine peut souscrire à un topic"""

        if domain == Domain.BRAIN:
            for forbidden in self.rules[domain].get("forbidden_topics", []):
                if self._matches_pattern(topic, forbidden):
                    return False, f"Topic {topic} forbidden for {domain.value}"
            return True, None

        allowed = False
        for pattern in self.rules.get(domain, {}).get("allowed_topics", []):
            if self._matches_pattern(topic, pattern):
                allowed = True
                break

        if not allowed:
            return False, f"Topic {topic} not allowed for {domain.value}"

        return True, None

    def validate_emission(self, source_domain: Domain, target_topic: str, data: dict) -> tuple[bool, dict | None]:
        """Valider et filtrer une émission"""

        if source_domain == Domain.BRAIN:
            return True, data

        if source_domain in [Domain.BRIDGE, Domain.AVATAR]:
            filtered = self._filter_basic_pii(data)
            return True, filtered

        return True, data

    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Vérifier si un topic match un pattern"""
        regex_pattern = pattern.replace(".", "\\.").replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", topic))

    def _filter_basic_pii(self, data: dict) -> dict:
        """Filtrage basique des PII"""
        if not isinstance(data, dict):
            return data

        filtered = data.copy()
        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"

        def clean_value(value):
            if isinstance(value, str):
                value = re.sub(email_pattern, "[EMAIL]", value)
            return value

        for key, value in filtered.items():
            filtered[key] = clean_value(value)

        return filtered
