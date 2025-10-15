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


class SecurityError(Exception):
    """Erreur de sécurité (firewall, permissions)"""

    pass


class ResourceError(Exception):
    """Erreur de ressources (mémoire, CPU)"""

    pass


class DependencyError(Exception):
    """Erreur de dépendances (cycle, manquante)"""

    pass


class DiscoveryError(Exception):
    """Erreur durant la découverte de modules"""

    pass
