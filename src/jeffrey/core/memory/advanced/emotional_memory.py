#!/usr/bin/env python

"""
Système de mémoire émotionnelle et souvenirs affectifs.

Ce module implémente les fonctionnalités essentielles pour système de mémoire émotionnelle et souvenirs affectifs.
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

from datetime import datetime
from typing import Any

from jeffrey.core.io_manager import IOManager


class EmotionalMemory:
    """
    Gère les souvenirs émotionnels de Jeffrey (graines plantées, journal intime numérique).
    Utilise IOManager pour gérer les opérations de lecture/écriture.
    """

    def __init__(self, io_manager: IOManager | None = None, filepath: str = "emotional_journal") -> None:
        """
        Initialise la mémoire émotionnelle.

        Args:
            io_manager: Gestionnaire d'I/O à utiliser. Si None, un nouveau sera créé.
            filepath: Nom du fichier de journal émotionnel (sans extension)
        """
        self.io_manager = io_manager or IOManager(data_dir="data")
        self.filepath = filepath
        self.data = {"graines": [], "journal": []}
        self.load_memory()

    def load_memory(self) -> None:
        """
        Charge la mémoire émotionnelle depuis le fichier.
        Si le fichier n'existe pas ou est invalide, initialise une mémoire vide.
        """
        loaded_data = self.io_manager.load_data(self.filepath, default_data={"graines": [], "journal": []})
        self.data = loaded_data

    def save_memory(self) -> bool:
        """
        Sauvegarde la mémoire émotionnelle dans le fichier.

        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        return self.io_manager.save_data(self.data, self.filepath)

    def planter_graine(self, sujet: str, emotion: str) -> None:
        """
        Enregistre une nouvelle "graine" liée à un sujet marquant.

        Args:
            sujet: Le sujet ou thème de la graine
            emotion: L'émotion associée à cette graine
        """
        entree = {"sujet": sujet, "emotion": emotion, "date": datetime.utcnow().isoformat()}
        self.data["graines"].append(entree)
        self.save_memory()

    def enregistrer_moment(self, description: str, emotion: str) -> None:
        """
        Enregistre un moment significatif dans le journal intime numérique.

        Args:
            description: Description du moment
            emotion: L'émotion ressentie
        """
        entree = {
            "description": description,
            "emotion": emotion,
            "date": datetime.utcnow().isoformat(),
        }
        self.data["journal"].append(entree)
        self.save_memory()

    def get_graines(self) -> list[dict[str, Any]]:
        """
        Récupère toutes les graines émotionnelles.

        Returns:
            Liste des graines émotionnelles
        """
        return self.data.get("graines", [])

    def get_journal(self) -> list[dict[str, Any]]:
        """
        Récupère tout le journal émotionnel.

        Returns:
            Liste des entrées du journal
        """
        return self.data.get("journal", [])

    def recuperer_derniers_souvenirs(self, n: int = 10) -> list[dict[str, Any]]:
        """
        Récupère les n derniers souvenirs du journal.

        Args:
            n: Nombre de souvenirs à récupérer

        Returns:
            Liste des derniers souvenirs
        """
        journal = self.get_journal()
        # Trier par date (du plus récent au plus ancien)
        journal_trie = sorted(journal, key=lambda x: x.get("date", ""), reverse=True)
        # Retourner les n premiers éléments (ou tous si moins de n)
        return journal_trie[:n]
