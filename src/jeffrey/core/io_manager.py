#!/usr/bin/env python

"""
Module de gestion des opérations d'entrée/sortie pour l'Orchestrateur_IA.
Gère toutes les interactions avec le système de fichiers pour le stockage et la récupération de données.
# 📄 Ce module gère les opérations I/O de Jeffrey, dont la journalisation via append_log().
"""

import json
import os
from datetime import datetime
from typing import Any


class IOManager:
    """
    Gestionnaire d'opérations d'entrée/sortie pour l'Orchestrateur_IA.
    Cette classe centralise toutes les opérations de lecture/écriture dans des fichiers.
    """

    def __init__(self, data_dir: str | None = None):
        """
        Initialise le gestionnaire I/O avec un répertoire de données optionnel.

        Args:
            data_dir: Répertoire où stocker les données. Par défaut: ~/.jeffrey_data
        """
        if data_dir is None:
            self.data_dir = os.path.expanduser("~/.jeffrey_data")
        else:
            self.data_dir = data_dir

        # Créer le répertoire s'il n'existe pas
        os.makedirs(self.data_dir, exist_ok=True)

    def get_filepath(self, filename: str) -> str:
        """
        Renvoie le chemin complet pour un nom de fichier donné.

        Args:
            filename: Nom du fichier (avec ou sans l'extension)

        Returns:
            Chemin complet du fichier
        """
        # Ajouter l'extension .json si non présente
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        return os.path.join(self.data_dir, filename)

    def save_data(self, data: Any, filename: str) -> bool:
        """
        Sauvegarde des données dans un fichier JSON.

        Args:
            data: Données à sauvegarder (doit être sérialisable en JSON)
            filename: Nom du fichier de destination

        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        filepath = self.get_filepath(filename)

        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des données dans {filepath}: {e}")
            return False

    def load_data(self, filename: str, default_data: Any = None) -> Any:
        """
        Charge des données depuis un fichier JSON.

        Args:
            filename: Nom du fichier à charger
            default_data: Données par défaut à retourner si le fichier n'existe pas ou est invalide

        Returns:
            Les données chargées ou default_data si échec
        """
        filepath = self.get_filepath(filename)

        if not os.path.exists(filepath):
            return default_data

        try:
            with open(filepath, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des données depuis {filepath}: {e}")
            return default_data

    def append_data(self, new_item: Any, filename: str, is_list: bool = True) -> bool:
        """
        Ajoute un élément à des données existantes (liste ou dictionnaire).

        Args:
            new_item: Nouvel élément à ajouter
            filename: Nom du fichier cible
            is_list: True si les données sont une liste, False pour un dictionnaire

        Returns:
            True si l'opération a réussi, False sinon
        """
        # Charger les données existantes
        if is_list:
            data = self.load_data(filename, default_data=[])
            data.append(new_item)
        else:
            data = self.load_data(filename, default_data={})
            if isinstance(new_item, dict):
                data.update(new_item)
            else:
                return False  # Impossible d'ajouter un élément non-dict à un dict

        # Sauvegarder les données mises à jour
        return self.save_data(data, filename)

    def file_exists(self, filename: str) -> bool:
        """
        Vérifie si un fichier existe.

        Args:
            filename: Nom du fichier à vérifier

        Returns:
            True si le fichier existe, False sinon
        """
        filepath = self.get_filepath(filename)
        return os.path.exists(filepath)

    def delete_file(self, filename: str) -> bool:
        """
        Supprime un fichier.

        Args:
            filename: Nom du fichier à supprimer

        Returns:
            True si le fichier a été supprimé, False sinon
        """
        filepath = self.get_filepath(filename)

        if not os.path.exists(filepath):
            return False

        try:
            os.remove(filepath)
            return True
        except Exception as e:
            print(f"Erreur lors de la suppression du fichier {filepath}: {e}")
            return False

    def append_log(self, log_entry: str, log_filename: str = "jeffrey_log") -> bool:
        """
        Ajoute une entrée de journal dans un fichier de logs.

        Args:
            log_entry: Entrée de journal à ajouter
            log_filename: Nom du fichier de logs

        Returns:
            True si l'opération a réussi, False sinon
        """
        timestamp = datetime.now().isoformat()
        log_item = {"timestamp": timestamp, "entry": log_entry}

        return self.append_data(log_item, log_filename, is_list=True)

    def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Nettoie les fichiers plus anciens qu'un certain nombre de jours.

        Args:
            days_old: Nombre de jours pour considérer un fichier comme vieux

        Returns:
            Nombre de fichiers supprimés
        """
        import time

        current_time = time.time()
        deleted_count = 0

        try:
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    file_age_days = file_age / (24 * 3600)

                    if file_age_days > days_old:
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                        except Exception:
                            pass  # Ignorer les erreurs de suppression

        except Exception as e:
            print(f"Erreur lors du nettoyage des fichiers: {e}")

        return deleted_count
