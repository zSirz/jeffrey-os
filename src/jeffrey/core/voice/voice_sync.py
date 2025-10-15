#!/usr/bin/env python

"""
Module de synchronisation vocale pour Jeffrey.
Gère la mémoire vocale partagée entre les appareils via iCloud.
"""

import csv
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from jeffrey.utils.env_loader import detecter_plateforme

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('jeffrey_voice_sync.log', encoding='utf-8'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class VoiceSync:
    """Gestion de la synchronisation vocale de Jeffrey"""

    def __init__(self):
        """Initialise le système de synchronisation vocale"""
        self.platform_name, self.base_path = detecter_plateforme()
        self.sync_file = os.path.join(self.base_path, "jeffrey_voice_sync.json")
        self.local_sync_file = os.path.expanduser("~/Documents/Jeffrey_Local/jeffrey_voice_sync.json")

        # Créer les répertoires s'ils n'existent pas
        Path(os.path.dirname(self.sync_file)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.local_sync_file)).mkdir(parents=True, exist_ok=True)

        # Initialiser les données
        self.voice_history: list[dict[str, Any]] = []
        self._charger_historique()

    def _charger_historique(self) -> None:
        """Charge l'historique vocal depuis le fichier de synchronisation"""
        try:
            # Essayer d'abord le fichier iCloud
            if os.path.exists(self.sync_file):
                with open(self.sync_file, encoding='utf-8') as f:
                    data = json.load(f)
                    self.voice_history = data.get('history', [])
                logger.info(f"Historique vocal chargé depuis {self.sync_file}")
                return

            # Fallback sur le fichier local
            if os.path.exists(self.local_sync_file):
                with open(self.local_sync_file, encoding='utf-8') as f:
                    data = json.load(f)
                    self.voice_history = data.get('history', [])
                logger.info(f"Historique vocal chargé depuis {self.local_sync_file}")
                return

        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique : {e}")
            self.voice_history = []

    def _sauvegarder_historique(self) -> None:
        """Sauvegarde l'historique vocal dans les fichiers"""
        try:
            data = {'history': self.voice_history}

            # Sauvegarde principale (iCloud)
            with open(self.sync_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Sauvegarde locale (fallback)
            with open(self.local_sync_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug("Historique vocal sauvegardé avec succès")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique : {e}")

    def log_phrase(
        self, texte: str, emotion: str, timestamp: datetime | None = None, vocal_id: str | None = None, **metadata
    ) -> str:
        """
        Enregistre une nouvelle phrase vocale dans l'historique.

        Args:
            texte (str): Le texte prononcé
            emotion (str): L'émotion exprimée
            timestamp (datetime, optional): Horodatage de la phrase
            vocal_id (str, optional): Identifiant unique de la vocalisation
            **metadata: Métadonnées supplémentaires (tonalité, voix, etc.)

        Returns:
            str: L'identifiant unique de la vocalisation
        """
        if timestamp is None:
            timestamp = datetime.now()

        if vocal_id is None:
            vocal_id = str(uuid.uuid4())

        entry = {
            'id': vocal_id,
            'timestamp': timestamp.isoformat(),
            'text': texte,
            'emotion': emotion,
            'platform': self.platform_name,
            **metadata,
        }

        self.voice_history.append(entry)
        self._sauvegarder_historique()

        logger.info(f"Nouvelle entrée vocale enregistrée : {vocal_id}")
        return vocal_id

    def get_last_spoken(self) -> dict[str, Any] | None:
        """
        Récupère la dernière phrase vocale enregistrée.

        Returns:
            Optional[Dict[str, Any]]: Dernière entrée vocale ou None si vide
        """
        if not self.voice_history:
            return None
        return self.voice_history[-1]

    def get_history(self, limit: int | None = None, emotion_filter: str | None = None) -> list[dict[str, Any]]:
        """
        Récupère l'historique des phrases vocales.

        Args:
            limit (int, optional): Nombre maximum d'entrées à retourner
            emotion_filter (str, optional): Filtrer par émotion

        Returns:
            List[Dict[str, Any]]: Liste des entrées vocales triées par date
        """
        # Trier par date décroissante
        history = sorted(self.voice_history, key=lambda x: x['timestamp'], reverse=True)

        # Filtrer par émotion si demandé
        if emotion_filter:
            history = [entry for entry in history if entry['emotion'].lower() == emotion_filter.lower()]

        # Limiter le nombre d'entrées si demandé
        if limit is not None:
            history = history[:limit]

        return history

    def export_to_csv(self, output_path: str) -> bool:
        """
        Exporte l'historique vocal au format CSV.

        Args:
            output_path (str): Chemin du fichier CSV de sortie

        Returns:
            bool: True si l'export a réussi, False sinon
        """
        try:
            # Créer le répertoire de sortie si nécessaire
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Préparer les en-têtes
            if not self.voice_history:
                logger.warning("Aucune donnée à exporter")
                return False

            # Déterminer tous les champs possibles
            fields = set()
            for entry in self.voice_history:
                fields.update(entry.keys())
            fields = sorted(list(fields))

            # Écrire le CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.voice_history)

            logger.info(f"Historique vocal exporté vers {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de l'export CSV : {e}")
            return False

    def fallback_local(self) -> None:
        """Force l'utilisation du fichier local en cas de problème avec iCloud"""
        try:
            if os.path.exists(self.local_sync_file):
                with open(self.local_sync_file, encoding='utf-8') as f:
                    data = json.load(f)
                    self.voice_history = data.get('history', [])
                logger.info("Mode fallback local activé")
            else:
                logger.warning("Aucun fichier local disponible pour le fallback")
        except Exception as e:
            logger.error(f"Erreur lors du fallback local : {e}")

    def get_entries_by_emotion(self, emotion: str, limit: int = 5) -> list[str]:
        """
        Récupère les dernières phrases vocales associées à une émotion.

        Args:
            emotion (str): Émotion à rechercher
            limit (int): Nombre maximum d'entrées à retourner

        Returns:
            List[str]: Liste des phrases vocales triées par date (plus récentes d'abord)
        """
        # Récupérer l'historique filtré par émotion
        entries = self.get_history(emotion_filter=emotion, limit=limit)

        # Extraire uniquement les textes
        return [entry['text'] for entry in entries]


if __name__ == "__main__":
    # Test du module
    sync = VoiceSync()

    # Générer quelques entrées de test
    test_entries = [
        ("Bonjour, je suis Jeffrey !", "joie", {"tonalite": "aigue", "voix": "default"}),
        ("Je suis un peu triste aujourd'hui...", "tristesse", {"tonalite": "grave", "voix": "sad"}),
        ("Je suis très curieuse de te connaître !", "curiosite", {"tonalite": "neutre", "voix": "curious"}),
    ]

    print("\n=== Test du module VoiceSync ===")

    # Enregistrer les entrées de test
    for texte, emotion, metadata in test_entries:
        vocal_id = sync.log_phrase(texte, emotion, **metadata)
        print("\n✅ Entrée enregistrée :")
        print(f"   ID: {vocal_id}")
        print(f"   Texte: {texte}")
        print(f"   Émotion: {emotion}")
        print(f"   Métadonnées: {metadata}")

    # Afficher la dernière entrée
    last = sync.get_last_spoken()
    if last:
        print("\n📝 Dernière entrée :")
        print(f"   Texte: {last['text']}")
        print(f"   Émotion: {last['emotion']}")
        print(f"   Date: {last['timestamp']}")

    # Exporter en CSV
    csv_path = os.path.expanduser("~/Documents/Jeffrey_Local/voice_history.csv")
    if sync.export_to_csv(csv_path):
        print(f"\n📊 Export CSV réussi : {csv_path}")

    print("\n=== Test terminé ===")
