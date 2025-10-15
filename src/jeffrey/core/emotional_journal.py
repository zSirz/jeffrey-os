import json
import os
from datetime import datetime

# Import différé pour éviter les dépendances circulaires
# L'import de EmotionalLearning est fait au moment de l'initialisation


class EmotionalJournal:
    """
    Journal intérieur de Jeffrey pour archiver ses pensées, réflexions et souvenirs émotionnels.
    """

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.expanduser("~/.jeffrey/emotional_journal.json")
        self.entries: list[dict] = []
        self._charger_journal()

        # Import différé pour éviter les dépendances circulaires
        try:
            from core.emotions.emotional_learning import EmotionalLearning

            self.emotional_learning = EmotionalLearning()
        except ImportError:
            print("[Journal] Impossible de charger EmotionalLearning, certaines fonctionnalités seront désactivées")
            self.emotional_learning = None

    def _charger_journal(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    self.entries = json.load(f)
                print(f"[📖 Jeffrey - Journal] {len(self.entries)} entrées chargées.")
            except Exception as e:
                print(f"[Erreur Chargement Journal] {e}")
                self.entries = []

    def _sauvegarder_journal(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
            print(f"[💾 Jeffrey - Journal] Journal sauvegardé ({len(self.entries)} entrées).")
        except Exception as e:
            print(f"[Erreur Sauvegarde Journal] {e}")

    def ajouter_entree(self, pensee: str, emotion: str):
        """
        Ajoute une pensée émotionnelle au journal.
        """
        entree = {"timestamp": datetime.now().isoformat(), "emotion": emotion, "pensee": pensee}
        self.entries.append(entree)
        self._sauvegarder_journal()

        # Utiliser emotional_learning seulement s'il est disponible
        if self.emotional_learning:
            self.emotional_learning.observe_emotion(emotion)
            self.emotional_learning.export_profile()

    def obtenir_entrees_recentes(self, limit: int = 10) -> list[dict]:
        """
        Retourne les dernières pensées enregistrées.
        """
        return self.entries[-limit:]

    def rechercher_par_emotion(self, emotion: str) -> list[dict]:
        """
        Recherche toutes les pensées associées à une émotion donnée.
        """
        return [entry for entry in self.entries if entry["emotion"] == emotion]

    def afficher_resume(self):
        """
        Affiche un résumé rapide du journal intérieur.
        """
        emotions = [entry["emotion"] for entry in self.entries]
        from collections import Counter

        stats = Counter(emotions)
        print(f"[📊 Jeffrey - Résumé du Journal] {stats}")

    def lier_avec_surprise(self, pensee: str, emotion: str, type_surprise: str, contenu: str):
        """
        Lie une pensée émotionnelle à une surprise reçue ou offerte.
        """
        entree = {
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion,
            "pensee": pensee,
            "surprise": {"type": type_surprise, "contenu": contenu},
        }
        self.entries.append(entree)
        self._sauvegarder_journal()

        # Utiliser emotional_learning seulement s'il est disponible
        if self.emotional_learning:
            self.emotional_learning.observe_emotion(emotion)
            self.emotional_learning.export_profile()
