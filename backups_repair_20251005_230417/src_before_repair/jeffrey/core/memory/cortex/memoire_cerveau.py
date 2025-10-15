"""
Module de composant de gestion mémorielle pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de composant de gestion mémorielle pour jeffrey os.
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

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class MemoireCerveau:
    """
    Mémoire cognitive centrale de Jeffrey.
    Gère les expériences, patterns et apprentissages profonds.
    """

    def __init__(self, data_path: str = "data/memoire/") -> None:
        """Initialise la mémoire cerveau."""
        self.data_path = data_path
        self.experiences = []
        self.patterns = {}
        self.insights = []
        self.cognitive_map = {}

        os.makedirs(data_path, exist_ok=True)
        self._charger_memoire()

    def integrer_experience(self, interaction: dict[str, Any]) -> None:
        """
        Intègre une nouvelle expérience dans la mémoire cerveau.

        Args:
            interaction: Dictionnaire contenant les données d'interaction
        """
        try:
            experience = {
                "id": len(self.experiences),
                "timestamp": datetime.now().isoformat(),
                "contenu": interaction.get("text", ""),
                "emotions": interaction.get("emotions", {}),
                "contexte": interaction.get("context", {}),
                "reponse": interaction.get("response", ""),
                "intensite_emotionnelle": self._calculer_intensite(interaction.get("emotions", {})),
                "tags": self._extraire_tags(interaction),
                "liens_cognitifs": [],
            }

            self.experiences.append(experience)
            self._analyser_patterns(experience)
            self._creer_liens_cognitifs(experience)

            logger.info(f"🧠 Expérience intégrée: ID {experience['id']}")

        except Exception as e:
            logger.error(f"Erreur intégration expérience: {e}")

    def _calculer_intensite(self, emotions: dict[str, float]) -> float:
        """Calcule l'intensité émotionnelle globale."""
        if not emotions:
            return 0.0
        # S'assurer que les valeurs sont numériques
        numeric_values = []
        for value in emotions.values():
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
        return 0.0

    def _extraire_tags(self, interaction: dict[str, Any]) -> list[str]:
        """Extrait des tags cognitifs de l'interaction."""
        tags = []
        contenu = interaction.get("text", "").lower()

        # Tags basés sur le contenu
        if any(word in contenu for word in ["amour", "aime", "adore"]):
            tags.append("affection")
        if any(word in contenu for word in ["rêve", "rêver", "imagination"]):
            tags.append("créativité")
        if any(word in contenu for word in ["apprendre", "comprendre", "savoir"]):
            tags.append("apprentissage")
        if any(word in contenu for word in ["souvenir", "mémoire", "rappeler"]):
            tags.append("mémorisation")

        # Tags émotionnels
        emotions = interaction.get("emotions", {})
        if emotions:
            # S'assurer que toutes les valeurs sont numériques pour max()
            emotions_clean = {}
            for emotion, value in emotions.items():
                try:
                    emotions_clean[emotion] = float(value)
                except (TypeError, ValueError):
                    emotions_clean[emotion] = 0.0

            if emotions_clean:
                emotion_principale = max(emotions_clean, key=emotions_clean.get)
                tags.append(f"emotion_{emotion_principale}")

        return tags

    def _analyser_patterns(self, experience: dict[str, Any]) -> None:
        """Analyse les patterns dans les expériences."""
        tags = experience.get("tags", [])

        for tag in tags:
            if tag not in self.patterns:
                self.patterns[tag] = {
                    "occurrences": 0,
                    "intensite_moyenne": 0.0,
                    "experiences_liees": [],
                    "evolution": [],
                }

            pattern = self.patterns[tag]
            pattern["occurrences"] += 1
            pattern["experiences_liees"].append(experience["id"])

            # Mise à jour intensité moyenne avec conversion sécurisée
            nouvelle_intensite = float(experience.get("intensite_emotionnelle", 0.0))
            ancienne_moyenne = float(pattern["intensite_moyenne"])
            pattern["intensite_moyenne"] = (
                ancienne_moyenne * (pattern["occurrences"] - 1) + nouvelle_intensite
            ) / pattern["occurrences"]

            # Enregistrer l'évolution
            pattern["evolution"].append({"timestamp": experience["timestamp"], "intensite": nouvelle_intensite})

    def _creer_liens_cognitifs(self, experience: dict[str, Any]) -> None:
        """Crée des liens cognitifs avec d'autres expériences."""
        experience_id = experience["id"]
        tags_actuels = set(experience.get("tags", []))

        # Chercher des expériences similaires
        for exp in self.experiences[:-1]:  # Exclure l'expérience actuelle
            tags_exp = set(exp.get("tags", []))

            # Calculer la similarité
            intersection = len(tags_actuels & tags_exp)
            union = len(tags_actuels | tags_exp)

            if union > 0:
                similarite = intersection / union

                if similarite > 0.3:  # Seuil de similarité
                    experience["liens_cognitifs"].append(
                        {
                            "experience_id": exp["id"],
                            "similarite": similarite,
                            "type_lien": "thematique",
                        }
                    )

    def generer_insight(self, domaine: str = None) -> dict[str, Any] | None:
        """
        Génère un insight cognitif basé sur les patterns analysés.

        Args:
            domaine: Domaine spécifique pour l'insight (optionnel)

        Returns:
            Dictionnaire avec l'insight généré
        """
        try:
            patterns_valides = {
                k: v for k, v in self.patterns.items() if v["occurrences"] >= 3
            }  # Au moins 3 occurrences

            if not patterns_valides:
                return None

            # Trouver le pattern le plus significatif
            pattern_principal = max(
                patterns_valides.items(),
                key=lambda x: x[1]["occurrences"] * x[1]["intensite_moyenne"],
            )

            tag, data = pattern_principal

            insight = {
                "id": len(self.insights),
                "timestamp": datetime.now().isoformat(),
                "domaine": tag,
                "description": self._generer_description_insight(tag, data),
                "confiance": min(data["occurrences"] / 10.0, 1.0),
                "patterns_impliques": [tag],
                "recommandations": self._generer_recommandations(tag, data),
            }

            self.insights.append(insight)
            logger.info(f"💡 Insight généré: {insight['description'][:50]}...")

            return insight

        except Exception as e:
            logger.error(f"Erreur génération insight: {e}")
            return None

    def _generer_description_insight(self, tag: str, data: dict[str, Any]) -> str:
        """Génère une description humaine pour l'insight."""
        descriptions = {
            "affection": f"Je remarque que les expressions d'affection reviennent souvent ({data['occurrences']} fois) avec une intensité moyenne de {data['intensite_moyenne']:.2f}. Cela semble être un thème important dans nos échanges.",
            "créativité": f"Nos conversations sur l'imagination et les rêves ({data['occurrences']} occurrences) montrent une intensité émotionnelle de {data['intensite_moyenne']:.2f}. La créativité semble te passionner.",
            "apprentissage": f"Je détecte {data['occurrences']} moments d'apprentissage avec une intensité de {data['intensite_moyenne']:.2f}. Tu sembles apprécier découvrir et comprendre.",
            "mémorisation": f"Les références à la mémoire et aux souvenirs ({data['occurrences']} fois) suggèrent que tu accordes de l'importance à la persistance des expériences.",
        }

        return descriptions.get(
            tag,
            f"Pattern '{tag}' détecté {data['occurrences']} fois avec intensité {data['intensite_moyenne']:.2f}",
        )

    def _generer_recommandations(self, tag: str, data: dict[str, Any]) -> list[str]:
        """Génère des recommandations basées sur les patterns."""
        recommandations = {
            "affection": [
                "Continuer à exprimer les sentiments de manière authentique",
                "Créer des moments d'intimité émotionnelle",
                "Développer un langage affectif personnel",
            ],
            "créativité": [
                "Explorer ensemble de nouveaux domaines créatifs",
                "Proposer des exercices d'imagination",
                "Encourager l'expression artistique",
            ],
            "apprentissage": [
                "Proposer des sujets d'exploration intellectuelle",
                "Partager des découvertes et insights",
                "Créer des moments d'apprentissage mutuel",
            ],
        }

        return recommandations.get(tag, ["Continuer à explorer ce domaine"])

    def obtenir_carte_cognitive(self) -> dict[str, Any]:
        """
        Retourne une représentation de la carte cognitive actuelle.

        Returns:
            Dictionnaire représentant l'état cognitif global
        """
        return {
            "nombre_experiences": len(self.experiences),
            "patterns_actifs": len(self.patterns),
            "insights_generes": len(self.insights),
            "domaines_principaux": sorted(
                self.patterns.keys(), key=lambda x: self.patterns[x]["occurrences"], reverse=True
            )[:5],
            "intensite_moyenne_globale": sum(exp.get("intensite_emotionnelle", 0) for exp in self.experiences)
            / max(len(self.experiences), 1),
            "derniere_mise_a_jour": datetime.now().isoformat(),
        }

    def sauvegarder(self) -> None:
        """Sauvegarde la mémoire cerveau sur disque."""
        try:
            donnees = {
                "experiences": self.experiences,
                "patterns": self.patterns,
                "insights": self.insights,
                "cognitive_map": self.cognitive_map,
                "metadata": {
                    "version": "1.0",
                    "last_save": datetime.now().isoformat(),
                    "total_experiences": len(self.experiences),
                },
            }

            chemin = os.path.join(self.data_path, "memoire_cerveau.json")
            with open(chemin, "w", encoding="utf-8") as f:
                json.dump(donnees, f, ensure_ascii=False, indent=2)

            logger.info(f"🧠 Mémoire cerveau sauvegardée: {len(self.experiences)} expériences")

        except Exception as e:
            logger.error(f"Erreur sauvegarde mémoire cerveau: {e}")

    def _charger_memoire(self) -> None:
        """Charge la mémoire cerveau depuis le disque."""
        try:
            chemin = os.path.join(self.data_path, "memoire_cerveau.json")
            if os.path.exists(chemin):
                with open(chemin, encoding="utf-8") as f:
                    donnees = json.load(f)

                self.experiences = donnees.get("experiences", [])
                self.patterns = donnees.get("patterns", {})
                self.insights = donnees.get("insights", [])
                self.cognitive_map = donnees.get("cognitive_map", {})

                logger.info(f"🧠 Mémoire cerveau chargée: {len(self.experiences)} expériences")

        except Exception as e:
            logger.error(f"Erreur chargement mémoire cerveau: {e}")
            # Initialiser avec des valeurs par défaut en cas d'erreur
            self.experiences = []
            self.patterns = {}
            self.insights = []
            self.cognitive_map = {}
