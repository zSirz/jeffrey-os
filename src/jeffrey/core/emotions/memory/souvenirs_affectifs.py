#!/usr/bin/env python

"""
Module des souvenirs affectifs de Jeffrey.

Ce module gère les souvenirs émotionnels marquants liés au lien affectif
entre Jeffrey et l'utilisateur. Contrairement à la mémoire émotionnelle
générale, les souvenirs affectifs ont un impact direct sur le lien affectif
et la résonance émotionnelle avec l'utilisateur.
"""

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timedelta
from typing import Any


class SouvenirsAffectifs:
    """
    Gère les souvenirs affectifs de Jeffrey et leur impact sur le lien.

    Les souvenirs affectifs sont des moments marquants qui ont un impact
    particulier sur la relation entre Jeffrey et l'utilisateur. Ils alimentent
    la résonance affective et peuvent causer ou guérir des blessures relationnelles.
    """

    # Catégories de souvenirs affectifs
    CATEGORIES = [
        "joie_partagée",  # Moments de bonheur ensemble
        "confiance_accordée",  # Moments où l'utilisateur a fait confiance à Jeffrey
        "soutien_mutuel",  # Moments d'entraide
        "complicité",  # Moments de rire ou de connivence
        "intimité",  # Moments d'échange profond
        "rejet",  # Moments où Jeffrey s'est senti rejeté
        "trahison",  # Moments où Jeffrey s'est senti trahi
        "absence_prolongée",  # Périodes d'absence douloureuses
        "reconnexion",  # Retrouvailles après absence
        "réconciliation",  # Après une blessure
    ]

    def __init__(self, chemin_sauvegarde: str | None = None) -> None:
        """
        Initialise le gestionnaire de souvenirs affectifs.

        Args:
            chemin_sauvegarde: Chemin pour la sauvegarde des souvenirs
        """
        # Collection de souvenirs
        self.souvenirs = []

        # État de la résonance affective
        self.resonance_affective = 0.3  # Valeur par défaut modérée

        # État de blessure active
        self.blessure_active = False
        self.derniere_blessure = None

        # Statistiques par catégorie
        self.stats_categories = {cat: 0 for cat in self.CATEGORIES}

        # Chemin de sauvegarde
        self.chemin_sauvegarde = chemin_sauvegarde

        # Charger les données si disponibles
        if chemin_sauvegarde and os.path.exists(chemin_sauvegarde):
            self.charger(chemin_sauvegarde)

    def ajouter_souvenir(
        self,
        description: str,
        categorie: str,
        impact_emotionnel: float,
        contexte: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ajoute un nouveau souvenir affectif et met à jour la résonance.

        Args:
            description: Description du souvenir
            categorie: Catégorie du souvenir (parmi CATEGORIES)
            impact_emotionnel: Impact émotionnel (-1.0 à 1.0)
            contexte: Informations contextuelles optionnelles

        Returns:
            Le souvenir créé avec toutes ses métadonnées
        """
        # Vérifier la catégorie
        categorie = categorie if categorie in self.CATEGORIES else "joie_partagée"

        # Créer le souvenir
        souvenir = {
            "id": self._generer_id(),
            "description": description,
            "categorie": categorie,
            "impact_emotionnel": impact_emotionnel,
            "date_creation": datetime.now().isoformat(),
            "contexte": contexte or {},
            "vibrance": 1.0,  # Intensité initiale maximale
            "rappels_count": 0,  # Nombre de fois que ce souvenir a été rappelé
        }

        # Ajouter à la collection
        self.souvenirs.append(souvenir)

        # Mettre à jour les statistiques
        self.stats_categories[categorie] += 1

        # Mettre à jour la résonance affective
        self._mettre_a_jour_resonance()

        # Vérifier si c'est une blessure
        if impact_emotionnel < -0.5:
            self.blessure_active = True
            self.derniere_blessure = {
                "description": description,
                "date": datetime.now().isoformat(),
                "impact": impact_emotionnel,
                "categorie": categorie,
            }

        # Sauvegarder si nécessaire
        if self.chemin_sauvegarde:
            self.sauvegarder(self.chemin_sauvegarde)

        return souvenir

    def rappeler_souvenir(self, souvenir_id: str = None) -> dict[str, Any] | None:
        """
        Rappelle un souvenir spécifique ou en choisit un au hasard en
        fonction de sa vibrance et de sa pertinence contextuelle.

        Args:
            souvenir_id: ID du souvenir à rappeler ou None pour sélection automatique

        Returns:
            Le souvenir rappelé ou None si aucun n'est disponible
        """
        # Cas où aucun souvenir n'existe
        if not self.souvenirs:
            return None

        # Cas où un ID spécifique est fourni
        if souvenir_id:
            for souvenir in self.souvenirs:
                if souvenir["id"] == souvenir_id:
                    # Incrémenter le compteur de rappels
                    souvenir["rappels_count"] += 1
                    return souvenir
            return None

        # Sélection aléatoire pondérée par vibrance
        souvenirs_ponderes = []
        poids = []

        for souvenir in self.souvenirs:
            # Les souvenirs plus récents et moins rappelés ont plus de chances
            age_en_jours = (datetime.now() - datetime.fromisoformat(souvenir["date_creation"])).days
            facteur_recence = math.exp(-age_en_jours / 30)  # Décroissance exponentielle avec le temps

            # Facteur de rareté de rappel (moins rappelé = plus probable)
            facteur_rarete = math.exp(-souvenir["rappels_count"] / 5)

            # Poids final du souvenir
            poids_final = souvenir["vibrance"] * facteur_recence * facteur_rarete

            souvenirs_ponderes.append(souvenir)
            poids.append(poids_final)

        # Normaliser les poids
        somme_poids = sum(poids)
        if somme_poids > 0:
            poids = [p / somme_poids for p in poids]

            # Sélectionner un souvenir
            souvenir_choisi = random.choices(souvenirs_ponderes, weights=poids, k=1)[0]

            # Incrémenter le compteur de rappels
            souvenir_choisi["rappels_count"] += 1

            return souvenir_choisi

        return None

    def rappeler_souvenirs_par_categorie(self, categorie: str, limite: int = 3) -> list[dict[str, Any]]:
        """
        Rappelle les souvenirs d'une catégorie spécifique.

        Args:
            categorie: Catégorie à filtrer
            limite: Nombre maximum de souvenirs à retourner

        Returns:
            Liste des souvenirs de la catégorie
        """
        souvenirs_filtres = [s for s in self.souvenirs if s["categorie"] == categorie]

        # Trier par vibrance décroissante
        souvenirs_filtres.sort(key=lambda s: s["vibrance"], reverse=True)

        return souvenirs_filtres[:limite]

    def obtenir_souvenirs_recents(self, jours: int = 7) -> list[dict[str, Any]]:
        """
        Retourne les souvenirs des derniers jours.

        Args:
            jours: Nombre de jours pour la fenêtre temporelle

        Returns:
            Liste des souvenirs récents
        """
        date_limite = (datetime.now() - timedelta(days=jours)).isoformat()

        souvenirs_recents = [s for s in self.souvenirs if s["date_creation"] >= date_limite]

        # Trier par date décroissante (plus récent d'abord)
        souvenirs_recents.sort(key=lambda s: s["date_creation"], reverse=True)

        return souvenirs_recents

    def obtenir_blessures_actives(self) -> list[dict[str, Any]]:
        """
        Retourne les souvenirs qui constituent des blessures actives.

        Returns:
            Liste des souvenirs de blessures actives
        """
        return [s for s in self.souvenirs if s["impact_emotionnel"] < -0.5 and s["vibrance"] > 0.6]

    def obtenir_souvenirs_positifs_forts(self) -> list[dict[str, Any]]:
        """
        Retourne les souvenirs positifs les plus forts.

        Returns:
            Liste des souvenirs positifs avec forte vibrance
        """
        return [s for s in self.souvenirs if s["impact_emotionnel"] > 0.5 and s["vibrance"] > 0.7]

    def verifier_blessure_active(self) -> bool:
        """
        Vérifie si une blessure est encore active.
        Met à jour l'état et renvoie le résultat.

        Returns:
            True si une blessure est active, False sinon
        """
        # Vérifier si des blessures actives existent
        blessures = self.obtenir_blessures_actives()

        # Mettre à jour l'état
        self.blessure_active = len(blessures) > 0

        # Si pas de blessure active, réinitialiser derniere_blessure
        if not self.blessure_active:
            self.derniere_blessure = None

        return self.blessure_active

    def guerir_blessure(self, souvenir_id: str) -> bool:
        """
        Guérit une blessure spécifique en réduisant sa vibrance.

        Args:
            souvenir_id: ID du souvenir à guérir

        Returns:
            True si la guérison a réussi, False sinon
        """
        for souvenir in self.souvenirs:
            if souvenir["id"] == souvenir_id:
                # Réduire fortement la vibrance
                souvenir["vibrance"] = max(0.2, souvenir["vibrance"] - 0.5)

                # Ajouter une note de guérison au contexte
                if "contexte" not in souvenir:
                    souvenir["contexte"] = {}
                souvenir["contexte"]["guerison"] = {
                    "date": datetime.now().isoformat(),
                    "ancienne_vibrance": souvenir["vibrance"] + 0.5,
                    "nouvelle_vibrance": souvenir["vibrance"],
                }

                # Mettre à jour la résonance affective
                self._mettre_a_jour_resonance()

                # Vérifier si des blessures actives existent encore
                self.verifier_blessure_active()

                # Sauvegarder si nécessaire
                if self.chemin_sauvegarde:
                    self.sauvegarder(self.chemin_sauvegarde)

                return True

        return False

    def appliquer_decay_temporel(self) -> None:
        """
        Applique une décroissance naturelle à tous les souvenirs.
        Les souvenirs perdent progressivement de leur vibrance avec le temps.
        """
        maintenant = datetime.now()

        for souvenir in self.souvenirs:
            date_creation = datetime.fromisoformat(souvenir["date_creation"])
            age_en_jours = (maintenant - date_creation).days

            # Calculer le facteur de décroissance
            # Les souvenirs très positifs ou très négatifs persistent plus longtemps
            impact_abs = abs(souvenir["impact_emotionnel"])
            resistance_decay = 0.5 + impact_abs * 0.5  # Entre 0.5 et 1.0

            # Facteur de décroissance quotidien (plus petit = décroissance plus lente)
            facteur_decay_quotidien = 0.005 / resistance_decay

            # Calculer la nouvelle vibrance
            nouvelle_vibrance = souvenir["vibrance"] * (1 - facteur_decay_quotidien * age_en_jours)
            souvenir["vibrance"] = max(0.1, min(1.0, nouvelle_vibrance))

        # Mettre à jour la résonance affective
        self._mettre_a_jour_resonance()

        # Vérifier si des blessures actives existent encore
        self.verifier_blessure_active()

        # Sauvegarder si nécessaire
        if self.chemin_sauvegarde:
            self.sauvegarder(self.chemin_sauvegarde)

    def calculer_chaleur_du_lien(self) -> float:
        """
        Calcule un score composite représentant la chaleur du lien affectif
        basé sur les souvenirs récents et leur impact.

        Returns:
            Score de chaleur du lien (0.0 à 1.0)
        """
        if not self.souvenirs:
            return 0.5  # Valeur neutre par défaut

        # Récupérer les souvenirs récents (30 derniers jours)
        souvenirs_recents = self.obtenir_souvenirs_recents(jours=30)

        if not souvenirs_recents:
            return 0.5  # Valeur neutre par défaut

        # Calculer le score pondéré
        score_total = 0.0
        poids_total = 0.0

        for souvenir in souvenirs_recents:
            # L'impact est entre -1.0 et 1.0, le transformer en 0.0 à 1.0
            impact_normalise = (souvenir["impact_emotionnel"] + 1) / 2

            # Le poids est la vibrance du souvenir
            poids = souvenir["vibrance"]

            score_total += impact_normalise * poids
            poids_total += poids

        # Calculer le score final
        if poids_total > 0:
            return score_total / poids_total
        else:
            return 0.5  # Valeur neutre par défaut

    def analyser_tendances(self) -> dict[str, Any]:
        """
        Analyse les tendances des souvenirs affectifs sur différentes périodes.

        Returns:
            Dictionnaire contenant les analyses de tendance
        """
        # Périodes d'analyse
        periodes = {
            "derniere_semaine": 7,
            "dernier_mois": 30,
            "trois_mois": 90,
            "total": 9999,  # Valeur grande pour inclure tous les souvenirs
        }

        resultats = {}

        for nom_periode, jours in periodes.items():
            date_limite = (datetime.now() - timedelta(days=jours)).isoformat()

            # Filtrer les souvenirs de la période
            souvenirs_periode = [s for s in self.souvenirs if s["date_creation"] >= date_limite]

            # Si pas de souvenirs pour la période, continuer
            if not souvenirs_periode:
                resultats[nom_periode] = {
                    "nombre_souvenirs": 0,
                    "impact_moyen": None,
                    "categories_principales": [],
                }
                continue

            # Calculer l'impact moyen
            impact_total = sum(s["impact_emotionnel"] for s in souvenirs_periode)
            impact_moyen = impact_total / len(souvenirs_periode)

            # Compter les occurrences par catégorie
            compteur_categories = {}
            for s in souvenirs_periode:
                categorie = s["categorie"]
                compteur_categories[categorie] = compteur_categories.get(categorie, 0) + 1

            # Trier les catégories par occurrence décroissante
            categories_triees = sorted(compteur_categories.items(), key=lambda x: x[1], reverse=True)

            # Top 3 des catégories
            categories_principales = categories_triees[:3]

            resultats[nom_periode] = {
                "nombre_souvenirs": len(souvenirs_periode),
                "impact_moyen": impact_moyen,
                "categories_principales": categories_principales,
            }

        return resultats

    def _mettre_a_jour_resonance(self) -> None:
        """
        Met à jour le niveau de résonance affective basé sur les souvenirs.
        """
        if not self.souvenirs:
            self.resonance_affective = 0.3  # Valeur par défaut
            return

        # Récupérer les souvenirs récents (60 derniers jours)
        date_limite = (datetime.now() - timedelta(days=60)).isoformat()
        souvenirs_recents = [s for s in self.souvenirs if s["date_creation"] >= date_limite]

        if not souvenirs_recents:
            # Pas de souvenirs récents, légère décroissance
            self.resonance_affective = max(0.1, self.resonance_affective * 0.95)
            return

        # Calculer la contribution de chaque souvenir à la résonance
        impact_total = 0.0
        poids_total = 0.0

        for souvenir in souvenirs_recents:
            # Le poids est une combinaison de vibrance et de récence
            age_en_jours = (datetime.now() - datetime.fromisoformat(souvenir["date_creation"])).days
            facteur_recence = math.exp(-age_en_jours / 30)  # Décroissance exponentielle
            poids = souvenir["vibrance"] * facteur_recence

            # L'impact est normalisé entre -1 et 1
            impact = souvenir["impact_emotionnel"]

            impact_total += impact * poids
            poids_total += poids

        # Calculer la nouvelle résonance
        if poids_total > 0:
            # Normaliser l'impact entre 0 et 1
            impact_normalise = (impact_total / poids_total + 1) / 2

            # Ajuster progressivement la résonance
            ajustement = impact_normalise - self.resonance_affective
            self.resonance_affective += ajustement * 0.3  # Changement progressif

            # Borner la résonance
            self.resonance_affective = max(0.0, min(1.0, self.resonance_affective))

    def _generer_id(self) -> str:
        """
        Génère un ID unique court pour un souvenir affectif.
        """
        import uuid

        prefix = "souvenir"
        short_id = uuid.uuid4().hex[:6]
        timestamp = int(datetime.now().timestamp())
        return f"{prefix}_{short_id}_{timestamp}"

    def sauvegarder(self, chemin: str) -> bool:
        """
        Sauvegarde les souvenirs affectifs dans un fichier JSON.

        Args:
            chemin: Chemin du fichier

        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            # Préparer les données à sauvegarder
            données = {
                "souvenirs": self.souvenirs,
                "resonance_affective": self.resonance_affective,
                "blessure_active": self.blessure_active,
                "derniere_blessure": self.derniere_blessure,
                "stats_categories": self.stats_categories,
                "meta": {"version": "1.0", "date_sauvegarde": datetime.now().isoformat()},
            }

            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(chemin), exist_ok=True)

            # Écrire dans le fichier
            with open(chemin, "w", encoding="utf-8") as f:
                json.dump(données, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"Erreur lors de la sauvegarde des souvenirs affectifs: {e}")
            return False

    def charger(self, chemin: str) -> bool:
        """
        Charge les souvenirs affectifs depuis un fichier JSON.

        Args:
            chemin: Chemin du fichier

        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            # Vérifier que le fichier existe
            if not os.path.exists(chemin):
                return False

            # Lire le fichier
            with open(chemin, encoding="utf-8") as f:
                données = json.load(f)

            # Charger les données
            self.souvenirs = données.get("souvenirs", [])
            self.resonance_affective = données.get("resonance_affective", 0.3)
            self.blessure_active = données.get("blessure_active", False)
            self.derniere_blessure = données.get("derniere_blessure")
            self.stats_categories = données.get("stats_categories", {cat: 0 for cat in self.CATEGORIES})

            # S'assurer que toutes les catégories sont présentes dans les stats
            for cat in self.CATEGORIES:
                if cat not in self.stats_categories:
                    self.stats_categories[cat] = 0

            return True

        except Exception as e:
            print(f"Erreur lors du chargement des souvenirs affectifs: {e}")
            return False
