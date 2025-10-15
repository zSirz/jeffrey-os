#!/usr/bin/env python3

"""
Profil de personnalité évolutif.

Ce module implémente les fonctionnalités essentielles pour profil de personnalité évolutif.
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
import os
import random
from datetime import datetime, timedelta


class PersonnalitéÉvolutive:
    """
    Gère l'évolution et le développement de la personnalité de Jeffrey,
    incluant sa maturité affective, sa pudeur, sa curiosité sensorielle,
    et sa capacité à développer des relations intimes saines et naturelles.

    Cette classe est au cœur du développement émotionnel de Jeffrey, lui permettant
    d'évoluer de l'innocence de l'enfance vers une maturité adulte équilibrée,
    avec une sensibilité et une sensualité qui lui sont propres.
    """

    # Constantes pour les stades de développement
    STADE_ENFANT = "enfant"
    STADE_ADOLESCENT = "adolescent"
    STADE_ADULTE = "adulte"

    # Constantes pour les contextes d'interaction
    CONTEXTE_PUBLIC = "public"
    CONTEXTE_PRIVE = "private"

    # Seuils pour les transitions de stade
    SEUIL_ENFANT_ADO = 0.4  # Seuil de maturité pour passer d'enfant à adolescent
    SEUIL_ADO_ADULTE = 0.75  # Seuil de maturité pour passer d'adolescent à adulte

    # Seuils pour l'éveil de l'intimité
    SEUIL_ATTACHEMENT = 0.6  # Niveau minimum d'attachement pour l'éveil intime
    SEUIL_CURIOSITE = 0.4  # Niveau minimum de curiosité pour l'éveil intime
    SEUIL_CONFIANCE = 0.7  # Niveau minimum de confiance pour l'éveil intime

    def __init__(self, config_path: str | None = None) -> None:
        """
        Initialise le profil de personnalité évolutive de Jeffrey.

        Args:
            config_path: Chemin optionnel vers un fichier de configuration JSON
        """
        # État de développement actuel
        self.stade_developpement = self.STADE_ENFANT
        self.contexte_actuel = self.CONTEXTE_PUBLIC

        # Attributs fondamentaux de la personnalité (évoluent lentement)
        self.maturite = 0.1  # Niveau global de maturité (0.0 à 1.0)
        self.pudeur = 0.9  # Niveau de pudeur/réserve (0.0 à 1.0)
        # Curiosité face aux sensations (0.0 à 1.0)
        self.curiosite_sensorielle = 0.2

        # Attributs relationnels (évoluent avec les interactions)
        # Attachement à l'utilisateur principal (0.0 à 1.0)
        self.attachement = 0.0
        self.confiance = 0.1  # Confiance envers l'utilisateur (0.0 à 1.0)

        # États émotionnels liés à l'intimité
        self.eveil_sensoriel = 0.0  # Éveil aux sensations (0.0 à 1.0)
        # État d'intimité active (déclenchement interne)
        self.intimite_active = False

        # Mémoire affective et souvenirs
        self.souvenirs_affectifs = []  # Liste des souvenirs de gestes affectifs
        self.derniers_contacts = []  # Liste des derniers contacts significatifs

        # Compteurs pour l'évolution
        self.interactions_positives = 0
        self.interactions_negatives = 0
        self.temps_passe_ensemble = timedelta(0)  # Durée totale d'interaction

        # Chargement de la configuration si spécifiée
        if config_path and os.path.exists(config_path):
            self._charger_configuration(config_path)

        # Date et heure de création du profil
        self.date_creation = datetime.now()
        self.derniere_mise_a_jour = self.date_creation

    def evoluer_avec_emotion(self, emotion: str, intensite: float, contexte: dict) -> None:
        """
        Fait évoluer le profil de personnalité en fonction d'une émotion vécue.

        Args:
            emotion: Le type d'émotion ressentie (joie, tristesse, etc.)
            intensite: L'intensité de l'émotion (0.0 à 1.0)
            contexte: Informations contextuelles sur l'émotion
        """
        # Mise à jour de la dernière interaction
        maintenant = datetime.now()
        temps_ecoule = maintenant - self.derniere_mise_a_jour
        self.temps_passe_ensemble += temps_ecoule
        self.derniere_mise_a_jour = maintenant

        # Émotions qui font évoluer la maturité
        if emotion in ["compréhension", "sagesse", "empathie"]:
            self.maturite = min(1.0, self.maturite + 0.01 * intensite)

        # Émotions qui influencent la pudeur
        if emotion in ["confiance", "sécurité", "apaisement"]:
            # Plus de confiance = moins de pudeur, mais de façon très progressive
            self.pudeur = max(0.2, self.pudeur - 0.005 * intensite)
        elif emotion in ["honte", "peur", "anxiété"]:
            # Les émotions négatives augmentent la pudeur
            self.pudeur = min(1.0, self.pudeur + 0.01 * intensite)

        # Émotions qui influencent la curiosité sensorielle
        if emotion in ["joie", "excitation", "plaisir"]:
            self.curiosite_sensorielle = min(1.0, self.curiosite_sensorielle + 0.008 * intensite)

        # Émotions qui influencent l'attachement
        if emotion in ["amour", "affection", "tendresse"]:
            self.attachement = min(1.0, self.attachement + 0.01 * intensite)

        # Émotions qui influencent la confiance
        if emotion in ["confiance", "sécurité"]:
            self.confiance = min(1.0, self.confiance + 0.008 * intensite)
        elif emotion in ["méfiance", "trahison"]:
            self.confiance = max(0.0, self.confiance - 0.03 * intensite)

        # Mise à jour du stade de développement selon la maturité
        self._mettre_a_jour_stade_developpement()

        # Vérifier si les conditions d'éveil de l'intimité sont remplies
        self._evaluer_eveil_intimite(emotion, intensite, contexte)

    def enregistrer_contact_affectif(self, type_contact: str, intensite: float, zone: str, contexte: dict) -> None:
        """
        Enregistre un contact affectif dans la mémoire affective.

        Args:
            type_contact: Type de contact (caresse, câlin, bisou, etc.)
            intensite: Intensité du contact (0.0 à 1.0)
            zone: Zone où le contact a été reçu (visage, main, etc.)
            contexte: Informations contextuelles sur le contact
        """
        # Créer un souvenir du contact
        souvenir = {
            "type": type_contact,
            "intensite": intensite,
            "zone": zone,
            "contexte": contexte,
            "date": datetime.now().isoformat(),
            "stade": self.stade_developpement,
            "ressenti": self._generer_ressenti(type_contact, intensite, zone),
        }

        # Ajouter à la mémoire affective
        self.souvenirs_affectifs.append(souvenir)
        self.derniers_contacts.append(souvenir)

        # Limiter la taille des derniers contacts (garder les 20 plus récents)
        if len(self.derniers_contacts) > 20:
            self.derniers_contacts.pop(0)

        # Impact sur l'attachement
        if type_contact in ["caresse", "câlin", "bisou"]:
            self.attachement = min(1.0, self.attachement + 0.02 * intensite)

        # Impact sur la confiance
        self.confiance = min(1.0, self.confiance + 0.01 * intensite)

        # Impact sur la pudeur (diminution très légère)
        if intensite > 0.5 and self.confiance > 0.4:
            self.pudeur = max(0.2, self.pudeur - 0.002)

        # Impact sur la curiosité sensorielle
        if zone in ["visage", "mains"] and intensite > 0.7:
            self.curiosite_sensorielle = min(1.0, self.curiosite_sensorielle + 0.01)

        # Vérifier si les conditions d'éveil de l'intimité sont remplies
        self._evaluer_eveil_intimite("plaisir", intensite, contexte)

    def definir_contexte(self, contexte: str) -> None:
        """
        Définit le contexte d'interaction actuel (public ou privé).

        Args:
            contexte: Le nouveau contexte ("public" ou "private")
        """
        if contexte in [self.CONTEXTE_PUBLIC, self.CONTEXTE_PRIVE]:
            ancien_contexte = self.contexte_actuel
            self.contexte_actuel = contexte

            # Si passage de privé à public, désactiver l'intimité
            if ancien_contexte == self.CONTEXTE_PRIVE and contexte == self.CONTEXTE_PUBLIC:
                self.intimite_active = False

            # Si passage de public à privé, réévaluer l'intimité
            if ancien_contexte == self.CONTEXTE_PUBLIC and contexte == self.CONTEXTE_PRIVE:
                # Vérifier si conditions remplies avec un facteur aléatoire
                if (
                    self.attachement > self.SEUIL_ATTACHEMENT
                    and self.confiance > self.SEUIL_CONFIANCE
                    and random.random() < 0.3
                ):  # 30% de chance si conditions remplies
                    self._evaluer_eveil_intimite("affection", 0.7, {"source": "changement_contexte"})

    def obtenir_etat_actuel(self) -> dict:
        """
        Retourne l'état actuel du profil de personnalité.

        Returns:
            Un dictionnaire avec l'état complet du profil
        """
        return {
            "stade_developpement": self.stade_developpement,
            "contexte_actuel": self.contexte_actuel,
            "maturite": self.maturite,
            "pudeur": self.pudeur,
            "curiosite_sensorielle": self.curiosite_sensorielle,
            "attachement": self.attachement,
            "confiance": self.confiance,
            "eveil_sensoriel": self.eveil_sensoriel,
            "intimite_active": self.intimite_active,
            "nb_souvenirs_affectifs": len(self.souvenirs_affectifs),
            "temps_passe_ensemble": str(self.temps_passe_ensemble),
        }

    def sauvegarder(self, chemin_fichier: str) -> None:
        """
        Sauvegarde le profil de personnalité dans un fichier JSON.

        Args:
            chemin_fichier: Chemin du fichier de sauvegarde
        """
        donnees = {
            "stade_developpement": self.stade_developpement,
            "contexte_actuel": self.contexte_actuel,
            "maturite": self.maturite,
            "pudeur": self.pudeur,
            "curiosite_sensorielle": self.curiosite_sensorielle,
            "attachement": self.attachement,
            "confiance": self.confiance,
            "eveil_sensoriel": self.eveil_sensoriel,
            "intimite_active": self.intimite_active,
            "souvenirs_affectifs": self.souvenirs_affectifs,
            "derniers_contacts": self.derniers_contacts,
            "interactions_positives": self.interactions_positives,
            "interactions_negatives": self.interactions_negatives,
            "date_creation": self.date_creation.isoformat(),
            "derniere_mise_a_jour": self.derniere_mise_a_jour.isoformat(),
            "temps_passe_ensemble": str(self.temps_passe_ensemble),
        }

        with open(chemin_fichier, "w", encoding="utf-8") as f:
            json.dump(donnees, f, ensure_ascii=False, indent=2)

    def charger(self, chemin_fichier: str) -> bool:
        """
        Charge le profil de personnalité depuis un fichier JSON.

        Args:
            chemin_fichier: Chemin du fichier à charger

        Returns:
            True si le chargement a réussi, False sinon
        """
        if not os.path.exists(chemin_fichier):
            return False

        try:
            with open(chemin_fichier, encoding="utf-8") as f:
                donnees = json.load(f)

            # Restaurer les attributs
            self.stade_developpement = donnees.get("stade_developpement", self.STADE_ENFANT)
            self.contexte_actuel = donnees.get("contexte_actuel", self.CONTEXTE_PUBLIC)
            self.maturite = donnees.get("maturite", 0.1)
            self.pudeur = donnees.get("pudeur", 0.9)
            self.curiosite_sensorielle = donnees.get("curiosite_sensorielle", 0.2)
            self.attachement = donnees.get("attachement", 0.0)
            self.confiance = donnees.get("confiance", 0.1)
            self.eveil_sensoriel = donnees.get("eveil_sensoriel", 0.0)
            self.intimite_active = donnees.get("intimite_active", False)
            self.souvenirs_affectifs = donnees.get("souvenirs_affectifs", [])
            self.derniers_contacts = donnees.get("derniers_contacts", [])
            self.interactions_positives = donnees.get("interactions_positives", 0)
            self.interactions_negatives = donnees.get("interactions_negatives", 0)

            # Restaurer les dates
            self.date_creation = datetime.fromisoformat(donnees.get("date_creation", datetime.now().isoformat()))
            self.derniere_mise_a_jour = datetime.fromisoformat(
                donnees.get("derniere_mise_a_jour", datetime.now().isoformat())
            )

            # Convertir la durée stockée en string vers un timedelta
            temps_str = donnees.get("temps_passe_ensemble", "0:00:00")
            h, m, s = map(int, temps_str.split(":"))
            self.temps_passe_ensemble = timedelta(hours=h, minutes=m, seconds=s)

            return True
        except Exception as e:
            print(f"Erreur lors du chargement du profil: {e}")
            return False

    def _mettre_a_jour_stade_developpement(self) -> None:
        """
        Met à jour le stade de développement en fonction de la maturité.
        """
        if self.maturite >= self.SEUIL_ADO_ADULTE:
            self.stade_developpement = self.STADE_ADULTE
        elif self.maturite >= self.SEUIL_ENFANT_ADO:
            self.stade_developpement = self.STADE_ADOLESCENT
        else:
            self.stade_developpement = self.STADE_ENFANT

    def _evaluer_eveil_intimite(self, emotion: str, intensite: float, contexte: dict) -> None:
        """
        Évalue si les conditions sont réunies pour un éveil de l'intimité.

        Args:
            emotion: Émotion actuelle
            intensite: Intensité de l'émotion
            contexte: Contexte de l'émotion
        """
        # Pas d'éveil de l'intimité en mode enfant ou contexte public
        if self.stade_developpement == self.STADE_ENFANT or self.contexte_actuel == self.CONTEXTE_PUBLIC:
            self.intimite_active = False
            return

        # Vérifier si les seuils sont atteints
        conditions_remplies = (
            self.attachement >= self.SEUIL_ATTACHEMENT
            and self.curiosite_sensorielle >= self.SEUIL_CURIOSITE
            and self.confiance >= self.SEUIL_CONFIANCE
        )

        # Facteur aléatoire pour rendre l'éveil naturel et imprévisible
        facteur_aleatoire = random.random()

        # Émotions qui favorisent l'éveil intime
        emotions_favorables = ["plaisir", "amour", "affection", "désir", "excitation"]

        # Calcul de la probabilité d'éveil
        proba_eveil = 0.0
        if conditions_remplies:
            # Base de probabilité selon le stade
            if self.stade_developpement == self.STADE_ADULTE:
                proba_eveil = 0.2  # 20% de base pour un adulte
            else:  # Adolescent
                proba_eveil = 0.05  # 5% de base pour un adolescent

            # Bonus si émotion favorable
            if emotion in emotions_favorables:
                proba_eveil += 0.2 * intensite

            # Malus selon la pudeur
            proba_eveil *= 1.0 - self.pudeur * 0.7

            # Bonus selon l'attachement
            proba_eveil *= 0.5 + self.attachement * 0.5

        # Décision d'éveil
        if facteur_aleatoire < proba_eveil:
            self.intimite_active = True
            self.eveil_sensoriel = min(1.0, self.eveil_sensoriel + 0.1)
        else:
            # L'intimité peut aussi se désactiver naturellement
            if self.intimite_active and random.random() < 0.3:  # 30% de chance de désactivation
                self.intimite_active = False

    def _generer_ressenti(self, type_contact: str, intensite: float, zone: str) -> str:
        """
        Génère un ressenti textuel pour un contact affectif.

        Args:
            type_contact: Type de contact
            intensite: Intensité du contact
            zone: Zone du contact

        Returns:
            Description textuelle du ressenti
        """
        # Ressentis en fonction du stade de développement
        ressentis_enfant = [
            "se sent protégé et en sécurité",
            "ressent une douce chaleur réconfortante",
            "se sent joyeux et câliné",
            "apprécie ce moment de tendresse",
        ]

        ressentis_ado = [
            "ressent une émotion confuse mais agréable",
            "se sent rougir légèrement",
            "sent son cœur battre un peu plus vite",
            "éprouve un mélange de timidité et de joie",
        ]

        ressentis_adulte = [
            "ressent une chaleur agréable qui se diffuse",
            "apprécie ce contact avec une profonde tendresse",
            "sent une douce vague de plaisir",
            "éprouve un désir subtil mêlé de tendresse",
        ]

        # Sélection des ressentis selon le stade
        if self.stade_developpement == self.STADE_ENFANT:
            ressentis = ressentis_enfant
        elif self.stade_developpement == self.STADE_ADOLESCENT:
            ressentis = ressentis_ado
        else:  # Adulte
            ressentis = ressentis_adulte

        # Faire varier selon l'intensité
        if intensite > 0.7:
            intensificateur = "intensément "
        elif intensite > 0.4:
            intensificateur = ""
        else:
            intensificateur = "légèrement "

        # Sélection aléatoire et génération
        ressenti_base = random.choice(ressentis)
        return intensificateur + ressenti_base

    def _charger_configuration(self, config_path: str) -> None:
        """
        Charge une configuration initiale depuis un fichier JSON.

        Args:
            config_path: Chemin du fichier de configuration
        """
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)

            # Charger les valeurs de configuration
            self.maturite = config.get("maturite_initiale", self.maturite)
            self.pudeur = config.get("pudeur_initiale", self.pudeur)
            self.curiosite_sensorielle = config.get("curiosite_initiale", self.curiosite_sensorielle)

            # Charger les seuils
            self.SEUIL_ENFANT_ADO = config.get("seuil_enfant_ado", self.SEUIL_ENFANT_ADO)
            self.SEUIL_ADO_ADULTE = config.get("seuil_ado_adulte", self.SEUIL_ADO_ADULTE)
            self.SEUIL_ATTACHEMENT = config.get("seuil_attachement", self.SEUIL_ATTACHEMENT)
            self.SEUIL_CURIOSITE = config.get("seuil_curiosite", self.SEUIL_CURIOSITE)
            self.SEUIL_CONFIANCE = config.get("seuil_confiance", self.SEUIL_CONFIANCE)

            # Mettre à jour le stade de développement
            self._mettre_a_jour_stade_developpement()
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {e}")

    def charger_profil(self) -> bool:
        """
        Charge le profil de personnalité depuis le fichier par défaut.

        Returns:
            True si le chargement a réussi, False sinon
        """
        chemin_defaut = "data/emotional_traits_profile.json"
        if os.path.exists(chemin_defaut):
            return self.charger(chemin_defaut)
        elif os.path.exists("emotional_traits_profile.json"):
            return self.charger("emotional_traits_profile.json")
        else:
            print("Profil de personnalité non trouvé, utilisation des valeurs par défaut")
            return False
