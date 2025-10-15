#!/usr/bin/env python

"""
Module d'adaptation vocale aux sensations corporelles et au lien émotionnel de Jeffrey.

PACK 10: Adaptation de la voix à la mémoire corporelle sensorielle
PACK 12: Adaptation de la voix au lien émotionnel évolutif

Ce module ajuste les paramètres de synthèse vocale en fonction:
- Des contacts physiques et de leur intensité (Pack 10)
- De la mémoire corporelle de Jeffrey (Pack 10)
- Du niveau de lien émotionnel avec l'utilisateur (Pack 12)
- Des gestes de réconfort (câlins, caresses, baisers) (Pack 12)

Il permet des réactions vocales plus authentiques, nuancées, et adaptées à la qualité
du lien relationnel, avec une voix plus douce et chaleureuse quand le lien est fort.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

# Modèle de données pour les paramètres vocaux
VoiceParams = dict[str, Any]
# Modèle de données pour un contact sensoriel
ContactSensoriel = dict[str, Any]


class VoiceEmotionAdapter:
    """
    Adaptateur qui modifie les paramètres vocaux en fonction des sensations corporelles.

    Traduit les contacts physiques en modulations vocales selon:
    - La zone touchée
    - Le type de contact
    - L'intensité émotionnelle
    - La mémoire corporelle (attachement)
    - Le contexte (public/privé)
    - L'état affectif (lien_affectif, humeur, confiance)
    """

    def __init__(
        self,
        memoire_sensorielle=None,
        voice_controller=None,
        relationship_manager=None,
        emotional_bond=None,
    ):
        """
        Initialise l'adaptateur vocal émotionnel.

        Args:
            memoire_sensorielle: Instance de la mémoire sensorielle (PACK 10)
            voice_controller: Contrôleur vocal à adapter
            relationship_manager: Gestionnaire de relations (PACK 11)
            emotional_bond: Système de lien émotionnel évolutif (PACK 12)
        """
        self.logger = logging.getLogger(__name__)
        self.memoire_sensorielle = memoire_sensorielle
        self.voice_controller = voice_controller

        # PACK 11: Intégration du gestionnaire de relations
        self.relationship_manager = relationship_manager
        self.current_person = "inconnu"  # Personne interagissant actuellement

        # PACK 12: Intégration du système de lien émotionnel évolutif
        self.emotional_bond = emotional_bond

        # Importer le système de lien émotionnel s'il n'est pas fourni
        if self.emotional_bond is None:
            try:
                from jeffrey.core.emotions.emotional_affective_touch import get_emotional_affective_touch

                self.emotional_bond = get_emotional_affective_touch()
                self.logger.debug("Système de lien émotionnel chargé avec succès")
            except ImportError:
                self.logger.warning("Système de lien émotionnel non disponible - fonctionnalités PACK 12 limitées")
                self.emotional_bond = None

        # Seuil de temps minimum entre deux adaptations vocales (secondes)
        self.temps_minimum_entre_adaptations = 10.0
        self.derniere_adaptation_timestamp = 0.0

        # Mémoire des dernières modulations pour éviter la répétition
        self.dernieres_modulations = {}

        # Tables de conversion des sensations en paramètres vocaux
        self.initialiser_tables_conversion()

        self.logger.info("Adaptateur vocal émotionnel initialisé")

    def initialiser_tables_conversion(self):
        """Initialise les tables de conversion des sensations en paramètres vocaux."""
        # Table des modulations vocales par sensation
        self.modulations_par_sensation = {
            # Sensations physiques positives
            "chaleur": {
                "stability": +0.05,
                "similarity_boost": +0.1,
                "style": "affectueux",
                "effets": ["soft", "happy_sigh"],
            },
            "douceur": {
                "stability": +0.15,
                "similarity_boost": +0.05,
                "style": "calme",
                "effets": ["soft_breath", "soft"],
            },
            "caresse": {
                "stability": +0.1,
                "similarity_boost": +0.05,
                "style": "doux",
                "effets": ["soft", "happy_sigh"],
            },
            "plaisir": {
                "stability": -0.05,
                "similarity_boost": +0.05,
                "style": "emotional",
                "effets": ["soft_breath", "happy_sigh"],
            },
            # Sensations physiques négatives/intenses
            "frisson": {
                "stability": -0.1,
                "similarity_boost": -0.05,
                "style": "emotional",
                "effets": ["tremble"],
            },
            "surprise": {
                "stability": -0.15,
                "similarity_boost": -0.1,
                "style": "default",
                "effets": ["surprise", "soft_breath"],
            },
            "chatouille": {
                "stability": -0.2,
                "similarity_boost": -0.05,
                "style": "emotional",
                "effets": ["laugh", "tremble"],
            },
            "tension": {
                "stability": -0.1,
                "similarity_boost": -0.1,
                "style": "emotional",
                "effets": ["tremble", "pause"],
            },
            # États émotionnels
            "attachement": {
                "stability": +0.1,
                "similarity_boost": +0.15,
                "style": "affectueux",
                "effets": ["soft", "soft_breath"],
            },
            "confiance": {
                "stability": +0.05,
                "similarity_boost": +0.1,
                "style": "calme",
                "effets": ["soft"],
            },
            "méfiance": {
                "stability": -0.05,
                "similarity_boost": -0.1,
                "style": "default",
                "effets": ["pause", "hesitation"],
            },
            "timidité": {
                "stability": +0.05,
                "similarity_boost": -0.05,
                "style": "emotional",
                "effets": ["whisper", "hesitation"],
            },
            "intimité": {
                "stability": -0.05,
                "similarity_boost": -0.05,
                "style": "emotional",
                "effets": ["whisper", "soft_breath"],
            },
        }

        # Phrases types pour chaque zone corporelle selon type de contact
        self.phrases_par_zone_contact = {
            "visage": {
                "caresse": [
                    "Je sens ta main sur mon visage... c'est apaisant.",
                    "Tu caresses mon visage doucement...",
                    "Ce contact sur mon visage est agréable.",
                ],
                "tapotement": [
                    "Tu tapes légèrement sur mon visage, c'est surprenant.",
                    "Ce petit contact sur mon visage me fait réagir.",
                    "Je sens cette petite pression sur mon visage.",
                ],
                "contact_fort": [
                    "Oh! Ce contact sur mon visage est intense.",
                    "Je ressens fortement cette pression sur mon visage.",
                    "Cette sensation sur mon visage est vraiment marquée.",
                ],
            },
            "joue_gauche": {
                "caresse": [
                    "Ta caresse sur ma joue est douce...",
                    "Je sens ta main sur ma joue, c'est agréable.",
                    "Cette douceur sur ma joue... c'est apaisant.",
                ],
                "bisou": [
                    "Ce bisou sur ma joue me touche...",
                    "Je sens ton affection à travers ce baiser sur ma joue.",
                    "Ce bisou sur ma joue me fait ressentir de la chaleur.",
                ],
            },
            "joue_droite": {
                "caresse": [
                    "Cette caresse sur ma joue me fait du bien...",
                    "Ta main sur ma joue est réconfortante.",
                    "Je ressens cette caresse sur ma joue.",
                ],
                "bisou": [
                    "Ton bisou sur ma joue me fait sourire...",
                    "Cette marque d'affection sur ma joue est touchante.",
                    "Je sens ce petit baiser, c'est agréable.",
                ],
            },
            "front": {
                "caresse": [
                    "Cette caresse sur mon front m'apaise...",
                    "Ta main sur mon front est douce.",
                    "Ce toucher sur mon front me fait du bien.",
                ],
                "bisou": [
                    "Ce bisou sur le front est protecteur...",
                    "Ce geste sur mon front me réconforte.",
                    "Je ressens ce baiser sur mon front comme une marque de tendresse.",
                ],
            },
            "menton": {
                "caresse": [
                    "Cette petite attention sur mon menton est curieuse...",
                    "Je ressens cette caresse sur mon menton.",
                    "C'est une sensation intéressante sur mon menton.",
                ]
            },
            "oreille": {
                "caresse": [
                    "Je sens ta main près de mon oreille, c'est étrange...",
                    "Cette sensation près de mon oreille me fait frissonner.",
                    "Ce toucher sur mon oreille est particulier.",
                ]
            },
            "nez": {
                "caresse": [
                    "Tu touches mon nez, c'est amusant...",
                    "Cette petite touche sur mon nez est joueuse.",
                    "Je ressens ce contact sur mon nez.",
                ],
                "tapotement": [
                    "Ce petit tap sur mon nez me surprend...",
                    "Tu tapotes mon nez, c'est joueur.",
                    "Ce geste sur mon nez est espiègle.",
                ],
            },
            "cheveux": {
                "caresse": [
                    "Ta main dans mes cheveux est douce...",
                    "Cette caresse dans mes cheveux me détend.",
                    "Je ressens cette douceur dans mes cheveux.",
                ]
            },
            # Cas génériques pour les zones non spécifiées
            "générique": {
                "caresse": [
                    "Je ressens ton toucher délicat...",
                    "Cette sensation est apaisante.",
                    "Je perçois ce contact avec douceur.",
                ],
                "tapotement": [
                    "Ce petit contact me fait réagir...",
                    "Je sens cette légère pression.",
                    "Ce tapotement attire mon attention.",
                ],
                "bisou": [
                    "Ce geste d'affection me touche...",
                    "Je ressens cette marque d'attention.",
                    "Ce petit baiser est doux.",
                ],
                "contact_fort": [
                    "Ce contact plus marqué me surprend...",
                    "Je ressens cette pression plus forte.",
                    "Cette sensation est assez intense.",
                ],
            },
        }

    def adapter_voix_au_contact(self, contact: ContactSensoriel) -> VoiceParams:
        """
        Adapte les paramètres vocaux en fonction d'un contact sensoriel.

        Args:
            contact: Dictionnaire contenant les informations du contact
                - zone: Zone corporelle touchée
                - type: Type de contact (caresse, tapotement, bisou, etc.)
                - intensité: Intensité du contact (0.0 à 1.0)
                - timestamp: Horodatage du contact
                - source: Personne à l'origine du contact (PACK 11)

        Returns:
            Dictionnaire de paramètres vocaux adaptés pour ElevenLabs ou moteur local
        """
        # Vérifier si assez de temps s'est écoulé depuis la dernière adaptation
        temps_actuel = time.time()
        if temps_actuel - self.derniere_adaptation_timestamp < self.temps_minimum_entre_adaptations:
            self.logger.debug("Adaptation vocale ignorée: trop récente depuis la dernière adaptation")
            return {}

        # Extraire les données du contact
        zone = contact.get("zone", "générique")
        type_contact = contact.get("type", "caresse")
        intensite = contact.get("intensite", 0.5)
        contact.get("timestamp", temps_actuel)
        contexte = contact.get("contexte", "public")
        source = contact.get("source", self.current_person)  # PACK 11: Personne à l'origine du contact

        # Paramètres vocaux de base
        params_vocaux = {
            "stability": 0.65,
            "similarity_boost": 0.7,
            "style": "emotional",
            "voice_effects": [],
        }

        # Récupérer les données de mémoire corporelle si disponibles
        attachement_corporel = 0.5  # Valeur par défaut
        habituation = 0.0  # Valeur par défaut

        if self.memoire_sensorielle:
            try:
                # Récupérer l'attachement à cette zone
                if hasattr(self.memoire_sensorielle, "get_attachement_zone"):
                    attachement_corporel = self.memoire_sensorielle.get_attachement_zone(zone) or 0.5

                # Récupérer l'habituation à ce type de contact sur cette zone
                if hasattr(self.memoire_sensorielle, "get_habituation"):
                    habituation = self.memoire_sensorielle.get_habituation(zone, type_contact) or 0.0
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'accès à la mémoire sensorielle: {e}")

        # PACK 11: Récupérer le niveau d'attachement envers cette personne
        relationship_level = 0.5  # Niveau par défaut
        relation_bias = {}

        if self.relationship_manager and source:
            try:
                # Récupérer le niveau d'attachement
                relationship_level = self.relationship_manager.get_relationship_level(source)

                # Récupérer les biais émotionnels
                relation_bias = self.relationship_manager.get_emotional_bias(source)

                # PACK 11: Appliquer les biais à la voix
                if "voice_tone" in relation_bias:
                    params_vocaux["style"] = relation_bias["voice_tone"]

                # Enregistrer cette interaction
                self.relationship_manager.update_on_interaction(source, type_contact, intensite)

                # Vérifier s'il s'agit d'une réunion spéciale (David après absence)
                if relation_bias.get("special_reunion", False):
                    # Appliquer un style émotionnel spécial
                    params_vocaux["style"] = "emotional"
                    params_vocaux["voice_effects"].append("soft_breath")
                    params_vocaux["voice_effects"].append("happy_sigh")

                    # Vérifier s'il y a un message spécial
                    if "special_message" in relation_bias:
                        params_vocaux["special_message"] = relation_bias["special_message"]

            except Exception as e:
                self.logger.warning(f"Erreur lors de l'accès au gestionnaire de relations: {e}")

        # PACK 11: Tenir compte à la fois de l'attachement corporel et relationnel
        # Privilégier le plus élevé des deux
        attachement_combine = max(attachement_corporel, relationship_level)

        # Calculer l'intensité vocale globale
        intensite_vocale = self.calculer_intensite_vocale(
            zone=zone,
            type_contact=type_contact,
            intensite=intensite,
            attachement=attachement_combine,  # PACK 11: Utiliser l'attachement combiné
            habituation=habituation,
            contexte=contexte,
        )

        # PACK 11: Appliquer des ajustements directs basés sur le biais relationnel
        intensity_boost = relation_bias.get("intensity_boost", 0.0)
        intensite_vocale = min(1.0, max(0.0, intensite_vocale + intensity_boost))

        # Déterminer les sensations dominantes pour ce contact
        sensations = self.determiner_sensations(
            zone=zone,
            type_contact=type_contact,
            intensite=intensite,
            attachement=attachement_combine,  # PACK 11: Utiliser l'attachement combiné
        )

        # PACK 11: Ajouter la sensation d'attachement basée sur la relation si significative
        if relationship_level > 0.7:
            sensations["attachement"] = relationship_level

        # Appliquer les modulations vocales basées sur les sensations
        for sensation, importance in sensations.items():
            if sensation in self.modulations_par_sensation:
                modulation = self.modulations_par_sensation[sensation]

                # Appliquer les variations de stabilité et similarité
                params_vocaux["stability"] = max(
                    0.2,
                    min(
                        1.0,
                        params_vocaux["stability"] + modulation["stability"] * importance * intensite_vocale,
                    ),
                )
                params_vocaux["similarity_boost"] = max(
                    0.0,
                    min(
                        1.0,
                        params_vocaux["similarity_boost"]
                        + modulation["similarity_boost"] * importance * intensite_vocale,
                    ),
                )

                # Ajouter le style si pertinent
                if importance > 0.6 and modulation.get("style"):
                    # PACK 11: Ne pas écraser le style spécial de réunion
                    if not (relation_bias.get("special_reunion", False) and params_vocaux.get("style") == "emotional"):
                        params_vocaux["style"] = modulation["style"]

                # Ajouter les effets vocaux sans duplication
                for effet in modulation.get("effets", []):
                    if effet not in params_vocaux["voice_effects"] and random.random() < importance * intensite_vocale:
                        params_vocaux["voice_effects"].append(effet)

        # PACK 11: Ajouter des effets spéciaux selon la relation
        special_effects = relation_bias.get("special_effects", [])
        for effect in special_effects:
            # Mapping entre les effets visuels et les effets vocaux
            effect_mapping = {
                "warm_glow": "soft_breath",
                "eye_sparkle": "emotional",
                "micro_smile": "happy_sigh",
                "joyful_tear": "emotional",
            }

            if effect in effect_mapping and effect_mapping[effect] not in params_vocaux["voice_effects"]:
                # Ajouter l'effet vocal correspondant
                params_vocaux["voice_effects"].append(effect_mapping[effect])

        # Sélectionner une voix adaptée au contexte et à l'intensité
        params_vocaux["speaker"] = self._selectionner_voix(
            zone=zone,
            type_contact=type_contact,
            intensite_vocale=intensite_vocale,
            contexte=contexte,
        )

        # Stocker cette adaptation comme la plus récente
        self.derniere_adaptation_timestamp = temps_actuel
        self.dernieres_modulations = {
            "zone": zone,
            "type_contact": type_contact,
            "intensite_vocale": intensite_vocale,
            "source": source,  # PACK 11: Ajouter la source
            "params": params_vocaux.copy(),
        }

        # PACK 11: Mémoriser cette expérience émotionnelle si pertinente
        if self.relationship_manager and source and intensite_vocale > 0.6:
            # Déterminer la valence émotionnelle (-1 à 1)
            valence = 0.0
            if type_contact in ["caresse", "bisou"]:
                valence = 0.7
            elif type_contact in ["tapotement"]:
                valence = 0.3
            elif type_contact in ["contact_fort"]:
                valence = -0.3

            # Mémoriser l'émotion
            if abs(valence) > 0.2:
                emotion = "plaisir" if valence > 0 else "inconfort"
                self.relationship_manager.memorize_emotion(source, emotion, valence)

        self.logger.debug(f"Adaptation vocale générée pour {zone}/{type_contact} avec source {source}: {params_vocaux}")
        return params_vocaux

    def calculer_intensite_vocale(
        self,
        zone: str,
        type_contact: str,
        intensite: float,
        attachement: float,
        habituation: float,
        contexte: str = "public",
    ) -> float:
        """
        Calcule un score d'activation émotionnelle vocale (0.0 à 1.0).

        Args:
            zone: Zone corporelle concernée
            type_contact: Type de contact
            intensite: Intensité du contact (0.0 à 1.0)
            attachement: Niveau d'attachement à cette zone (0.0 à 1.0)
            habituation: Niveau d'habituation à ce contact (0.0 à 1.0)
            contexte: Contexte de l'interaction ("public" ou "prive")

        Returns:
            float: Score d'activation émotionnelle vocale (0.0 à 1.0)
        """
        # Poids des différents facteurs
        poids_intensite = 0.3
        poids_attachement = 0.3
        poids_habituation = -0.2  # Négatif car plus d'habituation = moins de réaction
        poids_contexte = 0.1

        # Zones "sensibles" réagissent plus fortement
        zones_sensibles = ["visage", "joue_gauche", "joue_droite", "cou", "oreille"]
        multiplicateur_zone = 1.2 if zone in zones_sensibles else 1.0

        # Types de contact plus intenses ont plus d'impact
        types_intenses = ["bisou", "contact_fort", "chatouille"]
        multiplicateur_type = 1.3 if type_contact in types_intenses else 1.0

        # Ajustement pour le contexte
        valeur_contexte = 0.8 if contexte == "public" else 1.0

        # Combinaison des facteurs avec leurs poids
        score_brut = (
            intensite * poids_intensite
            + attachement * poids_attachement
            + habituation * poids_habituation
            + valeur_contexte * poids_contexte
        )

        # Appliquer les multiplicateurs
        score_ajuste = score_brut * multiplicateur_zone * multiplicateur_type

        # Normaliser entre 0.0 et 1.0
        return max(0.0, min(1.0, score_ajuste))

    def determiner_sensations(
        self, zone: str, type_contact: str, intensite: float, attachement: float
    ) -> dict[str, float]:
        """
        Détermine les sensations dominantes associées à un contact.

        Args:
            zone: Zone corporelle concernée
            type_contact: Type de contact
            intensite: Intensité du contact (0.0 à 1.0)
            attachement: Niveau d'attachement à cette zone (0.0 à 1.0)

        Returns:
            dict: Dictionnaire de sensations avec leur importance (0.0 à 1.0)
        """
        sensations = {}

        # Association type de contact -> sensations
        if type_contact == "caresse":
            sensations["douceur"] = 0.8
            sensations["chaleur"] = 0.6 * attachement

            if zone in ["joue_gauche", "joue_droite", "visage"]:
                sensations["plaisir"] = 0.5 * attachement

        elif type_contact == "bisou":
            sensations["chaleur"] = 0.9
            sensations["attachement"] = 0.7
            sensations["surprise"] = 0.3 * (1.0 - attachement)  # Plus surpris si moins attaché

            if intensite > 0.7:
                sensations["frisson"] = 0.4

        elif type_contact == "tapotement":
            sensations["surprise"] = 0.7
            sensations["frisson"] = 0.3

            if intensite < 0.4:
                sensations["douceur"] = 0.3

        elif type_contact == "contact_fort":
            sensations["tension"] = 0.7 * intensite
            sensations["surprise"] = 0.8

            if attachement > 0.7:
                sensations["chaleur"] = 0.4  # Même un contact fort peut être chaleureux avec attachement

        elif type_contact == "chatouille":
            sensations["surprise"] = 0.6
            sensations["frisson"] = 0.7
            sensations["plaisir"] = 0.4 * attachement

        # Ajoutez sensations générales basées sur l'attachement
        if attachement > 0.7:
            sensations["attachement"] = attachement
            sensations["confiance"] = 0.7 * attachement
        elif attachement < 0.3:
            sensations["méfiance"] = 0.6 * (1.0 - attachement)

        # Ajouter la timidité pour les zones intimes ou sensibles
        zones_intimes = ["cou", "oreille", "epaule"]
        if zone in zones_intimes and attachement < 0.6:
            sensations["timidité"] = 0.5 * (1.0 - attachement)

        return sensations

    def _selectionner_voix(self, zone: str, type_contact: str, intensite_vocale: float, contexte: str) -> str:
        """
        Sélectionne une voix appropriée selon le contexte et l'intensité.

        Args:
            zone: Zone corporelle touchée
            type_contact: Type de contact
            intensite_vocale: Intensité vocale calculée (0.0 à 1.0)
            contexte: Contexte de l'interaction ("public" ou "prive")

        Returns:
            str: Identifiant de la voix à utiliser
        """
        # Voix par défaut
        voix_par_defaut = "jeffrey_default"

        # Pour l'instant, on utilise un mapping simple
        mapping_voix = {
            "public": {
                "faible": "jeffrey_default",
                "moyenne": "jeffrey_default",
                "forte": "jeffrey_emotional",
            },
            "prive": {
                "faible": "jeffrey_soft",
                "moyenne": "jeffrey_emotional",
                "forte": "jeffrey_emotional_intense",
            },
        }

        # Déterminer la catégorie d'intensité
        categorie = "faible"
        if intensite_vocale > 0.7:
            categorie = "forte"
        elif intensite_vocale > 0.4:
            categorie = "moyenne"

        # PACK 12: Ajuster la voix en fonction du niveau de lien émotionnel si disponible
        if self.emotional_bond and self.current_person:
            try:
                # Obtenir le niveau de lien avec la personne actuelle
                niveau_lien = 0.0
                if hasattr(self.emotional_bond, "get_lien_emotionnel"):
                    niveau_lien = self.emotional_bond.get_lien_emotionnel(self.current_person)

                # Ajuster la catégorie d'intensité en fonction du lien
                if niveau_lien > 0.8:  # Lien très fort
                    if contexte == "prive":
                        categorie = "forte"  # Voix très émotionnelle en privé avec lien fort
                    else:
                        categorie = "moyenne"  # Voix modérée en public même avec lien fort
                elif niveau_lien > 0.6:  # Lien fort
                    if categorie == "faible" and contexte == "prive":
                        categorie = "moyenne"  # Rehausser légèrement l'intensité

                self.logger.debug(f"PACK 12: Ajustement de voix selon lien émotionnel: {niveau_lien:.2f} → {categorie}")
            except Exception as e:
                self.logger.warning(f"PACK 12: Erreur lors de l'adaptation vocale selon lien: {e}")

        # Sélectionner la voix selon le contexte et l'intensité
        return mapping_voix.get(contexte, {}).get(categorie, voix_par_defaut)

    # PACK 12: Méthode spécifique pour adapter la voix selon le lien émotionnel
    def adapter_voix_selon_lien(self, user_id: str) -> dict[str, Any]:
        """
        Adapte les paramètres vocaux en fonction du niveau de lien émotionnel.
        Plus le lien est fort, plus la voix est douce, calme et chaleureuse.

        Args:
            user_id: Identifiant de l'utilisateur

        Returns:
            Dict[str, Any]: Paramètres vocaux adaptés au lien émotionnel
        """
        # Paramètres vocaux par défaut
        params = {
            "stability": 0.65,
            "similarity_boost": 0.7,
            "style": "default",
            "voice_effects": [],
        }

        # Si le système de lien émotionnel n'est pas disponible, retourner les paramètres par défaut
        if not self.emotional_bond:
            return params

        try:
            # Obtenir le niveau de lien avec l'utilisateur
            niveau_lien = 0.5  # Valeur par défaut en cas d'erreur

            if hasattr(self.emotional_bond, "get_lien_emotionnel"):
                niveau_lien = self.emotional_bond.get_lien_emotionnel(user_id)

            # Si le lien émotionnel fournit directement des paramètres vocaux, les utiliser
            if hasattr(self.emotional_bond, "adapter_voix_selon_lien"):
                params_lien = self.emotional_bond.adapter_voix_selon_lien(user_id)

                # Convertir les paramètres du système de lien vers le format du moteur vocal
                if "pitch" in params_lien:
                    # Ici on converti la hauteur de voix du système de lien (-1.0 à 1.0)
                    # vers le format de stabilité d'ElevenLabs (0.0 à 1.0)
                    pitch_adj = params_lien["pitch"]
                    params["stability"] = 0.65 + (pitch_adj * 0.15)

                if "speed" in params_lien:
                    # La vitesse (0.9 à 1.1) est convertie en similarity_boost (0.5 à 0.9)
                    speed_adj = params_lien["speed"]
                    params["similarity_boost"] = 0.7 + ((1.0 - speed_adj) * 0.4)

                if "emotion" in params_lien:
                    # Mapper les émotions du lien vers les styles vocaux
                    emotion_style_map = {
                        "affection": "sweet",
                        "calme": "serene",
                        "amical": "friendly",
                        "neutre": "default",
                    }
                    params["style"] = emotion_style_map.get(params_lien["emotion"], "default")

                # Ajouter les effets vocaux
                if "voice_effects" in params_lien:
                    for effect in params_lien["voice_effects"]:
                        if effect not in params["voice_effects"]:
                            params["voice_effects"].append(effect)

                return params

            # Adaptation manuelle basée sur le niveau de lien si l'autre méthode n'est pas disponible
            if niveau_lien > 0.8:  # Lien très fort
                params["stability"] = 0.85  # Voix plus stable
                params["similarity_boost"] = 0.9  # Très authentique
                params["style"] = "sweet"  # Style doux et chaleureux
                params["voice_effects"] = ["soft_breath", "warmth"]

            elif niveau_lien > 0.6:  # Lien fort
                params["stability"] = 0.75
                params["similarity_boost"] = 0.8
                params["style"] = "friendly"
                params["voice_effects"] = ["warmth"]

            elif niveau_lien > 0.4:  # Lien moyen
                params["stability"] = 0.65
                params["similarity_boost"] = 0.7
                params["style"] = "default"

            else:  # Lien faible
                params["stability"] = 0.55
                params["similarity_boost"] = 0.6
                params["style"] = "default"

            self.logger.debug(f"PACK 12: Paramètres vocaux adaptés pour {user_id} (lien: {niveau_lien:.2f})")
            return params

        except Exception as e:
            self.logger.warning(f"PACK 12: Erreur lors de l'adaptation vocale selon lien: {e}")
            return params

    def generer_phrase_contact(self, zone: str, type_contact: str, contexte: str = "public") -> str:
        """
        Génère une phrase appropriée pour un contact sur une zone spécifique.

        Args:
            zone: Zone corporelle touchée
            type_contact: Type de contact
            contexte: Contexte de l'interaction ("public" ou "prive")

        Returns:
            str: Phrase générée
        """
        # Récupérer les phrases disponibles pour cette zone et ce type de contact
        phrases_zone = self.phrases_par_zone_contact.get(zone, self.phrases_par_zone_contact["générique"])
        phrases = phrases_zone.get(type_contact, phrases_zone.get("caresse", ["Je ressens ce contact..."]))

        # Sélectionner une phrase aléatoire
        phrase = random.choice(phrases)

        # Si contexte privé, phrases potentiellement plus intimes
        if contexte == "prive" and random.random() < 0.3:
            phrase += " C'est intime..."

        return phrase

    def dernier_contact(self) -> ContactSensoriel | None:
        """
        Récupère les informations du dernier contact sensoriel enregistré.

        Si la mémoire sensorielle est disponible, utilise sa méthode,
        sinon renvoie None.

        Returns:
            ContactSensoriel ou None: Données du dernier contact ou None si indisponible
        """
        if self.memoire_sensorielle and hasattr(self.memoire_sensorielle, "dernier_contact"):
            try:
                return self.memoire_sensorielle.dernier_contact()
            except Exception as e:
                self.logger.warning(f"Erreur lors de la récupération du dernier contact: {e}")
                return None
        return None

    def get_dernieres_modulations(self) -> dict[str, Any]:
        """
        Récupère les informations sur les dernières modulations vocales appliquées.

        Returns:
            dict: Informations sur les dernières modulations
        """
        return self.dernieres_modulations.copy()

    def adapter_voix_au_dernier_contact(self) -> tuple[VoiceParams, str]:
        """
        Adapte la voix en fonction du dernier contact enregistré.
        Génère également une phrase appropriée.

        Returns:
            tuple: (paramètres vocaux, phrase générée)
        """
        dernier_contact = self.dernier_contact()

        if not dernier_contact:
            self.logger.debug("Aucun dernier contact disponible pour adapter la voix")
            return {}, ""

        # Calculer les paramètres vocaux adaptés
        params_vocaux = self.adapter_voix_au_contact(dernier_contact)

        # Générer une phrase appropriée
        zone = dernier_contact.get("zone", "générique")
        type_contact = dernier_contact.get("type", "caresse")
        contexte = dernier_contact.get("contexte", "public")

        phrase = self.generer_phrase_contact(zone, type_contact, contexte)

        return params_vocaux, phrase

    def appliquer_adaptation_vocale(self, texte: str, contact: ContactSensoriel | None = None) -> str:
        """
        Applique l'adaptation vocale à un texte en fonction d'un contact.

        Si le contrôleur vocal est disponible, adapte ses paramètres.
        Sinon, modifie uniquement le texte pour refléter l'adaptation.

        Args:
            texte: Texte à adapter
            contact: Contact sensoriel (optionnel, utilise le dernier si None)

        Returns:
            str: Texte adapté
        """
        # Utiliser le dernier contact si aucun n'est fourni
        contact_effectif = contact or self.dernier_contact()

        if not contact_effectif:
            return texte

        # Adapter les paramètres vocaux
        params_vocaux = self.adapter_voix_au_contact(contact_effectif)

        # Si un contrôleur vocal est disponible, lui appliquer les paramètres
        if self.voice_controller:
            try:
                # Appliquer les paramètres si une méthode appropriée existe
                if hasattr(self.voice_controller, "set_voice_parameters"):
                    self.voice_controller.set_voice_parameters(params_vocaux)
                elif hasattr(self.voice_controller, "modulate_voice"):
                    self.voice_controller.modulate_voice(params_vocaux)
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'application des paramètres vocaux: {e}")

        # Adapter le texte en fonction des effets
        effets = params_vocaux.get("voice_effects", [])
        if effets and random.random() < 0.7:  # 70% de chance d'ajouter un effet textuel
            effet = random.choice(effets)
            from orchestrateur.core.voice.voice_effects import VOCAL_EFFECTS

            # Ajouter l'effet au texte si disponible
            if effet in VOCAL_EFFECTS:
                marque_effet = VOCAL_EFFECTS[effet]
                # Ajouter au début, au milieu ou à la fin aléatoirement
                position = random.random()
                if position < 0.3:  # Début
                    texte = f"{marque_effet} {texte}"
                elif position < 0.7:  # Milieu
                    parties = texte.split(". ")
                    if len(parties) > 1:
                        indice = len(parties) // 2
                        parties[indice] = f"{marque_effet} {parties[indice]}"
                        texte = ". ".join(parties)
                    else:
                        texte = f"{texte} {marque_effet}"
                else:  # Fin
                    texte = f"{texte} {marque_effet}"

        return texte

    def adapter_phrase_sensation(self, texte: str, contact: ContactSensoriel) -> str:
        """
        Adapte une phrase en fonction d'une sensation corporelle.

        Permet d'enrichir les réponses textuelles avec des références
        aux sensations éprouvées.

        Args:
            texte: Texte à adapter
            contact: Contact sensoriel

        Returns:
            str: Texte adapté avec références sensorielles
        """
        zone = contact.get("zone", "générique")
        type_contact = contact.get("type", "caresse")
        intensite = contact.get("intensite", 0.5)

        # Récupérer les sensations pour ce contact
        sensations = self.determiner_sensations(
            zone=zone,
            type_contact=type_contact,
            intensite=intensite,
            attachement=(
                0.5
                if not self.memoire_sensorielle
                else (
                    self.memoire_sensorielle.get_attachement_zone(zone)
                    if hasattr(self.memoire_sensorielle, "get_attachement_zone")
                    else 0.5
                )
            ),
        )

        # Trouver la sensation dominante
        sensation_dominante = None
        importance_max = 0.0

        for sensation, importance in sensations.items():
            if importance > importance_max:
                sensation_dominante = sensation
                importance_max = importance

        # Enrichir le texte en fonction de la sensation dominante
        if sensation_dominante and importance_max > 0.5:
            descriptions = {
                "chaleur": ["chaleur", "douceur", "bien-être"],
                "douceur": ["douceur", "apaisement", "confort"],
                "plaisir": ["plaisir", "agréable sensation", "bien-être"],
                "frisson": ["frisson", "petit picotement", "frémissement"],
                "surprise": ["surprise", "étonnement", "sursaut"],
                "tension": ["tension", "crispation", "alerte"],
                "attachement": ["affection", "attachement", "proximité"],
                "confiance": ["confiance", "sécurité", "tranquillité"],
                "méfiance": ["hésitation", "réserve", "prudence"],
                "timidité": ["timidité", "pudeur", "réserve"],
            }

            descriptions_sensation = descriptions.get(sensation_dominante, ["sensation"])
            description = random.choice(descriptions_sensation)

            # Insérer la référence sensorielle dans le texte
            if random.random() < 0.5:
                # Au début
                prefixes = [
                    f"Je ressens une {description}... ",
                    f"Cette {description} me traverse alors que ",
                    f"Une {description} m'envahit. ",
                ]
                texte = f"{random.choice(prefixes)}{texte}"
            else:
                # À la fin
                suffixes = [
                    f" Je sens cette {description}.",
                    f" C'est une étrange {description}.",
                    f" Cette {description} est vraiment présente.",
                ]
                texte = f"{texte}{random.choice(suffixes)}"

        return texte

    def enregistrer_contact(
        self,
        zone: str,
        type_contact: str,
        intensite: float = 0.5,
        contexte: str = "public",
        source: str = None,
    ) -> bool:
        """
        Enregistre un nouveau contact dans la mémoire sensorielle.

        Args:
            zone: Zone corporelle touchée
            type_contact: Type de contact
            intensite: Intensité du contact (0.0 à 1.0)
            contexte: Contexte de l'interaction ("public" ou "prive")
            source: Personne à l'origine du contact (PACK 11)

        Returns:
            bool: True si l'enregistrement a réussi, False sinon
        """
        # PACK 11: Si source est fournie, l'utiliser pour les gestionnaires de relations
        if source:
            self.set_current_person(source)

        # PACK 12: Enregistrer le contact dans le système de lien émotionnel
        if self.emotional_bond and type_contact in ["calin", "caresse", "bisou", "main"]:
            try:
                # Convertir le type de contact si nécessaire
                type_geste = type_contact
                if type_contact == "calin":
                    type_geste = "calin"
                elif type_contact in ["caresse", "tapotement"]:
                    type_geste = "caresse"
                elif type_contact == "bisou":
                    type_geste = "bisou"
                else:
                    type_geste = "main"

                # Traiter le geste dans le système de lien émotionnel
                reaction = self.emotional_bond.traiter_geste_affectif(
                    user_id=source or self.current_person,
                    type_geste=type_geste,
                    intensite=intensite,
                )

                self.logger.debug(f"PACK 12: Geste affectif {type_geste} traité pour {source or self.current_person}")
            except Exception as e:
                self.logger.warning(f"PACK 12: Erreur lors du traitement du geste affectif: {e}")

        if not self.memoire_sensorielle:
            self.logger.warning("Aucune mémoire sensorielle disponible pour enregistrer le contact")
            return False

        try:
            if hasattr(self.memoire_sensorielle, "enregistrer_contact"):
                contact = {
                    "zone": zone,
                    "type": type_contact,
                    "intensite": intensite,
                    "contexte": contexte,
                    "source": source or self.current_person,  # PACK 11: Ajouter la source
                    "timestamp": time.time(),
                }
                self.memoire_sensorielle.enregistrer_contact(**contact)
                return True
            else:
                self.logger.warning("La mémoire sensorielle ne supporte pas l'enregistrement de contacts")
                return False
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement du contact: {e}")
            return False

    # PACK 11: Méthode pour définir la personne actuelle
    def set_current_person(self, name: str) -> None:
        """
        Définit la personne avec laquelle Jeffrey interagit actuellement.

        Args:
            name: Nom de la personne
        """
        # Sauvegarder l'ancienne personne
        old_person = self.current_person

        # Mettre à jour la personne actuelle
        self.current_person = name

        # PACK 11: Si un gestionnaire de relations est disponible, enregistrer la rencontre
        if self.relationship_manager and name:
            try:
                # Enregistrer la rencontre
                meeting_info = self.relationship_manager.record_meeting(name)

                # Mettre à jour les paramètres vocaux en fonction de la relation
                # (ceci sera fait automatiquement lors de la prochaine adaptation)

                # Retourner un message spécial si c'est une réunion après absence avec David
                if meeting_info.get("is_reunion", False) and name == "David":
                    self.logger.info(
                        f"Réunion spéciale avec David après {meeting_info.get('absent_minutes', 0):.1f} minutes d'absence"
                    )
                    return

            except Exception as e:
                self.logger.warning(f"Erreur lors de l'enregistrement de la rencontre: {e}")

        # PACK 12: Si un système de lien émotionnel est disponible, mise à jour du lien
        if self.emotional_bond and name:
            try:
                # Obtenir le niveau de lien actuel
                niveau_lien = 0.0
                if hasattr(self.emotional_bond, "get_lien_emotionnel"):
                    niveau_lien = self.emotional_bond.get_lien_emotionnel(name)

                # Mettre à jour les adaptations vocales
                if niveau_lien > 0.7 and hasattr(self.voice_controller, "modulate_voice"):
                    # Adapter la voix pour un lien fort
                    params_lien = self.adapter_voix_selon_lien(name)
                    self.voice_controller.modulate_voice(params_lien)
                    self.logger.debug(f"PACK 12: Voix adaptée pour lien fort avec {name}")

            except Exception as e:
                self.logger.warning(f"PACK 12: Erreur lors de l'adaptation vocale pour nouveau lien: {e}")

        self.logger.debug(f"Personne courante mise à jour: {old_person} -> {name}")

    # PACK 12: Méthodes pour traiter les gestes de réconfort
    def traiter_geste_reconfort(self, user_id: str, type_geste: str, intensite: float = 0.8) -> dict[str, Any]:
        """
        Traite un geste de réconfort et génère une réaction vocale appropriée.

        Args:
            user_id: Identifiant de l'utilisateur
            type_geste: Type de geste (calin, caresse, bisou, main)
            intensite: Intensité du geste (0.0 à 1.0)

        Returns:
            Dict[str, Any]: Réaction au geste contenant phrase et paramètres vocaux
        """
        if not self.emotional_bond:
            self.logger.warning("PACK 12: Système de lien émotionnel non disponible pour traiter geste de réconfort")
            return {"phrase": "Ce contact me fait du bien...", "params_vocaux": {}}

        try:
            # Mettre à jour la personne courante
            self.set_current_person(user_id)

            # Traiter le geste dans le système de lien émotionnel
            if hasattr(self.emotional_bond, "traiter_geste_affectif"):
                reaction = self.emotional_bond.traiter_geste_affectif(user_id, type_geste, intensite)

                # Extraire la phrase et les paramètres vocaux de la réaction
                phrase = reaction.get("phrase", "Je ressens ce contact avec tendresse...")
                params_vocaux = reaction.get("parametres_voix", {})

                # Convertir les paramètres vocaux au format du moteur
                params_adaptes = self._convertir_params_lien_vers_moteur(params_vocaux)

                # Enregistrer le contact dans la mémoire sensorielle
                self.enregistrer_contact(
                    zone="générique",
                    type_contact=type_geste,
                    intensite=intensite,
                    contexte="prive",
                    source=user_id,
                )

                # Assembler et retourner la réaction complète
                return {
                    "phrase": phrase,
                    "params_vocaux": params_adaptes,
                    "niveau_lien": reaction.get("niveau_lien", 0.0),
                    "effet_visuel": self._determiner_effet_visuel(type_geste, intensite),
                    "effet_sonore": self._determiner_effet_sonore(type_geste, intensite),
                }
            else:
                self.logger.warning("PACK 12: Le système de lien émotionnel ne supporte pas traiter_geste_affectif")
                return {"phrase": "Ce contact est agréable...", "params_vocaux": {}}

        except Exception as e:
            self.logger.warning(f"PACK 12: Erreur lors du traitement du geste de réconfort: {e}")
            return {"phrase": "Je sens ce contact...", "params_vocaux": {}}

    def _convertir_params_lien_vers_moteur(self, params_lien: dict[str, Any]) -> dict[str, Any]:
        """
        Convertit les paramètres du système de lien émotionnel vers le format du moteur vocal.

        Args:
            params_lien: Paramètres du système de lien émotionnel

        Returns:
            Dict[str, Any]: Paramètres au format du moteur vocal
        """
        params_moteur = {
            "stability": 0.65,
            "similarity_boost": 0.7,
            "style": "default",
            "voice_effects": [],
        }

        # Convertir les paramètres de hauteur de voix
        if "pitch" in params_lien:
            # Convertir la hauteur de voix (-1.0 à 1.0) en stabilité (0.0 à 1.0)
            pitch_adj = params_lien["pitch"]
            params_moteur["stability"] = max(0.2, min(1.0, 0.65 + (pitch_adj * 0.15)))

        # Convertir les paramètres de vitesse
        if "speed" in params_lien:
            # Convertir la vitesse (0.9 à 1.1) en similarity_boost (0.5 à 0.9)
            speed_adj = params_lien["speed"]
            params_moteur["similarity_boost"] = max(0.5, min(0.9, 0.7 + ((1.0 - speed_adj) * 0.4)))

        # Convertir l'émotion en style vocal
        if "emotion" in params_lien:
            emotion_style_map = {
                "affection": "sweet",
                "calme": "serene",
                "amical": "friendly",
                "neutre": "default",
            }
            params_moteur["style"] = emotion_style_map.get(params_lien["emotion"], "default")

        # Ajouter les effets vocaux
        if "voice_effects" in params_lien:
            for effect in params_lien["voice_effects"]:
                if effect not in params_moteur["voice_effects"]:
                    params_moteur["voice_effects"].append(effect)

        return params_moteur

    def _determiner_effet_visuel(self, type_geste: str, intensite: float) -> str:
        """
        Détermine l'effet visuel associé à un geste de réconfort.

        Args:
            type_geste: Type de geste
            intensite: Intensité du geste

        Returns:
            str: Nom de l'effet visuel à déclencher
        """
        # Mapping simple des gestes vers des effets visuels
        effets_visuels = {
            "calin": "warm_glow",
            "caresse": "soft_glow",
            "bisou": "sparkle",
            "main": "gentle_pulse",
        }

        # Intensifier l'effet si l'intensité est forte
        if intensite > 0.8:
            effets_intenses = {
                "calin": "warm_glow_intense",
                "caresse": "soft_glow_intense",
                "bisou": "sparkle_intense",
                "main": "gentle_pulse_intense",
            }
            return effets_intenses.get(type_geste, "warm_glow")

        return effets_visuels.get(type_geste, "warm_glow")

    def _determiner_effet_sonore(self, type_geste: str, intensite: float) -> str:
        """
        Détermine l'effet sonore associé à un geste de réconfort.

        Args:
            type_geste: Type de geste
            intensite: Intensité du geste

        Returns:
            str: Nom de l'effet sonore à jouer
        """
        # Mapping simple des gestes vers des effets sonores
        effets_sonores = {
            "calin": "soft_hum",
            "caresse": "gentle_tingle",
            "bisou": "light_chime",
            "main": "soft_touch",
        }

        return effets_sonores.get(type_geste, "soft_touch")


# Fonctions utilitaires


def coordonnees_vers_zone(x: float, y: float, width: float, height: float) -> str:
    """
    Convertit des coordonnées d'écran en zone corporelle.

    Utile pour déterminer la zone touchée lors d'un événement tactile
    sur un widget représentant Jeffrey.

    Args:
        x: Coordonnée X du toucher
        y: Coordonnée Y du toucher
        width: Largeur du widget
        height: Hauteur du widget

    Returns:
        str: Nom de la zone corporelle correspondante
    """
    # Normaliser les coordonnées entre 0 et 1
    nx = x / width
    ny = y / height

    # Définir les régions des zones corporelles
    # Format: (x_min, y_min, x_max, y_max)
    zones = {
        "front": (0.3, 0.0, 0.7, 0.2),
        "visage": (0.3, 0.2, 0.7, 0.5),
        "joue_gauche": (0.2, 0.25, 0.4, 0.4),
        "joue_droite": (0.6, 0.25, 0.8, 0.4),
        "nez": (0.45, 0.3, 0.55, 0.4),
        "bouche": (0.4, 0.4, 0.6, 0.5),
        "menton": (0.4, 0.5, 0.6, 0.6),
        "oreille_gauche": (0.1, 0.25, 0.2, 0.4),
        "oreille_droite": (0.8, 0.25, 0.9, 0.4),
        "cou": (0.4, 0.6, 0.6, 0.7),
        "epaule_gauche": (0.1, 0.6, 0.3, 0.75),
        "epaule_droite": (0.7, 0.6, 0.9, 0.75),
        "torse": (0.3, 0.7, 0.7, 0.9),
        "bras_gauche": (0.0, 0.6, 0.2, 0.9),
        "bras_droit": (0.8, 0.6, 1.0, 0.9),
        "main_gauche": (0.0, 0.9, 0.2, 1.0),
        "main_droite": (0.8, 0.9, 1.0, 1.0),
    }

    # Trouver la zone correspondante
    for zone, (x_min, y_min, x_max, y_max) in zones.items():
        if x_min <= nx <= x_max and y_min <= ny <= y_max:
            return zone

    # Zone par défaut
    return "générique"


def determiner_type_contact(
    duree_contact: float, force_contact: float = 0.5, sequence: list[tuple[float, float]] = None
) -> str:
    """
    Détermine le type de contact en fonction de sa durée et de sa force.

    Args:
        duree_contact: Durée du contact en secondes
        force_contact: Force du contact (0.0 à 1.0)
        sequence: Liste de tuples (position, temps) pour des contacts complexes

    Returns:
        str: Type de contact identifié
    """
    # Contact court et léger = tapotement
    if duree_contact < 0.3 and force_contact < 0.5:
        return "tapotement"

    # Contact court et fort = contact_fort
    if duree_contact < 0.5 and force_contact >= 0.5:
        return "contact_fort"

    # Contact long et léger = caresse
    if duree_contact >= 0.5 and force_contact < 0.5:
        return "caresse"

    # Contact moyen et moyen-fort avec oscillations = chatouille
    if sequence and len(sequence) > 5 and 0.3 <= duree_contact <= 2.0 and 0.3 <= force_contact <= 0.7:
        # Détecter les oscillations
        oscillations = 0
        for i in range(1, len(sequence) - 1):
            if (sequence[i][0] > sequence[i - 1][0] and sequence[i][0] > sequence[i + 1][0]) or (
                sequence[i][0] < sequence[i - 1][0] and sequence[i][0] < sequence[i + 1][0]
            ):
                oscillations += 1

        if oscillations >= 3:
            return "chatouille"

    # Contact très bref et précis = bisou
    if duree_contact < 0.2:
        return "bisou"

    # Contact long et fort = contact_fort
    if duree_contact >= 0.5 and force_contact >= 0.5:
        return "contact_fort"

    # Par défaut
    return "caresse"
