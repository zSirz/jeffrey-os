#!/usr/bin/env python
"""
energy_face_memory.py - Gestion de la mémoire du visage de Jeffrey
Partie de la refactorisation du fichier energy_face.py d'origine (PACK 18)

Ce module gère les interactions du visage avec la mémoire sensorielle :
- Mémoire des zones touchées
- Mémorisation des interactions sensorielles
- Mémorisation des contacts affectifs
- Calcul d'habituation sensorielle
"""

import logging
import math
import time

from kivy.clock import Clock


class MemoryHandler:
    """
    Gestionnaire de la mémoire sensorielle pour le visage de Jeffrey.
    Contrôle les interactions avec la mémoire et l'apprentissage sensoriel.
    """

    def __init__(self, face_widget):
        """
        Initialise le gestionnaire de mémoire.

        Args:
            face_widget: Widget du visage (EnergyFaceCoreWidget)
        """
        self.face = face_widget

        # Variables pour la mémoire sensorielle
        self._touch_memory = {}  # Mémoire des zones touchées
        self._active_sensory_responses = {}  # Réponses sensorielles actives

        # Variables pour la mémoire des zones corporelles
        self._zone_memory = {}  # Mémoire des zones touchées

        # Planifier la sauvegarde régulière de la mémoire sensorielle
        Clock.schedule_interval(self.save_sensory_memory, 60.0)  # Toutes les 60 secondes

    def save_sensory_memory(self, dt):
        """
        Sauvegarde périodique de la mémoire sensorielle.

        Args:
            dt: Delta temps depuis la dernière sauvegarde
        """
        if hasattr(self.face, "memoire_sensorielle"):
            try:
                self.face.memoire_sensorielle.sauvegarder()
                logging.info("Mémoire sensorielle sauvegardée")
            except Exception as e:
                logging.error(f"Erreur lors de la sauvegarde de la mémoire sensorielle: {e}")

    def process_touch(self, x: float, y: float, touch_type: str = "caresse", intensity: float = 0.5):
        """
        Traite un toucher et met à jour la mémoire sensorielle.

        Args:
            x: Coordonnée X du toucher
            y: Coordonnée Y du toucher
            touch_type: Type de toucher (caresse, tapotement, etc.)
            intensity: Intensité du toucher (0.0 à 1.0)
        """
        # Identifier la zone touchée
        zone = self._identify_touch_zone(x, y)

        # Pas de traitement si zone non identifiée
        if not zone:
            return

        # Créer l'entrée dans la mémoire des zones si elle n'existe pas
        if zone not in self._zone_memory:
            self._zone_memory[zone] = {
                "last_touch": None,
                "touch_count": 0,
                "touch_types": {},
                "average_intensity": 0.0,
            }

        # Mettre à jour la mémoire de la zone
        memory = self._zone_memory[zone]
        memory["last_touch"] = time.time()
        memory["touch_count"] += 1

        # Mettre à jour le compteur par type
        memory["touch_types"][touch_type] = memory["touch_types"].get(touch_type, 0) + 1

        # Mettre à jour l'intensité moyenne (moyenne mobile)
        memory["average_intensity"] = (memory["average_intensity"] * (memory["touch_count"] - 1) + intensity) / memory[
            "touch_count"
        ]

        # Transmettre à la mémoire sensorielle si disponible
        if hasattr(self.face, "memoire_sensorielle"):
            # Convertir la zone pour la mémoire sensorielle
            zone_sensorielle = self._convert_zone_to_sensory(zone)

            # Personne qui interagit (si connue)
            source = getattr(self.face, "current_person", "inconnu")

            # Mode de contexte
            contexte = getattr(self.face, "context_mode", "public")

            # Créer le contact sensoriel
            contact = {
                "zone": zone_sensorielle,
                "type": touch_type,
                "intensite": intensity,
                "source": source,
                "contexte": contexte,
                "timestamp": time.time(),
            }

            # Enregistrer le contact dans la mémoire sensorielle
            try:
                self.face.memoire_sensorielle.enregistrer_contact(contact)

                # Déclencher une réponse sensorielle si la zone est sensible
                sensibilite = self.face.memoire_sensorielle.obtenir_sensibilite_zone(zone_sensorielle)
                if sensibilite > 0.4:
                    self._trigger_sensory_response(zone, sensibilite, intensity)
            except Exception as e:
                logging.error(f"Erreur lors de l'enregistrement du contact: {e}")

    def _identify_touch_zone(self, x: float, y: float) -> str | None:
        """
        Identifie la zone corporelle touchée en fonction des coordonnées.

        Args:
            x: Coordonnée X du toucher
            y: Coordonnée Y du toucher

        Returns:
            Nom de la zone touchée ou None si hors zones
        """
        # Coordonnées relatives au centre du visage
        rel_x = x - self.face.center_x
        rel_y = y - self.face.center_y
        dist = math.sqrt(rel_x**2 + rel_y**2)

        # Identifier les zones en fonction des coordonnées relatives
        if dist < 100:  # Dans la zone du visage
            # Zones du visage
            if abs(rel_x) > 30 and rel_y < 0:
                # Joues (gauche/droite selon rel_x)
                return "joue_gauche" if rel_x < 0 else "joue_droite"
            elif abs(rel_y) > 30 and rel_y > 0:
                # Front
                return "front"
            elif abs(rel_y) > 30 and rel_y < 0:
                # Menton
                return "menton"
            elif abs(rel_x) < 20 and abs(rel_y) < 10:
                # Nez (centre)
                return "nez"
            elif abs(rel_x) < 30 and rel_y < -20 and rel_y > -40:
                # Bouche
                return "lèvres"
            else:
                # Reste du visage
                return "visage"
        elif dist < 150:
            # Tête (au-delà du visage mais proche)
            return "tête"

        # Hors des zones définies
        return None

    def _convert_zone_to_sensory(self, zone: str) -> str:
        """
        Convertit une zone d'interface en zone pour la mémoire sensorielle.

        Args:
            zone: Zone de l'interface

        Returns:
            Zone correspondante dans le format de la mémoire sensorielle
        """
        # Mapping des zones de l'interface vers la mémoire sensorielle
        zone_mapping = {
            "joue_gauche": "joue_gauche",
            "joue_droite": "joue_droite",
            "front": "front",
            "menton": "menton",
            "nez": "nez",
            "lèvres": "lèvres",
            "visage": "visage",
            "tête": "tête",
        }

        return zone_mapping.get(zone, "inconnu")

    def _trigger_sensory_response(self, zone: str, sensibilite: float, intensity: float):
        """
        Déclenche une réponse visuelle à un toucher sensoriel.

        Args:
            zone: Zone touchée
            sensibilite: Sensibilité de la zone (0.0 à 1.0)
            intensity: Intensité du toucher (0.0 à 1.0)
        """
        # La réponse dépend de la zone et de sa sensibilité
        if zone in ["joue_gauche", "joue_droite"]:
            # Rougissement des joues
            if hasattr(self.face.effects, "add_blush"):
                left = zone == "joue_gauche"
                right = zone == "joue_droite"
                blush_intensity = sensibilite * intensity * 0.8
                self.face.effects.add_blush(intensity=blush_intensity, left=left, right=right)

        elif zone == "lèvres":
            # Forte sensibilité pour les lèvres - effet special
            if sensibilite > 0.7 and intensity > 0.6:
                if hasattr(self.face.effects, "apply_frisson"):
                    frisson_intensity = sensibilite * intensity * 0.6
                    self.face.effects.apply_frisson(intensity=frisson_intensity, duration=1.0)

        elif zone in ["front", "visage"]:
            # Effet de brillance dans les yeux
            if sensibilite > 0.5 and intensity > 0.5:
                if hasattr(self.face.effects, "add_eye_reflection"):
                    reflection_intensity = sensibilite * intensity * 0.7
                    self.face.effects.add_eye_reflection(intensity=reflection_intensity)

    def get_zone_sensitivity(self, zone: str) -> float:
        """
        Récupère la sensibilité d'une zone.

        Args:
            zone: Nom de la zone

        Returns:
            Niveau de sensibilité (0.0 à 1.0)
        """
        # Sensibilité de base par zone
        base_sensitivity = {
            "joue_gauche": 0.7,
            "joue_droite": 0.7,
            "front": 0.5,
            "menton": 0.4,
            "nez": 0.6,
            "lèvres": 0.9,
            "visage": 0.6,
            "tête": 0.5,
        }.get(zone, 0.5)

        # Si la mémoire sensorielle est disponible, utiliser sa sensibilité
        if hasattr(self.face, "memoire_sensorielle"):
            zone_sensorielle = self._convert_zone_to_sensory(zone)
            try:
                return self.face.memoire_sensorielle.obtenir_sensibilite_zone(zone_sensorielle)
            except Exception:
                pass

        # Sinon, ajuster selon l'historique local des contacts
        if zone in self._zone_memory:
            memory = self._zone_memory[zone]

            # La sensibilité augmente avec le nombre de contacts (jusqu'à un certain point)
            touch_factor = min(1.0, memory["touch_count"] / 10.0)

            # L'intensité moyenne des contacts précédents influence aussi
            intensity_factor = memory.get("average_intensity", 0.5)

            # Calculer la sensibilité ajustée
            adjusted = base_sensitivity * (1.0 + touch_factor * 0.3) * (1.0 + intensity_factor * 0.2)
            return min(1.0, adjusted)

        return base_sensitivity

    def get_zone_habituation(self, zone: str, user_id: str | None = None) -> float:
        """
        Récupère le niveau d'habituation pour une zone.

        Args:
            zone: Zone corporelle concernée
            user_id: Identifiant de l'utilisateur (utilise l'utilisateur actuel si None)

        Returns:
            Niveau d'habituation (0.0 à 1.0)
        """
        if not hasattr(self.face, "memoire_sensorielle"):
            return 0.0

        # Utiliser l'utilisateur actuel si non spécifié
        user_id = user_id or getattr(self.face, "current_person", "inconnu")

        # Convertir la zone
        zone_sensorielle = self._convert_zone_to_sensory(zone)

        try:
            # Récupérer l'habituation de la mémoire sensorielle
            return self.face.memoire_sensorielle.obtenir_habituation_zone(zone_sensorielle, user_id)
        except Exception:
            return 0.0

    def get_contact_history(self, zone: str | None = None) -> list[dict]:
        """
        Récupère l'historique des contacts pour une zone ou toutes les zones.

        Args:
            zone: Zone spécifique (ou None pour toutes les zones)

        Returns:
            Liste des contacts enregistrés
        """
        if not hasattr(self.face, "memoire_sensorielle"):
            return []

        try:
            if zone:
                # Convertir la zone
                zone_sensorielle = self._convert_zone_to_sensory(zone)
                return self.face.memoire_sensorielle.obtenir_historique_contacts(zone_sensorielle)
            else:
                # Toutes les zones
                return self.face.memoire_sensorielle.obtenir_historique_tous_contacts()
        except Exception:
            return []

    def get_preferred_zones(self, count: int = 3) -> list[dict]:
        """
        Récupère les zones préférées (les plus sensibles).

        Args:
            count: Nombre de zones à retourner

        Returns:
            Liste des zones préférées avec leurs infos
        """
        if not hasattr(self.face, "memoire_sensorielle"):
            return []

        try:
            return self.face.memoire_sensorielle.obtenir_zones_préférées(count)
        except Exception:
            return []

    def reset_memory_for_zone(self, zone: str):
        """
        Réinitialise la mémoire pour une zone spécifique.

        Args:
            zone: Zone à réinitialiser
        """
        # Réinitialiser la mémoire locale
        if zone in self._zone_memory:
            del self._zone_memory[zone]

        # Réinitialiser dans la mémoire sensorielle si disponible
        if hasattr(self.face, "memoire_sensorielle"):
            zone_sensorielle = self._convert_zone_to_sensory(zone)
            try:
                self.face.memoire_sensorielle.réinitialiser_zone(zone_sensorielle)
            except Exception as e:
                logging.error(f"Erreur lors de la réinitialisation de la zone: {e}")

    def create_sensory_memory_report(self) -> dict:
        """
        Crée un rapport sur l'état actuel de la mémoire sensorielle.

        Returns:
            Dictionnaire avec les informations de la mémoire sensorielle
        """
        report = {
            "zones": {},
            "preferred_zones": [],
            "total_contacts": 0,
            "users": {},
            "last_contact": None,
        }

        # Vérifier si la mémoire sensorielle est disponible
        if not hasattr(self.face, "memoire_sensorielle"):
            return report

        try:
            # Récupérer les informations sur toutes les zones
            zones_info = self.face.memoire_sensorielle.obtenir_info_toutes_zones()
            report["zones"] = zones_info

            # Récupérer les zones préférées
            report["preferred_zones"] = self.get_preferred_zones(5)

            # Nombre total de contacts
            all_contacts = self.face.memoire_sensorielle.obtenir_historique_tous_contacts()
            report["total_contacts"] = len(all_contacts)

            # Dernier contact
            if all_contacts:
                report["last_contact"] = all_contacts[0]

            # Informations par utilisateur
            users = set(contact.get("source", "inconnu") for contact in all_contacts)
            for user in users:
                user_contacts = [c for c in all_contacts if c.get("source") == user]
                report["users"][user] = {
                    "contacts_count": len(user_contacts),
                    "last_contact": user_contacts[0] if user_contacts else None,
                }
        except Exception as e:
            logging.error(f"Erreur lors de la création du rapport de mémoire sensorielle: {e}")

        return report
