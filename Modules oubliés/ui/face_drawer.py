#!/usr/bin/env python
"""
Module de dessin du visage de Jeffrey.
Contient la classe FaceDrawer responsable uniquement du rendu graphique du visage.
"""

import math

from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, Line, PopMatrix, PushMatrix, Rotate


class FaceDrawer:
    """
    Classe responsable du dessin du visage de Jeffrey.
    S'occupe uniquement des aspects graphiques sans logique d'animation.
    """

    def __init__(self, parent_widget):
        """
        Initialise le module de dessin.

        Args:
            parent_widget: Widget parent (EnergyFaceWidget) contenant le canvas
        """
        self.parent = parent_widget
        # Référence au widget parent pour accéder au canvas et aux propriétés

        # Dictionnaire des formes de bouche (importé depuis le parent)
        self.mouth_shapes = {
            "A": {"width_factor": 1.0, "height_factor": 1.5},  # grande ouverte
            "E": {"width_factor": 1.2, "height_factor": 0.8},  # mi-ouverte
            "I": {"width_factor": 1.5, "height_factor": 0.5},  # étirée sur les côtés
            "O": {"width_factor": 0.8, "height_factor": 0.8},  # arrondie
            "M": {"width_factor": 0.9, "height_factor": 0.2},  # fermée
            "F": {"width_factor": 1.1, "height_factor": 0.4},  # dents visibles sur lèvre inférieure
            "J": {"width_factor": 1.2, "height_factor": 0.6},  # dents visibles
            "S": {"width_factor": 1.0, "height_factor": 0.4},  # dents légèrement visibles
            "X": {"width_factor": 0.9, "height_factor": 0.3},  # neutre/fermée
            "OU": {"width_factor": 0.6, "height_factor": 0.8},  # bouche très arrondie
            "AI": {"width_factor": 1.3, "height_factor": 0.6},  # étirée mais ouverte
            "IN": {"width_factor": 1.2, "height_factor": 0.5},  # bouche légèrement étirée
            "ON": {"width_factor": 0.7, "height_factor": 0.9},  # arrondie et tombante
            "UI": {"width_factor": 0.9, "height_factor": 0.6},  # pincée
            "AN": {"width_factor": 1.0, "height_factor": 0.7},  # bouche ovale
            "EN": {"width_factor": 1.0, "height_factor": 0.6},  # bouche basse
            "EU": {"width_factor": 0.85, "height_factor": 0.7},  # arrondie pincée
            "AU": {"width_factor": 0.75, "height_factor": 0.9},  # très arrondie
            "EI": {"width_factor": 1.3, "height_factor": 0.7},  # sourire vocalique
            "OE": {"width_factor": 0.8, "height_factor": 0.6},  # intermédiaire entre O et E
            "GN": {"width_factor": 1.0, "height_factor": 0.5},  # étroite avec repli
            "CH": {"width_factor": 1.1, "height_factor": 0.4},  # lèvres projetées
            "UI": {"width_factor": 0.9, "height_factor": 0.6},  # pincée (déjà présent, renforcé)
        }

    # ====================================================
    # MÉTHODE PRINCIPALE DE DESSIN
    # ====================================================

    def draw_face(self):
        """
        Méthode principale de dessin du visage.
        Coordonne l'appel de toutes les sous-méthodes de dessin.
        """
        with self.parent.canvas:
            # PACK 9: Overlay de blessure affective (dessiné en premier)
            if self.parent.blessure_overlay_opacity > 0.01:
                # Couleur subtile, légèrement bleutée pour la tristesse
                blessure_color = (0.8, 0.85, 0.95, self.parent.blessure_overlay_opacity)
                Color(*blessure_color)

                # Overlay sur tout le visage, légèrement plus grand
                overlay_size = 230  # Taille légèrement plus grande que le visage
                Ellipse(
                    pos=(
                        self.parent.center_x - overlay_size / 2,
                        self.parent.center_y - overlay_size / 2,
                    ),
                    size=(overlay_size, overlay_size),
                )

            # PACK 9: Effet de résonance affective
            if self.parent.resonance_active:
                # Aura qui respire doucement autour du visage
                resonance_opacity = 0.04 + 0.03 * math.sin(self.parent.resonance_phase)
                resonance_scale = 1.0 + 0.07 * math.sin(self.parent.resonance_phase * 0.5)

                # Couleur douce et chaleureuse avec une teinte dorée
                resonance_color = (1.0, 0.95, 0.85, resonance_opacity)
                Color(*resonance_color)

                # Aura qui entoure tout le visage
                resonance_size = 250 * resonance_scale
                Ellipse(
                    pos=(
                        self.parent.center_x - resonance_size / 2,
                        self.parent.center_y - resonance_size / 2,
                    ),
                    size=(resonance_size, resonance_size),
                )

                # Deuxième couche plus diffuse
                Color(1.0, 0.9, 0.8, resonance_opacity * 0.6)
                outer_resonance_size = resonance_size * 1.2
                Ellipse(
                    pos=(
                        self.parent.center_x - outer_resonance_size / 2,
                        self.parent.center_y - outer_resonance_size / 2,
                    ),
                    size=(outer_resonance_size, outer_resonance_size),
                )

            # Dessiner les couches de beauté en arrière-plan
            self.draw_beauty_layers()

            # Dessin du halo principal
            self.draw_halo()

            # PACK 8: Aura du lien affectif
            if self.parent.lien_heart_glow:
                # Créer un halo scintillant autour de la région du cœur (bas du visage)
                heart_glow_opacity = 0.06 + 0.04 * math.sin(self.parent.lien_affectif_phase)
                heart_glow_scale = 1.0 + 0.05 * math.sin(self.parent.lien_affectif_phase * 0.6)

                # Couleur rose-doré très douce et chaleureuse
                # PACK 9: Moduler la couleur avec la résonance
                if self.parent.resonance_affective > 0.7:
                    # Plus doré si forte résonance
                    Color(1.0, 0.85, 0.8, heart_glow_opacity)
                else:
                    # Rose standard sinon
                    Color(0.95, 0.9, 0.85, heart_glow_opacity)

                # Taille du halo environ 2/3 du visage, centré un peu plus bas
                heart_size = 140 * heart_glow_scale
                heart_y_offset = -35  # Déplacer vers le bas légèrement
                Ellipse(
                    pos=(
                        self.parent.center_x - heart_size / 2,
                        self.parent.center_y - heart_size / 2 + heart_y_offset,
                    ),
                    size=(heart_size, heart_size),
                )

            # Base du visage (forme ovale principale)
            # Calculer le facteur d'échelle de respiration
            breath_factor = 1.0 + self.parent.scale - 1.0  # Normaliser par rapport à l'échelle de base

            # Adapter légèrement la couleur selon l'émotion
            face_color = (0.95, 0.88, 0.82, 1.0)  # Teint de base

            if self.parent.emotion == "colère":
                face_color = (0.95, 0.8, 0.78, 1.0)  # Plus rouge pour la colère
            elif self.parent.emotion == "peur":
                face_color = (0.92, 0.86, 0.8, 1.0)  # Plus pâle pour la peur
            elif self.parent.emotion == "tristesse":
                face_color = (0.9, 0.85, 0.83, 1.0)  # Plus bleuté pour la tristesse

            # PACK 5: Modulation du teint selon l'état d'intimité
            if hasattr(self.parent, "intimite_active") and self.parent.intimite_active:
                # Légère rougeur du visage en cas d'intimité active
                face_color = (
                    min(1.0, face_color[0] + 0.05),
                    min(1.0, face_color[1] - 0.05),
                    min(1.0, face_color[2] - 0.02),
                    face_color[3],
                )

            # Modulation du teint selon la fatigue
            if hasattr(self.parent, "fatigue_level") and self.parent.fatigue_level > 0.3:
                # Teint plus terne avec la fatigue
                fatigue_factor = self.parent.fatigue_level * 0.15
                face_color = (
                    max(0.7, face_color[0] - fatigue_factor),
                    max(0.7, face_color[1] - fatigue_factor),
                    max(0.7, face_color[2] - fatigue_factor * 0.5),
                    face_color[3],
                )

            # Dessiner le visage avec la couleur adaptée
            Color(*face_color)

            # Visage légèrement ovale
            face_width = 200 * breath_factor
            face_height = 220 * breath_factor
            Ellipse(
                pos=(self.parent.center_x - face_width / 2, self.parent.center_y - face_height / 2),
                size=(face_width, face_height),
            )

            # Adaptation de l'humeur et des traits principaux
            # Appliquer des micro-mouvements pour plus de vie
            # Dessiner les différentes parties du visage dans l'ordre

            # Sourcils
            self.draw_eyebrows()

            # Joues
            self.draw_cheeks()

            # Yeux
            self.draw_eyes()

            # Bouche
            self.draw_mouth()

            # PACK 9: Larmes discrètes si blessure active
            if self.parent.larmes_discretes:
                self.draw_discrete_tears()

    # ====================================================
    # MÉTHODES DE DESSIN DE COMPOSANTS SPÉCIFIQUES
    # ====================================================

    def draw_beauty_layers(self):
        """
        Dessine les couches de beauté qui ajoutent de la profondeur au visage.
        """
        # Si les couches de beauté sont définies
        if hasattr(self.parent, "beauty_layers") and self.parent.beauty_layers:
            for layer in self.parent.beauty_layers:
                # Calculer l'opacité en fonction de la phase et l'échelle
                current_time = Clock.get_time()
                phase_adjusted = current_time * layer["speed"] + layer["phase_offset"]
                opacity_factor = 0.7 + 0.3 * math.sin(phase_adjusted)

                # Couleur de base du visage avec variation de l'opacité
                base_opacity = layer["opacity"] * opacity_factor

                # Créer un halo légèrement coloré par l'émotion actuelle
                if self.parent.emotion == "joie":
                    Color(0.97, 0.94, 0.82, base_opacity)  # Doré chaleureux
                elif self.parent.emotion == "tristesse":
                    Color(0.85, 0.90, 0.95, base_opacity)  # Bleuté subtil
                elif self.parent.emotion == "colère":
                    Color(0.95, 0.85, 0.82, base_opacity)  # Légèrement rougeâtre
                elif self.parent.emotion == "sérénité":
                    Color(0.9, 0.95, 0.9, base_opacity)  # Verdâtre apaisant
                else:
                    Color(0.92, 0.92, 0.94, base_opacity)  # Neutre légèrement bleuté

                # Taille et position adaptées à l'échelle
                size = 210 * layer["scale"]
                pos = (self.parent.center_x - size / 2, self.parent.center_y - size / 2)

                # Dessiner l'ellipse
                Ellipse(pos=pos, size=(size, size))

    def draw_eyebrows(self):
        """Dessine les sourcils en fonction de l'émotion."""
        # Couleur des sourcils (brun foncé)
        Color(0.4, 0.3, 0.25, 1.0)

        # Position de base des sourcils
        eyebrow_offset = 60  # Distance horizontale entre les sourcils
        eyebrow_y = self.parent.center_y + 45  # Hauteur des sourcils

        # Variations selon l'émotion
        eyebrow_angle_left = 0  # Angle en degrés
        eyebrow_angle_right = 0
        eyebrow_curvature = 0  # Courbure (positive = haut, négative = bas)
        eyebrow_y_offset = 0  # Décalage vertical

        if self.parent.emotion == "joie":
            eyebrow_angle_left = 15
            eyebrow_angle_right = -15
            eyebrow_curvature = 5
            eyebrow_y_offset = 2
        elif self.parent.emotion == "tristesse":
            eyebrow_angle_left = -15
            eyebrow_angle_right = 15
            eyebrow_curvature = -5
            eyebrow_y_offset = -2
        elif self.parent.emotion == "colère":
            eyebrow_angle_left = -20
            eyebrow_angle_right = 20
            eyebrow_curvature = -8
            eyebrow_y_offset = -5
        elif self.parent.emotion == "surprise":
            eyebrow_angle_left = 5
            eyebrow_angle_right = -5
            eyebrow_curvature = 10
            eyebrow_y_offset = 8
        elif self.parent.emotion == "peur":
            eyebrow_angle_left = -10
            eyebrow_angle_right = 10
            eyebrow_curvature = 8
            eyebrow_y_offset = 5

        # Modulation des sourcils pour l'intimité (Pack 5)
        if hasattr(self.parent, "intimite_active") and self.parent.intimite_active:
            # En mode intimité, sourcils légèrement relevés avec courbure subtile
            eyebrow_angle_left = max(-5, eyebrow_angle_left)
            eyebrow_angle_right = min(5, eyebrow_angle_right)
            eyebrow_curvature = max(2, eyebrow_curvature)
            eyebrow_y_offset = max(1, eyebrow_y_offset)

        # PACK 4: Modulation des sourcils pour les spasmes musculaires
        if hasattr(self.parent, "_muscle_tensions_active") and self.parent._muscle_tensions_active:
            if hasattr(self.parent, "_eyebrow_spasm_offset"):
                # Appliquer le spasme actuel
                eyebrow_curvature += self.parent._eyebrow_spasm_offset

        # PACK 8: Regard triste si lien blessé
        if hasattr(self.parent, "lien_sad_eyes") and self.parent.lien_sad_eyes:
            eyebrow_angle_left = -10
            eyebrow_angle_right = 10
            eyebrow_curvature = -3
            eyebrow_y_offset = -2

        # Dessiner les sourcils
        left_eyebrow_x = self.parent.center_x - eyebrow_offset
        right_eyebrow_x = self.parent.center_x + eyebrow_offset
        eyebrow_length = 30
        eyebrow_thickness = 5

        # Sourcil gauche avec rotation
        with self.parent.canvas:
            PushMatrix()
            Rotate(angle=eyebrow_angle_left, origin=(left_eyebrow_x, eyebrow_y + eyebrow_y_offset))

            # Points de contrôle pour la courbe
            x1 = left_eyebrow_x - eyebrow_length / 2
            y1 = eyebrow_y + eyebrow_y_offset
            x2 = left_eyebrow_x
            y2 = eyebrow_y + eyebrow_y_offset + eyebrow_curvature
            x3 = left_eyebrow_x + eyebrow_length / 2
            y3 = eyebrow_y + eyebrow_y_offset

            # Dessiner le sourcil courbe
            points = []
            steps = 10
            for i in range(steps):
                t = i / (steps - 1)
                # Formule de Bézier quadratique
                x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * x2 + t**2 * x3
                y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * y2 + t**2 * y3
                points.extend([x, y])

            Line(points=points, width=eyebrow_thickness, cap="round")
            PopMatrix()

        # Sourcil droit avec rotation
        with self.parent.canvas:
            PushMatrix()
            Rotate(angle=eyebrow_angle_right, origin=(right_eyebrow_x, eyebrow_y + eyebrow_y_offset))

            # Points de contrôle pour la courbe
            x1 = right_eyebrow_x - eyebrow_length / 2
            y1 = eyebrow_y + eyebrow_y_offset
            x2 = right_eyebrow_x
            y2 = eyebrow_y + eyebrow_y_offset + eyebrow_curvature
            x3 = right_eyebrow_x + eyebrow_length / 2
            y3 = eyebrow_y + eyebrow_y_offset

            # Dessiner le sourcil courbe
            points = []
            steps = 10
            for i in range(steps):
                t = i / (steps - 1)
                # Formule de Bézier quadratique
                x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * x2 + t**2 * x3
                y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * y2 + t**2 * y3
                points.extend([x, y])

            Line(points=points, width=eyebrow_thickness, cap="round")
            PopMatrix()

    def draw_cheeks(self):
        """
        Dessine les joues avec d'éventuels effets émotionnels.
        """
        # Position des joues
        cheek_offset_x = 55
        cheek_offset_y = -15
        cheek_size = 30

        # Couleur de base des joues
        cheek_opacity = 0.15

        # Ajustement selon l'émotion
        if self.parent.emotion == "joie":
            cheek_opacity = 0.25
        elif self.parent.emotion == "excitation":
            cheek_opacity = 0.3
        elif self.parent.emotion == "timidité":
            cheek_opacity = 0.35
        elif self.parent.emotion == "timide":  # alias
            cheek_opacity = 0.35

        # PACK 5B: Rougissement en cas d'intimité active
        if hasattr(self.parent, "intimite_active") and self.parent.intimite_active:
            # Renforcer l'intensité des joues
            cheek_opacity = max(0.3, cheek_opacity * 1.5)

        # PACK 5B: Rougissement sensuel
        if hasattr(self.parent, "blushing_intensity") and self.parent.blushing_intensity > 0:
            cheek_opacity = max(cheek_opacity, self.parent.blushing_intensity * 0.5)

        # PACK 5A: Effet de plaisir affectif
        if hasattr(self.parent, "pleasure_level") and self.parent.pleasure_level > 0.3:
            cheek_opacity = max(cheek_opacity, self.parent.pleasure_level * 0.3)

        # Couleur rose douce pour les joues
        Color(0.9, 0.5, 0.5, cheek_opacity)

        # Joue gauche
        left_cheek_pos = (
            self.parent.center_x - cheek_offset_x - cheek_size / 2,
            self.parent.center_y + cheek_offset_y - cheek_size / 2,
        )
        Ellipse(pos=left_cheek_pos, size=(cheek_size, cheek_size))

        # Joue droite
        right_cheek_pos = (
            self.parent.center_x + cheek_offset_x - cheek_size / 2,
            self.parent.center_y + cheek_offset_y - cheek_size / 2,
        )
        Ellipse(pos=right_cheek_pos, size=(cheek_size, cheek_size))

        # Joues renforcées si nécessaire (pour des effets spéciaux)
        if hasattr(self.parent, "_enhanced_cheek_left_intensity") and self.parent._enhanced_cheek_left_intensity > 0:
            Color(0.9, 0.6, 0.6, self.parent._enhanced_cheek_left_intensity)
            enhanced_size = cheek_size * (1 + self.parent._enhanced_cheek_left_intensity * 0.5)
            enhanced_pos = (
                self.parent.center_x - cheek_offset_x - enhanced_size / 2,
                self.parent.center_y + cheek_offset_y - enhanced_size / 2,
            )
            Ellipse(pos=enhanced_pos, size=(enhanced_size, enhanced_size))

        if hasattr(self.parent, "_enhanced_cheek_right_intensity") and self.parent._enhanced_cheek_right_intensity > 0:
            Color(0.9, 0.6, 0.6, self.parent._enhanced_cheek_right_intensity)
            enhanced_size = cheek_size * (1 + self.parent._enhanced_cheek_right_intensity * 0.5)
            enhanced_pos = (
                self.parent.center_x + cheek_offset_x - enhanced_size / 2,
                self.parent.center_y + cheek_offset_y - enhanced_size / 2,
            )
            Ellipse(pos=enhanced_pos, size=(enhanced_size, enhanced_size))

    def draw_halo(self):
        """Dessine le halo autour du visage."""
        # Paramètres du halo
        halo_size = 250
        base_opacity = 0.1

        # Moduler l'opacité selon l'émotion
        if self.parent.emotion == "joie":
            base_opacity = 0.15
        elif self.parent.emotion == "émerveillement":
            base_opacity = 0.18
        elif self.parent.emotion == "tristesse":
            base_opacity = 0.08

        # Moduler l'opacité avec l'animation du halo
        if hasattr(self.parent, "halo_animation"):
            base_opacity = base_opacity * (1 + 0.3 * math.sin(self.parent.halo_animation))

        # Dessiner le halo avec une couleur adaptée à l'émotion
        if self.parent.emotion == "joie" or self.parent.emotion == "émerveillement":
            Color(1.0, 0.95, 0.85, base_opacity)  # Doré chaud
        elif self.parent.emotion == "tristesse" or self.parent.emotion == "mélancolie":
            Color(0.85, 0.9, 1.0, base_opacity)  # Bleu clair
        elif self.parent.emotion == "colère":
            Color(1.0, 0.85, 0.85, base_opacity)  # Rouge subtil
        elif self.parent.emotion == "sérénité":
            Color(0.85, 1.0, 0.9, base_opacity)  # Vert apaisant
        else:
            Color(0.95, 0.95, 0.95, base_opacity)  # Blanc lumineux

        # PACK 5A: Halo modifié en cas de plaisir affectif
        if hasattr(self.parent, "pleasure_halo_active") and self.parent.pleasure_halo_active:
            # Couleur rose douce pour le plaisir affectif
            Color(0.95, 0.8, 0.95, base_opacity * 1.5)
            # Halo légèrement plus grand
            halo_size = 260

        # PACK 5B: Aura plus intense en cas d'intimité
        if hasattr(self.parent, "intimite_active") and self.parent.intimite_active:
            # Couleur plus chaude et intense
            if hasattr(self.parent, "intimite_phase"):
                # Utiliser la phase d'intimité pour l'animation
                intensity = 0.15 + 0.05 * math.sin(self.parent.intimite_phase)
                Color(0.98, 0.85, 0.9, intensity)
            else:
                Color(0.98, 0.85, 0.9, base_opacity * 1.5)
            # Halo légèrement plus grand
            halo_size = 260

        # Dessiner le halo
        halo_pos = (self.parent.center_x - halo_size / 2, self.parent.center_y - halo_size / 2)
        Ellipse(pos=halo_pos, size=(halo_size, halo_size))

    def draw_eyes(self):
        """Dessine les yeux avec expression et réflexion."""
        # Position et taille de base des yeux
        eye_offset = 40
        eye_size = 28
        iris_size = 24  # Légèrement plus petit que l'œil
        pupil_size = 10  # Taille de base de la pupille

        # Calculer les positions des yeux, de l'iris et des pupilles
        left_eye_x = self.parent.center_x - eye_offset
        right_eye_x = self.parent.center_x + eye_offset
        eye_y = self.parent.center_y + 20

        # Direction du regard (par défaut, regarde vers l'avant)
        iris_offset_x = 0
        iris_offset_y = 0

        # Dilatation de la pupille selon l'émotion
        pupil_dilation = 1.0

        if self.parent.emotion == "excitation" or self.parent.emotion == "surprise":
            pupil_dilation = 1.3  # Pupilles dilatées
        elif self.parent.emotion == "peur":
            pupil_dilation = 1.5  # Pupilles très dilatées
        elif self.parent.emotion == "concentration":
            pupil_dilation = 0.8  # Pupilles contractées

        # Ajuster la position de l'iris selon l'émotion
        if self.parent.emotion == "méfiance":
            iris_offset_x = -3  # Regard légèrement sur le côté
        elif self.parent.emotion == "pensif":
            iris_offset_y = 3  # Regard légèrement vers le haut
        elif self.parent.emotion == "intérêt":
            iris_offset_x = 4  # Regard dans la direction de l'intérêt

        # Couleur de l'iris selon l'état émotionnel
        iris_color = [0.3, 0.6, 0.7, 0.9]  # Bleu-vert par défaut

        if self.parent.emotion == "colère":
            iris_color = [0.5, 0.3, 0.3, 0.9]  # Rouge-brun
        elif self.parent.emotion == "joie":
            iris_color = [0.3, 0.5, 0.7, 0.9]  # Bleu plus vif
        elif self.parent.emotion == "tristesse":
            iris_color = [0.4, 0.5, 0.6, 0.8]  # Bleu-gris plus terne

        # PACK 8: Effets spéciaux pour le lien affectif
        if hasattr(self.parent, "lien_eye_shine") and self.parent.lien_eye_shine:
            # Yeux plus brillants avec une teinte spéciale si fort lien
            iris_brightness = 0.1 + 0.05 * math.sin(self.parent.lien_affectif_phase * 0.7)
            iris_color = (
                min(1.0, iris_color[0] + iris_brightness),
                min(1.0, iris_color[1] + iris_brightness * 0.5),
                min(1.0, iris_color[2] + iris_brightness * 0.8),
                min(1.0, iris_color[3] + 0.05),
            )
            # Augmenter légèrement la dilatation des pupilles
            pupil_dilation *= 1.1
        elif hasattr(self.parent, "lien_sad_eyes") and self.parent.lien_sad_eyes:
            # Yeux plus foncés et tristes si le lien est blessé
            iris_color = (
                iris_color[0] * 0.85,
                iris_color[1] * 0.85,
                iris_color[2] * 0.9,
                iris_color[3] * 0.95,
            )
            # Diminuer légèrement la dilatation des pupilles
            pupil_dilation *= 0.9
            # Ajuster le décalage vertical pour un regard plus baissé
            iris_offset_y = -2

        # PACK 5: Modifications des yeux en mode intimité
        if hasattr(self.parent, "intimite_active") and self.parent.intimite_active:
            # Pupilles légèrement plus dilatées
            pupil_dilation *= 1.2
            # Couleur légèrement modifiée (plus brillante)
            iris_color = (
                min(1.0, iris_color[0] + 0.05),
                min(1.0, iris_color[1] + 0.05),
                min(1.0, iris_color[2] + 0.1),
                min(1.0, iris_color[3] + 0.05),
            )
            # Regard légèrement plus bas
            if iris_offset_y >= 0:
                iris_offset_y = -1

        # PACK 4: Effets de fatigue
        if hasattr(self.parent, "fatigue_level") and self.parent.fatigue_level > 0.5:
            # Yeux plus ternes et fatigués
            iris_color = (
                iris_color[0] * 0.9,
                iris_color[1] * 0.9,
                iris_color[2] * 0.9,
                iris_color[3] * 0.9,
            )
            # Pupilles légèrement contractées
            pupil_dilation *= 0.9

        # Calculer la taille finale de la pupille
        adjusted_pupil_size = pupil_size * pupil_dilation

        # Blancs des yeux
        Color(0.95, 0.95, 0.95, 1)

        # Modifier légèrement l'ouverture des yeux selon l'émotion
        eye_scale_y = 1.0

        if self.parent.emotion == "joie":
            eye_scale_y = 0.9  # Yeux légèrement plissés de joie
        elif self.parent.emotion == "surprise":
            eye_scale_y = 1.2  # Yeux grand ouverts
        elif self.parent.emotion == "fatigué":
            eye_scale_y = 0.7  # Yeux fatigués

        # PACK 4: Paupières ajustées pour la fatigue
        if hasattr(self.parent, "fatigue_level"):
            # Réduire l'ouverture des yeux avec la fatigue
            eye_scale_y *= max(0.6, 1.0 - self.parent.fatigue_level * 0.3)

        # Gestion du clignement
        if hasattr(self.parent, "eyelid_openness"):
            eye_scale_y *= self.parent.eyelid_openness

        # Œil gauche (forme ovale)
        adjusted_eye_height = eye_size * eye_scale_y
        left_eye_pos = (left_eye_x - eye_size / 2, eye_y - adjusted_eye_height / 2)
        Ellipse(pos=left_eye_pos, size=(eye_size, adjusted_eye_height))

        # Œil droit (forme ovale)
        right_eye_pos = (right_eye_x - eye_size / 2, eye_y - adjusted_eye_height / 2)
        Ellipse(pos=right_eye_pos, size=(eye_size, adjusted_eye_height))

        # Si les yeux sont suffisamment ouverts, dessiner iris et pupilles
        if eye_scale_y * self.parent.eyelid_openness > 0.3:
            # Iris gauche avec position ajustée
            Color(*iris_color)
            left_iris_pos = (
                left_eye_x - iris_size / 2 + iris_offset_x,
                eye_y - iris_size / 2 + iris_offset_y,
            )
            Ellipse(pos=left_iris_pos, size=(iris_size, iris_size))

            # Iris droit avec position ajustée
            right_iris_pos = (
                right_eye_x - iris_size / 2 + iris_offset_x,
                eye_y - iris_size / 2 + iris_offset_y,
            )
            Ellipse(pos=right_iris_pos, size=(iris_size, iris_size))

            # Pupille gauche
            Color(0.1, 0.1, 0.1, 1)
            left_pupil_pos = (
                left_eye_x - adjusted_pupil_size / 2 + iris_offset_x,
                eye_y - adjusted_pupil_size / 2 + iris_offset_y,
            )
            Ellipse(pos=left_pupil_pos, size=(adjusted_pupil_size, adjusted_pupil_size))

            # Pupille droite
            right_pupil_pos = (
                right_eye_x - adjusted_pupil_size / 2 + iris_offset_x,
                eye_y - adjusted_pupil_size / 2 + iris_offset_y,
            )
            Ellipse(pos=right_pupil_pos, size=(adjusted_pupil_size, adjusted_pupil_size))

            # Reflets dans les yeux (petits points blancs)
            Color(1, 1, 1, 0.8)
            reflection_size = iris_size * 0.2

            # Reflet principal (en haut à droite de la pupille)
            left_reflection_pos = (
                left_eye_x - reflection_size / 2 + iris_offset_x + iris_size * 0.2,
                eye_y - reflection_size / 2 + iris_offset_y + iris_size * 0.2,
            )
            Ellipse(pos=left_reflection_pos, size=(reflection_size, reflection_size))

            right_reflection_pos = (
                right_eye_x - reflection_size / 2 + iris_offset_x + iris_size * 0.2,
                eye_y - reflection_size / 2 + iris_offset_y + iris_size * 0.2,
            )
            Ellipse(pos=right_reflection_pos, size=(reflection_size, reflection_size))

            # Petit reflet secondaire (plus petit, en bas à gauche)
            small_reflection_size = reflection_size * 0.6
            left_small_reflection_pos = (
                left_eye_x - small_reflection_size / 2 + iris_offset_x - iris_size * 0.15,
                eye_y - small_reflection_size / 2 + iris_offset_y - iris_size * 0,
            )
            right_small_reflection_pos = (
                right_eye_x - small_reflection_size / 2 + iris_offset_x - iris_size * 0.15,
                eye_y - small_reflection_size / 2 + iris_offset_y - iris_size * 0,
            )

            Ellipse(pos=left_small_reflection_pos, size=(small_reflection_size, small_reflection_size))
            Ellipse(pos=right_small_reflection_pos, size=(small_reflection_size, small_reflection_size))

    def draw_mouth(self):
        """
        Dessine la bouche en fonction de l'émotion et de la forme actuelle.
        """
        # Position de base de la bouche
        mouth_y = self.parent.center_y - 30

        # Obtenir les facteurs de dimension de la forme actuelle
        current_shape = self.parent.current_mouth_shape
        mouth_shape = self.mouth_shapes.get(current_shape, self.mouth_shapes["X"])
        width_factor = mouth_shape["width_factor"]
        height_factor = mouth_shape["height_factor"]

        # Taille de base de la bouche, ajustée par les facteurs
        mouth_width = 60 * width_factor
        mouth_height = 20 * height_factor

        # Ajustements pour les émotions
        if self.parent.emotion == "joie":
            mouth_width *= 1.2
            mouth_height *= 0.7  # Bouche plus large mais moins haute
        elif self.parent.emotion == "tristesse":
            mouth_width *= 0.8
            mouth_height *= 0.5  # Bouche plus étroite et fine
        elif self.parent.emotion == "surprise":
            mouth_height *= 1.5  # Bouche plus ouverte
        elif self.parent.emotion == "colère":
            mouth_width *= 0.9
            mouth_height *= 0.7  # Bouche plus serrée
            mouth_y -= 5  # Légèrement descendue

        # PACK 4: Ajustements pour les effets spéciaux
        if hasattr(self.parent, "_muscle_tensions_active") and self.parent._muscle_tensions_active:
            if hasattr(self.parent, "_jaw_spasm_offset"):
                # Appliquer le spasme de la mâchoire
                mouth_y += self.parent._jaw_spasm_offset

        if hasattr(self.parent, "_vibration_active") and self.parent._vibration_active:
            if hasattr(self.parent, "_vibration_offset_x"):
                # Appliquer le tremblement latéral
                mouth_x_offset = self.parent._vibration_offset_x
            else:
                mouth_x_offset = 0
        else:
            mouth_x_offset = 0

        # Position finale de la bouche
        mouth_pos = (
            self.parent.center_x - mouth_width / 2 + mouth_x_offset,
            mouth_y - mouth_height / 2,
        )

        # Couleur des lèvres
        Color(0.8, 0.4, 0.4, 0.9)  # Rose modéré

        # Dessin de la bouche selon la forme
        if self.parent.is_speaking:
            # Bouche ouverte avec une forme plus circulaire
            Ellipse(pos=mouth_pos, size=(mouth_width, mouth_height))

            # Intérieur de la bouche (plus sombre)
            Color(0.6, 0.3, 0.3, 0.7)
            inner_mouth_width = mouth_width * 0.8
            inner_mouth_height = mouth_height * 0.8
            inner_mouth_pos = (
                self.parent.center_x - inner_mouth_width / 2 + mouth_x_offset,
                mouth_y - inner_mouth_height / 2,
            )
            Ellipse(pos=inner_mouth_pos, size=(inner_mouth_width, inner_mouth_height))
        else:
            # Bouche fermée ou légèrement entrouverte
            if mouth_height < 5:  # Bouche fermée
                # Ligne simple pour les lèvres fermées
                Line(points=[mouth_pos[0], mouth_y, mouth_pos[0] + mouth_width, mouth_y], width=2)
            else:
                # Bouche légèrement entrouverte
                Ellipse(pos=mouth_pos, size=(mouth_width, mouth_height))

                # Ligne de séparation des lèvres
                Color(0.5, 0.2, 0.2, 0.6)
                Line(
                    points=[
                        mouth_pos[0] + mouth_width * 0.1,
                        mouth_y,
                        mouth_pos[0] + mouth_width * 0.9,
                        mouth_y,
                    ],
                    width=1,
                )

    def draw_energy_particles(self):
        """Dessine les particules d'énergie autour du visage."""
        if hasattr(self.parent, "particles") and self.parent.particles:
            # Parcourir toutes les particules
            for particle in self.parent.particles:
                # Extraire les propriétés de la particule
                x, y = particle.get("pos", (0, 0))
                size = particle.get("size", 2)
                opacity = particle.get("opacity", 0.7)
                color = particle.get("color", [0.8, 0.9, 1.0])

                # Dessiner la particule
                Color(color[0], color[1], color[2], opacity)
                particle_pos = (x - size / 2, y - size / 2)
                Ellipse(pos=particle_pos, size=(size, size))

    def draw_discrete_tears(self):
        """
        PACK 9: Dessine des larmes discrètes pour la blessure émotionnelle.
        """
        with self.parent.canvas:
            # Calculer la phase de mouvement des larmes
            current_time = Clock.get_time()

            # Calculer l'opacité en fonction de l'état de blessure et de l'émotion
            opacity_base = min(0.4, self.parent.blessure_overlay_opacity * 2.5)
            opacity = opacity_base * (0.7 + 0.3 * math.sin(current_time * 0.6))

            # Couleur subtile pour les larmes
            Color(0.8, 0.9, 1.0, opacity)

            # Position de départ sous les yeux
            eye_offset = 40
            tear_start_y = self.parent.center_y + 15  # Juste sous les yeux

            # Larme gauche
            tear_left_x = self.parent.center_x - eye_offset + 3

            # Larme droite
            tear_right_x = self.parent.center_x + eye_offset - 3

            # Calculer une légère oscillation pour le mouvement
            offset_y = 2 * math.sin(current_time * 0.3)

            # Dessiner les larmes (très petites et subtiles)
            tear_size = (3, 6)

            # Larme gauche
            left_tear_pos = (
                tear_left_x - tear_size[0] / 2,
                tear_start_y - tear_size[1] - 6 + offset_y,
            )
            Ellipse(pos=left_tear_pos, size=tear_size)

            # Larme droite (légèrement décalée pour l'asymétrie naturelle)
            right_tear_pos = (
                tear_right_x - tear_size[0] / 2,
                tear_start_y - tear_size[1] - 8 - offset_y,
            )
            Ellipse(pos=right_tear_pos, size=tear_size)

            # Traînée de larme très subtile (gauche)
            Color(0.8, 0.9, 1.0, opacity * 0.5)
            left_trail_height = 10 + 5 * math.sin(current_time * 0.1)
            left_trail_pos = (
                tear_left_x - 1,
                tear_start_y - tear_size[1] - 10 + offset_y - left_trail_height,
            )
            Line(
                points=[
                    tear_left_x,
                    tear_start_y - tear_size[1] - 6 + offset_y,
                    tear_left_x,
                    tear_start_y - tear_size[1] - 6 + offset_y - left_trail_height,
                ],
                width=0.5,
            )

    def draw_enhanced_cheeks(self, left=False, right=False, intensity=0.5):
        """
        Dessine des joues avec une intensité renforcée pour des effets spéciaux.

        Args:
            left: Si True, dessine la joue gauche renforcée
            right: Si True, dessine la joue droite renforcée
            intensity: Intensité de l'effet (0.0 à 1.0)
        """
        # Position des joues
        cheek_offset_x = 55
        cheek_offset_y = -15

        # Taille ajustée à l'intensité
        cheek_size = 30 * (1 + intensity * 0.5)

        # Couleur rose plus vive avec opacité basée sur l'intensité
        Color(0.95, 0.5, 0.5, intensity * 0.6)

        # Dessiner les joues renforcées selon les paramètres
        if left:
            left_cheek_pos = (
                self.parent.center_x - cheek_offset_x - cheek_size / 2,
                self.parent.center_y + cheek_offset_y - cheek_size / 2,
            )
            Ellipse(pos=left_cheek_pos, size=(cheek_size, cheek_size))

        if right:
            right_cheek_pos = (
                self.parent.center_x + cheek_offset_x - cheek_size / 2,
                self.parent.center_y + cheek_offset_y - cheek_size / 2,
            )
            Ellipse(pos=right_cheek_pos, size=(cheek_size, cheek_size))
