"""
Module: jeffrey.widgets
Description: Composants visuels et effets pour l'interface émotionnelle de Jeffrey

Ce module contient les widgets Kivy pour l'affichage des émotions :
- EmotionParticle : Classe de particule individuelle
- EmotionParticleEmitter : Système d'émission de particules émotionnelles
- EmotionParticles : Widget d'intégration pour les écrans
"""

from jeffrey.widgets.emotion_particles import (
    EmotionParticle,
    EmotionParticleEmitter,
    EmotionParticles,
)

__all__ = [
    "EmotionParticle",
    "EmotionParticleEmitter",
    "EmotionParticles",
]

__version__ = "1.0.0"
