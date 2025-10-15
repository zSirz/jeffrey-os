"""
jeffrey.ml - Module d'apprentissage automatique pour Jeffrey OS Phase 1
Système de détection émotionnelle auto-apprenant avec architecture encoder/proto/feedback.
"""

from .encoder import SentenceEncoder, create_default_encoder
from .feedback import FeedbackEvent, FeedbackStore
from .proto import EmotionPrediction, ProtoClassifier

__all__ = [
    'SentenceEncoder',
    'create_default_encoder',
    'ProtoClassifier',
    'EmotionPrediction',
    'FeedbackStore',
    'FeedbackEvent',
]
