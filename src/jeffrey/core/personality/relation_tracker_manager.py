# core/personality/relation_tracker_manager.py

from jeffrey.core.personality.relation_tracker import RelationTracker

# Singleton global du gestionnaire de relation
_relation_tracker_instance = None


def get_relation_tracker():
    """Retourne l’instance unique du RelationTracker."""
    global _relation_tracker_instance
    if _relation_tracker_instance is None:
        _relation_tracker_instance = RelationTracker()
    return _relation_tracker_instance


def enregistrer_interaction(event_type: str, value: float = 1.0):
    """Enregistre une interaction avec Jeffrey et met à jour la relation."""
    tracker = get_relation_tracker()
    tracker.update_relationship(event_type, value)


def get_niveau_relation():
    """Retourne le niveau actuel de la relation (entre 0 et 1)."""
    tracker = get_relation_tracker()
    return tracker.get_relation_level()


def get_profil_relation():
    """Retourne une description textuelle du lien actuel."""
    tracker = get_relation_tracker()
    return tracker.describe_relation()
