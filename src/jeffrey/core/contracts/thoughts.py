"""
Thought Contract - Format standardisé pour toutes les pensées
Idée de GPT pour éviter les surprises de format
"""
from typing import TypedDict, List, Any, Optional
from enum import Enum
from datetime import datetime

class ThoughtState(Enum):
    """États possibles d'une pensée"""
    AWARE = "aware"
    REFLECTIVE = "reflective"
    DREAMING = "dreaming"
    CONFUSED = "confused"
    CREATIVE = "creative"
    PROCESSING = "processing"
    RESTING = "resting"

class Thought(TypedDict, total=False):
    """
    Contrat standard pour une pensée générée par la conscience
    TypedDict permet validation et auto-completion
    """
    # Champs obligatoires
    state: str  # ThoughtState value
    timestamp: str

    # Champs optionnels mais recommandés
    context_size: int
    mode: str
    summary: str
    confidence: float

    # Métadonnées enrichies
    emotion_context: str
    memory_references: List[str]
    related_thoughts: List[str]

    # Pour debug et monitoring
    processing_time_ms: float
    source_module: str

def create_thought(
    state: ThoughtState = ThoughtState.AWARE,
    summary: str = "",
    **kwargs
) -> Thought:
    """Factory pour créer une pensée valide"""

    thought = {
        "state": state.value,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary,
        "source_module": kwargs.get("source_module", "consciousness_engine"),
        **kwargs
    }
    return thought

def validate_thought(thought: Any) -> bool:
    """Valide qu'un objet respecte le contrat Thought"""
    if not isinstance(thought, dict):
        return False

    # Vérifier les champs obligatoires
    required = ["state", "timestamp"]
    return all(field in thought for field in required)

def ensure_thought_format(thought_data: Any) -> Thought:
    """
    Garantit qu'une donnée respecte le format Thought
    Convertit ou crée une pensée valide si nécessaire
    """
    if validate_thought(thought_data):
        return thought_data

    # Si c'est un dict mais incomplet, essayer de l'enrichir
    if isinstance(thought_data, dict):
        return create_thought(
            state=ThoughtState.PROCESSING,
            summary=thought_data.get("summary", "Processed thought"),
            **{k: v for k, v in thought_data.items() if k not in ["state", "timestamp"]}
        )

    # Si ce n'est pas un dict, créer une pensée de base
    return create_thought(
        state=ThoughtState.AWARE,
        summary=str(thought_data) if thought_data else "Empty thought"
    )