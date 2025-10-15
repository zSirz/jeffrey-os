"""
Configuration de connexion robuste pour NeuralBus
"""

import os
from typing import Any


def get_robust_connection_options() -> dict[str, Any]:
    """Retourne les options de connexion NATS robustes"""

    # Récupérer le namespace de la session
    namespace = os.getenv("NB_NS", "dev")

    return {
        "servers": [os.getenv("NATS_URL", "nats://127.0.0.1:4222")],
        # Reconnexion infinie avec backoff
        "reconnect_time_wait": 1,  # 1s entre tentatives
        "max_reconnect_attempts": -1,  # Infini
        # Health checks
        "ping_interval": 10,  # Ping toutes les 10s
        "max_outstanding_pings": 3,  # Max 3 pings sans réponse
        # Timeouts
        "drain_timeout": 2,  # 2s pour drain
        "flush_timeout": 2,  # 2s pour flush
        "connect_timeout": 5,  # 5s pour connexion initiale
        # Autres options
        "allow_reconnect": True,
        "verbose": os.getenv("NATS_VERBOSE", "false").lower() == "true",
        "pedantic": False,
        "name": f"jeffrey_{namespace}",
        # User data pour tracking
        "user_data": {"namespace": namespace, "component": "neuralbus"},
    }


def get_subject_with_namespace(base_subject: str) -> str:
    """Ajoute le namespace au subject"""
    namespace = os.getenv("NB_NS", "dev")
    return f"{namespace}.{base_subject}"
