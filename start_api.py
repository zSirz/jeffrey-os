#!/usr/bin/env python3
"""
Script de démarrage de l'API Jeffrey
Lance l'API REST avec le BrainKernel
"""

import os
import sys
from pathlib import Path

# Désactiver Kivy pour l'API
os.environ["KIVY_NO_ARGS"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "error"

# Setup path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Lance l'API Jeffrey"""
    print(
        """
    ╔════════════════════════════════════╗
    ║                                    ║
    ║    🧠 JEFFREY BRAIN API            ║
    ║       Version 2.0                  ║
    ║                                    ║
    ║    http://localhost:8000           ║
    ║    http://localhost:8000/docs      ║
    ║                                    ║
    ╚════════════════════════════════════╝
    """
    )

    try:
        import uvicorn
    except ImportError:
        print("❌ Erreur: uvicorn n'est pas installé")
        print("Installez-le avec: pip install uvicorn")
        sys.exit(1)

    try:
        # Vérifier que l'API existe
        from jeffrey.api.jeffrey_api import app
    except ImportError as e:
        print(f"❌ Erreur: Impossible d'importer l'API Jeffrey: {e}")
        print("Assurez-vous que src/jeffrey/api/jeffrey_api.py existe")
        sys.exit(1)

    uvicorn.run("src.jeffrey.api.jeffrey_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    main()
