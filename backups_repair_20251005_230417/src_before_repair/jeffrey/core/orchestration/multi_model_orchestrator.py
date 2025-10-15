"""
Orchestrateur principal du système cognitif.

Ce module implémente les fonctionnalités essentielles pour orchestrateur principal du système cognitif.
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

import asyncio

from core.conversation.conversation_memory import ConversationMemory
from core.orchestration.clients_factory import build_ia_clients
from core.orchestration.fusion_engine import fuse_responses
from core.orchestration.prompt_manager import PromptManager


class MultiModelOrchestrator:
    """
    Classe MultiModelOrchestrator pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self, ia_clients: dict[str, object] = None) -> None:
        """
        Args:
            ia_clients (dict, optional): Dictionnaire {nom_modèle: client_api_instance}.
            Si None, les clients sont automatiquement construits via la factory.
        """
        if ia_clients is None:
            self.ia_clients = build_ia_clients()
        else:
            self.ia_clients = ia_clients

        self.prompt_manager = PromptManager()
        self.memory = ConversationMemory()

    async def send_to_model(self, model_name: str, prompt: str) -> dict[str, str]:
        """Envoie le prompt à un modèle spécifique et récupère la réponse."""
        try:
            client = self.ia_clients.get(model_name)
            if client is None:
                return {"model": model_name, "response": "Client non configuré."}

            optimized_prompt = self.prompt_manager.optimize_prompt(model_name, prompt)
            response = await client.ask(optimized_prompt)
            return {"model": model_name, "response": response}

        except Exception as e:
            return {"model": model_name, "response": f"Erreur: {str(e)}"}

    async def send_to_all_models(self, prompt: str) -> list[dict[str, str]]:
        """Envoie la même requête à tous les modèles en parallèle."""
        tasks = [self.send_to_model(model, prompt) for model in self.ia_clients]
        results = await asyncio.gather(*tasks)
        return results

    def ask_all(self, prompt: str) -> list[dict[str, str]]:
        """Méthode synchrone pour interagir avec l'orchestrateur."""
        return asyncio.run(self.send_to_all_models(prompt))

    def ask_all_fused(self, prompt: str) -> str:
        """
        Envoie une requête à toutes les IA et retourne une réponse fusionnée.

        Args:
            prompt (str): Demande de l'utilisateur.

        Returns:
            str: Réponse fusionnée finale.
        """
        raw_responses = asyncio.run(self.send_to_all_models(prompt))
        fused_response = fuse_responses(raw_responses)
        self.memory.add_interaction(prompt, raw_responses, fused_response)
        return fused_response

    def show_conversation_history(self) -> None:
        """
        Affiche l'historique complet de la session de conversation.
        """
        history = self.memory.get_full_history()
        if not history:
            print("\n📜 Aucun historique disponible pour cette session.\n")
            return

        print("\n📜 Historique de la session :\n")
        for idx, interaction in enumerate(history, start=1):
            print(f"--- Interaction {idx} ---")
            print(f"📝 Prompt utilisateur : {interaction['prompt']}")
            print("🤖 Réponses individuelles :")
            for r in interaction["responses"]:
                print(f"   - {r['model']}: {r['response']}")
            print(f"🎯 Réponse fusionnée : {interaction['fused_response']}")
            print("")

    def save_conversation_history(self, file_path: str = "conversation_history.json") -> None:
        """
        Sauvegarde l'historique de la session dans un fichier JSON.

        Args:
            file_path (str): Chemin du fichier où sauvegarder l'historique.
        """
        import json

        history = self.memory.get_full_history()
        if not history:
            print("\n📦 Aucun historique à sauvegarder.\n")
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4, ensure_ascii=False)
            print(f"\n✅ Historique sauvegardé dans : {file_path}\n")
        except Exception as e:
            print(f"\n❌ Erreur lors de la sauvegarde de l'historique : {e}\n")

    def load_conversation_history(self, file_path: str = "conversation_history.json") -> None:
        """
        Recharge l'historique d'une session sauvegardée depuis un fichier JSON.

        Args:
            file_path (str): Chemin du fichier JSON à charger.
        """
        import json
        import os

        if not os.path.exists(file_path):
            print(f"\n⚠️ Fichier introuvable : {file_path}\n")
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                loaded_history = json.load(f)

            if isinstance(loaded_history, list):
                self.memory.history = loaded_history
                print(f"\n✅ Historique chargé depuis : {file_path}\n")
            else:
                print(f"\n❌ Format d'historique invalide dans : {file_path}\n")

        except Exception as e:
            print(f"\n❌ Erreur lors du chargement de l'historique : {e}\n")

    def analyze_conversation_history(self) -> dict[str, int]:
        """
        Analyse l'historique de conversation pour extraire des statistiques simples :
        - nombre d'interactions
        - nombre de mots par prompt
        - nombre de mots par réponse fusionnée

        Returns:
            dict: Résultats de l'analyse.
        """
        history = self.memory.get_full_history()
        if not history:
            print("\n📊 Aucun historique à analyser.\n")
            return {}

        total_prompts = len(history)
        total_prompt_words = sum(len(interaction["prompt"].split()) for interaction in history)
        total_response_words = sum(len(interaction["fused_response"].split()) for interaction in history)

        stats = {
            "total_interactions": total_prompts,
            "total_prompt_words": total_prompt_words,
            "total_response_words": total_response_words,
            "average_prompt_length": (total_prompt_words / total_prompts if total_prompts else 0),
            "average_response_length": (total_response_words / total_prompts if total_prompts else 0),
        }

        print("\n📊 Statistiques de la session :")
        for key, value in stats.items():
            print(f"- {key.replace('_', ' ').capitalize()}: {value}")

        return stats
