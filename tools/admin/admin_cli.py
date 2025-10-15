#!/usr/bin/env python3
"""
Interface d'administration CLI pour l'Orchestrateur IA (Jeffrey)

Ce module fournit une interface en ligne de commande pour gérer:
- Le registre des modèles d'IA
- Les quotas et crédits
- Les profils vocaux et d'émotions
"""

import argparse
import json
import sys
from typing import Any

import yaml
from core.config import Config

# Import des modules de l'Orchestrateur
from credit_system.credit_manager import CreditManager


class IARegistryManager:
    """
    Gestionnaire de registre des modèles IA.
    Permet de lister, activer/désactiver, modifier et ajouter des modèles au registre.
    """

    def __init__(self, registry_path: str = "ia_registry.yaml"):
        """Initialise le gestionnaire de registre."""
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Charge le registre des modèles IA depuis le fichier YAML."""
        try:
            with open(self.registry_path, encoding='utf-8') as f:
                registry = yaml.safe_load(f)
                return registry if registry else {"models": {}}
        except FileNotFoundError:
            print(f"Le fichier de registre {self.registry_path} n'existe pas. Création d'un nouveau registre.")
            return {"models": {}}
        except Exception as e:
            print(f"Erreur lors du chargement du registre: {e}")
            sys.exit(1)

    def _save_registry(self) -> bool:
        """Sauvegarde le registre dans le fichier YAML."""
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.registry, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du registre: {e}")
            return False

    def list_models(self, detailed: bool = False) -> list[dict[str, Any]]:
        """
        Liste tous les modèles disponibles.

        Args:
            detailed: Si True, affiche les informations détaillées pour chaque modèle

        Returns:
            Liste des modèles avec leurs informations
        """
        models = self.registry.get("models", {})
        model_list = []

        for model_id, model_info in models.items():
            # Extraire les informations de base
            model_data = {
                "id": model_id,
                "name": model_info.get("name", model_id),
                "provider": model_info.get("provider", "Unknown"),
                "eco_score": model_info.get("eco_score", "N/A"),
                "status": model_info.get("status", "active"),
            }

            # Ajouter les métriques si demandé
            if detailed:
                metrics = model_info.get("metrics", {})
                model_data["metrics"] = metrics
                model_data["description"] = model_info.get("description", "")
                model_data["capabilities"] = model_info.get("capabilities", [])

            model_list.append(model_data)

        return model_list

    def change_model_status(self, model_id: str, activate: bool) -> bool:
        """
        Active ou désactive un modèle.

        Args:
            model_id: Identifiant du modèle à modifier
            activate: True pour activer, False pour désactiver

        Returns:
            True si réussi, False sinon
        """
        models = self.registry.get("models", {})

        if model_id not in models:
            print(f"Modèle {model_id} non trouvé dans le registre.")
            return False

        status = "active" if activate else "disabled"
        models[model_id]["status"] = status

        if self._save_registry():
            print(f"Modèle {model_id} {'activé' if activate else 'désactivé'} avec succès.")
            return True
        return False

    def update_model_cost(self, model_id: str, cost_per_token: float) -> bool:
        """
        Modifie le coût par token d'un modèle.

        Args:
            model_id: Identifiant du modèle à modifier
            cost_per_token: Nouveau coût par token (0 à 1)

        Returns:
            True si réussi, False sinon
        """
        models = self.registry.get("models", {})

        if model_id not in models:
            print(f"Modèle {model_id} non trouvé dans le registre.")
            return False

        # Vérifier que le coût est dans les limites valides
        if cost_per_token < 0 or cost_per_token > 1:
            print("Le coût par token doit être entre 0 et 1.")
            return False

        # Mettre à jour le coût dans les métriques
        if "metrics" not in models[model_id]:
            models[model_id]["metrics"] = {}

        models[model_id]["metrics"]["cost"] = cost_per_token

        if self._save_registry():
            print(f"Coût du modèle {model_id} mis à jour à {cost_per_token}.")
            return True
        return False

    def add_model(
        self,
        model_id: str,
        name: str,
        provider: str,
        category: str = "general",
        status: str = "active",
        metrics: dict[str, float] | None = None,
    ) -> bool:
        """
        Ajoute un nouveau modèle au registre.

        Args:
            model_id: Identifiant du modèle
            name: Nom du modèle
            provider: Fournisseur du modèle
            category: Catégorie du modèle
            status: Statut initial du modèle
            metrics: Métriques du modèle

        Returns:
            True si réussi, False sinon
        """
        models = self.registry.get("models", {})

        if model_id in models:
            print(f"Modèle {model_id} existe déjà dans le registre.")
            return False

        # Créer l'entrée du modèle
        models[model_id] = {
            "name": name,
            "provider": provider,
            "description": f"Modèle {category} de {provider}",
            "category": category,
            "status": status,
            "metrics": metrics or {"quality": 0.7, "latency": 0.7, "cost": 0.7, "stability": 0.7},
            "capabilities": [category, "text-generation"],
            "eco_score": 0.5,
        }

        if self._save_registry():
            print(f"Modèle {model_id} ajouté avec succès.")
            return True
        return False

    def export_to_json(self, output_path: str = "ia_registry_export.json") -> bool:
        """
        Exporte le registre au format JSON.

        Args:
            output_path: Chemin du fichier JSON de sortie

        Returns:
            True si réussi, False sinon
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
            print(f"Registre exporté avec succès vers {output_path}")
            return True
        except Exception as e:
            print(f"Erreur lors de l'exportation du registre: {e}")
            return False


class CreditQuotaManager:
    """
    Gestionnaire de quotas et crédits pour les modèles IA.
    """

    def __init__(self):
        """Initialise le gestionnaire de crédits."""
        self.credit_manager = CreditManager()

    def get_credit_balance(self) -> int:
        """
        Récupère le solde de crédits actuel.

        Returns:
            Solde de crédits
        """
        return self.credit_manager.get_balance()

    def reset_credits(self, amount: int = None) -> bool:
        """
        Réinitialise les crédits à un montant spécifié.

        Args:
            amount: Nouveau montant de crédits (utilise Config.DEFAULT_INITIAL_CREDITS si None)

        Returns:
            True si réussi
        """
        # Réinitialisation en créant une nouvelle instance avec le montant spécifié
        self.credit_manager = CreditManager(initial_credits=amount)
        print(f"Crédits réinitialisés à {self.credit_manager.get_balance()}.")
        return True

    def add_credits(self, amount: int) -> bool:
        """
        Ajoute des crédits au solde actuel.

        Args:
            amount: Montant de crédits à ajouter

        Returns:
            True si réussi
        """
        if amount <= 0:
            print("Le montant de crédits à ajouter doit être positif.")
            return False

        # Ajouter les crédits au balance actuel
        self.credit_manager.balance += amount
        print(f"Ajout de {amount} crédits. Nouveau solde: {self.credit_manager.get_balance()}.")
        return True

    def get_credit_status(self) -> dict[str, Any]:
        """
        Obtient le statut complet des crédits.

        Returns:
            Dictionnaire avec les informations sur les crédits
        """
        return {
            "balance": self.credit_manager.get_balance(),
            "reserved": self.credit_manager.reserved,
            "default_initial": Config.DEFAULT_INITIAL_CREDITS,
        }


def main():
    """Fonction principale de l'interface d'administration CLI."""
    parser = argparse.ArgumentParser(description="Interface d'administration pour l'Orchestrateur IA")

    # Créer des sous-parseurs pour les différentes commandes
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # Commande: list
    list_parser = subparsers.add_parser("list", help="Lister les modèles IA disponibles")
    list_parser.add_argument("--detailed", "-d", action="store_true", help="Afficher les informations détaillées")
    list_parser.add_argument("--json", "-j", action="store_true", help="Sortie au format JSON")

    # Commande: status
    status_parser = subparsers.add_parser("status", help="Changer le statut d'un modèle")
    status_parser.add_argument("model_id", help="ID du modèle")
    status_group = status_parser.add_mutually_exclusive_group(required=True)
    status_group.add_argument("--enable", action="store_true", help="Activer le modèle")
    status_group.add_argument("--disable", action="store_true", help="Désactiver le modèle")

    # Commande: cost
    cost_parser = subparsers.add_parser("cost", help="Modifier le coût par token d'un modèle")
    cost_parser.add_argument("model_id", help="ID du modèle")
    cost_parser.add_argument("cost", type=float, help="Nouveau coût par token (0-1)")

    # Commande: add
    add_parser = subparsers.add_parser("add", help="Ajouter un nouveau modèle")
    add_parser.add_argument("model_id", help="ID du modèle")
    add_parser.add_argument("name", help="Nom du modèle")
    add_parser.add_argument("provider", help="Fournisseur du modèle")
    add_parser.add_argument("--category", "-c", default="general", help="Catégorie du modèle")
    add_parser.add_argument(
        "--status", "-s", choices=["active", "disabled"], default="active", help="Statut initial du modèle"
    )

    # Commande: export
    export_parser = subparsers.add_parser("export", help="Exporter le registre au format JSON")
    export_parser.add_argument("--output", "-o", default="ia_registry_export.json", help="Chemin du fichier de sortie")

    # Commande: credits
    credits_parser = subparsers.add_parser("credits", help="Gérer les crédits et quotas")
    credits_subparsers = credits_parser.add_subparsers(dest="credits_command", help="Commande de gestion des crédits")

    # Sous-commande: credits status
    credits_status_parser = credits_subparsers.add_parser("status", help="Afficher le statut des crédits")
    credits_status_parser.add_argument("--json", "-j", action="store_true", help="Sortie au format JSON")

    # Sous-commande: credits reset
    credits_reset_parser = credits_subparsers.add_parser("reset", help="Réinitialiser les crédits")
    credits_reset_parser.add_argument(
        "--amount", "-a", type=int, help="Montant à définir (utilise la valeur par défaut si non spécifié)"
    )

    # Sous-commande: credits add
    credits_add_parser = credits_subparsers.add_parser("add", help="Ajouter des crédits")
    credits_add_parser.add_argument("amount", type=int, help="Montant de crédits à ajouter")

    # Traiter les arguments
    args = parser.parse_args()

    # Initialiser les gestionnaires
    registry_manager = IARegistryManager()
    credit_manager = CreditQuotaManager()

    # Exécuter la commande appropriée
    if args.command == "list":
        models = registry_manager.list_models(detailed=args.detailed)
        if args.json:
            print(json.dumps(models, indent=2))
        else:
            # Affichage formaté en tableau
            if not models:
                print("Aucun modèle trouvé dans le registre.")
                return

            # Déterminer les colonnes à afficher
            if args.detailed:
                headers = ["ID", "Nom", "Fournisseur", "Éco-score", "Statut", "Qualité", "Latence", "Coût", "Stabilité"]

                # Formater les données
                rows = []
                for model in models:
                    metrics = model.get("metrics", {})
                    row = [
                        model["id"],
                        model["name"],
                        model["provider"],
                        model.get("eco_score", "N/A"),
                        model.get("status", "active"),
                        metrics.get("quality", "N/A"),
                        metrics.get("latency", "N/A"),
                        metrics.get("cost", "N/A"),
                        metrics.get("stability", "N/A"),
                    ]
                    rows.append(row)
            else:
                headers = ["ID", "Nom", "Fournisseur", "Éco-score", "Statut"]

                # Formater les données
                rows = []
                for model in models:
                    row = [
                        model["id"],
                        model["name"],
                        model["provider"],
                        model.get("eco_score", "N/A"),
                        model.get("status", "active"),
                    ]
                    rows.append(row)

            # Afficher le tableau
            # Calculer la largeur de chaque colonne
            col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]

            # Afficher les en-têtes
            header_row = " | ".join(f"{headers[i]:{col_widths[i]}}" for i in range(len(headers)))
            print(header_row)
            print("-" * len(header_row))

            # Afficher les données
            for row in rows:
                print(" | ".join(f"{str(row[i]):{col_widths[i]}}" for i in range(len(row))))

    elif args.command == "status":
        registry_manager.change_model_status(args.model_id, args.enable)

    elif args.command == "cost":
        registry_manager.update_model_cost(args.model_id, args.cost)

    elif args.command == "add":
        registry_manager.add_model(args.model_id, args.name, args.provider, args.category, args.status)

    elif args.command == "export":
        registry_manager.export_to_json(args.output)

    elif args.command == "credits":
        if args.credits_command == "status":
            status = credit_manager.get_credit_status()
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"Solde actuel: {status['balance']} crédits")
                print(f"Crédits réservés: {status['reserved']} crédits")
                print(f"Valeur initiale par défaut: {status['default_initial']} crédits")

        elif args.credits_command == "reset":
            credit_manager.reset_credits(args.amount)

        elif args.credits_command == "add":
            credit_manager.add_credits(args.amount)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
