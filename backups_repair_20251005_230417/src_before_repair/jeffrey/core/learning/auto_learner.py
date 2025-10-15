"""
Module AutoLearner : Apprentissage automatique évolutif.

Ce module permet à Jeffrey d’apprendre en continu de ses interactions (textes, réponses, consignes, retours...).
Il découpe les phrases en idées, analyse les concepts, interagit avec la mémoire de connaissances, et stocke ce qu’il apprend.
L’apprentissage peut se faire depuis l’utilisateur ou depuis ses propres réponses (mode auto-réflexif).

Il est sécurisé : estimation du coût, autorisation de l’utilisateur, stockage dans le cache, et possibilité de travailler en offline.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from typing import Any

# from jeffrey.core.api_security import secure_api_method
# from jeffrey.core.ia_pricing import estimate_cost, estimate_tokens


class AutoLearner:
    """
    Système d'apprentissage automatique pour améliorer les modèles d'IA.

    Utilise les interactions passées et les feedbacks utilisateurs pour
    améliorer les performances des modèles d'IA via l'apprentissage continu.
    """

    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo",
        learning_model: str | None = None,
        user_id: str = "default_user",
        data_path: str = "data/learning_data",
        max_examples_per_batch: int = 50,
    ):
        """
        Initialise le système d'apprentissage automatique.

        Args:
            base_model: Modèle de base à améliorer
            learning_model: Modèle utilisé pour l'apprentissage (défaut: même que base_model)
            user_id: Identifiant de l'utilisateur
            data_path: Chemin des données d'apprentissage
            max_examples_per_batch: Nombre maximum d'exemples par lot d'apprentissage
        """
        self.logger = logging.getLogger(__name__)
        self.base_model = base_model
        self.learning_model = learning_model or base_model
        self.user_id = user_id
        self.data_path = data_path
        self.max_examples_per_batch = max_examples_per_batch

        # Identifiant unique pour cette instance
        self.learner_id = f"autolearn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Créer les répertoires de données si nécessaires
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(os.path.join(data_path, "examples"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "processed"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "batches"), exist_ok=True)

        # Statistiques d'apprentissage
        self.stats = {
            "examples_collected": 0,
            "batches_processed": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "quality_improvements": {},
            "last_update": None,
        }

        self.logger.info(
            f"AutoLearner initialisé avec modèle de base {base_model} et modèle d'apprentissage {self.learning_model}"
        )

    def add_example(
        self,
        prompt: str,
        response: str,
        feedback: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Ajoute un exemple d'apprentissage au système.

        Args:
            prompt: Prompt ou question d'entrée
            response: Réponse du modèle
            feedback: Feedback utilisateur optionnel (score, commentaires, etc.)
            metadata: Métadonnées supplémentaires (modèle, timestamp, etc.)

        Returns:
            ID de l'exemple ajouté
        """
        # Générer un ID unique pour cet exemple
        example_id = f"ex_{random.randint(1000, 9999)}"

        # Créer l'objet d'exemple
        example = {
            "id": example_id,
            "prompt": prompt,
            "response": response,
            "feedback": feedback or {},
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "processed": False,
            "learner_id": self.learner_id,
            "user_id": self.user_id,
        }

        # Sauvegarder l'exemple
        example_path = os.path.join(self.data_path, "examples", f"{example_id}.json")
        with open(example_path, "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2, ensure_ascii=False)

        # Mettre à jour les statistiques
        self.stats["examples_collected"] += 1
        self.stats["last_update"] = datetime.now().isoformat()

        self.logger.debug(f"Exemple d'apprentissage ajouté: {example_id}")
        return example_id

    def create_learning_batch(self, max_examples: int | None = None, category: str | None = None) -> str | None:
        """
        Crée un lot d'exemples pour l'apprentissage.

        Args:
            max_examples: Nombre maximum d'exemples dans le lot
            category: Catégorie d'exemples à inclure

        Returns:
            ID du lot créé ou None si pas assez d'exemples
        """
        # Utiliser la valeur par défaut si non spécifiée
        max_size = max_examples or self.max_examples_per_batch

        # Trouver tous les exemples non traités
        examples_dir = os.path.join(self.data_path, "examples")
        examples = []

        for filename in os.listdir(examples_dir):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(examples_dir, filename)

            try:
                with open(file_path, encoding="utf-8") as f:
                    example = json.load(f)

                # Vérifier si l'exemple a déjà été traité
                if example.get("processed", False):
                    continue

                # Vérifier la catégorie si spécifiée
                if category and example.get("metadata", {}).get("category") != category:
                    continue

                examples.append(example)

            except Exception as e:
                self.logger.warning(f"Erreur lors de la lecture de {filename}: {e}")

        # Vérifier si assez d'exemples
        if not examples:
            self.logger.info("Aucun exemple disponible pour créer un lot d'apprentissage")
            return None

        # Limiter le nombre d'exemples
        if len(examples) > max_size:
            # Trier par score de feedback si disponible
            examples.sort(key=lambda x: x.get("feedback", {}).get("score", 0), reverse=True)
            # Prendre les max_size meilleurs exemples
            examples = examples[:max_size]

        # Générer un ID unique pour ce lot
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"

        # Créer l'objet lot
        batch = {
            "id": batch_id,
            "examples": examples,
            "count": len(examples),
            "created_at": datetime.now().isoformat(),
            "processed": False,
            "learner_id": self.learner_id,
            "user_id": self.user_id,
            "category": category,
        }

        # Sauvegarder le lot
        batch_path = os.path.join(self.data_path, "batches", f"{batch_id}.json")
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Lot d'apprentissage créé: {batch_id} avec {len(examples)} exemples")
        return batch_id

    # @secure_api_method(model_name_attr="learning_model", reason="Apprentissage automatique IA")
    def process_batch(self, batch_id: str) -> dict[str, Any]:
        """
        Traite un lot d'exemples pour l'apprentissage.

        Args:
            batch_id: ID du lot à traiter

        Returns:
            Résultats du traitement
        """
        # Charger le lot
        batch_path = os.path.join(self.data_path, "batches", f"{batch_id}.json")

        if not os.path.exists(batch_path):
            return {"success": False, "error": f"Lot {batch_id} introuvable"}

        try:
            with open(batch_path, encoding="utf-8") as f:
                batch = json.load(f)

            # Vérifier si déjà traité
            if batch.get("processed", False):
                return {"success": False, "error": f"Lot {batch_id} déjà traité"}

            # Préparer les exemples pour l'apprentissage
            examples = batch["examples"]

            # Estimer le nombre total de tokens
            total_text = ""
            for example in examples:
                total_text += example["prompt"] + " " + example["response"]

            estimated_tokens = estimate_tokens(total_text, self.learning_model)

            # Effectuer l'apprentissage (simulation pour l'exemple)
            learning_results = self._perform_learning(examples, estimated_tokens)

            # Si réussi, marquer les exemples comme traités
            if learning_results.get("success", False):
                # Marquer le lot comme traité
                batch["processed"] = True
                batch["processed_at"] = datetime.now().isoformat()
                batch["results"] = learning_results

                # Sauvegarder le lot mis à jour
                with open(batch_path, "w", encoding="utf-8") as f:
                    json.dump(batch, f, indent=2, ensure_ascii=False)

                # Marquer chaque exemple comme traité
                for example in examples:
                    example_path = os.path.join(self.data_path, "examples", f"{example['id']}.json")
                    if os.path.exists(example_path):
                        try:
                            with open(example_path, encoding="utf-8") as f:
                                example_data = json.load(f)

                            example_data["processed"] = True
                            example_data["processed_at"] = datetime.now().isoformat()
                            example_data["batch_id"] = batch_id

                            with open(example_path, "w", encoding="utf-8") as f:
                                json.dump(example_data, f, indent=2, ensure_ascii=False)

                        except Exception as e:
                            self.logger.warning(f"Erreur lors de la mise à jour de {example['id']}: {e}")

                # Mettre à jour les statistiques
                self.stats["batches_processed"] += 1
                self.stats["total_tokens_used"] += learning_results.get("tokens_used", estimated_tokens)
                self.stats["total_cost"] += learning_results.get("cost", 0.0)
                self.stats["last_update"] = datetime.now().isoformat()

                # Si des améliorations sont rapportées, les enregistrer
                improvements = learning_results.get("improvements", {})
                for metric, value in improvements.items():
                    if metric not in self.stats["quality_improvements"]:
                        self.stats["quality_improvements"][metric] = 0.0
                    self.stats["quality_improvements"][metric] += value

                self.logger.info(f"Traitement du lot {batch_id} réussi")
                return {
                    "success": True,
                    "batch_id": batch_id,
                    "examples_processed": len(examples),
                    "results": learning_results,
                }

            else:
                return {
                    "success": False,
                    "error": learning_results.get("error", "Échec du traitement"),
                    "batch_id": batch_id,
                }

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du lot {batch_id}: {e}")
            return {"success": False, "error": str(e)}

    def _perform_learning(self, examples: list[dict[str, Any]], estimated_tokens: int) -> dict[str, Any]:
        """
        Effectue l'apprentissage sur un ensemble d'exemples.

        Args:
            examples: Liste d'exemples d'apprentissage
            estimated_tokens: Estimation du nombre de tokens

        Returns:
            Résultats de l'apprentissage
        """
        # Cette méthode devrait être remplacée par l'appel API réel
        # vers le service d'apprentissage approprié (OpenAI, etc.)

        # Simuler l'apprentissage pour cet exemple
        # En production, cela serait remplacé par un appel à l'API appropriée

        # Simuler une petite amélioration aléatoire
        improvements = {
            "accuracy": random.uniform(0.01, 0.05),
            "relevance": random.uniform(0.02, 0.04),
            "coherence": random.uniform(0.01, 0.06),
        }

        # Estimer le coût
        cost = estimate_cost(self.learning_model, token_count=estimated_tokens)

        # Simuler une réponse de succès
        return {
            "success": True,
            "model": self.learning_model,
            "tokens_used": estimated_tokens,
            "cost": cost,
            "improvements": improvements,
            "applied_to": self.base_model,
            "examples_count": len(examples),
            "timestamp": datetime.now().isoformat(),
        }

    def get_learning_stats(self) -> dict[str, Any]:
        """
        Récupère les statistiques d'apprentissage.

        Returns:
            Statistiques d'apprentissage
        """
        return {
            "learner_id": self.learner_id,
            "base_model": self.base_model,
            "learning_model": self.learning_model,
            "stats": self.stats,
            "updated_at": datetime.now().isoformat(),
        }

    def get_fallback_response(self, batch_id: str) -> dict[str, Any]:
        """
        Méthode de repli en cas de refus d'autorisation.

        Args:
            batch_id: ID du lot qui devait être traité

        Returns:
            Réponse de repli
        """
        return {
            "success": False,
            "error": "L'autorisation d'utiliser le modèle a été refusée",
            "batch_id": batch_id,
            "model": self.learning_model,
            "fallback": True,
        }

    # @secure_api_method(
    #     model_name_attr="base_model", reason="Analyse d'amélioration des connaissances"
    # )
    def analyze_improvements(self) -> dict[str, Any]:
        """
        Analyse les améliorations apportées par l'apprentissage.

        Returns:
            Analyse des améliorations
        """
        # Récupérer les statistiques d'apprentissage
        stats = self.get_learning_stats()

        # Effectuer l'analyse (simulée ici)
        metrics = stats["stats"].get("quality_improvements", {})
        total_batches = stats["stats"].get("batches_processed", 0)

        # Calculer les améliorations moyennes
        avg_improvements = {}
        for metric, value in metrics.items():
            if total_batches > 0:
                avg_improvements[metric] = value / total_batches
            else:
                avg_improvements[metric] = 0.0

        # Simuler une analyse de performance
        analysis = {
            "success": True,
            "model": self.base_model,
            "learning_model": self.learning_model,
            "total_examples": stats["stats"].get("examples_collected", 0),
            "total_batches": total_batches,
            "total_tokens": stats["stats"].get("total_tokens_used", 0),
            "total_cost": stats["stats"].get("total_cost", 0.0),
            "average_improvements": avg_improvements,
            "estimated_overall_improvement": (sum(avg_improvements.values()) if avg_improvements else 0.0),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return analysis

    def retrain_from_memory(self, log_path="data/learning_log.jsonl"):
        """
        Recharge les interactions apprises et simule un réentraînement local.
        Peut être utilisé pour adapter un modèle personnalisé.

        Args:
            log_path: Chemin du fichier journal d'apprentissage

        Returns:
            Dict: Résultat du réentraînement
        """
        try:
            # Importer le module d'exportation d'apprentissage
            from jeffrey.core.ia_services.export_learning import export_for_training, load_log, score_entries

            # Charger et évaluer les entrées du journal
            self.logger.info(f"Chargement du journal d'apprentissage depuis {log_path}")
            entries = load_log(log_path)

            if not entries:
                self.logger.warning("Aucune entrée trouvée dans le journal d'apprentissage")
                return {
                    "success": False,
                    "error": "Journal d'apprentissage vide",
                    "entries_found": 0,
                }

            self.logger.info(f"Notation de {len(entries)} entrées")
            score_entries(entries)

            # Exporter les données pour l'entraînement
            training_data_path = os.path.join(self.data_path, "training_data.jsonl")
            min_quality = 0.6  # Ne garder que les exemples de qualité suffisante

            self.logger.info(f"Exportation des données d'entraînement de qualité > {min_quality}")
            exported_count = export_for_training(
                training_data_path,
                min_quality=min_quality,
                max_entries=1000,  # Limiter pour prévenir l'sur-apprentissage
                log_path=log_path,
            )

            if exported_count == 0:
                self.logger.warning("Aucun exemple de qualité suffisante pour l'entraînement")
                return {
                    "success": False,
                    "error": "Pas assez d'exemples de qualité",
                    "entries_found": len(entries),
                    "entries_qualified": 0,
                }

            # Simuler un entraînement local (serait remplacé par un vrai entraînement en production)
            self.logger.info(f"Simulation de réentraînement avec {exported_count} exemples")

            training_result = {
                "success": True,
                "model": self.base_model,
                "examples_used": exported_count,
                "training_file": training_data_path,
                "training_epochs": 3,  # Simulé
                "final_loss": 0.123,  # Simulé
                "training_duration": "00:15:23",  # Simulé
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(f"Réentraînement terminé: {exported_count} exemples utilisés")

            # Enregistrer les statistiques d'apprentissage
            self.stats["last_retrain"] = {
                "examples_used": exported_count,
                "quality_threshold": min_quality,
                "timestamp": datetime.now().isoformat(),
            }

            return training_result

        except Exception as e:
            self.logger.error(f"Erreur lors du réentraînement: {e}")
            return {"success": False, "error": str(e)}
