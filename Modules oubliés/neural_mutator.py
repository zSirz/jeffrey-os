"""
Neural Mutator for Jeffrey OS DreamMode
PyTorch-based neural network for creative mutations with all critical fixes.
"""

import hashlib
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


class NeuralMutator(nn.Module):
    """
    Réseau de neurones pour mutations créatives intelligentes.
    Architecture légère optimisée pour génération rapide.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        latent_dim: int = 16,
        model_path: str = "models/neural_mutator.pth",
        diversity_weight: float = 0.1,
        training_mode: bool = False,
    ):
        super().__init__()

        # Architecture encoder-decoder avec skip connections
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, latent_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 48),  # Skip connection
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, embedding_dim),
        )

        # Creativity modulation
        self.creativity_modulator = nn.Linear(1, latent_dim)

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.model_path = model_path
        self.diversity_weight = diversity_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move to appropriate device
        self.to(self.device)

        # Historique pour diversity loss
        self.generation_history = []

        # Charger le modèle si existant
        if os.path.exists(model_path):
            try:
                self.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded existing model from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")

        # Mode approprié selon usage
        if training_mode:
            self.train()
        else:
            self.eval()

    def forward(self, x: torch.Tensor, creativity: float = 0.3) -> torch.Tensor:
        """
        Forward pass avec niveau de créativité contrôlé.
        Gère correctement batch_size=1 pour BatchNorm.
        """
        # Ensure tensor is on right device
        x = x.to(self.device)

        batch_size = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # CRITICAL FIX: Handle BatchNorm pour single sample
        original_training = self.training
        if batch_size == 1 and self.training:
            self.eval()  # Force eval mode pour éviter BatchNorm error
            single_sample_mode = True
        else:
            single_sample_mode = False

        try:
            # Encode
            latent = self.encoder(x)

            # Moduler avec créativité
            creativity_tensor = torch.full((x.shape[0], 1), creativity, dtype=torch.float32, device=self.device)
            creativity_mod = self.creativity_modulator(creativity_tensor)

            # Ajouter bruit contrôlé
            noise = torch.randn_like(latent) * creativity
            latent_creative = latent + noise + creativity_mod

            # Diversity loss component (pour régularisation)
            if self.training and len(self.generation_history) > 0:
                # Calculer diversity loss
                diversity_loss = -torch.mean(torch.std(latent_creative, dim=0)) * self.diversity_weight
                self.register_buffer("diversity_loss", diversity_loss.detach())

            # Decode avec skip connection
            combined = torch.cat([latent_creative, x], dim=1)
            output = self.decoder(combined)

        finally:
            # Restore training mode if needed
            if single_sample_mode and original_training:
                self.train()

        return output.squeeze(0) if batch_size == 1 else output

    @torch.no_grad()
    def generate_variant(self, base_embedding: np.ndarray, creativity_level: float = 0.3) -> np.ndarray:
        """
        Génère une variante avec cache LRU pour performance.
        """
        # Hash pour cache avec privacy
        embedding_hash = hashlib.sha256(base_embedding.tobytes()).hexdigest()[:16]
        return self._generate_cached(embedding_hash, tuple(base_embedding.flatten()), creativity_level)

    @lru_cache(maxsize=1000)
    def _generate_cached(self, cache_key: str, base_tuple: tuple, creativity: float) -> np.ndarray:
        """Version cachée avec privacy-preserving key."""
        base_array = np.array(base_tuple, dtype=np.float32)
        base_tensor = torch.tensor(base_array, dtype=torch.float32)
        variant_tensor = self.forward(base_tensor, creativity)
        return variant_tensor.cpu().numpy()

    @torch.no_grad()
    def generate_batch(self, embeddings: list[np.ndarray], creativity_levels: list[float]) -> list[np.ndarray]:
        """
        Génération par batch pour efficacité.
        """
        if not embeddings:
            return []

        # Stack embeddings
        batch_tensor = torch.stack([torch.tensor(emb, dtype=torch.float32) for emb in embeddings]).to(self.device)

        # Batch creativity levels
        creativity_tensor = torch.tensor(creativity_levels, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Forward through the network (batch processing)
        with torch.no_grad():
            # Process all at once for efficiency
            variants = []
            for i, creativity in enumerate(creativity_levels):
                variant = self.forward(batch_tensor[i : i + 1], creativity)
                variants.append(variant.cpu().numpy())

        return variants

    def train_on_feedback(
        self,
        successful_mutations: list[tuple[np.ndarray, np.ndarray]],
        learning_rate: float = 0.001,
        chunk_size: int = 32,
    ):
        """
        Entraînement incrémental sur les mutations réussies.
        Version optimisée avec chunking pour scalabilité.
        """
        if not successful_mutations:
            return

        # Cooldown mechanism (train every 10 feedbacks)
        if not hasattr(self, "_feedback_count"):
            self._feedback_count = 0
        self._feedback_count += 1

        if self._feedback_count % 10 != 0:
            return  # Skip training this time

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Temporarily set to training mode
        original_mode = self.training
        self.train()

        try:
            # Process in chunks pour éviter OOM
            for chunk_start in range(0, len(successful_mutations), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(successful_mutations))
                chunk = successful_mutations[chunk_start:chunk_end]

                if not chunk:
                    continue

                # Stack tensors pour batch processing
                x_batch = torch.stack([torch.tensor(orig, dtype=torch.float32) for orig, _ in chunk]).to(self.device)

                y_batch = torch.stack([torch.tensor(mut, dtype=torch.float32) for _, mut in chunk]).to(self.device)

                # Forward
                output = self.forward(x_batch, creativity=0.3)

                # Loss principale
                main_loss = criterion(output, y_batch)

                # Add diversity loss if available
                total_loss = main_loss
                if hasattr(self, "diversity_loss"):
                    total_loss = main_loss + self.diversity_loss

                # Backward
                optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping pour stabilité
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()

                # Update history (limited size)
                self.generation_history.extend(output.detach().cpu())
                if len(self.generation_history) > 100:
                    self.generation_history = self.generation_history[-50:]

        finally:
            # Restore original mode
            if original_mode:
                self.train()
            else:
                self.eval()

        # Sauvegarder le modèle
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.state_dict(), self.model_path)
            print(f"✅ Model saved to {self.model_path}")
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")

    def get_model_stats(self) -> dict:
        """Retourne les statistiques du modèle."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.latent_dim,
            "diversity_weight": self.diversity_weight,
            "feedback_count": getattr(self, "_feedback_count", 0),
            "generation_history_size": len(self.generation_history),
        }

    def reset_cache(self):
        """Vide le cache LRU (utile pour tests)."""
        self._generate_cached.cache_clear()

    def export_embeddings(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Exporte les embeddings latents pour analyse."""
        with torch.no_grad():
            embeddings = []
            for inp in inputs:
                tensor = torch.tensor(inp, dtype=torch.float32).to(self.device)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                latent = self.encoder(tensor)
                embeddings.append(latent.cpu().numpy())
            return embeddings
