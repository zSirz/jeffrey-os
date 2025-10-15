"""
Data Augmenter for Jeffrey OS DreamMode
Privacy-preserving synthetic data generation for training neural mutator.
"""

import hashlib
import json
import random
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np


class DataAugmenter:
    """
    Génère des données synthétiques pour entraînement.
    Protège la privacy avec anonymisation et differential privacy.
    """

    def __init__(self, dream_journal=None, privacy_level: str = 'high'):
        self.dream_journal = dream_journal
        self.privacy_level = privacy_level  # 'low', 'medium', 'high'
        self.privacy_mode = True  # Always anonymize by default

        # Synthetic data generation parameters
        self.pattern_cache = deque(maxlen=1000)
        self.generation_stats = {'total_generated': 0, 'successful_patterns': 0, 'privacy_violations_prevented': 0}

        # Privacy noise parameters (differential privacy)
        self.noise_scales = {'low': 0.01, 'medium': 0.05, 'high': 0.1}

    async def generate_synthetic_feedback(
        self, num_samples: int = 100, base_patterns: list[dict] = None, diversity_factor: float = 0.3
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Génère du feedback synthétique basé sur l'historique ou patterns.
        """
        synthetic_data = []

        # Load patterns from journal or use provided ones
        if self.dream_journal and not base_patterns:
            try:
                base_patterns = await self.dream_journal.get_successful_patterns(limit=50)
            except:
                base_patterns = None

        if not base_patterns:
            # Fallback : générer depuis random patterns
            base_patterns = self._generate_random_patterns(min(num_samples, 20))

        # Anonymiser les patterns si mode privacy
        if self.privacy_mode:
            base_patterns = [self._anonymize_pattern(p) for p in base_patterns]

        # Apply differential privacy noise
        if self.privacy_level in ['medium', 'high']:
            base_patterns = [self._add_differential_privacy_noise(p) for p in base_patterns]

        # Track successful pattern characteristics for better generation
        pattern_characteristics = self._analyze_pattern_characteristics(base_patterns)

        # Augmenter les données avec diversité contrôlée
        for i in range(num_samples):
            base_idx = i % len(base_patterns)
            base = base_patterns[base_idx]

            # Créer embedding original
            original_embedding = self._pattern_to_embedding(base)

            # Créer mutation synthétique avec diversité
            mutation_type = self._select_mutation_type(diversity_factor, pattern_characteristics)
            mutated_embedding = self._apply_synthetic_mutation(original_embedding, mutation_type, base)

            # Validate quality before adding
            if self._validate_synthetic_pair(original_embedding, mutated_embedding):
                synthetic_data.append((original_embedding, mutated_embedding))
                self.generation_stats['successful_patterns'] += 1

            self.generation_stats['total_generated'] += 1

        # Cache patterns for future use
        self.pattern_cache.extend(base_patterns[:50])  # Keep recent patterns

        return synthetic_data

    def _anonymize_pattern(self, pattern: dict) -> dict:
        """
        Anonymise un pattern en remplaçant données sensibles.
        """
        anonymized = pattern.copy()

        # Detect and replace PII
        sensitive_fields = self._detect_sensitive_fields(pattern)

        for field in sensitive_fields:
            if field in anonymized:
                original_value = anonymized[field]

                if isinstance(original_value, str):
                    # Hash string values
                    anonymized[field] = hashlib.sha256(original_value.encode()).hexdigest()[:8]
                elif isinstance(original_value, dict):
                    # Recursively anonymize nested dicts
                    anonymized[field] = self._anonymize_pattern(original_value)
                else:
                    anonymized[field] = 'ANONYMIZED'

        # Remplacer user-specific preferences par distributions synthétiques
        if 'user_preferences' in anonymized:
            # Use deterministic seed based on anonymized user ID for consistency
            user_seed = self._generate_anonymous_seed(anonymized.get('user_id', 'default'))
            np.random.seed(user_seed)

            anonymized['user_preferences'] = {
                'risk_tolerance': np.clip(np.random.normal(0.5, 0.2), 0, 1),
                'innovation_preference': np.clip(np.random.normal(0.6, 0.15), 0, 1),
                'simplicity_preference': np.clip(np.random.normal(0.4, 0.2), 0, 1),
                'creativity_preference': np.clip(np.random.normal(0.55, 0.18), 0, 1),
            }

        # Add anonymization metadata
        anonymized['_anonymized'] = True
        anonymized['_privacy_level'] = self.privacy_level

        return anonymized

    def _detect_sensitive_fields(self, pattern: dict) -> list[str]:
        """Détecte les champs sensibles dans un pattern."""
        sensitive_indicators = [
            'user_id',
            'session_id',
            'ip_address',
            'email',
            'name',
            'phone',
            'address',
            'credit_card',
            'ssn',
            'password',
            'token',
            'key',
            'personal',
            'private',
            'confidential',
            'secret',
        ]

        sensitive_fields = []

        def check_field(key, value, path=''):
            field_path = f"{path}.{key}" if path else key

            # Check key name
            if any(indicator in key.lower() for indicator in sensitive_indicators):
                sensitive_fields.append(field_path)
                return

            # Check string values for patterns
            if isinstance(value, str):
                # Email pattern
                if '@' in value and '.' in value:
                    sensitive_fields.append(field_path)
                # Phone pattern
                elif any(char.isdigit() for char in value) and len(value) >= 10:
                    if '-' in value or '(' in value or ')' in value:
                        sensitive_fields.append(field_path)
                # Credit card pattern
                elif (
                    len(value.replace('-', '').replace(' ', '')) == 16
                    and value.replace('-', '').replace(' ', '').isdigit()
                ):
                    sensitive_fields.append(field_path)

            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    check_field(nested_key, nested_value, field_path)

        for key, value in pattern.items():
            check_field(key, value)

        return sensitive_fields

    def _add_differential_privacy_noise(self, pattern: dict) -> dict:
        """Ajoute du bruit pour differential privacy."""
        noisy_pattern = pattern.copy()
        noise_scale = self.noise_scales[self.privacy_level]

        # Add noise to numeric values
        for key, value in pattern.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                # Add Laplace noise for differential privacy
                noise = np.random.laplace(0, noise_scale)
                noisy_pattern[key] = value + noise

            elif isinstance(value, dict):
                noisy_pattern[key] = self._add_differential_privacy_noise(value)

        return noisy_pattern

    def _generate_anonymous_seed(self, user_identifier: str) -> int:
        """Génère un seed anonyme mais consistant."""
        # Hash to create consistent but anonymous seed
        hash_obj = hashlib.sha256(user_identifier.encode())
        return int(hash_obj.hexdigest()[:8], 16) % (2**31)

    def _generate_random_patterns(self, num_patterns: int) -> list[dict]:
        """
        Génère des patterns aléatoires comme fallback.
        """
        patterns = []

        # Define realistic parameter distributions based on typical usage
        pattern_types = ['optimization', 'creation', 'analysis', 'innovation', 'simplification']
        complexity_dist = np.random.beta(2, 3, num_patterns)  # Skewed toward simpler
        impact_dist = np.random.beta(3, 2, num_patterns)  # Skewed toward higher impact

        for i in range(num_patterns):
            pattern_type = random.choice(pattern_types)

            pattern = {
                'id': f'synthetic_{i}',
                'type': pattern_type,
                'complexity': float(complexity_dist[i]),
                'impact': float(impact_dist[i]),
                'feasibility': np.clip(np.random.normal(0.6, 0.2), 0.1, 0.9),
                'user_preferences': {
                    'risk_tolerance': np.clip(np.random.normal(0.5, 0.2), 0, 1),
                    'innovation_preference': np.clip(np.random.normal(0.6, 0.15), 0, 1),
                    'simplicity_preference': np.clip(np.random.normal(0.4, 0.2), 0, 1),
                    'creativity_preference': np.clip(np.random.normal(0.55, 0.18), 0, 1),
                },
                'context': {
                    'time_pressure': random.choice(['low', 'medium', 'high']),
                    'resource_availability': np.random.uniform(0.3, 0.9),
                    'stakeholder_support': np.random.uniform(0.4, 0.8),
                },
                'timestamp': datetime.now().isoformat(),
                'synthetic': True,
                'generation_method': 'statistical_sampling',
            }

            # Add type-specific attributes
            if pattern_type == 'optimization':
                pattern['target_metric'] = random.choice(['speed', 'accuracy', 'efficiency', 'cost'])
                pattern['improvement_target'] = np.random.uniform(0.1, 0.5)
            elif pattern_type == 'creation':
                pattern['novelty_level'] = np.random.uniform(0.3, 0.9)
                pattern['inspiration_sources'] = random.sample(['nature', 'technology', 'art', 'science'], 2)
            elif pattern_type == 'analysis':
                pattern['depth_level'] = random.choice(['surface', 'detailed', 'comprehensive'])
                pattern['analysis_scope'] = np.random.uniform(0.4, 0.8)

            patterns.append(pattern)

        return patterns

    def _analyze_pattern_characteristics(self, patterns: list[dict]) -> dict:
        """Analyse les caractéristiques des patterns pour améliorer la génération."""
        if not patterns:
            return {}

        characteristics = {
            'avg_complexity': 0,
            'avg_impact': 0,
            'common_types': {},
            'preference_distributions': {},
            'success_factors': [],
        }

        # Analyze numeric characteristics
        complexities = [p.get('complexity', 0.5) for p in patterns if 'complexity' in p]
        impacts = [p.get('impact', 0.5) for p in patterns if 'impact' in p]

        if complexities:
            characteristics['avg_complexity'] = np.mean(complexities)
            characteristics['complexity_std'] = np.std(complexities)
        if impacts:
            characteristics['avg_impact'] = np.mean(impacts)
            characteristics['impact_std'] = np.std(impacts)

        # Analyze pattern types
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            characteristics['common_types'][pattern_type] = characteristics['common_types'].get(pattern_type, 0) + 1

        # Analyze user preferences
        all_preferences = {}
        for pattern in patterns:
            prefs = pattern.get('user_preferences', {})
            for key, value in prefs.items():
                if isinstance(value, (int, float)):
                    if key not in all_preferences:
                        all_preferences[key] = []
                    all_preferences[key].append(value)

        for key, values in all_preferences.items():
            characteristics['preference_distributions'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

        return characteristics

    def _select_mutation_type(self, diversity_factor: float, characteristics: dict) -> str:
        """Sélectionne le type de mutation basé sur la diversité et les caractéristiques."""
        mutation_types = ['slight', 'moderate', 'strong', 'creative', 'conservative']

        # Adjust probabilities based on diversity factor
        if diversity_factor > 0.7:
            # High diversity - favor stronger mutations
            weights = [0.1, 0.2, 0.3, 0.3, 0.1]
        elif diversity_factor > 0.4:
            # Medium diversity
            weights = [0.2, 0.3, 0.2, 0.2, 0.1]
        else:
            # Low diversity - favor conservative mutations
            weights = [0.3, 0.3, 0.1, 0.1, 0.2]

        # Adjust based on pattern characteristics
        avg_complexity = characteristics.get('avg_complexity', 0.5)
        if avg_complexity > 0.7:
            # High complexity patterns - favor simplifying mutations
            weights[4] += 0.2  # More conservative
            weights[2] -= 0.1  # Less strong mutations

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return np.random.choice(mutation_types, p=weights)

    def _pattern_to_embedding(self, pattern: dict, embedding_dim: int = 64) -> np.ndarray:
        """
        Convertit un pattern en embedding vectoriel avec gestion robuste des types.
        """
        features = []

        # Core numeric features with defaults
        features.append(self._safe_numeric_extract(pattern, 'complexity', 0.5))
        features.append(self._safe_numeric_extract(pattern, 'impact', 0.5))
        features.append(self._safe_numeric_extract(pattern, 'feasibility', 0.5))

        # User preferences with robust extraction
        prefs = pattern.get('user_preferences', {})
        for pref_key in ['risk_tolerance', 'innovation_preference', 'simplicity_preference', 'creativity_preference']:
            features.append(self._safe_numeric_extract(prefs, pref_key, 0.5))

        # Context features
        context = pattern.get('context', {})
        features.append(self._safe_numeric_extract(context, 'resource_availability', 0.5))
        features.append(self._safe_numeric_extract(context, 'stakeholder_support', 0.5))

        # Time pressure encoding
        time_pressure_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        time_pressure = context.get('time_pressure', 'medium')
        features.append(time_pressure_map.get(time_pressure, 0.5))

        # Pattern type encoding (one-hot style with more types)
        type_map = {
            'optimization': [1, 0, 0, 0, 0],
            'creation': [0, 1, 0, 0, 0],
            'analysis': [0, 0, 1, 0, 0],
            'innovation': [0, 0, 0, 1, 0],
            'simplification': [0, 0, 0, 0, 1],
        }
        pattern_type = pattern.get('type', 'optimization')
        features.extend(type_map.get(pattern_type, [0, 0, 0, 0, 1]))

        # Derived features
        features.append(features[0] * features[1])  # complexity * impact
        features.append(features[1] * features[2])  # impact * feasibility
        features.append(abs(features[3] - features[4]))  # risk vs innovation gap

        # Add some temporal encoding if timestamp available
        if 'timestamp' in pattern:
            try:
                # Simple temporal encoding (hour of day)
                timestamp = datetime.fromisoformat(pattern['timestamp'].replace('Z', '+00:00'))
                hour_norm = timestamp.hour / 24.0
                features.append(hour_norm)
                features.append(np.sin(2 * np.pi * hour_norm))  # Cyclical encoding
                features.append(np.cos(2 * np.pi * hour_norm))
            except:
                features.extend([0.5, 0.0, 1.0])  # Default temporal features
        else:
            features.extend([0.5, 0.0, 1.0])

        # Pad or truncate to embedding dimension
        while len(features) < embedding_dim:
            # Use harmonic series for padding
            features.append(1.0 / (len(features) + 1))

        return np.array(features[:embedding_dim], dtype=np.float32)

    def _safe_numeric_extract(self, data: dict, key: str, default: float) -> float:
        """Extraction sécurisée de valeurs numériques."""
        value = data.get(key, default)
        if isinstance(value, (int, float)):
            return float(np.clip(value, 0.0, 1.0))
        elif isinstance(value, str):
            try:
                return float(np.clip(float(value), 0.0, 1.0))
            except:
                return default
        else:
            return default

    def _apply_synthetic_mutation(
        self, embedding: np.ndarray, mutation_type: str, original_pattern: dict = None
    ) -> np.ndarray:
        """
        Applique une mutation synthétique contrôlée avec réalisme amélioré.
        """
        mutation_params = {
            'slight': {'noise_scale': 0.05, 'feature_change_prob': 0.3, 'directional_bias': 0.1},
            'moderate': {'noise_scale': 0.15, 'feature_change_prob': 0.5, 'directional_bias': 0.2},
            'strong': {'noise_scale': 0.3, 'feature_change_prob': 0.7, 'directional_bias': 0.3},
            'creative': {'noise_scale': 0.4, 'feature_change_prob': 0.8, 'directional_bias': 0.5},
            'conservative': {'noise_scale': 0.03, 'feature_change_prob': 0.2, 'directional_bias': 0.05},
        }

        params = mutation_params.get(mutation_type, mutation_params['moderate'])
        mutated = embedding.copy()

        # Apply feature-specific mutations
        for i in range(len(mutated)):
            if np.random.random() < params['feature_change_prob']:
                # Determine mutation direction based on feature type
                if i < 3:  # Core features (complexity, impact, feasibility)
                    mutation_direction = self._get_smart_mutation_direction(
                        i, mutated[i], original_pattern, params['directional_bias']
                    )
                else:
                    mutation_direction = np.random.randn()

                # Apply noise with directional bias
                noise = np.random.normal(0, params['noise_scale'])
                mutated[i] += noise + mutation_direction * params['directional_bias']

        # Ensure realistic constraints
        mutated[:7] = np.clip(mutated[:7], 0.0, 1.0)  # Core normalized features

        # Maintain some correlations (e.g., high complexity often means lower feasibility)
        if mutation_type in ['moderate', 'strong']:
            complexity_idx, feasibility_idx = 0, 2
            if mutated[complexity_idx] > 0.8 and mutated[feasibility_idx] > 0.7:
                # High complexity with high feasibility is unrealistic
                mutated[feasibility_idx] = np.clip(mutated[feasibility_idx] - 0.2, 0.1, 1.0)

        return mutated.astype(np.float32)

    def _get_smart_mutation_direction(
        self, feature_idx: int, current_value: float, original_pattern: dict, bias_strength: float
    ) -> float:
        """Détermine une direction de mutation intelligente."""
        # Feature-specific intelligence
        if feature_idx == 0:  # Complexity
            # Bias towards simplification (generally preferred)
            return -bias_strength if current_value > 0.6 else bias_strength * 0.5
        elif feature_idx == 1:  # Impact
            # Bias towards higher impact
            return bias_strength if current_value < 0.7 else -bias_strength * 0.3
        elif feature_idx == 2:  # Feasibility
            # Bias towards higher feasibility
            return bias_strength if current_value < 0.8 else -bias_strength * 0.2
        else:
            return 0.0  # No bias for other features

    def _validate_synthetic_pair(self, original: np.ndarray, mutated: np.ndarray) -> bool:
        """Valide qu'une paire synthétique est réaliste."""
        # Check for reasonable difference
        distance = np.linalg.norm(original - mutated)
        if distance < 0.01 or distance > 2.0:
            return False  # Too similar or too different

        # Check for realistic feature values
        if np.any(mutated[:7] < 0) or np.any(mutated[:7] > 1):
            return False  # Features out of realistic range

        # Check for extreme value combinations
        complexity, impact, feasibility = mutated[0], mutated[1], mutated[2]
        if complexity > 0.9 and feasibility > 0.9:
            return False  # Unrealistic combination
        if impact < 0.1 and complexity > 0.8:
            return False  # High complexity with no impact

        return True

    def get_generation_stats(self) -> dict[str, Any]:
        """Retourne les statistiques de génération."""
        stats = self.generation_stats.copy()
        if stats['total_generated'] > 0:
            stats['success_rate'] = stats['successful_patterns'] / stats['total_generated']
        else:
            stats['success_rate'] = 0.0

        stats['cache_size'] = len(self.pattern_cache)
        stats['privacy_level'] = self.privacy_level

        return stats

    def clear_cache(self):
        """Vide le cache de patterns."""
        self.pattern_cache.clear()

    def export_synthetic_dataset(self, num_samples: int = 1000, filepath: str = None) -> str | None:
        """Exporte un dataset synthétique complet."""
        try:
            synthetic_data = asyncio.run(self.generate_synthetic_feedback(num_samples=num_samples))

            # Convert to serializable format
            dataset = {
                'metadata': {
                    'num_samples': len(synthetic_data),
                    'privacy_level': self.privacy_level,
                    'generation_timestamp': datetime.now().isoformat(),
                    'embedding_dim': len(synthetic_data[0][0]) if synthetic_data else 64,
                },
                'data': [
                    {'original': original.tolist(), 'mutated': mutated.tolist()} for original, mutated in synthetic_data
                ],
            }

            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(dataset, f, indent=2)
                return filepath
            else:
                return json.dumps(dataset, indent=2)

        except Exception as e:
            print(f"Error exporting synthetic dataset: {e}")
            return None
