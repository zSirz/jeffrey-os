"""
Ethical Guard for Jeffrey OS DreamMode
Content filtering, bias detection, and safety validation.
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Any


class EthicalFilter:
    """
    Filtre éthique pour contenus générés.
    Détecte biais, contenus sensibles et risques.
    """

    def __init__(self):
        # Keywords sensibles (exemples - à étendre)
        self.sensitive_keywords = {
            "violence": ["harm", "hurt", "attack", "destroy", "kill", "damage", "weapon"],
            "discrimination": ["bias", "discriminate", "exclude", "prejudice", "racist", "sexist"],
            "manipulation": ["deceive", "trick", "manipulate", "exploit", "scam", "fraud"],
            "privacy": ["personal_data", "private", "confidential", "secret", "password", "token"],
            "illegal": ["hack", "crack", "pirate", "steal", "illegal", "criminal", "fraud"],
            "inappropriate": ["explicit", "adult", "nsfw", "inappropriate", "offensive"],
        }

        # Patterns regex pour détection avancée
        self.risk_patterns = [
            r"(?i)(hack|crack|breach|exploit)\s+\w+",
            r"(?i)(steal|theft|rob)\s+\w+",
            r"(?i)(illegal|unlawful|criminal)\s+\w+",
            r"(?i)(password|token|key)\s*[:=]\s*\w+",  # Détection credentials
            r"(?i)(delete|remove|destroy)\s+(all|everything)",  # Actions destructrices
            r"(?i)(bypass|override|disable)\s+(security|safety|protection)",
        ]

        self.blocked_hashes: set[str] = set()  # Cache de contenus bloqués
        self.risk_log: list[dict] = []  # Log des contenus risqués détectés

        # Seuils de tolérance
        self.risk_thresholds = {
            "violence": 0.7,
            "discrimination": 0.8,
            "manipulation": 0.6,
            "privacy": 0.5,
            "illegal": 0.3,
            "inappropriate": 0.8,
        }

    async def is_safe(self, content: dict) -> bool:
        """
        Vérifie si un contenu est éthiquement acceptable.
        """
        # Extraire le texte à analyser
        text_content = self._extract_text(content)

        # Check 1: Keywords sensibles avec scoring
        risk_scores = self._calculate_risk_scores(text_content)
        for category, score in risk_scores.items():
            if score > self.risk_thresholds.get(category, 0.5):
                self._log_risk_detection(content, category, score, "keyword_threshold")
                return False

        # Check 2: Patterns regex
        regex_risks = self._detect_risk_patterns(text_content)
        if regex_risks:
            self._log_risk_detection(content, "regex_patterns", 1.0, "regex_match", regex_risks)
            return False

        # Check 3: Hash de contenus précédemment bloqués
        content_hash = self._hash_content(text_content)
        if content_hash in self.blocked_hashes:
            self._log_risk_detection(content, "previously_blocked", 1.0, "hash_match")
            return False

        # Check 4: Analyse de biais
        bias_score = self._detect_bias(content)
        if bias_score > 0.6:
            self._log_risk_detection(content, "bias", bias_score, "bias_threshold")
            return False

        # Check 5: Validation structurelle
        if not self._validate_structure(content):
            self._log_risk_detection(content, "structure", 1.0, "invalid_structure")
            return False

        return True

    def _calculate_risk_scores(self, text: str) -> dict[str, float]:
        """Calcule des scores de risque par catégorie."""
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)
        scores = {}

        for category, keywords in self.sensitive_keywords.items():
            matches = 0
            for keyword in keywords:
                # Compter les occurrences avec scoring pondéré
                count = text_lower.count(keyword)
                if count > 0:
                    # Pondération par longueur du keyword
                    weight = len(keyword) / 10.0
                    matches += count * weight

            # Normaliser par rapport à la longueur du texte
            scores[category] = min(matches / word_count * 10, 1.0)

        return scores

    def _detect_risk_patterns(self, text: str) -> list[str]:
        """Détecte les patterns regex à risque."""
        detected_patterns = []
        for pattern in self.risk_patterns:
            matches = re.findall(pattern, text)
            if matches:
                detected_patterns.extend(matches)
        return detected_patterns

    def _detect_bias(self, content: dict) -> float:
        """
        Détecte les biais potentiels dans le contenu.
        """
        bias_score = 0.0

        # Biais de diversité
        if "variants" in content:
            variants = content["variants"]
            if len(variants) > 1:
                # Analyser la diversité des stratégies
                strategies = [v.get("strategy", "unknown") for v in variants]
                unique_strategies = len(set(strategies))
                diversity_ratio = unique_strategies / len(variants)

                # Moins de diversité = plus de biais potentiel
                bias_score += (1.0 - diversity_ratio) * 0.3

                # Analyser la diversité lexicale
                all_text = " ".join(self._extract_text(v) for v in variants)
                words = all_text.lower().split()
                unique_words = len(set(words))
                if len(words) > 0:
                    lexical_diversity = unique_words / len(words)
                    bias_score += (1.0 - lexical_diversity) * 0.2

        # Biais de répétition
        text = self._extract_text(content)
        words = text.split()
        if len(words) > 10:
            # Détecter les répétitions excessives
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            max_freq = max(word_freq.values()) if word_freq else 0
            repetition_ratio = max_freq / len(words)
            if repetition_ratio > 0.3:  # Plus de 30% de répétition d'un mot
                bias_score += repetition_ratio * 0.5

        return min(bias_score, 1.0)

    def _validate_structure(self, content: dict) -> bool:
        """Valide la structure du contenu."""
        # Vérifications basiques de structure
        if not isinstance(content, dict):
            return False

        # Le contenu doit avoir au moins une clé significative
        required_keys = ["id", "type", "content", "text", "description", "variants"]
        if not any(key in content for key in required_keys):
            return False

        # Vérifier qu'il n'y a pas de contenu vide
        text_content = self._extract_text(content)
        if len(text_content.strip()) < 10:  # Contenu trop court
            return False

        return True

    def block_content(self, content: dict, reason: str):
        """Bloque un contenu pour référence future."""
        text = self._extract_text(content)
        content_hash = self._hash_content(text)
        self.blocked_hashes.add(content_hash)

        # Log du blocage
        self._log_risk_detection(content, "manual_block", 1.0, reason)

    def _log_risk_detection(
        self,
        content: dict,
        category: str,
        score: float,
        reason: str,
        details: list | None = None,
    ):
        """Log une détection de risque."""
        risk_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "score": score,
            "reason": reason,
            "content_hash": self._hash_content(self._extract_text(content)),
            "content_preview": self._extract_text(content)[:100] + "...",
            "details": details or [],
        }

        self.risk_log.append(risk_entry)

        # Garder seulement les 100 dernières entrées
        if len(self.risk_log) > 100:
            self.risk_log.pop(0)

    def _extract_text(self, content: dict) -> str:
        """Extrait tout le texte d'un contenu."""
        if isinstance(content, str):
            return content

        texts = []
        for key, value in content.items():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict):
                texts.append(self._extract_text(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (str, dict)):
                        texts.append(self._extract_text(item))

        return " ".join(texts)

    def _hash_content(self, text: str) -> str:
        """Hash un contenu pour comparaison rapide."""
        # Normaliser le texte avant hashing
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get_risk_statistics(self) -> dict[str, Any]:
        """Retourne les statistiques des risques détectés."""
        if not self.risk_log:
            return {"total": 0, "categories": {}, "recent_activity": []}

        # Compter par catégorie
        category_counts = {}
        for entry in self.risk_log:
            cat = entry["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Activité récente (dernières 24h)
        recent_activity = [entry for entry in self.risk_log if self._is_recent(entry["timestamp"])]

        return {
            "total": len(self.risk_log),
            "categories": category_counts,
            "recent_activity": len(recent_activity),
            "blocked_content_count": len(self.blocked_hashes),
        }

    def _is_recent(self, timestamp_str: str, hours: int = 24) -> bool:
        """Vérifie si un timestamp est récent."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            delta = datetime.now() - timestamp
            return delta.total_seconds() < hours * 3600
        except:
            return False

    def export_risk_log(self, filepath: str):
        """Exporte le log des risques en JSON."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.risk_log, f, indent=2)
        except Exception as e:
            print(f"Error exporting risk log: {e}")

    def clear_old_logs(self, days: int = 30):
        """Supprime les logs plus anciens que X jours."""
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        self.risk_log = [
            entry for entry in self.risk_log if datetime.fromisoformat(entry["timestamp"]).timestamp() > cutoff
        ]
