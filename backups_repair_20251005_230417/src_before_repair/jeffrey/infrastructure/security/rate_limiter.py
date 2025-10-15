"""
Limiteur de débit adaptatif.

Ce module implémente les fonctionnalités essentielles pour limiteur de débit adaptatif.
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

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types de limitations de débit"""

    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_ENDPOINT = "per_endpoint"
    GLOBAL = "global"


class RateLimitPeriod(Enum):
    """Périodes de limitation"""

    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


@dataclass
class RateLimitRule:
    """Règle de limitation de débit"""

    name: str
    limit: int
    period: RateLimitPeriod
    limit_type: RateLimitType
    burst_limit: int | None = None
    block_duration: int = 300  # 5 minutes par défaut
    enabled: bool = True

    def period_seconds(self) -> int:
        """Retourne la période en secondes"""
        return {
            RateLimitPeriod.SECOND: 1,
            RateLimitPeriod.MINUTE: 60,
            RateLimitPeriod.HOUR: 3600,
            RateLimitPeriod.DAY: 86400,
        }[self.period]


@dataclass
class RateLimitStatus:
    """Statut de limitation pour un identifiant"""

    key: str
    requests: deque
    blocked_until: datetime | None = None
    total_requests: int = 0
    violations: int = 0
    last_request: datetime | None = None


class RateLimiter:
    """Gestionnaire de limitation de débit"""

    # Règles par défaut
    DEFAULT_RULES = {
        "api_default": RateLimitRule(
            name="api_default",
            limit=10,
            period=RateLimitPeriod.MINUTE,
            limit_type=RateLimitType.PER_IP,
            burst_limit=20,
        ),
        "login_attempts": RateLimitRule(
            name="login_attempts",
            limit=5,
            period=RateLimitPeriod.MINUTE,
            limit_type=RateLimitType.PER_IP,
            block_duration=900,  # 15 minutes
        ),
        "password_reset": RateLimitRule(
            name="password_reset",
            limit=3,
            period=RateLimitPeriod.HOUR,
            limit_type=RateLimitType.PER_IP,
            block_duration=3600,  # 1 heure
        ),
        "file_upload": RateLimitRule(
            name="file_upload",
            limit=20,
            period=RateLimitPeriod.HOUR,
            limit_type=RateLimitType.PER_USER,
            burst_limit=5,
        ),
        "database_queries": RateLimitRule(
            name="database_queries",
            limit=100,
            period=RateLimitPeriod.MINUTE,
            limit_type=RateLimitType.PER_USER,
            burst_limit=150,
        ),
        "api_calls": RateLimitRule(
            name="api_calls",
            limit=1000,
            period=RateLimitPeriod.HOUR,
            limit_type=RateLimitType.PER_USER,
            burst_limit=1200,
        ),
        "transaction_creation": RateLimitRule(
            name="transaction_creation",
            limit=50,
            period=RateLimitPeriod.MINUTE,
            limit_type=RateLimitType.PER_USER,
            block_duration=120,
        ),
        "bank_connection": RateLimitRule(
            name="bank_connection",
            limit=5,
            period=RateLimitPeriod.DAY,
            limit_type=RateLimitType.PER_USER,
            block_duration=86400,  # 24 heures
        ),
        "export_data": RateLimitRule(
            name="export_data",
            limit=10,
            period=RateLimitPeriod.DAY,
            limit_type=RateLimitType.PER_USER,
            burst_limit=15,
        ),
        "global_requests": RateLimitRule(
            name="global_requests",
            limit=10000,
            period=RateLimitPeriod.MINUTE,
            limit_type=RateLimitType.GLOBAL,
            burst_limit=12000,
        ),
    }

    def __init__(self, storage_file: str = ".rate_limits.json") -> None:
        self.rules: dict[str, RateLimitRule] = self.DEFAULT_RULES.copy()
        self.status: dict[str, RateLimitStatus] = {}
        self.storage_file = storage_file
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

        # Charger les données persistantes
        self._load_storage()

    def _load_storage(self):
        """Charge les données de rate limiting depuis le fichier"""
        try:
            with open(self.storage_file) as f:
                data = json.load(f)

            # Reconstruire les statuts
            for key, status_data in data.get("status", {}).items():
                requests = deque(status_data.get("requests", []))
                blocked_until = None
                if status_data.get("blocked_until"):
                    blocked_until = datetime.fromisoformat(status_data["blocked_until"])

                self.status[key] = RateLimitStatus(
                    key=key,
                    requests=requests,
                    blocked_until=blocked_until,
                    total_requests=status_data.get("total_requests", 0),
                    violations=status_data.get("violations", 0),
                    last_request=(
                        datetime.fromisoformat(status_data["last_request"]) if status_data.get("last_request") else None
                    ),
                )

            logger.info(f"Données de rate limiting chargées: {len(self.status)} entrées")
        except FileNotFoundError:
            logger.info("Aucun fichier de rate limiting existant")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")

    def _save_storage(self):
        """Sauvegarde les données de rate limiting"""
        try:
            data = {"status": {}, "timestamp": datetime.now().isoformat()}

            # Sérialiser les statuts
            for key, status in self.status.items():
                data["status"][key] = {
                    "requests": list(status.requests),
                    "blocked_until": (status.blocked_until.isoformat() if status.blocked_until else None),
                    "total_requests": status.total_requests,
                    "violations": status.violations,
                    "last_request": (status.last_request.isoformat() if status.last_request else None),
                }

            with open(self.storage_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")

    def _get_key(self, identifier: str, rule: RateLimitRule) -> str:
        """
        Génère une clé unique pour un identifiant et une règle

        Args:
            identifier: Identifiant (IP, user_id, etc.)
            rule: Règle de limitation

        Returns:
            str: Clé unique
        """
        if rule.limit_type == RateLimitType.GLOBAL:
            return f"global:{rule.name}"
        else:
            # Hasher l'identifiant pour la confidentialité
            hashed_id = hashlib.sha256(identifier.encode()).hexdigest()[:16]
            return f"{rule.limit_type.value}:{rule.name}:{hashed_id}"

    def _cleanup_old_requests(self):
        """Nettoie les anciennes requêtes"""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return

        now = time.time()
        cleaned_count = 0

        for key, status in list(self.status.items()):
            # Nettoyer les requêtes expirées
            initial_size = len(status.requests)
            while status.requests and status.requests[0] < now - 86400:  # 24 heures
                status.requests.popleft()

            # Débloquer si nécessaire
            if status.blocked_until and datetime.now() > status.blocked_until:
                status.blocked_until = None

            # Supprimer les statuts vides
            if not status.requests and not status.blocked_until:
                del self.status[key]
                cleaned_count += 1
            elif len(status.requests) < initial_size:
                cleaned_count += 1

        self.last_cleanup = now

        if cleaned_count > 0:
            logger.debug(f"Nettoyage effectué: {cleaned_count} entrées")
            self._save_storage()

    def check_rate_limit(self, identifier: str, rule_name: str) -> tuple[bool, dict[str, any]]:
        """
        Vérifie si une requête est autorisée

        Args:
            identifier: Identifiant (IP, user_id, etc.)
            rule_name: Nom de la règle à appliquer

        Returns:
            Tuple[bool, Dict]: (autorisé, informations)
        """
        self._cleanup_old_requests()

        if rule_name not in self.rules:
            logger.warning(f"Règle de rate limiting non trouvée: {rule_name}")
            return True, {"rule": "not_found"}

        rule = self.rules[rule_name]

        if not rule.enabled:
            return True, {"rule": "disabled"}

        key = self._get_key(identifier, rule)
        status = self.status.get(key)

        if not status:
            status = RateLimitStatus(key=key, requests=deque())
            self.status[key] = status

        now = time.time()
        current_time = datetime.now()

        # Vérifier si bloqué
        if status.blocked_until and current_time < status.blocked_until:
            remaining = int((status.blocked_until - current_time).total_seconds())
            return False, {
                "rule": rule_name,
                "blocked": True,
                "remaining_time": remaining,
                "reason": "blocked",
            }

        # Nettoyer les requêtes expirées pour cette règle
        period_seconds = rule.period_seconds()
        cutoff_time = now - period_seconds

        while status.requests and status.requests[0] < cutoff_time:
            status.requests.popleft()

        # Vérifier la limite
        current_requests = len(status.requests)

        # Vérifier la limite normale
        if current_requests >= rule.limit:
            # Vérifier la limite burst si disponible
            if rule.burst_limit and current_requests >= rule.burst_limit:
                # Bloquer temporairement
                status.blocked_until = current_time + timedelta(seconds=rule.block_duration)
                status.violations += 1

                self._save_storage()

                logger.warning(f"Rate limit dépassé pour {key}: {current_requests}/{rule.limit}")

                return False, {
                    "rule": rule_name,
                    "blocked": True,
                    "remaining_time": rule.block_duration,
                    "reason": "burst_limit_exceeded",
                    "current_requests": current_requests,
                    "limit": rule.limit,
                    "burst_limit": rule.burst_limit,
                }
            elif not rule.burst_limit:
                # Pas de limite burst, refuser immédiatement
                return False, {
                    "rule": rule_name,
                    "blocked": False,
                    "remaining_time": period_seconds,
                    "reason": "limit_exceeded",
                    "current_requests": current_requests,
                    "limit": rule.limit,
                }

        # Autoriser la requête
        return True, {
            "rule": rule_name,
            "blocked": False,
            "current_requests": current_requests,
            "limit": rule.limit,
            "remaining_quota": rule.limit - current_requests,
        }

    def record_request(self, identifier: str, rule_name: str) -> bool:
        """
        Enregistre une requête

        Args:
            identifier: Identifiant
            rule_name: Nom de la règle

        Returns:
            bool: True si enregistré avec succès
        """
        if rule_name not in self.rules:
            return False

        rule = self.rules[rule_name]
        key = self._get_key(identifier, rule)
        status = self.status.get(key)

        if not status:
            status = RateLimitStatus(key=key, requests=deque())
            self.status[key] = status

        now = time.time()
        status.requests.append(now)
        status.total_requests += 1
        status.last_request = datetime.now()

        # Sauvegarder périodiquement
        if status.total_requests % 10 == 0:
            self._save_storage()

        return True

    def reset_limits(self, identifier: str, rule_name: str = None) -> bool:
        """
        Remet à zéro les limites pour un identifiant

        Args:
            identifier: Identifiant
            rule_name: Nom de la règle (optionnel, toutes si None)

        Returns:
            bool: True si réussi
        """
        if rule_name:
            if rule_name not in self.rules:
                return False

            rule = self.rules[rule_name]
            key = self._get_key(identifier, rule)

            if key in self.status:
                del self.status[key]
                self._save_storage()
                return True
        else:
            # Réinitialiser toutes les règles pour cet identifiant
            keys_to_remove = []
            for key, status in self.status.items():
                if identifier in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.status[key]

            if keys_to_remove:
                self._save_storage()
                return True

        return False

    def get_status(self, identifier: str, rule_name: str) -> dict[str, any] | None:
        """
        Récupère le statut de limitation pour un identifiant

        Args:
            identifier: Identifiant
            rule_name: Nom de la règle

        Returns:
            Dict: Informations de statut
        """
        if rule_name not in self.rules:
            return None

        rule = self.rules[rule_name]
        key = self._get_key(identifier, rule)
        status = self.status.get(key)

        if not status:
            return {
                "rule": rule_name,
                "current_requests": 0,
                "limit": rule.limit,
                "remaining_quota": rule.limit,
                "blocked": False,
            }

        # Nettoyer les requêtes expirées
        now = time.time()
        period_seconds = rule.period_seconds()
        cutoff_time = now - period_seconds

        while status.requests and status.requests[0] < cutoff_time:
            status.requests.popleft()

        current_requests = len(status.requests)
        blocked = status.blocked_until and datetime.now() < status.blocked_until

        return {
            "rule": rule_name,
            "current_requests": current_requests,
            "limit": rule.limit,
            "remaining_quota": max(0, rule.limit - current_requests),
            "blocked": blocked,
            "blocked_until": status.blocked_until.isoformat() if status.blocked_until else None,
            "total_requests": status.total_requests,
            "violations": status.violations,
            "last_request": status.last_request.isoformat() if status.last_request else None,
        }

    def add_rule(self, rule: RateLimitRule):
        """
        Ajoute une nouvelle règle

        Args:
            rule: Règle à ajouter
        """
        self.rules[rule.name] = rule
        logger.info(f"Règle ajoutée: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """
        Supprime une règle

        Args:
            rule_name: Nom de la règle

        Returns:
            bool: True si supprimée
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Règle supprimée: {rule_name}")
            return True
        return False

    def get_statistics(self) -> dict[str, any]:
        """
        Récupère les statistiques de rate limiting

        Returns:
            Dict: Statistiques
        """
        stats = {
            "total_rules": len(self.rules),
            "active_limits": len(self.status),
            "blocked_identifiers": 0,
            "total_requests": 0,
            "total_violations": 0,
            "rules": {},
        }

        now = datetime.now()

        for key, status in self.status.items():
            if status.blocked_until and now < status.blocked_until:
                stats["blocked_identifiers"] += 1

            stats["total_requests"] += status.total_requests
            stats["total_violations"] += status.violations

        for rule_name, rule in self.rules.items():
            stats["rules"][rule_name] = {
                "limit": rule.limit,
                "period": rule.period.value,
                "enabled": rule.enabled,
                "active_limiters": len([k for k in self.status.keys() if rule_name in k]),
            }

        return stats


# Instance globale du rate limiter
rate_limiter = RateLimiter()


def rate_limit(rule_name: str, identifier_func: callable = None):
    """
    Décorateur pour appliquer un rate limiting

    Args:
        rule_name: Nom de la règle à appliquer
        identifier_func: Fonction pour obtenir l'identifiant (optionnel)

    Returns:
        function: Fonction décorée
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Obtenir l'identifiant
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                # Utiliser l'IP par défaut (simulation)
                identifier = "127.0.0.1"

            # Vérifier le rate limit
            allowed, info = rate_limiter.check_rate_limit(identifier, rule_name)

            if not allowed:
                from .security_validator import ValidationError

                raise ValidationError(f"Rate limit dépassé: {info.get('reason', 'unknown')}")

            # Enregistrer la requête
            rate_limiter.record_request(identifier, rule_name)

            # Exécuter la fonction
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Fonctions utilitaires
def check_api_limit(identifier: str) -> tuple[bool, dict]:
    """Vérifie la limite API"""
    return rate_limiter.check_rate_limit(identifier, "api_default")


def check_login_limit(identifier: str) -> tuple[bool, dict]:
    """Vérifie la limite de connexion"""
    return rate_limiter.check_rate_limit(identifier, "login_attempts")


def record_api_call(identifier: str):
    """Enregistre un appel API"""
    return rate_limiter.record_request(identifier, "api_calls")


def record_login_attempt(identifier: str):
    """Enregistre une tentative de connexion"""
    return rate_limiter.record_request(identifier, "login_attempts")


def get_rate_limit_status(identifier: str, rule_name: str) -> dict | None:
    """Récupère le statut de rate limiting"""
    return rate_limiter.get_status(identifier, rule_name)


def reset_rate_limits(identifier: str, rule_name: str = None) -> bool:
    """Remet à zéro les limites"""
    return rate_limiter.reset_limits(identifier, rule_name)
