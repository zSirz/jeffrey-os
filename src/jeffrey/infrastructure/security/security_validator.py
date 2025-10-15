"""
Validateur de sécurité multi-niveaux.

Ce module implémente les fonctionnalités essentielles pour validateur de sécurité multi-niveaux.
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
import logging
import re
import secrets
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception pour les erreurs de validation"""

    pass


class SecurityLevel(Enum):
    """Niveaux de sécurité pour la validation"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Règle de validation"""

    name: str
    pattern: str
    min_length: int = 0
    max_length: int = 1000
    required: bool = True
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    custom_validator: callable | None = None


class SecurityValidator:
    """Validateur sécurisé pour tous les inputs utilisateur"""

    # Patterns de validation sécurisés
    PATTERNS = {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "phone": r"^\+?[1-9]\d{1,14}$",
        "username": r"^[a-zA-Z0-9_.-]{3,30}$",
        "password": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$",
        "account_id": r"^[a-zA-Z0-9_-]{1,50}$",
        "transaction_id": r"^[a-zA-Z0-9_-]{1,100}$",
        "amount": r"^\d{1,10}(\.\d{1,2})?$",
        "date": r"^\d{4}-\d{2}-\d{2}$",
        "datetime": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$",
        "bank_routing": r"^\d{9}$",
        "bank_account": r"^[a-zA-Z0-9]{4,17}$",
        "currency": r"^[A-Z]{3}$",
        "safe_text": r"^[a-zA-Z0-9\s\-_.,!?()]{0,500}$",
        "safe_filename": r"^[a-zA-Z0-9_.-]{1,255}$",
        "url": r"^https?://[^\s/$.?#].[^\s]*$",
        "json_key": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        "sql_safe": r"^[a-zA-Z0-9_\s\-.,!?()]{0,1000}$",
    }

    # Règles de validation par défaut
    DEFAULT_RULES = {
        "email": ValidationRule("email", PATTERNS["email"], 5, 254, True, SecurityLevel.HIGH),
        "phone": ValidationRule("phone", PATTERNS["phone"], 8, 15, False, SecurityLevel.MEDIUM),
        "username": ValidationRule("username", PATTERNS["username"], 3, 30, True, SecurityLevel.HIGH),
        "password": ValidationRule("password", PATTERNS["password"], 8, 128, True, SecurityLevel.CRITICAL),
        "account_id": ValidationRule("account_id", PATTERNS["account_id"], 1, 50, True, SecurityLevel.HIGH),
        "transaction_id": ValidationRule(
            "transaction_id", PATTERNS["transaction_id"], 1, 100, True, SecurityLevel.HIGH
        ),
        "amount": ValidationRule("amount", PATTERNS["amount"], 1, 15, True, SecurityLevel.MEDIUM),
        "date": ValidationRule("date", PATTERNS["date"], 10, 10, True, SecurityLevel.MEDIUM),
        "datetime": ValidationRule("datetime", PATTERNS["datetime"], 19, 24, True, SecurityLevel.MEDIUM),
        "bank_routing": ValidationRule("bank_routing", PATTERNS["bank_routing"], 9, 9, True, SecurityLevel.CRITICAL),
        "bank_account": ValidationRule("bank_account", PATTERNS["bank_account"], 4, 17, True, SecurityLevel.CRITICAL),
        "currency": ValidationRule("currency", PATTERNS["currency"], 3, 3, True, SecurityLevel.MEDIUM),
        "safe_text": ValidationRule("safe_text", PATTERNS["safe_text"], 0, 500, False, SecurityLevel.MEDIUM),
        "safe_filename": ValidationRule("safe_filename", PATTERNS["safe_filename"], 1, 255, True, SecurityLevel.MEDIUM),
        "url": ValidationRule("url", PATTERNS["url"], 10, 2048, False, SecurityLevel.MEDIUM),
        "json_key": ValidationRule("json_key", PATTERNS["json_key"], 1, 50, True, SecurityLevel.MEDIUM),
        "sql_safe": ValidationRule("sql_safe", PATTERNS["sql_safe"], 0, 1000, False, SecurityLevel.HIGH),
    }

    # Mots-clés dangereux SQL
    SQL_DANGEROUS_KEYWORDS = [
        "drop",
        "delete",
        "insert",
        "update",
        "alter",
        "create",
        "truncate",
        "exec",
        "execute",
        "union",
        "select",
        "script",
        "javascript",
        "eval",
        "onclick",
        "onload",
        "onerror",
        "onmouseover",
        "onfocus",
        "onblur",
    ]

    # Caractères à échapper
    ESCAPE_CHARS = {
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "&": "&amp;",
        "/": "&#x2F;",
        "\\": "&#x5C;",
    }

    def __init__(self) -> None:
        self.rules = self.DEFAULT_RULES.copy()
        self.failed_attempts = {}
        self.rate_limits = {}

    def validate(self, value: Any, rule_name: str, field_name: str = None) -> str:
        """
        Valide une valeur selon une règle

        Args:
            value: Valeur à valider
            rule_name: Nom de la règle à appliquer
            field_name: Nom du champ (pour les erreurs)

        Returns:
            str: Valeur validée et nettoyée

        Raises:
            ValidationError: Si la validation échoue
        """
        if rule_name not in self.rules:
            raise ValidationError(f"Règle de validation '{rule_name}' non trouvée")

        rule = self.rules[rule_name]
        field_name = field_name or rule_name

        # Convertir en string si nécessaire
        if value is None:
            if rule.required:
                raise ValidationError(f"Le champ '{field_name}' est requis")
            return ""

        str_value = str(value).strip()

        # Vérifier la longueur
        if len(str_value) < rule.min_length:
            raise ValidationError(f"Le champ '{field_name}' doit contenir au moins {rule.min_length} caractères")

        if len(str_value) > rule.max_length:
            raise ValidationError(f"Le champ '{field_name}' ne peut pas dépasser {rule.max_length} caractères")

        # Vérifier le pattern
        if str_value and not re.match(rule.pattern, str_value):
            raise ValidationError(f"Le champ '{field_name}' contient des caractères invalides")

        # Vérification custom
        if rule.custom_validator:
            try:
                str_value = rule.custom_validator(str_value)
            except Exception as e:
                raise ValidationError(f"Validation custom échouée pour '{field_name}': {str(e)}")

        # Sécurité supplémentaire selon le niveau
        if rule.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            str_value = self._advanced_sanitize(str_value)

        # Détection d'injection SQL
        if self._detect_sql_injection(str_value):
            raise ValidationError(f"Contenu potentiellement dangereux détecté dans '{field_name}'")

        return str_value

    def validate_dict(self, data: dict[str, Any], rules_map: dict[str, str]) -> dict[str, str]:
        """
        Valide un dictionnaire selon un mapping de règles

        Args:
            data: Données à valider
            rules_map: Mapping champ -> règle

        Returns:
            Dict[str, str]: Données validées

        Raises:
            ValidationError: Si une validation échoue
        """
        validated = {}

        for field_name, rule_name in rules_map.items():
            value = data.get(field_name)
            try:
                validated[field_name] = self.validate(value, rule_name, field_name)
            except ValidationError as e:
                logger.warning(f"Validation échouée pour {field_name}: {e}")
                raise

        return validated

    def sanitize_html(self, text: str) -> str:
        """
        Nettoie le HTML pour éviter les injections XSS

        Args:
            text: Texte à nettoyer

        Returns:
            str: Texte nettoyé
        """
        if not text:
            return ""

        # Échapper les caractères HTML
        for char, escape in self.ESCAPE_CHARS.items():
            text = text.replace(char, escape)

        return text

    def sanitize_sql(self, text: str) -> str:
        """
        Nettoie le texte pour éviter les injections SQL

        Args:
            text: Texte à nettoyer

        Returns:
            str: Texte nettoyé
        """
        if not text:
            return ""

        # Échapper les guillemets
        text = text.replace("'", "''")
        text = text.replace('"', '""')

        return text

    def _advanced_sanitize(self, text: str) -> str:
        """
        Nettoyage avancé pour les champs sensibles

        Args:
            text: Texte à nettoyer

        Returns:
            str: Texte nettoyé
        """
        if not text:
            return ""

        # Supprimer les caractères de contrôle
        text = "".join(char for char in text if ord(char) >= 32 or char in "\t\n\r")

        # Nettoyer HTML et SQL
        text = self.sanitize_html(text)
        text = self.sanitize_sql(text)

        return text

    def _detect_sql_injection(self, text: str) -> bool:
        """
        Détecte les tentatives d'injection SQL

        Args:
            text: Texte à analyser

        Returns:
            bool: True si une injection est détectée
        """
        if not text:
            return False

        text_lower = text.lower()

        # Vérifier les mots-clés dangereux
        for keyword in self.SQL_DANGEROUS_KEYWORDS:
            if keyword in text_lower:
                return True

        # Vérifier les patterns d'injection communs
        injection_patterns = [
            r"'.*or.*'.*=.*'",
            r'".*or.*".*=.*"',
            r";\s*drop\s+table",
            r";\s*delete\s+from",
            r"union\s+select",
            r"--\s*",
            r"/\*.*\*/",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def add_custom_rule(self, name: str, rule: ValidationRule):
        """
        Ajoute une règle de validation personnalisée

        Args:
            name: Nom de la règle
            rule: Règle de validation
        """
        self.rules[name] = rule

    def get_rule(self, name: str) -> ValidationRule | None:
        """
        Récupère une règle de validation

        Args:
            name: Nom de la règle

        Returns:
            ValidationRule: Règle ou None si non trouvée
        """
        return self.rules.get(name)


# Instance globale du validateur
validator = SecurityValidator()


def validate_input(rule_name: str, field_name: str = None):
    """
    Décorateur pour valider les inputs de fonction

    Args:
        rule_name: Nom de la règle de validation
        field_name: Nom du champ (optionnel)

    Returns:
        function: Fonction décorée
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Valider le premier argument si présent
            if args:
                validated_arg = validator.validate(args[0], rule_name, field_name)
                args = (validated_arg,) + args[1:]

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_inputs(rules_map: dict[str, str]):
    """
    Décorateur pour valider plusieurs inputs

    Args:
        rules_map: Mapping argument -> règle

    Returns:
        function: Fonction décorée
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Valider kwargs selon le mapping
            validated_kwargs = {}
            for key, value in kwargs.items():
                if key in rules_map:
                    validated_kwargs[key] = validator.validate(value, rules_map[key], key)
                else:
                    validated_kwargs[key] = value

            return func(*args, **validated_kwargs)

        return wrapper

    return decorator


# Fonctions utilitaires
def is_safe_filename(filename: str) -> bool:
    """Vérifie si un nom de fichier est sécurisé"""
    try:
        validator.validate(filename, "safe_filename")
        return True
    except ValidationError:
        return False


def is_valid_email(email: str) -> bool:
    """Vérifie si un email est valide"""
    try:
        validator.validate(email, "email")
        return True
    except ValidationError:
        return False


def is_strong_password(password: str) -> bool:
    """Vérifie si un mot de passe est fort"""
    try:
        validator.validate(password, "password")
        return True
    except ValidationError:
        return False


def clean_user_input(text: str) -> str:
    """Nettoie un input utilisateur"""
    return validator.sanitize_html(validator.sanitize_sql(text))


def generate_csrf_token() -> str:
    """Génère un token CSRF sécurisé"""
    return secrets.token_urlsafe(32)


def hash_password(password: str) -> str:
    """Hache un mot de passe de manière sécurisée"""
    salt = secrets.token_hex(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return salt + key.hex()


def verify_password(password: str, hashed: str) -> bool:
    """Vérifie un mot de passe haché"""
    try:
        salt = hashed[:64]
        key = hashed[64:]
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return key == new_key.hex()
    except:
        return False
