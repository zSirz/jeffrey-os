"""
Module de sécurité pour les appels API coûteux.

Ce module fournit un décorateur et des utilitaires pour sécuriser tous les appels
aux API IA coûteuses (Claude, GPT, Grok, ElevenLabs) en ajoutant automatiquement:
- Estimation du coût via estimate_cost
- Demande d'autorisation utilisateur via ask_user_authorization
- Blocage automatique si l'utilisateur refuse

Usage:
    @secure_api_call(model_name="gpt-4", reason="Génération de texte")
        def ma_fonction(texte):
        # Appel API sécurisé

    # ou avec token_count dynamique
    @secure_api_call(model_name="claude-3", reason="Analyse de sentiment")
            def ma_fonction(texte):
        token_count = len(texte.split()) * 1.3  # Estimation
        # Le décorateur utilisera cette valeur
"""

import functools
import inspect
import logging
from collections.abc import Callable

from Orchestrateur_IA.core.api_guard import ask_user_authorization
from Orchestrateur_IA.core.ia_pricing import estimate_cost, estimate_tokens

logger = logging.getLogger(__name__)


def secure_api_call(
    model_name: str,
    reason: str,
    token_count: int | None = None,
    fallback_func: Callable | None = None,
    user_id: str = "default_user",
):
    """
    Décorateur qui sécurise les appels API en ajoutant estimation de coût et autorisation.

    Args:
        model_name: Nom du modèle IA utilisé (ex: "gpt-4", "claude-3", "elevenlabs")
        reason: Raison de l'appel API (sera affichée à l'utilisateur)
        token_count: Nombre de tokens estimé (si None, utilisera 800 par défaut ou tentera d'estimer)
        fallback_func: Fonction à appeler si l'autorisation est refusée
        user_id: Identifiant de l'utilisateur

    Returns:
        Le décorateur configuré
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Déterminer le nombre de tokens
            actual_token_count = token_count

            # 1. Vérifier si un paramètre nommé 'token_count' existe dans les kwargs            if "token_count" in kwargs:

        if "token_count" in kwargs:
            actual_token_count = kwargs["token_count"]

            # 2. Chercher le paramètre 'text' ou 'prompt' pour estimation
        elif actual_token_count is None:
            text = None
            for param_name in ["text", "prompt", "content", "message", "query"]:
                if param_name in kwargs:
                    text = kwargs[param_name]
                    break

                # Recherche dans les arguments positionnels (selon la signature de la fonction)
        if text is None:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            # Vérifier que nous avons assez d'arguments positionnels
        for i, param_name in enumerate(param_names):
            if i < len(args) and param_name in [
                "text",
                "prompt",
                "content",
                "message",
                "query",
            ]:
                text = args[i]
                break

        # Si texte trouvé, estimer le nombre de tokens
        if text is not None and isinstance(text, str):
            actual_token_count = estimate_tokens(text, model_name)
            logger.debug(f"Tokens estimés pour '{model_name}': {actual_token_count} (basé sur texte)")

        # 3. Utiliser une valeur par défaut si aucune estimation possible
        if actual_token_count is None:
            actual_token_count = 800
            logger.debug(f"Utilisation de la valeur par défaut de {actual_token_count} tokens pour '{model_name}'")

            # Estimer le coût
            estimated_cost = estimate_cost(model_name, token_count=actual_token_count)

        # Demander l'autorisation
        if ask_user_authorization(model_name, estimated_cost, reason=reason, user_id=user_id):
            logger.info(f"Appel API autorisé pour '{model_name}' (coût estimé: {estimated_cost:.4f}$)")
            return func(*args, **kwargs)
        else:
            logger.warning(f"Appel API refusé pour '{model_name}' (coût estimé: {estimated_cost:.4f}$)")

            # Si une fonction de repli est fournie, l'appeler
        if fallback_func:
            logger.info(f"Utilisation de la fonction de repli: {fallback_func.__name__}")
        return fallback_func(*args, **kwargs)

        # Sinon, retourner une valeur par défaut appropriée
        return {
            "success": False,
            "error": "Autorisation refusée par l'utilisateur",
            "model": model_name,
            "cost_estimate": estimated_cost,
            "reason": reason,
        }

        return wrapper

        return decorator


def secure_api_method(
    model_name_attr: str = "model_name",
    reason: str = "Appel API",
    token_count_method: str | None = None,
    fallback_method: str | None = None,
):
    """
    Décorateur pour méthodes de classe qui sécurise les appels API.
    Utilise les attributs de la classe pour configurer la sécurité.

    Args:
        model_name_attr: Nom de l'attribut contenant le nom du modèle (défaut: "model_name")
        reason: Raison fixe ou format pour la raison (peut utiliser {attr} pour insérer des attributs)
        token_count_method: Nom de la méthode estimant le nombre de tokens (optionnel)
        fallback_method: Nom de la méthode à appeler si autorisation refusée (optionnel)

    Returns:
        Le décorateur configuré pour méthodes
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # Obtenir le nom du modèle depuis l'attribut
            model_name = getattr(self, model_name_attr, None)

        if not model_name:
            logger.warning(f"Attribut '{model_name_attr}' non trouvé, impossible de sécuriser l'appel API")
        return method(self, *args, **kwargs)

        # Déterminer la raison de l'appel en formatant avec les attributs si nécessaire
        actual_reason = reason
        if "{" in reason and "}" in reason:
            try:
                format_dict = {
                    attr: getattr(self, attr)
                    for attr in dir(self)
                    if not attr.startswith("_") and not callable(getattr(self, attr))
                }
                actual_reason = reason.format(**format_dict)
            except Exception as e:
                logger.warning(f"Erreur de formatage de la raison: {e}")

            # Estimation du nombre de tokens
            token_count = None
            if token_count_method and hasattr(self, token_count_method):
                try:
                    token_count_func = getattr(self, token_count_method)
                    token_count = token_count_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Erreur lors de l'appel de {token_count_method}: {e}")

            # Fonction de repli
            fallback = None
            if fallback_method and hasattr(self, fallback_method):
                fallback = getattr(self, fallback_method)

            # Obtenir l'ID utilisateur si disponible
            user_id = getattr(self, "user_id", "default_user")

            # Créer et appliquer le décorateur
            secure_decorator = secure_api_call(
                model_name=model_name,
                reason=actual_reason,
                token_count=token_count,
                fallback_func=fallback,
                user_id=user_id,
            )

            # Appliquer le décorateur à la méthode
            secured_method = secure_decorator(method)
            return secured_method(self, *args, **kwargs)

            return wrapper

            return decorator


def get_default_fallback_response():
    """
    Retourne une réponse par défaut quand l'autorisation est refusée.

    Returns:
        Dict contenant la réponse par défaut
    """
    return {
        "success": False,
        "error": "Autorisation refusée par l'utilisateur",
        "response": "L'opération a été annulée car l'autorisation a été refusée.",
        "fallback": True,
    }
