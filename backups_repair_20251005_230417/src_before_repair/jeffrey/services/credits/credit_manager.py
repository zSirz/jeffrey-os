"""
Gestionnaire de crédits et quotas.

Ce module implémente les fonctionnalités essentielles pour gestionnaire de crédits et quotas.
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

import logging

from core.config import Config

logger = logging.getLogger("credit_manager")


class CreditManager:
    """
    Manages credits for AI operations, including reserving and consuming credits.
    """

    def __init__(self, initial_credits=None) -> None:
        """
        Initialize the credit manager with a starting balance.

        Args:
            initial_credits (int, optional): Initial credit balance.
                                           Defaults to Config.DEFAULT_INITIAL_CREDITS.
        """
        self.balance = initial_credits or Config.DEFAULT_INITIAL_CREDITS
        self.reserved = 0
        logger.info("Credit Manager initialized with %d credits", self.balance)

    def get_balance(self) -> Any:
        """
        Get the current available credit balance.

        Returns:
            int: The available credit balance
        """
        return self.balance

    def reserve_credits(self, amount):
        """
        Reserve credits for an operation.

        Args:
            amount (int): Number of credits to reserve

        Returns:
            bool: True if reservation successful, False otherwise
        """
        if amount <= 0:
            logger.warning("Invalid credit reservation amount: %d", amount)
            return False

        if self.balance < amount:
            logger.warning("Insufficient credits: %d < %d", self.balance, amount)
            return False

        self.balance -= amount
        self.reserved += amount
        logger.info("Reserved %d credits. Balance: %d, Reserved: %d", amount, self.balance, self.reserved)
        return True

    def confirm_usage(self, amount):
        """
        Confirm the usage of reserved credits.

        Args:
            amount (int): Number of reserved credits to confirm using

        Returns:
            bool: True if confirmation successful, False otherwise
        """
        if amount <= 0 or amount > self.reserved:
            logger.warning("Invalid credit confirmation: %d (reserved: %d)", amount, self.reserved)
            return False

        self.reserved -= amount
        logger.info("Confirmed usage of %d credits. Reserved: %d", amount, self.reserved)
        return True

    def cancel_reservation(self, amount):
        """
        Cancel a credit reservation and return credits to the balance.

        Args:
            amount (int): Number of reserved credits to return

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        if amount <= 0 or amount > self.reserved:
            logger.warning("Invalid credit cancellation: %d (reserved: %d)", amount, self.reserved)
            return False

        self.balance += amount
        self.reserved -= amount
        logger.info(
            "Cancelled reservation of %d credits. Balance: %d, Reserved: %d",
            amount,
            self.balance,
            self.reserved,
        )
        return True
