"""
Module système pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module système pour jeffrey os.
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
import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BudgetExceededException(Exception):
    """Exception quand le budget est dépassé"""

    pass


@dataclass
class APICall:
    """Structure d'un appel API"""

    timestamp: datetime
    api_name: str
    endpoint: str
    parameters: dict[str, Any]
    response_time: float
    estimated_cost: float
    success: bool
    error_message: str | None = None
    model_used: str | None = None
    tokens_used: int | None = None
    user_id: str | None = None
    correlation_id: str = field(default_factory=lambda: hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16])


class EnhancedAuditLogger:
    """Gestionnaire d'audit avec persistance SQLite"""

    def __init__(self, db_path: str | None = None, daily_limit: float = 10.0, monthly_limit: float = 200.0) -> None:
        self.db_path = Path(db_path or "data/audit/jeffrey_audit.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.active_transactions: dict[str, list[APICall]] = {}
        self.daily_spent = 0.0
        self.monthly_spent = 0.0
        self._lock = asyncio.Lock()
        self._init_database()
        logger.info("EnhancedAuditLogger initialisé")

    def _init_database(self):
        """Initialise la base SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT UNIQUE,
                    transaction_id TEXT,
                    timestamp TEXT,
                    api_name TEXT,
                    endpoint TEXT,
                    parameters TEXT,
                    response_time REAL,
                    estimated_cost REAL,
                    success BOOLEAN,
                    status TEXT DEFAULT 'pending'
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_calls(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction ON api_calls(transaction_id)")
            conn.commit()

    async def log_api_call_with_rollback(self, api_call: APICall, transaction_id: str | None = None) -> str:
        """Enregistre un appel API avec support de rollback"""
        async with self._lock:
            if self.daily_spent + api_call.estimated_cost > self.daily_limit:
                raise BudgetExceededException("Budget quotidien dépassé")

            if not transaction_id:
                transaction_id = f"tx_{datetime.now().timestamp():.0f}"

            if transaction_id not in self.active_transactions:
                self.active_transactions[transaction_id] = []

            self.active_transactions[transaction_id].append(api_call)
            self.daily_spent += api_call.estimated_cost
            self.monthly_spent += api_call.estimated_cost

            return transaction_id

    async def commit_transaction(self, transaction_id: str):
        """Valide une transaction"""
        async with self._lock:
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]
                logger.debug(f"Transaction committed: {transaction_id}")

    async def get_current_budget_status(self) -> dict[str, Any]:
        """Retourne le statut actuel du budget"""
        async with self._lock:
            return {
                "daily_spent": self.daily_spent,
                "daily_limit": self.daily_limit,
                "monthly_spent": self.monthly_spent,
                "monthly_limit": self.monthly_limit,
                "utilization_percentage": ((self.daily_spent / self.daily_limit * 100) if self.daily_limit > 0 else 0),
                "remaining_daily": max(0, self.daily_limit - self.daily_spent),
                "remaining_monthly": max(0, self.monthly_limit - self.monthly_spent),
            }

    async def rollback_transaction(self, transaction_id: str, reason: str = ""):
        """Annule une transaction"""
        async with self._lock:
            if transaction_id in self.active_transactions:
                calls = self.active_transactions[transaction_id]
                total_cost = sum(call.estimated_cost for call in calls)
                self.daily_spent -= total_cost
                self.monthly_spent -= total_cost
                del self.active_transactions[transaction_id]
                logger.debug(f"Transaction rolled back: {transaction_id}")

    def get_cost_breakdown(self, period: str = "today") -> dict[str, Any]:
        """Retourne les métriques de coût"""
        return {
            "period": period,
            "daily_spent": round(self.daily_spent, 2),
            "daily_limit": self.daily_limit,
            "daily_remaining": round(self.daily_limit - self.daily_spent, 2),
            "monthly_spent": round(self.monthly_spent, 2),
            "monthly_limit": self.monthly_limit,
        }
