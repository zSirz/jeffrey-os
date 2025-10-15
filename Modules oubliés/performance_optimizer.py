"""
Optimiseur de performance pour Jeffrey OS
Implémente cache LRU, optimisations SQL et monitoring
"""

import hashlib
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from queue import Queue
from typing import Any

logger = logging.getLogger(__name__)


class LRUCache:
    """Cache LRU thread-safe"""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Any:
        """Récupère un élément du cache"""
        with self.lock:
            if key in self.cache:
                # Déplacer à la fin (plus récent)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None

    def set(self, key: str, value: Any) -> None:
        """Ajoute un élément au cache"""
        with self.lock:
            if key in self.cache:
                # Mettre à jour et déplacer à la fin
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Ajouter nouveau
                self.cache[key] = value

                # Supprimer le plus ancien si nécessaire
                if len(self.cache) > self.maxsize:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    self.evictions += 1

    def delete(self, key: str) -> bool:
        """Supprime un élément du cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Vide le cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def stats(self) -> dict[str, Any]:
        """Retourne les statistiques du cache"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total_requests,
            }


@dataclass
class QueryOptimization:
    """Optimisation de requête SQL"""

    original_query: str
    optimized_query: str
    columns: list[str]
    indexes_needed: list[str]
    performance_gain: float = 0.0


class SQLOptimizer:
    """Optimiseur de requêtes SQL"""

    # Colonnes communes par table
    TABLE_COLUMNS = {
        "transactions": [
            "id",
            "type_transaction",
            "montant",
            "compte_id",
            "date_transaction",
            "categorie_id",
            "description",
            "direction",
        ],
        "comptes": [
            "id",
            "nom",
            "type_compte",
            "solde",
            "devise",
            "banque",
            "numero_compte",
            "created_at",
            "updated_at",
        ],
        "categories": ["id", "nom", "type", "couleur", "icone", "parent_id"],
        "plaid_items": [
            "id",
            "item_id",
            "access_token",
            "institution_id",
            "user_id",
            "institution_name",
            "last_sync",
            "created_at",
        ],
        "plaid_accounts": [
            "id",
            "account_id",
            "item_id",
            "name",
            "type",
            "subtype",
            "current_balance",
            "available_balance",
            "iso_currency_code",
            "compte_id",
        ],
    }

    # Indexes recommandés
    RECOMMENDED_INDEXES = {
        "transactions": [
            "idx_transactions_compte_id",
            "idx_transactions_date",
            "idx_transactions_categorie_id",
            "idx_transactions_type",
        ],
        "comptes": ["idx_comptes_type", "idx_comptes_created_at"],
        "plaid_items": ["idx_plaid_items_user_id", "idx_plaid_items_institution_id"],
        "plaid_accounts": ["idx_plaid_accounts_item_id", "idx_plaid_accounts_compte_id"],
    }

    def __init__(self):
        self.optimizations = []
        self.query_cache = LRUCache(500)

    def optimize_query(self, query: str, table_name: str = None) -> QueryOptimization:
        """
        Optimise une requête SQL

        Args:
            query: Requête SQL à optimiser
            table_name: Nom de la table (optionnel)

        Returns:
            QueryOptimization: Optimisation suggérée
        """
        query_lower = query.lower().strip()

        # Détecter SELECT *
        if "select *" in query_lower:
            return self._optimize_select_star(query, table_name)

        # Détecter les requêtes sans WHERE sur de grandes tables
        if "select" in query_lower and "where" not in query_lower:
            return self._optimize_missing_where(query, table_name)

        # Détecter les requêtes sans ORDER BY avec LIMIT
        if "limit" in query_lower and "order by" not in query_lower:
            return self._optimize_missing_order_by(query)

        # Pas d'optimisation nécessaire
        return QueryOptimization(original_query=query, optimized_query=query, columns=[], indexes_needed=[])

    def _optimize_select_star(self, query: str, table_name: str) -> QueryOptimization:
        """Optimise les requêtes SELECT *"""
        if not table_name:
            # Essayer de détecter la table
            query_lower = query.lower()
            for table in self.TABLE_COLUMNS.keys():
                if f"from {table}" in query_lower:
                    table_name = table
                    break

        if table_name and table_name in self.TABLE_COLUMNS:
            columns = self.TABLE_COLUMNS[table_name]
            columns_str = ", ".join(columns)

            # Remplacer SELECT * par les colonnes spécifiques
            optimized_query = query.replace("SELECT *", f"SELECT {columns_str}")
            optimized_query = optimized_query.replace("select *", f"SELECT {columns_str}")

            return QueryOptimization(
                original_query=query,
                optimized_query=optimized_query,
                columns=columns,
                indexes_needed=self.RECOMMENDED_INDEXES.get(table_name, []),
                performance_gain=20.0,  # Estimation 20% d'amélioration
            )

        return QueryOptimization(original_query=query, optimized_query=query, columns=[], indexes_needed=[])

    def _optimize_missing_where(self, query: str, table_name: str) -> QueryOptimization:
        """Optimise les requêtes sans WHERE"""
        # Ajouter LIMIT si absent
        if "limit" not in query.lower():
            optimized_query = f"{query.rstrip(';')} LIMIT 1000"

            return QueryOptimization(
                original_query=query,
                optimized_query=optimized_query,
                columns=[],
                indexes_needed=[],
                performance_gain=30.0,
            )

        return QueryOptimization(original_query=query, optimized_query=query, columns=[], indexes_needed=[])

    def _optimize_missing_order_by(self, query: str) -> QueryOptimization:
        """Optimise les requêtes avec LIMIT sans ORDER BY"""
        # Ajouter ORDER BY sur une colonne d'index
        if "transactions" in query.lower():
            optimized_query = query.replace("LIMIT", "ORDER BY id DESC LIMIT")
        else:
            optimized_query = query.replace("LIMIT", "ORDER BY id LIMIT")

        return QueryOptimization(
            original_query=query,
            optimized_query=optimized_query,
            columns=[],
            indexes_needed=[],
            performance_gain=15.0,
        )


class ConnectionPool:
    """Pool de connexions SQLite"""

    def __init__(self, db_path: str, min_connections: int = 5, max_connections: int = 20):
        self.db_path = db_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool = Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()

        # Créer les connexions initiales
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialise le pool avec les connexions minimales"""
        for _ in range(self.min_connections):
            conn = self._create_connection()
            if conn:
                self.pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection | None:
        """Crée une nouvelle connexion"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Optimisations SQLite
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")

            with self.lock:
                self.active_connections += 1

            return conn
        except Exception as e:
            logger.error(f"Erreur lors de la création de connexion: {e}")
            return None

    @contextmanager
    def get_connection(self):
        """Récupère une connexion du pool"""
        conn = None
        try:
            # Essayer de récupérer une connexion existante
            try:
                conn = self.pool.get_nowait()
            except:
                # Créer une nouvelle connexion si possible
                if self.active_connections < self.max_connections:
                    conn = self._create_connection()
                else:
                    # Attendre qu'une connexion se libère
                    conn = self.pool.get(timeout=5)

            if conn:
                yield conn
            else:
                raise Exception("Impossible d'obtenir une connexion")

        finally:
            if conn:
                # Remettre la connexion dans le pool
                try:
                    self.pool.put_nowait(conn)
                except:
                    # Pool plein, fermer la connexion
                    conn.close()
                    with self.lock:
                        self.active_connections -= 1

    def close_all(self):
        """Ferme toutes les connexions"""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except:
                break

        with self.lock:
            self.active_connections = 0


class PerformanceMonitor:
    """Moniteur de performance"""

    def __init__(self):
        self.query_stats = {}
        self.slow_queries = []
        self.cache_stats = {}
        self.lock = threading.Lock()

    def record_query(self, query: str, execution_time: float, result_count: int = 0):
        """Enregistre les statistiques d'une requête"""
        with self.lock:
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]

            if query_hash not in self.query_stats:
                self.query_stats[query_hash] = {
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                    "total_results": 0,
                }

            stats = self.query_stats[query_hash]
            stats["count"] += 1
            stats["total_time"] += execution_time
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["min_time"] = min(stats["min_time"], execution_time)
            stats["max_time"] = max(stats["max_time"], execution_time)
            stats["total_results"] += result_count

            # Enregistrer les requêtes lentes
            if execution_time > 0.1:  # Plus de 100ms
                self.slow_queries.append(
                    {
                        "query": query,
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat(),
                        "result_count": result_count,
                    }
                )

                # Limiter le nombre de requêtes lentes stockées
                if len(self.slow_queries) > 100:
                    self.slow_queries = self.slow_queries[-50:]

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques de performance"""
        with self.lock:
            return {
                "query_count": len(self.query_stats),
                "slow_queries_count": len(self.slow_queries),
                "top_queries": sorted(self.query_stats.values(), key=lambda x: x["total_time"], reverse=True)[:10],
                "slowest_queries": sorted(self.slow_queries, key=lambda x: x["execution_time"], reverse=True)[:10],
            }


class PerformanceOptimizer:
    """Optimiseur de performance principal"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.cache = LRUCache(1000)
        self.sql_optimizer = SQLOptimizer()
        self.connection_pool = ConnectionPool(db_path)
        self.monitor = PerformanceMonitor()
        self.indexes_created = set()

        # Créer les indexes recommandés
        self._create_recommended_indexes()

    def _create_recommended_indexes(self):
        """Crée les indexes recommandés"""
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_transactions_compte_id ON transactions(compte_id)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date_transaction)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_categorie_id ON transactions(categorie_id)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(type_transaction)",
            "CREATE INDEX IF NOT EXISTS idx_comptes_type ON comptes(type_compte)",
            "CREATE INDEX IF NOT EXISTS idx_comptes_created_at ON comptes(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_plaid_items_user_id ON plaid_items(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_plaid_accounts_item_id ON plaid_accounts(item_id)",
        ]

        try:
            with self.connection_pool.get_connection() as conn:
                for sql in indexes_sql:
                    conn.execute(sql)
                    index_name = sql.split("idx_")[1].split(" ")[0]
                    self.indexes_created.add(f"idx_{index_name}")

                conn.commit()
                logger.info(f"Indexes créés: {len(indexes_sql)}")
        except Exception as e:
            logger.error(f"Erreur lors de la création des indexes: {e}")

    def execute_query(self, query: str, params: tuple = None, cache_key: str = None) -> list[dict]:
        """
        Exécute une requête avec optimisations

        Args:
            query: Requête SQL
            params: Paramètres de la requête
            cache_key: Clé de cache (optionnel)

        Returns:
            List[Dict]: Résultats de la requête
        """
        params = params or ()

        # Vérifier le cache
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Optimiser la requête
        optimization = self.sql_optimizer.optimize_query(query)
        optimized_query = optimization.optimized_query

        # Exécuter la requête
        start_time = time.time()
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(optimized_query, params)
                results = [dict(row) for row in cursor.fetchall()]

                execution_time = time.time() - start_time

                # Enregistrer les statistiques
                self.monitor.record_query(optimized_query, execution_time, len(results))

                # Mettre en cache si demandé
                if cache_key and execution_time > 0.01:  # Cache les requêtes > 10ms
                    self.cache.set(cache_key, results)

                return results

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Erreur lors de l'exécution de la requête: {e}")
            self.monitor.record_query(optimized_query, execution_time, 0)
            raise

    def invalidate_cache(self, pattern: str = None):
        """Invalide le cache"""
        if pattern:
            # Invalider les clés correspondant au pattern
            keys_to_delete = []
            for key in self.cache.cache.keys():
                if pattern in key:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                self.cache.delete(key)
        else:
            # Invalider tout le cache
            self.cache.clear()

    def get_performance_report(self) -> dict[str, Any]:
        """Génère un rapport de performance"""
        return {
            "cache_stats": self.cache.stats(),
            "monitor_stats": self.monitor.get_stats(),
            "indexes_created": list(self.indexes_created),
            "connection_pool_stats": {
                "active_connections": self.connection_pool.active_connections,
                "max_connections": self.connection_pool.max_connections,
            },
        }


# Instance globale
_optimizer = None


def get_optimizer(db_path: str = "cashzen.db") -> PerformanceOptimizer:
    """Récupère l'instance de l'optimiseur"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PerformanceOptimizer(db_path)
    return _optimizer


def cached_query(cache_key: str, ttl: int = 300):
    """
    Décorateur pour mettre en cache les résultats de requête

    Args:
        cache_key: Clé de cache
        ttl: Time-to-live en secondes

    Returns:
        function: Fonction décorée
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_optimizer()

            # Créer une clé unique avec les paramètres
            full_key = f"{cache_key}_{hash(str(args) + str(kwargs))}"

            # Vérifier le cache
            result = optimizer.cache.get(full_key)
            if result is not None:
                return result

            # Exécuter la fonction
            result = func(*args, **kwargs)

            # Mettre en cache
            optimizer.cache.set(full_key, result)

            return result

        return wrapper

    return decorator


# Fonctions utilitaires
def optimize_query(query: str, table_name: str = None) -> str:
    """Optimise une requête SQL"""
    optimizer = get_optimizer()
    optimization = optimizer.sql_optimizer.optimize_query(query, table_name)
    return optimization.optimized_query


def execute_optimized_query(query: str, params: tuple = None, cache_key: str = None) -> list[dict]:
    """Exécute une requête optimisée"""
    optimizer = get_optimizer()
    return optimizer.execute_query(query, params, cache_key)


def clear_cache(pattern: str = None):
    """Vide le cache"""
    optimizer = get_optimizer()
    optimizer.invalidate_cache(pattern)


def get_performance_stats() -> dict[str, Any]:
    """Récupère les statistiques de performance"""
    optimizer = get_optimizer()
    return optimizer.get_performance_report()
