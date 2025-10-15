"""
Memory Port - Interface normalisée pour toutes les implémentations mémoire
Combine l'idée de GPT (port pattern) et Grok (détection runtime)
"""
import logging
import inspect
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryPort:
    """
    Adaptateur universel pour les différentes interfaces mémoire
    Détecte automatiquement la bonne méthode à utiliser
    """

    def __init__(self, impl: Any):
        self.impl = impl
        self.method_name = None
        self.fallback_memory = []  # Buffer de secours
        self.stats = {
            "stores_attempted": 0,
            "stores_succeeded": 0,
            "stores_failed": 0,
            "method_used": None
        }
        self._detect_interface()

    def _detect_interface(self):
        """Détecte automatiquement quelle méthode utiliser"""
        if self.impl is None:
            logger.warning("⚠️ No memory implementation provided - using fallback buffer")
            return

        candidates = ['store', 'add', 'save', 'insert', 'persist', 'append', 'push']

        for method in candidates:
            if hasattr(self.impl, method):
                # Tester si la méthode est callable
                if callable(getattr(self.impl, method)):
                    self.method_name = method
                    logger.info(f"✅ Memory interface detected: '{method}'")
                    self.stats["method_used"] = method
                    break

        # Fallback: tenter __call__
        if not self.method_name and callable(self.impl):
            self.method_name = "__call__"
            logger.info("✅ Memory uses __call__ interface")
            self.stats["method_used"] = "__call__"

        if not self.method_name:
            logger.warning("⚠️ No standard memory interface found - using fallback buffer")

    def store(self, entry: Dict[str, Any]) -> bool:
        """
        Interface unifiée pour stocker en mémoire
        Retourne True si succès, False sinon
        """
        self.stats["stores_attempted"] += 1

        # Ajouter timestamp si absent
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.utcnow().isoformat()

        try:
            if self.impl is None:
                # Fallback buffer uniquement
                self.fallback_memory.append(entry)
                logger.debug(f"Entry stored in fallback buffer (size: {len(self.fallback_memory)})")
                self.stats["stores_succeeded"] += 1
                return True

            elif self.method_name == "__call__":
                result = self.impl(entry)
                self.stats["stores_succeeded"] += 1
                return True

            elif self.method_name:
                method = getattr(self.impl, self.method_name)
                result = method(entry)
                self.stats["stores_succeeded"] += 1
                return True

            else:
                # Fallback buffer
                self.fallback_memory.append(entry)
                logger.debug(f"Entry stored in fallback buffer (size: {len(self.fallback_memory)})")
                self.stats["stores_succeeded"] += 1
                return True

        except Exception as e:
            self.stats["stores_failed"] += 1
            logger.error(f"Memory store failed with {self.method_name}: {e}")

            # Tenter le fallback buffer en cas d'échec
            try:
                self.fallback_memory.append(entry)
                logger.info("Entry saved to fallback buffer after error")
                return False
            except Exception:
                return False

    def search(self, query: str = "", limit: int = 10) -> List[Dict]:
        """Interface unifiée pour rechercher en mémoire"""
        if self.impl is None:
            return self.fallback_memory[-limit:]

        # Chercher la méthode de recherche avec inspection robuste
        for name in ['search', 'find', 'query', 'retrieve', 'get']:
            if hasattr(self.impl, name):
                try:
                    method = getattr(self.impl, name)
                    if not callable(method):
                        continue

                    # Utiliser inspect.signature pour éviter les problèmes __code__
                    try:
                        sig = inspect.signature(method)
                        kwargs = {}

                        if "query" in sig.parameters:
                            kwargs["query"] = query
                        elif "q" in sig.parameters:
                            kwargs["q"] = query
                        else:
                            # Fallback argument positionnel
                            return method(query)

                        if "limit" in sig.parameters:
                            kwargs["limit"] = limit

                        return method(**kwargs)

                    except (ValueError, TypeError):
                        # Si signature inspection échoue, essayer des approches simples
                        try:
                            return method(query=query, limit=limit)
                        except TypeError:
                            try:
                                return method(query)
                            except Exception:
                                pass

                except Exception as e:
                    logger.debug(f"Search method {name} failed: {e}")
                    pass

        # Fallback: retourner le buffer
        return self.fallback_memory[-limit:]

    def get_stats(self) -> Dict:
        """Retourne les statistiques d'utilisation"""
        return {
            **self.stats,
            "fallback_buffer_size": len(self.fallback_memory),
            "success_rate": self.stats["stores_succeeded"] / max(1, self.stats["stores_attempted"]),
            "has_real_impl": self.impl is not None,
            "detected_method": self.method_name
        }