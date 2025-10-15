"""
Memory Federation V2 - Production-ready avec toutes les améliorations
Gère 22+ modules avec budgets, concurrence, privacy
"""

import asyncio
import copy
import hashlib
import hmac
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml

from ..interfaces.protocols import MemoryModule, memory_hash
from ..loaders.secure_module_loader import SecureModuleLoader
from ..utils.async_helpers import LatencyBudget, asyncify
from ..utils.privacy import PrivacyGuard

logger = logging.getLogger(__name__)


@dataclass
class MemoryLayer:
    """Représente une couche de mémoire avec configuration"""

    name: str
    priority: int = 5
    weight: float = 1.0
    timeout_ms: float = 100
    max_concurrency: int = 2
    enabled: bool = True
    lazy: bool = False
    modules: dict[str, Any] = field(default_factory=dict)
    initialized: bool = False
    semaphore: asyncio.Semaphore | None = None

    def __post_init__(self):
        self.semaphore = asyncio.Semaphore(self.max_concurrency)


class MemoryFederationV2:
    """
    Fédération de mémoire production-ready
    Avec budgets, privacy, concurrence contrôlée
    """

    def __init__(self, loader: SecureModuleLoader, config_path: str = "config/federation.yaml"):
        self.loader = loader
        self.config = self._load_config(config_path)
        self.layers = {}
        self.initialized = False
        self.bus = None
        self.privacy_guard = PrivacyGuard()

        # Cache de déduplication
        self.seen_hashes: set[str] = set()
        self.cache_size_limit = self._config_get("memory_federation.cache_size", 10000)

        # Search key for HMAC (lazy initialization)
        self._search_key_cache = None

        # Métriques
        self.stats = defaultdict(
            lambda: {
                "total_modules": 0,
                "loaded_modules": 0,
                "failed_modules": 0,
                "stores": 0,
                "recalls": 0,
                "searches": 0,
                "errors": 0,
                "timeouts": 0,
                "cache_hits": 0,
                "latency_ms_p50": 0,
                "latency_ms_p95": 0,
                "latency_ms_p99": 0,
            }
        )

        # Latences pour métriques
        self.latencies = defaultdict(list)

        # Trace ID courant
        self.current_trace_id = None

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration avec accès au YAML racine"""
        try:
            with open(config_path) as f:
                self.root_config = yaml.safe_load(f) or {}
            return self.root_config.get("memory_federation", {})
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
            self.root_config = {}
            return {"enabled": True, "budget_ms": 400, "layers": {}}

    def _get_search_key(self) -> bytes:
        """Get or generate HMAC key for search index with production safety"""
        import binascii

        # Check if we're in production with search enabled
        is_production = os.environ.get("JEFFREY_MODE") == "production"
        search_enabled = self._config_get("privacy.enable_search_index", False)

        v = os.environ.get("JEFFREY_SEARCH_KEY")

        if is_production and search_enabled and not v:
            raise ValueError(
                "JEFFREY_SEARCH_KEY is required in production when enable_search_index=true\n"
                "Generate with: python -c 'import secrets; print(secrets.token_hex(32))'"
            )

        if not v:
            logger.warning("Using ephemeral search key (DEV ONLY)")
            return os.urandom(32)

        # hex -> bytes if hex detected; otherwise take as-is (base64/bytes)
        try:
            return binascii.unhexlify(v)  # 64 hex chars -> 32 bytes
        except (binascii.Error, ValueError):
            return v.encode() if isinstance(v, str) else v

    def _normalize_for_indexing(self, text: str) -> list[str]:
        """Normalize and tokenize text for indexing"""
        import re
        import unicodedata

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text.lower())
        # Remove non-alphanumeric (keep simple for EN; adapt for FR if needed)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        # Filter short tokens (min length 3)
        tokens = [t for t in text.split() if len(t) >= 3]
        return tokens

    @property
    def search_key(self) -> bytes:
        """Lazy getter for search key"""
        if self._search_key_cache is None:
            self._search_key_cache = self._get_search_key()
        return self._search_key_cache

    def _token_tag(self, token: str, key: bytes = None) -> str:
        """Generate secure HMAC tag for search token"""
        if key is None:
            key = self.search_key
        if not key:
            return hashlib.sha256(token.encode()).hexdigest()[:16]  # Fallback
        return hmac.new(key, token.encode(), hashlib.sha256).hexdigest()[:16]

    def _maybe_decrypt_text(self, text: str, metadata: dict) -> str:
        """
        Decrypt text if necessary during recall

        Args:
            text: Potentially encrypted text
            metadata: Metadata containing encryption flags

        Returns:
            Decrypted or original text
        """
        # Check if text is encrypted
        if isinstance(text, str) and metadata and metadata.get("_original_text_encrypted") is True:
            try:
                # Get KID for key rotation support
                kid = metadata.get("_kid", "v1")

                # Decrypt with appropriate key
                if self.privacy_guard:
                    if hasattr(self.privacy_guard, "decrypt_with_kid"):
                        decrypted = self.privacy_guard.decrypt_with_kid(text.encode(), kid)
                    else:
                        # Fallback to current key
                        decrypted = self.privacy_guard.cipher.decrypt(text.encode())

                    return decrypted.decode()
                else:
                    logger.error("Privacy guard not initialized for decryption")
                    metadata["decryption_error"] = True
                    return "[DECRYPTION_ERROR]"

            except Exception as e:
                logger.error(f"Failed to decrypt (KID={metadata.get('_kid')}): {e}")
                metadata["decryption_error"] = True
                return "[REDACTED]"

        # Text not encrypted, return as-is
        return text

    def _search_encrypted(self, query: str, items: list[dict]) -> list[dict]:
        """
        Search in encrypted items via hashed term index with multi-SKID support

        Args:
            query: Search query
            items: List of potentially encrypted items

        Returns:
            Matching items sorted by relevance
        """
        if not query:
            return items

        # Check if search index is enabled
        if not self._config_get("privacy.enable_search_index", False):
            # If disabled, only return non-encrypted items
            return [item for item in items if not item.get("metadata", {}).get("_original_text_encrypted")]

        # Normalize and tokenize query using robust method
        query_tokens = self._normalize_for_indexing(query)

        # Calculate tags for ALL known keys (multi-SKID support)
        query_tags_by_skid = {}

        if hasattr(self.privacy_guard, "search_keyring"):
            for skid, search_key in self.privacy_guard.search_keyring.items():
                tags = set()
                for token in query_tokens:
                    tag = hmac.new(search_key, token.encode(), hashlib.sha256).hexdigest()[:16]
                    tags.add(tag)
                query_tags_by_skid[skid] = tags
        else:
            # Fallback to single key
            search_key = self._get_search_key()
            tags = set()
            for token in query_tokens:
                tag = hmac.new(search_key, token.encode(), hashlib.sha256).hexdigest()[:16]
                tags.add(tag)
            query_tags_by_skid["s1"] = tags

        # Score each item with the correct key
        scored_items = []
        for item in items:
            metadata = item.get("metadata", {})

            # If item is encrypted, use index
            if metadata.get("_original_text_encrypted"):
                # Get the item's SKID
                item_skid = metadata.get("_skid", "s1")

                # Use corresponding query tags
                if item_skid in query_tags_by_skid:
                    query_tags = query_tags_by_skid[item_skid]
                    search_terms = set(metadata.get("_search_terms", []))

                    score = len(query_tags & search_terms)

                    # Bonus for n-grams if available and enabled
                    if "_ngram_terms" in metadata and self._config_get("privacy.enable_ngram_search", False):
                        # Get the search key for this SKID
                        if hasattr(self.privacy_guard, "get_search_key_for_skid"):
                            key = self.privacy_guard.get_search_key_for_skid(item_skid)
                        else:
                            key = self._get_search_key()

                        # Generate n-gram tags for query
                        query_ngrams = set()
                        for token in query_tokens[:20]:  # Limit for performance
                            for n in [2, 3]:
                                for i in range(len(token) - n + 1):
                                    ngram = token[i : i + n]
                                    tag = hmac.new(key, ngram.encode(), hashlib.sha256).hexdigest()[:8]
                                    query_ngrams.add(tag)

                        ngram_terms = set(metadata.get("_ngram_terms", []))
                        ngram_score = len(query_ngrams & ngram_terms)
                        score += ngram_score * 0.5  # Reduced weight

                    if score > 0:
                        scored_items.append((score, item))

            else:
                # Non-encrypted item, normal search
                text = item.get("text", "").lower()
                score = sum(1 for token in query_tokens if token in text)
                if score > 0:
                    scored_items.append((score, item))

        # Sort by score (descending)
        scored_items.sort(key=lambda x: x[0], reverse=True)

        # Return just the items
        return [item for score, item in scored_items]

    def _define_layers(self):
        """Définit les couches depuis la configuration"""
        layer_configs = self.config.get("layers", {})

        # Définition des modules par couche (comme avant)
        layer_modules = {
            "cortex": {
                "memory_bridge": "src.jeffrey.core.memory.cortex.memory_bridge:MemoryBridge",
            },
            "unified": {
                "unified_memory": "src.jeffrey.core.memory.unified_memory:UnifiedMemory",
                "advanced_unified": "src.jeffrey.core.memory.advanced_unified_memory:AdvancedUnifiedMemory",
            },
            "working": {
                "working_memory": "src.jeffrey.core.memory.working.working_memory:WorkingMemory",
                "voice_memory": "src.jeffrey.core.memory.advanced.voice_memory_manager:VoiceMemoryManager",
                "contextual_memory": "src.jeffrey.core.memory.advanced.contextual_memory_manager:ContextualMemoryManager",
            },
            "living": {
                "living_memory": "src.jeffrey.core.memory.living.living_memory:LivingMemory",
                "consciousness_memory": "src.jeffrey.core.consciousness.jeffrey_living_memory:JeffreyLivingMemory",
                "human_memory": "src.jeffrey.core.memory.jeffrey_human_memory:JeffreyHumanMemory",
            },
            "sensory": {
                "sensorial_memory": "src.jeffrey.core.memory.sensory.sensorial_memory:SensorialMemory",
                "jeffrey_sensory": "src.jeffrey.core.memory.sensory.jeffrey_sensory_memory:JeffreySensoryMemory",
            },
            "emotional": {
                "emotional_memory": "src.jeffrey.core.emotions.memory.emotional_memory:EmotionalMemory",
                "emotional_advanced": "src.jeffrey.core.memory.advanced.emotional_memory:AdvancedEmotionalMemory",
            },
            "managers": {
                "memory_manager": "src.jeffrey.core.memory.memory_manager:MemoryManager",
                "advanced_manager": "src.jeffrey.core.memory.advanced.memory_manager:AdvancedMemoryManager",
            },
            "special": {
                "memory_rituals": "src.jeffrey.core.memory.memory_rituals:MemoryRituals",
                "memory_health": "src.jeffrey.core.memory.memory_health_check:MemoryHealthCheck",
                "memory_sync": "src.jeffrey.core.memory.sync.jeffrey_memory_sync:JeffreyMemorySync",
            },
        }

        # Créer les couches avec configuration
        for layer_name, modules in layer_modules.items():
            layer_config = layer_configs.get(layer_name, {})

            if not layer_config.get("enabled", True):
                logger.info(f"Layer {layer_name} disabled in config")
                continue

            layer = MemoryLayer(
                name=layer_name,
                priority=layer_config.get("priority", 5),
                weight=layer_config.get("weight", 1.0),
                timeout_ms=layer_config.get("timeout_ms", 100),
                max_concurrency=layer_config.get("max_concurrency", 2),
                enabled=layer_config.get("enabled", True),
                lazy=layer_config.get("lazy", False),
            )

            layer.modules = modules  # Import paths pour l'instant
            self.layers[layer_name] = layer

    async def initialize(self, bus=None, trace_id: str | None = None):
        """Initialise la fédération avec traçabilité"""
        self.bus = bus
        self.current_trace_id = trace_id

        logger.info(f"🧠 Initializing Memory Federation V2 (trace: {trace_id})")

        # Définir les couches depuis la config
        self._define_layers()

        # Charger les couches non-lazy par priorité
        sorted_layers = sorted(self.layers.items(), key=lambda x: x[1].priority)

        for layer_name, layer in sorted_layers:
            if not layer.lazy:
                await self._load_layer(layer_name, layer)

        # S'abonner aux événements si bus disponible
        if self.bus:
            await self._setup_bus_subscriptions()

        self.initialized = True

        # Log stats
        total_loaded = sum(len(l.modules) for l in self.layers.values() if l.initialized)
        logger.info(f"✅ Memory Federation initialized: {total_loaded} modules loaded")

    async def _load_layer(self, layer_name: str, layer: MemoryLayer):
        """Charge une couche avec budget et concurrence"""
        if not layer.enabled:
            return

        logger.info(f"Loading memory layer: {layer_name} (priority {layer.priority})")

        budget = LatencyBudget(layer.timeout_ms * len(layer.modules))
        loaded_instances = {}

        # Charger en parallèle avec limite de concurrence
        tasks = []
        for module_name, import_path in layer.modules.items():
            if not budget.has_budget():
                logger.warning(f"Budget exhausted for layer {layer_name}")
                break

            task = self._load_module_safe(module_name, import_path, layer.timeout_ms / 1000.0, layer.semaphore)
            tasks.append((module_name, task))

        # Attendre les résultats
        for module_name, task in tasks:
            try:
                instance = await task
                if instance:
                    loaded_instances[module_name] = self._wrap_module(instance, module_name)
                    self.stats[layer_name]["loaded_modules"] += 1
                    logger.info(f"  ✅ Loaded: {module_name}")
                else:
                    self.stats[layer_name]["failed_modules"] += 1
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load {module_name}: {e}")
                self.stats[layer_name]["failed_modules"] += 1

        # Stocker les instances
        layer.modules = loaded_instances
        layer.initialized = len(loaded_instances) > 0
        self.stats[layer_name]["total_modules"] = len(loaded_instances)

    async def _load_module_safe(
        self, module_name: str, import_path: str, timeout: float, semaphore: asyncio.Semaphore
    ) -> Any | None:
        """Charge un module de manière sécurisée avec semaphore"""
        async with semaphore:
            try:
                from ..loaders.secure_module_loader import ModuleSpec

                spec = ModuleSpec(name=module_name, import_path=import_path, enabled=True, critical=False)
                instance = await asyncify(self.loader._safe_import, spec, timeout=timeout)
                return instance
            except Exception as e:
                logger.error(f"Error loading {module_name}: {e}")
                return None

    def _wrap_module(self, instance: Any, module_name: str) -> MemoryModule:
        """Wrap un module pour conformité avec l'interface"""
        # Si déjà conforme, retourner tel quel
        if isinstance(instance, MemoryModule):
            return instance

        # Sinon, créer un adaptateur
        from .memory_adapter import MemoryAdapter

        return MemoryAdapter(instance, module_name)

    async def _setup_bus_subscriptions(self):
        """Configure les abonnements au bus avec namespacing"""
        await self.bus.subscribe("memory.request.store", self._handle_store)
        await self.bus.subscribe("memory.request.recall", self._handle_recall)
        await self.bus.subscribe("memory.request.search", self._handle_search)
        await self.bus.subscribe("memory.request.consolidate", self._handle_consolidate)
        await self.bus.subscribe("memory.request.forget", self._handle_forget)  # GDPR

    # === MÉTHODES DE REQUÊTE HIÉRARCHIQUES ===

    async def recall_fast(self, user_id: str, limit: int = 5, query: str = "") -> list[dict]:
        """
        Rappel rapide depuis les couches working/contextual uniquement
        Budget: 50ms max
        """
        budget = LatencyBudget(50)
        memories = []

        # Seulement les couches rapides
        fast_layers = ["working"]

        for layer_name in fast_layers:
            if not budget.has_budget(10):
                break

            layer = self.layers.get(layer_name)
            if not layer or not layer.initialized:
                continue

            layer_memories = await self._recall_from_layer(
                layer, user_id, limit, timeout=budget.remaining_ms() / 1000.0
            )
            memories.extend(layer_memories)

        # Apply search if query provided
        if query:
            memories = self._search_encrypted(query, memories)

        # Decrypt results using enumerate to avoid index() issues
        for i, item in enumerate(memories):
            # Deep copy to avoid mutating original
            item_copy = copy.deepcopy(item)

            # Decrypt if necessary
            if "text" in item_copy:
                item_copy["text"] = self._maybe_decrypt_text(item_copy.get("text", ""), item_copy.get("metadata", {}))

            # Clean metadata for user
            metadata = item_copy.get("metadata", {})
            if "_original_text_encrypted" in metadata:
                del metadata["_original_text_encrypted"]
                metadata["contains_pii"] = True
            if "_pii_detected" in metadata:
                del metadata["_pii_detected"]
            # Clean search-related metadata
            if "_search_terms" in metadata:
                del metadata["_search_terms"]
            if "_ngram_terms" in metadata:
                del metadata["_ngram_terms"]
            if "_kid" in metadata:
                del metadata["_kid"]
            if "_skid" in metadata:
                del metadata["_skid"]

            # Replace the item
            memories[i] = item_copy

        return self._deduplicate_memories(memories)[:limit]

    async def recall_deep(self, user_id: str, limit: int = 10, query: str = "") -> list[dict]:
        """
        Rappel profond depuis toutes les couches
        Budget: 400ms max
        """
        budget = LatencyBudget(self.config.get("budget_ms", 400))

        results = await self._recall_with_budget(user_id, limit, budget, include_lazy=True)

        # Apply search if query provided
        if query:
            results = self._search_encrypted(query, results)

        # Decrypt results
        decrypted_results = []
        for item in results:
            # Deep copy to avoid mutating original
            item_copy = copy.deepcopy(item)
            if "text" in item_copy:
                item_copy["text"] = self._maybe_decrypt_text(item_copy.get("text", ""), item_copy.get("metadata", {}))
            # Clean metadata for user
            metadata = item_copy.get("metadata", {})
            if "_original_text_encrypted" in metadata:
                del metadata["_original_text_encrypted"]
            if "_pii_detected" in metadata:
                metadata["contains_pii"] = True
                del metadata["_pii_detected"]
            if "_search_terms" in metadata:
                del metadata["_search_terms"]
            if "_ngram_terms" in metadata:
                del metadata["_ngram_terms"]
            if "_kid" in metadata:
                del metadata["_kid"]
            decrypted_results.append(item_copy)

        return decrypted_results[:limit]

    async def recall_from_all(
        self,
        query: str = "",
        user_id: str = "system",
        layer_types: list[str] | None = None,
        max_results: int = 10,
    ) -> list[dict]:
        """Recall with automatic decryption from all layers"""

        if not self.initialized:
            return []

        # Get all results
        all_results = await self.recall_deep(user_id, max_results * 2, query)

        # Filter by layer types if specified
        if layer_types:
            filtered_results = []
            for item in all_results:
                layer_type = item.get("metadata", {}).get("layer_type", "")
                if layer_type in layer_types:
                    filtered_results.append(item)
            all_results = filtered_results

        return all_results[:max_results]

    async def _recall_with_budget(
        self, user_id: str, limit: int, budget: LatencyBudget, include_lazy: bool = False
    ) -> list[dict]:
        """Rappel avec gestion du budget"""
        all_memories = []

        # Parcourir par priorité
        sorted_layers = sorted(self.layers.items(), key=lambda x: x[1].priority)

        for layer_name, layer in sorted_layers:
            if not budget.has_budget(20):
                logger.debug(f"Budget exhausted at layer {layer_name}")
                break

            if not layer.initialized:
                if layer.lazy and include_lazy:
                    await self._ensure_layer_loaded(layer_name)
                else:
                    continue

            memories = await self._recall_from_layer(
                layer, user_id, limit, timeout=min(layer.timeout_ms, budget.remaining_ms()) / 1000.0
            )
            all_memories.extend(memories)

        # Dédupliquer et trier
        return self._deduplicate_and_sort(all_memories, limit)

    async def _recall_from_layer(self, layer: MemoryLayer, user_id: str, limit: int, timeout: float) -> list[dict]:
        """Rappel depuis une couche avec timeout"""
        layer_memories = []

        # Rappel parallèle depuis tous les modules de la couche
        tasks = []
        for module_name, instance in layer.modules.items():
            if not instance:
                continue

            task = asyncify(instance.recall, user_id, limit, timeout=timeout)
            tasks.append((module_name, task))

        # Collecter les résultats
        for module_name, task in tasks:
            try:
                memories = await task
                if memories:
                    # Ajouter metadata
                    for memory in memories:
                        if isinstance(memory, dict):
                            memory["_source"] = f"{layer.name}.{module_name}"
                            memory["_weight"] = layer.weight
                    layer_memories.extend(memories)
            except Exception as e:
                logger.debug(f"Recall failed in {layer.name}.{module_name}: {e}")
                self.stats[layer.name]["errors"] += 1

        return layer_memories

    def _deduplicate_memories(self, memories: list[dict]) -> list[dict]:
        """Déduplique les mémoires par hash"""
        seen = set()
        unique = []

        for memory in memories:
            # Générer hash
            text = memory.get("text", "")
            user_id = memory.get("user_id", "")
            role = memory.get("role", "")

            if text:
                h = memory_hash(text, user_id, role)
                if h not in seen:
                    seen.add(h)
                    unique.append(memory)
                else:
                    self.stats["global"]["cache_hits"] += 1

        return unique

    def _deduplicate_and_sort(self, memories: list[dict], limit: int) -> list[dict]:
        """Déduplique et trie par timestamp et poids"""
        # Dédupliquer
        unique = self._deduplicate_memories(memories)

        # Trier par timestamp (récent d'abord) et poids
        def sort_key(m):
            timestamp = m.get("timestamp", datetime.min.isoformat())
            weight = m.get("_weight", 1.0)
            return (timestamp, -weight)  # Plus récent et plus de poids en premier

        unique.sort(key=sort_key, reverse=True)

        return unique[:limit]

    # === STORE AVEC PRIVACY ===

    async def store_to_relevant(
        self,
        user_id: str,
        role: str,
        text: str,
        metadata: dict = None,
        layer_types: list[str] | None = None,
    ) -> list[str]:
        """
        Store avec privacy, dédup et chiffrement complets
        """
        if not self.initialized:
            return []

        # Safe metadata copy
        metadata = (metadata or {}).copy()
        metadata.setdefault("trace_id", getattr(self, "current_trace_id", ""))

        # 1. ANONYMISATION USER_ID
        user_id_effective = user_id
        if self._config_get("privacy.anonymize_user_ids", False):
            # Hash user_id pour anonymisation
            import hashlib

            user_id_effective = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        # 2. NORMALISATION POUR DÉDUP
        def normalize_text_local(t: str) -> str:
            """Normalise pour dédup cohérente"""
            return " ".join(t.lower().split())  # lowercase + single spaces

        # 3. DÉDUPLICATION PAR HASH (AVANT chiffrement)
        # Try to import from utils, fallback to local
        try:
            from ..interfaces.protocols import memory_hash as hash_func
        except:

            def hash_func(text: str, user_id: str, role: str) -> str:
                import hashlib

                return hashlib.sha1(f"{text}|{user_id}|{role}".encode()).hexdigest()

        h = hash_func(normalize_text_local(text), user_id_effective, role)

        if h in self.seen_hashes:
            logger.debug(f"Duplicate skipped (hash: {h[:8]}...)")
            self.stats["duplicates_skipped"] = self.stats.get("duplicates_skipped", 0) + 1
            return []

        # Ajouter avec pruning FIFO
        self.seen_hashes.add(h)
        if len(self.seen_hashes) > self.cache_size_limit:
            oldest = next(iter(self.seen_hashes))
            self.seen_hashes.remove(oldest)

        # 4. PRIVACY CHECK & CHIFFREMENT
        text_to_store = text
        if self.privacy_guard and self.privacy_guard.detect_pii(text):
            logger.warning(f"PII detected (trace: {metadata.get('trace_id', '')})")

            # Chiffrer metadata
            metadata = self.privacy_guard.encrypt_if_pii(metadata)

            # Chiffrer TEXTE si configuré (UN SEUL FLAG)
            if self._config_get("privacy.encrypt_pii", False):
                try:
                    # Create search index BEFORE encryption (if enabled)
                    if self._config_get("privacy.enable_search_index", False):
                        # Get current SKID and corresponding key
                        skid = os.environ.get("JEFFREY_SEARCH_KID", "s1")
                        key = None
                        if hasattr(self.privacy_guard, "get_search_key_for_skid"):
                            key = self.privacy_guard.get_search_key_for_skid(skid)
                        else:
                            key = self._get_search_key()

                        # Normalize and tokenize using robust method
                        tokens = self._normalize_for_indexing(text)

                        # Generate HMAC tags for search
                        search_terms = []
                        for token in tokens[:200]:  # Limit to 200 tags
                            search_terms.append(self._token_tag(token, key))

                        metadata["_search_terms"] = search_terms
                        metadata["_skid"] = skid  # Store Search Key ID
                        logger.debug(f"Search index created with SKID: {skid}")

                        # Optional: add n-gram index
                        if self._config_get("privacy.enable_ngram_search", False):
                            ngrams = []
                            for token in tokens[:50]:  # Limit tokens for n-grams
                                # Bi-grams and tri-grams
                                for n in [2, 3]:
                                    for i in range(len(token) - n + 1):
                                        ngram = token[i : i + n]
                                        ngrams.append(self._token_tag(ngram, key)[:8])
                            metadata["_ngram_terms"] = ngrams[:50]  # Limit size

                    # Encrypt text
                    encrypted = self.privacy_guard.cipher.encrypt(text.encode())
                    text_to_store = encrypted.decode()
                    metadata["_original_text_encrypted"] = True
                    metadata["_pii_detected"] = True

                    # Add Key ID for rotation support
                    metadata["_kid"] = os.environ.get("JEFFREY_KID", "v1")

                    # Ne JAMAIS logger le texte brut avec PII
                    logger.info(f"Text encrypted with KID: {metadata['_kid']} (length: {len(text)})")
                except Exception as e:
                    logger.error(f"Encryption failed: {e}")
                    # En cas d'échec, ne pas stocker
                    return []

        # 5. BUDGET & DISTRIBUTION
        budget = LatencyBudget(self._config_get("memory_federation.budget_ms", 400))

        # Déterminer les couches pertinentes
        relevant = self._get_relevant_layers_for_content(text_to_store, metadata)

        # Filter by layer_types if specified
        layers_to_use = []
        for layer_name in relevant:
            layer = self.layers.get(layer_name)
            if layer and layer.initialized:
                if not layer_types or layer_name in layer_types:
                    layers_to_use.append((layer_name, layer))

        stored_in = []
        for layer_name, layer in layers_to_use:
            if not budget.has_budget(20):
                break

            success = await self._store_in_layer(
                layer,
                user_id_effective,
                role,
                text_to_store,
                metadata,
                timeout=min(layer.timeout_ms, budget.remaining_ms()) / 1000.0,
            )

            if success:
                stored_in.extend(success)

        self.stats["global"]["stores"] += 1
        return stored_in

    async def _store_in_layer(
        self, layer: MemoryLayer, user_id: str, role: str, text: str, metadata: dict, timeout: float
    ) -> list[str]:
        """Stocke dans une couche"""
        stored = []

        # Store parallèle dans tous les modules
        tasks = []
        for module_name, instance in layer.modules.items():
            if not instance:
                continue

            payload = {
                "user_id": user_id,
                "role": role,
                "text": text,
                "timestamp": datetime.now().isoformat(),
                **metadata,
            }

            task = asyncify(instance.store, payload, timeout=timeout)
            tasks.append((module_name, task))

        # Collecter les résultats
        for module_name, task in tasks:
            try:
                success = await task
                if success:
                    stored.append(f"{layer.name}.{module_name}")
            except Exception as e:
                logger.debug(f"Store failed in {layer.name}.{module_name}: {e}")
                self.stats[layer.name]["errors"] += 1

        return stored

    def _get_relevant_layers_for_content(self, text: str, metadata: dict) -> list[str]:
        """Détermine les couches pertinentes intelligemment"""
        relevant = []

        # Toujours unified et working
        relevant.extend(["unified", "working"])

        # Analyse du contenu
        text_lower = text.lower()

        # Émotions
        emotion_words = [
            "heureux",
            "triste",
            "content",
            "fâché",
            "peur",
            "joie",
            "anxieux",
            "stressé",
            "déçu",
            "excité",
        ]
        if any(word in text_lower for word in emotion_words):
            relevant.append("emotional")

        # Importance
        if metadata.get("importance", 0) > 0.7:
            relevant.append("living")
            relevant.append("cortex")

        # Sensoriel
        sensory_words = [
            "voir",
            "entendre",
            "sentir",
            "toucher",
            "goût",
            "couleur",
            "son",
            "odeur",
            "texture",
        ]
        if any(word in text_lower for word in sensory_words):
            relevant.append("sensory")

        return list(set(relevant))  # Unique

    # === CONSOLIDATION ET MAINTENANCE ===

    async def consolidate_all(self):
        """Consolide toutes les mémoires"""
        tasks = []

        for layer_name, layer in self.layers.items():
            if not layer.initialized:
                continue

            for module_name, instance in layer.modules.items():
                if hasattr(instance, "consolidate"):
                    task = asyncify(instance.consolidate, timeout=layer.timeout_ms / 1000.0)
                    tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_store(self, envelope):
        """Handle store request from bus"""
        data = envelope.get("data", {})
        result = await self.store_to_relevant(
            data.get("user_id"), data.get("role"), data.get("text"), data.get("metadata")
        )
        return {"stored_in": result}

    async def _handle_recall(self, envelope):
        """Handle recall request from bus"""
        data = envelope.get("data", {})
        result = await self.recall_deep(data.get("user_id"), data.get("limit", 10))
        return {"memories": result}

    async def _handle_search(self, envelope):
        """Handle search request from bus"""
        # Implementation for search
        return {"results": []}

    async def _handle_consolidate(self, envelope):
        """Handle consolidate request from bus"""
        await self.consolidate_all()
        return {"status": "completed"}

    async def _handle_forget(self, envelope):
        """GDPR - Oublie un utilisateur"""
        user_id = envelope.get("data", {}).get("user_id")
        if not user_id:
            return

        logger.warning(f"GDPR forget request for user {user_id}")

        # Parcourir toutes les couches
        for layer in self.layers.values():
            if not layer.initialized:
                continue

            for instance in layer.modules.values():
                if hasattr(instance, "forget_user"):
                    await asyncify(instance.forget_user, user_id)

    # === LAZY LOADING ===

    async def _ensure_layer_loaded(self, layer_name: str):
        """Charge une couche lazy si nécessaire"""
        layer = self.layers.get(layer_name)

        if not layer or layer.initialized:
            return

        if layer.lazy:
            logger.info(f"Lazy loading layer: {layer_name}")
            await self._load_layer(layer_name, layer)

    # === MÉTRIQUES ===

    def _update_latency_metrics(self, operation: str, latency_ms: float):
        """Met à jour les métriques de latence"""
        self.latencies[operation].append(latency_ms)

        # Garder seulement les 1000 dernières
        if len(self.latencies[operation]) > 1000:
            self.latencies[operation] = self.latencies[operation][-1000:]

        # Calculer les percentiles
        if self.latencies[operation]:
            sorted_latencies = sorted(self.latencies[operation])
            n = len(sorted_latencies)

            self.stats["global"][f"latency_{operation}_p50"] = sorted_latencies[n // 2]
            self.stats["global"][f"latency_{operation}_p95"] = sorted_latencies[int(n * 0.95)]
            self.stats["global"][f"latency_{operation}_p99"] = sorted_latencies[int(n * 0.99)]

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques détaillées"""
        layer_stats = {}

        for name, layer in self.layers.items():
            layer_stats[name] = {
                "initialized": layer.initialized,
                "modules": len(layer.modules),
                "priority": layer.priority,
                "weight": layer.weight,
                **self.stats.get(name, {}),
            }

        return {
            "layers": layer_stats,
            "global": dict(self.stats.get("global", {})),
            "initialized": self.initialized,
            "config": {
                "budget_ms": self.config.get("budget_ms", 400),
                "privacy_enabled": self._config_get("privacy.encrypt_pii", False),
            },
        }

    def _config_get(self, path: str, default: Any = None) -> Any:
        """Helper universel pour config nested (local puis root)"""

        def get_from(d: dict, p: str):
            v = d
            for k in p.split("."):
                if not isinstance(v, dict):
                    return None
                v = v.get(k)
            return v

        # Chercher dans federation config puis root
        result = get_from(self.config, path)
        if result is None:
            result = get_from(getattr(self, "root_config", {}), path)
        return result if result is not None else default
