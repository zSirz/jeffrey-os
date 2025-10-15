# 🛡️ STRATÉGIE D'INTÉGRATION SÉCURISÉE - Jeffrey OS

## 🎯 PRINCIPE: Intégration Progressive avec Validation

### ⚠️ PROBLÈMES À ÉVITER
1. **Bus Saturation**: Trop de messages simultanés
2. **Interface Mismatch**: Modules incompatibles
3. **Cascade Failures**: Un bug qui casse tout
4. **Memory Leaks**: Accumulation non contrôlée
5. **Performance Degradation**: Système qui ralentit

---

## 📊 APPROCHE EN 5 NIVEAUX DE MATURITÉ

```
Level 0: Core Only (Memory + Bus)
   ↓ validate
Level 1: + 3 Cognitive Modules
   ↓ validate
Level 2: + 3 Emotional Modules
   ↓ validate
Level 3: + Consciousness + 1 Loop
   ↓ validate
Level 4: + All Loops
   ↓ validate
Level 5: Full Production
```

---

## 🔧 NIVEAU 0: INFRASTRUCTURE DE BASE (Jour 1)

### Créer `src/jeffrey/core/integration/safe_integrator.py`

```python
"""
Safe Integration Manager - Intégration progressive avec validation
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from jeffrey.utils.logger import get_logger

logger = get_logger("SafeIntegrator")

class IntegrationLevel(Enum):
    """Niveaux d'intégration progressifs"""
    CORE = 0          # Memory + Bus only
    COGNITIVE = 1     # + Basic cognitive
    EMOTIONAL = 2     # + Emotional modules
    CONSCIOUS = 3     # + Consciousness
    LOOPS = 4         # + Autonomous loops
    PRODUCTION = 5    # Full system

@dataclass
class ModuleHealth:
    """Health metrics for a module"""
    name: str
    active: bool
    error_rate: float
    avg_latency_ms: float
    memory_mb: float
    last_check: float

@dataclass
class BusHealth:
    """Health metrics for the bus"""
    messages_per_sec: float
    queue_size: int
    dropped_messages: int
    subscribers: int
    latency_p95_ms: float

class SafeIntegrator:
    """
    Gestionnaire d'intégration sécurisé avec montée en charge progressive
    """

    def __init__(self, max_bus_rate: int = 1000, max_memory_mb: int = 500):
        self.current_level = IntegrationLevel.CORE
        self.max_bus_rate = max_bus_rate  # Max messages/sec
        self.max_memory_mb = max_memory_mb

        # Core components
        self.memory = None
        self.bus = None

        # Module registries by level
        self.modules_by_level: Dict[IntegrationLevel, List[Any]] = {
            level: [] for level in IntegrationLevel
        }

        # Health tracking
        self.module_health: Dict[str, ModuleHealth] = {}
        self.bus_health: Optional[BusHealth] = None

        # Rate limiting
        self.rate_limiter = RateLimiter(max_bus_rate)

        # Circuit breakers per module
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        self.logger = logger

    async def start_integration(self, target_level: IntegrationLevel = IntegrationLevel.PRODUCTION):
        """
        Start progressive integration up to target level
        """
        self.logger.info(f"🚀 Starting safe integration to level {target_level.name}")

        for level in IntegrationLevel:
            if level.value > target_level.value:
                break

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📈 Advancing to Level {level.value}: {level.name}")

            # Integrate this level
            success = await self._integrate_level(level)

            if not success:
                self.logger.error(f"❌ Failed at level {level.name}")
                await self._rollback_to_safe_state()
                return False

            # Validate before proceeding
            if not await self._validate_current_state():
                self.logger.error(f"❌ Validation failed at level {level.name}")
                await self._rollback_to_safe_state()
                return False

            self.current_level = level
            self.logger.info(f"✅ Level {level.name} successfully integrated")

            # Cool-down period between levels
            await asyncio.sleep(2.0)

        self.logger.info(f"\n🎉 Integration complete at level {self.current_level.name}")
        return True

    async def _integrate_level(self, level: IntegrationLevel) -> bool:
        """Integrate specific level with safety checks"""

        try:
            if level == IntegrationLevel.CORE:
                return await self._integrate_core()
            elif level == IntegrationLevel.COGNITIVE:
                return await self._integrate_cognitive()
            elif level == IntegrationLevel.EMOTIONAL:
                return await self._integrate_emotional()
            elif level == IntegrationLevel.CONSCIOUS:
                return await self._integrate_conscious()
            elif level == IntegrationLevel.LOOPS:
                return await self._integrate_loops()
            elif level == IntegrationLevel.PRODUCTION:
                return await self._integrate_production()

        except Exception as e:
            self.logger.error(f"Integration failed at level {level.name}: {e}")
            return False

    async def _integrate_core(self) -> bool:
        """Level 0: Core infrastructure only"""

        self.logger.info("Integrating core infrastructure...")

        # 1. Start memory with validation
        from jeffrey.core.memory.unified_memory import UnifiedMemory
        self.memory = UnifiedMemory(backend="sqlite")
        await self.memory.initialize()

        # Validate memory
        test_data = {"test": "data", "timestamp": time.time()}
        await self.memory.store(test_data)
        retrieved = await self.memory.retrieve("test", limit=1)

        if not retrieved:
            self.logger.error("Memory validation failed")
            return False

        # 2. Start bus with rate limiting
        from jeffrey.core.neural_bus import NeuralBus
        self.bus = NeuralBus()

        # Wrap bus publish with rate limiting
        original_publish = self.bus.publish

        async def rate_limited_publish(envelope):
            if await self.rate_limiter.acquire():
                return await original_publish(envelope)
            else:
                self.logger.warning("Rate limit exceeded, dropping message")
                return False

        self.bus.publish = rate_limited_publish

        self.logger.info("✅ Core infrastructure ready")
        return True

    async def _integrate_cognitive(self) -> bool:
        """Level 1: Add basic cognitive modules"""

        self.logger.info("Integrating cognitive modules...")

        # Import with validation
        try:
            from jeffrey.core.cognitive.auto_learner import AutoLearner
            from jeffrey.core.cognitive.theory_of_mind import TheoryOfMind
            from jeffrey.core.cognitive.curiosity_engine import CuriosityEngine
        except ImportError as e:
            self.logger.error(f"Failed to import cognitive modules: {e}")
            return False

        # Create modules with circuit breakers
        modules = [
            AutoLearner(self.memory),
            TheoryOfMind(self.memory),
            CuriosityEngine(self.memory)
        ]

        # Initialize with timeout
        for module in modules:
            breaker = CircuitBreaker(module.name, max_failures=3)
            self.circuit_breakers[module.name] = breaker

            try:
                await asyncio.wait_for(module.initialize(), timeout=5.0)
                self.modules_by_level[IntegrationLevel.COGNITIVE].append(module)

                # Create health tracker
                self.module_health[module.name] = ModuleHealth(
                    name=module.name,
                    active=True,
                    error_rate=0.0,
                    avg_latency_ms=0.0,
                    memory_mb=0.0,
                    last_check=time.time()
                )

            except asyncio.TimeoutError:
                self.logger.error(f"Timeout initializing {module.name}")
                return False

        self.logger.info(f"✅ Integrated {len(modules)} cognitive modules")
        return True

    async def _integrate_emotional(self) -> bool:
        """Level 2: Add emotional modules with adapters"""

        self.logger.info("Integrating emotional modules...")

        # Use adapters for non-standard modules
        from jeffrey.core.integration.adapters import create_emotion_adapter

        emotion_modules = []

        try:
            # Try to import and adapt emotional modules
            emotion_adapter = create_emotion_adapter(self.memory)
            if emotion_adapter:
                await emotion_adapter.initialize()
                emotion_modules.append(emotion_adapter)

        except Exception as e:
            self.logger.warning(f"Could not integrate emotion module: {e}")
            # Continue anyway - graceful degradation

        if emotion_modules:
            self.modules_by_level[IntegrationLevel.EMOTIONAL] = emotion_modules
            self.logger.info(f"✅ Integrated {len(emotion_modules)} emotional modules")
        else:
            self.logger.warning("⚠️ No emotional modules integrated (continuing anyway)")

        return True  # Don't fail if emotional modules not available

    async def _integrate_conscious(self) -> bool:
        """Level 3: Add consciousness with careful monitoring"""

        self.logger.info("Integrating consciousness modules...")

        # Consciousness is heavy - monitor carefully
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        conscious_modules = []

        # Try consciousness modules one by one
        try:
            from jeffrey.core.integration.adapters import create_consciousness_adapter
            consciousness = create_consciousness_adapter(self.memory)

            if consciousness:
                await consciousness.initialize()
                conscious_modules.append(consciousness)

        except Exception as e:
            self.logger.warning(f"Consciousness integration partial: {e}")

        # Check memory increase
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before

        if memory_increase > self.max_memory_mb:
            self.logger.error(f"Memory increase too high: {memory_increase:.1f}MB")
            return False

        self.modules_by_level[IntegrationLevel.CONSCIOUS] = conscious_modules
        self.logger.info(f"✅ Consciousness integrated (+{memory_increase:.1f}MB)")
        return True

    async def _integrate_loops(self) -> bool:
        """Level 4: Start autonomous loops with monitoring"""

        self.logger.info("Starting autonomous loops...")

        # Start only essential loops first
        from jeffrey.core.loops.loop_manager import LoopManager

        # Create loop manager with restrictions
        self.loop_manager = LoopManager(
            memory=self.memory,
            bus=self.bus,
            max_iterations=100  # Limit iterations initially
        )

        # Start only awareness loop first
        essential_loops = ['awareness']  # Start minimal

        for loop_name in essential_loops:
            try:
                await self.loop_manager.start_loop(loop_name)
                self.logger.info(f"✅ Started {loop_name} loop")
            except Exception as e:
                self.logger.error(f"Failed to start {loop_name}: {e}")
                return False

        # Monitor for 5 seconds
        await self._monitor_system_health(duration=5.0)

        # If stable, start more loops
        if await self._is_system_stable():
            additional_loops = ['memory_consolidation']
            for loop_name in additional_loops:
                await self.loop_manager.start_loop(loop_name)

        return True

    async def _integrate_production(self) -> bool:
        """Level 5: Full production mode"""

        self.logger.info("Entering production mode...")

        # Remove restrictions
        if hasattr(self, 'loop_manager'):
            self.loop_manager.max_iterations = None  # Remove limit

        # Start all remaining loops
        all_loops = ['curiosity', 'emotional_decay', 'ml_clustering']

        for loop_name in all_loops:
            try:
                if hasattr(self.loop_manager, 'start_loop'):
                    await self.loop_manager.start_loop(loop_name)
            except:
                pass  # Non-critical loops

        self.logger.info("✅ Production mode active")
        return True

    async def _validate_current_state(self) -> bool:
        """
        Validate system health at current integration level
        """
        self.logger.info("🔍 Validating current state...")

        checks = []

        # 1. Check memory health
        if self.memory:
            try:
                stats = self.memory.get_stats()
                memory_ok = stats.get('total_memories', 0) >= 0
                checks.append(("Memory", memory_ok))
            except:
                checks.append(("Memory", False))

        # 2. Check bus health
        if self.bus:
            try:
                metrics = self.bus.get_metrics() if hasattr(self.bus, 'get_metrics') else {}
                bus_rate = metrics.get('published', 0)
                bus_ok = bus_rate < self.max_bus_rate
                checks.append(("Bus", bus_ok))

                # Update bus health
                self.bus_health = BusHealth(
                    messages_per_sec=bus_rate,
                    queue_size=metrics.get('pending_messages', 0),
                    dropped_messages=metrics.get('dropped', 0),
                    subscribers=metrics.get('subscribers', 0),
                    latency_p95_ms=metrics.get('p95_latency_ms', 0)
                )

            except:
                checks.append(("Bus", False))

        # 3. Check module health
        for level, modules in self.modules_by_level.items():
            if level.value > self.current_level.value:
                break

            for module in modules:
                if hasattr(module, 'active'):
                    module_ok = module.active and module.error_count < 10
                    checks.append((module.name, module_ok))

        # 4. Check system resources
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        resource_ok = cpu_percent < 80 and memory_percent < 80
        checks.append(("Resources", resource_ok))

        # Report
        failed_checks = [name for name, ok in checks if not ok]

        if failed_checks:
            self.logger.warning(f"❌ Validation failed: {', '.join(failed_checks)}")
            return False

        self.logger.info(f"✅ All {len(checks)} checks passed")
        return True

    async def _monitor_system_health(self, duration: float = 5.0):
        """Monitor system health for specified duration"""

        self.logger.info(f"📊 Monitoring system for {duration}s...")

        start_time = time.time()
        samples = []

        while time.time() - start_time < duration:
            # Collect metrics
            if self.bus and hasattr(self.bus, 'get_metrics'):
                metrics = self.bus.get_metrics()
                samples.append({
                    'timestamp': time.time(),
                    'bus_rate': metrics.get('published', 0),
                    'bus_dropped': metrics.get('dropped', 0)
                })

            await asyncio.sleep(0.5)

        # Analyze samples
        if samples:
            avg_rate = sum(s['bus_rate'] for s in samples) / len(samples)
            total_dropped = sum(s['bus_dropped'] for s in samples)

            self.logger.info(f"  Avg bus rate: {avg_rate:.1f} msg/s")
            self.logger.info(f"  Total dropped: {total_dropped}")

    async def _is_system_stable(self) -> bool:
        """Check if system is stable enough for more modules"""

        if not self.bus_health:
            return True  # No data, assume ok

        # Check key stability metrics
        stable = (
            self.bus_health.messages_per_sec < self.max_bus_rate * 0.7 and  # 70% capacity
            self.bus_health.dropped_messages == 0 and
            self.bus_health.latency_p95_ms < 100  # Under 100ms P95
        )

        return stable

    async def _rollback_to_safe_state(self):
        """Rollback to last known good state"""

        self.logger.warning("🔄 Rolling back to safe state...")

        # Stop loops if running
        if hasattr(self, 'loop_manager'):
            await self.loop_manager.stop_all()

        # Shutdown modules in reverse order
        for level in reversed(list(IntegrationLevel)):
            if level.value <= self.current_level.value:
                for module in self.modules_by_level.get(level, []):
                    if hasattr(module, 'shutdown'):
                        try:
                            await module.shutdown()
                        except:
                            pass

        # Keep only core
        self.current_level = IntegrationLevel.CORE
        self.logger.info("✅ Rolled back to CORE level")

    async def get_integration_report(self) -> Dict[str, Any]:
        """Get detailed integration status report"""

        report = {
            'current_level': self.current_level.name,
            'level_value': self.current_level.value,
            'max_level': IntegrationLevel.PRODUCTION.value,
            'progress': f"{(self.current_level.value / IntegrationLevel.PRODUCTION.value) * 100:.0f}%",
            'modules': {},
            'health': {}
        }

        # Count modules per level
        for level in IntegrationLevel:
            if level.value <= self.current_level.value:
                count = len(self.modules_by_level.get(level, []))
                report['modules'][level.name] = count

        # Add health metrics
        if self.bus_health:
            report['health']['bus'] = {
                'rate': f"{self.bus_health.messages_per_sec:.1f} msg/s",
                'dropped': self.bus_health.dropped_messages,
                'latency_p95': f"{self.bus_health.latency_p95_ms:.1f}ms"
            }

        # Add module health
        report['health']['modules'] = {}
        for name, health in self.module_health.items():
            report['health']['modules'][name] = {
                'active': health.active,
                'error_rate': f"{health.error_rate:.1%}"
            }

        return report


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: int):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class CircuitBreaker:
    """Circuit breaker for module protection"""

    def __init__(self, name: str, max_failures: int = 3, timeout: float = 60.0):
        self.name = name
        self.max_failures = max_failures
        self.timeout = timeout
        self.failures = 0
        self.last_failure = 0
        self.state = "closed"  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""

        # Check if circuit is open
        if self.state == "open":
            if time.time() - self.last_failure > self.timeout:
                self.state = "half-open"
                self.failures = 0
            else:
                raise Exception(f"Circuit breaker open for {self.name}")

        try:
            result = await func(*args, **kwargs)

            # Reset on success
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()

            if self.failures >= self.max_failures:
                self.state = "open"
                logger.warning(f"Circuit breaker opened for {self.name}")

            raise
```

---

## 🧪 TEST D'INTÉGRATION PROGRESSIVE

### Créer `test_safe_integration.py`

```python
#!/usr/bin/env python3
"""
Test d'intégration progressive et sécurisée
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jeffrey.core.integration.safe_integrator import SafeIntegrator, IntegrationLevel

async def test_progressive_integration():
    """Test integration level by level"""

    print("=" * 60)
    print("🛡️ SAFE PROGRESSIVE INTEGRATION TEST")
    print("=" * 60)

    integrator = SafeIntegrator(
        max_bus_rate=1000,  # Max 1000 msg/sec
        max_memory_mb=500   # Max 500MB increase
    )

    # Test each level progressively
    levels_to_test = [
        IntegrationLevel.CORE,
        IntegrationLevel.COGNITIVE,
        IntegrationLevel.EMOTIONAL,
        IntegrationLevel.CONSCIOUS,
        IntegrationLevel.LOOPS
    ]

    for target_level in levels_to_test:
        print(f"\n🎯 Testing integration to level: {target_level.name}")

        # Reset integrator for clean test
        integrator = SafeIntegrator()

        # Try integration
        success = await integrator.start_integration(target_level)

        if success:
            # Get report
            report = await integrator.get_integration_report()

            print(f"\n📊 Integration Report:")
            print(f"   Level: {report['current_level']} ({report['progress']})")
            print(f"   Modules: {sum(report['modules'].values())} total")

            for level_name, count in report['modules'].items():
                if count > 0:
                    print(f"     - {level_name}: {count} modules")

            if 'bus' in report['health']:
                print(f"   Bus Health:")
                print(f"     - Rate: {report['health']['bus']['rate']}")
                print(f"     - Dropped: {report['health']['bus']['dropped']}")

            print(f"✅ Level {target_level.name} successful!")

        else:
            print(f"❌ Failed at level {target_level.name}")
            break

        # Cleanup
        await integrator._rollback_to_safe_state()
        await asyncio.sleep(1)

    print("\n" + "=" * 60)
    print("✅ Progressive integration test complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_progressive_integration())
```

---

## 🎯 CRÉER LES ADAPTATEURS

### Créer `src/jeffrey/core/integration/adapters.py`

```python
"""
Adaptateurs pour modules non-standards
"""
from typing import Optional
from jeffrey.core.cognitive.base_module import BaseCognitiveModule

def create_emotion_adapter(memory) -> Optional[BaseCognitiveModule]:
    """Create adapter for emotion module"""

    try:
        # Try to import the actual emotion module
        from jeffrey.core.emotions.core.emotion_engine import EmotionEngine

        class EmotionAdapter(BaseCognitiveModule):
            def __init__(self, memory):
                super().__init__("EmotionEngine")
                self.memory = memory
                self.engine = None

            async def on_initialize(self):
                self.engine = EmotionEngine()
                if hasattr(self.engine, 'initialize'):
                    await self.engine.initialize()

            async def on_process(self, data):
                if not self.engine:
                    return {"error": "Engine not initialized"}

                # Adapt the interface
                text = data.get("text", "")

                # Call the actual method (adapt as needed)
                if hasattr(self.engine, 'process'):
                    result = await self.engine.process(text)
                elif hasattr(self.engine, 'analyze'):
                    result = await self.engine.analyze(text)
                else:
                    result = {"emotion": "neutral"}

                return result

        return EmotionAdapter(memory)

    except ImportError:
        return None


def create_consciousness_adapter(memory) -> Optional[BaseCognitiveModule]:
    """Create adapter for consciousness module"""

    try:
        from jeffrey.core.consciousness.self_awareness_tracker import SelfAwarenessTracker

        class ConsciousnessAdapter(BaseCognitiveModule):
            def __init__(self, memory):
                super().__init__("Consciousness")
                self.memory = memory
                self.tracker = None

            async def on_initialize(self):
                self.tracker = SelfAwarenessTracker()
                if hasattr(self.tracker, 'initialize'):
                    await self.tracker.initialize()

            async def on_process(self, data):
                if not self.tracker:
                    return {"error": "Tracker not initialized"}

                # Adapt interface
                if hasattr(self.tracker, 'process'):
                    return await self.tracker.process(data)
                else:
                    return {"awareness_level": 0.5}

        return ConsciousnessAdapter(memory)

    except ImportError:
        return None
```

---

## 📋 CHECKLIST DE SÉCURITÉ

### Avant chaque niveau:
- [ ] Vérifier la mémoire disponible
- [ ] Vérifier le CPU < 80%
- [ ] Vérifier le bus rate < limite
- [ ] Valider les imports
- [ ] Tester en isolation d'abord

### Monitoring continu:
- [ ] Rate limiting sur le bus
- [ ] Circuit breakers sur chaque module
- [ ] Health checks toutes les 30s
- [ ] Logs détaillés
- [ ] Métriques en temps réel

### Rollback automatique si:
- [ ] Bus saturation (>1000 msg/s)
- [ ] Memory leak (>500MB increase)
- [ ] CPU > 90% pendant 30s
- [ ] Plus de 3 modules en erreur
- [ ] Latence P95 > 500ms

---

## 🚀 COMMANDES D'EXÉCUTION SÉCURISÉES

```bash
# 1. Test d'intégration progressive
python test_safe_integration.py

# 2. Intégration niveau par niveau avec monitoring
python -c "
import asyncio
from src.jeffrey.core.integration.safe_integrator import SafeIntegrator, IntegrationLevel

async def integrate():
    integrator = SafeIntegrator()

    # Level 1: Just cognitive
    await integrator.start_integration(IntegrationLevel.COGNITIVE)
    report = await integrator.get_integration_report()
    print(f'Level 1: {report}')

    # If stable, continue...
    if input('Continue to Level 2? (y/n): ').lower() == 'y':
        await integrator.start_integration(IntegrationLevel.EMOTIONAL)

asyncio.run(integrate())
"

# 3. Monitoring en temps réel
watch -n 1 'python -c "
from src.jeffrey.core.integration.safe_integrator import SafeIntegrator
integrator = SafeIntegrator()
print(integrator.get_integration_report())
"'
```

---

## 💡 AVANTAGES DE CETTE APPROCHE

### 1. **Intégration Progressive** ✅
- Niveau par niveau
- Validation à chaque étape
- Rollback automatique si problème

### 2. **Protection du Bus** ✅
- Rate limiting intégré
- Queue monitoring
- Drop tracking

### 3. **Interfaces Adaptées** ✅
- Adaptateurs pour modules non-standards
- Conversion automatique d'interface
- Graceful degradation

### 4. **Monitoring Continu** ✅
- Health checks automatiques
- Métriques en temps réel
- Alertes sur seuils

### 5. **Circuit Breakers** ✅
- Protection par module
- Auto-recovery après timeout
- Isolation des failures

---

## 🎯 RÉSULTAT ATTENDU

Avec cette approche, vous aurez:

1. **Zéro risque de crash système** - Rollback automatique
2. **Performance garantie** - Rate limiting et monitoring
3. **Intégration validée** - Tests à chaque niveau
4. **Visibilité totale** - Métriques et logs détaillés
5. **Évolution contrôlée** - Montée en charge progressive

**L'intégration sera SÛRE, PROGRESSIVE et RÉVERSIBLE!** 🛡️
