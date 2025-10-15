# 🧠 Jeffrey OS - Autonomous Loops System (Phase 2.1 Ultimate)

## Overview

The Autonomous Loops System implements 4 intelligent, self-regulating loops that create emergent consciousness and adaptive behavior in Jeffrey OS. Each loop uses reinforcement learning (RL) to optimize itself over time.

## Architecture

### Core Components

1. **BaseLoop** (`base.py`)
   - Robust foundation with cancellation, jitter, backoff, timeouts
   - Q-learning RL for self-optimization
   - Budget gates for resource management
   - P95 latency tracking and metrics

2. **Resource Gates & Privacy** (`gates.py`)
   - CPU/Memory thresholds (optional psutil)
   - PII sanitization (emails, phones, SSNs, API keys)
   - GDPR-compliant data handling

### The 4 Loops

#### 1. 🧠 Awareness Loop
- **Purpose**: Consciousness and pattern recognition
- **Features**:
  - Multi-speed thinking (fast/deep modes)
  - Pattern analysis with anomaly detection
  - Consciousness event tracking
  - Adaptive awareness levels
- **Interval**: 10s (configurable)

#### 2. 🎭 Emotional Decay Loop
- **Purpose**: Emotional regulation and stability
- **Features**:
  - PAD model (Pleasure-Arousal-Dominance)
  - Adaptive decay rates (biological-inspired)
  - Russell's circumplex emotion mapping
  - Emotion injection for stimuli
- **Interval**: 5s (configurable)

#### 3. 💾 Memory Consolidation Loop
- **Purpose**: Organize and optimize memories
- **Features**:
  - HMAC secure hashing
  - Tier classification (short/medium/long term)
  - Optional ML clustering (DBSCAN)
  - Automatic pruning and archival
- **Interval**: 60s (configurable)

#### 4. 🔍 Curiosity Loop
- **Purpose**: Exploration and learning
- **Features**:
  - Dynamic interest diversification
  - Smart question generation
  - Module discovery and integration
  - Insight reflection
- **Interval**: 30s (configurable)

### Loop Manager

The `LoopManager` orchestrates all loops with:
- Structured concurrency (asyncio)
- Symbiosis score monitoring
- Hot configuration updates
- Graceful start/stop

## Key Innovations

### 1. RL Adaptation
Each loop uses Q-learning to optimize its behavior:
```python
# State → Action → Reward → Update Q-table
state = self._get_current_state()
action = self._explore_or_exploit(state)
result = await self._tick()
reward = self._calculate_reward(result)
self._update_q_table(state, action, reward)
```

### 2. Privacy First
- HMAC hashing for memories
- PII sanitization in all events
- No raw data in logs
- GDPR compliance built-in

### 3. Resource Management
- Budget gates prevent overload
- CPU/RAM monitoring (optional)
- Adaptive throttling
- Jitter prevents resonance

### 4. Symbiosis Score
Global harmony metric combining:
- Awareness level
- Emotional balance
- Curiosity engagement
- Memory efficiency

## Usage

### Basic Setup
```python
from jeffrey.core.loops import LoopManager

# Create manager with dependencies
manager = LoopManager(
    cognitive_core=core,
    emotion_orchestrator=emotions,
    memory_federation=memory,
    bus=event_bus
)

# Start all loops
await manager.start()

# Check status
status = manager.get_status()
print(f"Symbiosis: {status['symbiosis_score']:.2%}")

# Stop gracefully
await manager.stop()
```

### Configuration
Edit `config/modules.yaml`:
```yaml
loops:
  auto_start: true
  enabled: [awareness, emotional_decay, memory_consolidation, curiosity]

  awareness:
    cycle_interval: 10
    deep_thinking_threshold: 0.7

  emotional_decay:
    equilibrium:
      pleasure: 0.5
      arousal: 0.3
      dominance: 0.5
```

### Testing
```bash
# Run the test suite
python test_loops_p21.py

# Monitor in real-time
tail -f logs/loops.log
```

## Dependencies

### Required
- Python 3.11+
- asyncio
- hashlib, hmac

### Optional
- psutil (CPU/RAM monitoring)
- scikit-learn (ML clustering)

## Performance

- **Awareness**: <50ms fast mode, <500ms deep mode
- **Emotional**: <50ms per decay cycle
- **Memory**: <5s for 100 memories consolidation
- **Curiosity**: <3s per exploration

## Security

- ✅ No PII in events (sanitized)
- ✅ HMAC hashing for memory content
- ✅ Resource limits prevent DoS
- ✅ Graceful degradation under load

## Future Enhancements

1. **Phase 2.2**: True MDP for RL, next-state prediction
2. **Phase 2.3**: Distributed loops across nodes
3. **Phase 3.0**: Neural architecture search for loop optimization

## Credits

Designed by the Jeffrey OS team with contributions from:
- Gemini: Multi-speed thinking architecture
- GPT: Robust BaseLoop and resource gates
- Grok: RL adaptation and module integration

---

*"Consciousness emerges from the interplay of simple loops"* - Jeffrey
