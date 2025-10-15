# NeuralBus - Jeffrey OS Event Bus

## 🚀 Installation

```bash
# Installation des dépendances obligatoires
pip install nats-py pydantic pydantic-settings python-dotenv pytest pytest-asyncio

# Installation des dépendances optionnelles (RECOMMANDÉ pour performance)
pip install msgpack uvloop redis psutil

# Configuration
cp .env.p2.example .env.p2  # Si nécessaire
```

## ✅ Tests & Benchmarks

```bash
# Run tests
make nb-test

# Run benchmark
make nb-benchmark

# Monitor stream
make nb-monitor
```

## 📊 Features

- ✅ **CloudEvents v1.0** compliant
- ✅ **Priority Lanes**: Critical, High, Normal, Low
- ✅ **Dead Letter Queue (DLQ)**: Automatic failed message handling
- ✅ **Deduplication**: Redis distributed or LRU local
- ✅ **Circuit Breaker**: Per-handler protection
- ✅ **Dynamic Batching**: Auto-adjusting batch size
- ✅ **Multi-tenant**: Tenant isolation built-in
- ✅ **Self-optimization**: Automatic purging and optimization

## 🎯 Performance

- **Target**: 5000+ events/sec
- **Latency P95**: < 10ms
- **Optimizations**: msgpack, uvloop, dynamic batching

## 📝 Usage Example

```python
from jeffrey.core.neuralbus import neural_bus, CloudEvent, EventMeta, EventPriority

# Initialize
await neural_bus.initialize()

# Publish
event = CloudEvent(
    meta=EventMeta(
        type="user.created",
        tenant_id="tenant1",
        priority=EventPriority.HIGH
    ),
    data={"user_id": "123", "name": "John"}
)
await neural_bus.publish(event)

# Consume
consumer = neural_bus.create_consumer("MY_CONSUMER", "events.user.>")

async def handler(event, headers):
    print(f"Processing {event.meta.type}: {event.data}")

consumer.register_handler("user.created", handler)
await consumer.connect()
await consumer.run()
```

## 🔧 Configuration

All configuration via environment variables with `NEURALBUS_` prefix:

- `NEURALBUS_LOG_LEVEL`: INFO, DEBUG, ERROR (default: INFO)
- `NEURALBUS_USE_REDIS_DEDUP`: true/false (default: false)
- `NEURALBUS_ENABLE_ENCRYPTION`: true/false (default: false)
- `NEURALBUS_BATCH_DYNAMIC`: true/false (default: true)

## 📋 Status

✅ **Production Ready** with all fixes from GPT applied:
- Pydantic v2 field_validator
- Synchronous NATS callbacks
- Redis asyncio connection fix
- Headers None protection

## 🏗️ Architecture

```
NeuralBus
├── Publisher (with deduplication)
├── Consumer (with circuit breaker)
├── Stream (NATS JetStream)
├── Priority Lanes (4 levels)
├── DLQ (automatic retry)
└── Self-Optimization (background)
```

## 📜 License

Proprietary - Jeffrey OS
