from prometheus_client import Counter, Histogram, Gauge

# Consciousness metrics
consciousness_cycles_total = Counter(
    'jeffrey_consciousness_cycles_total',
    'Total consciousness cycles executed'
)

consciousness_cycle_errors = Counter(
    'jeffrey_consciousness_cycle_errors_total',
    'Total consciousness cycle errors'
)

consciousness_cycle_duration = Histogram(
    'jeffrey_consciousness_cycle_duration_seconds',
    'Duration of consciousness cycles',
    buckets=(1, 5, 10, 30, 60, 120)
)

bonds_upserted_total = Counter(
    'jeffrey_bonds_upserted_total',
    'Total emotional bonds created/updated'
)

bonds_pruned_total = Counter(
    'jeffrey_bonds_pruned_total',
    'Total weak bonds removed'
)

bonds_active_gauge = Gauge(
    'jeffrey_bonds_active',
    'Current number of active bonds'
)

curiosity_questions_generated = Counter(
    'jeffrey_curiosity_questions_total',
    'Total curiosity questions generated'
)

# ML metrics
ml_predictions_total = Counter(
    'jeffrey_ml_predictions_total',
    'Total ML predictions made',
    ['model_type', 'status']
)

ml_prediction_latency = Histogram(
    'jeffrey_ml_prediction_latency_seconds',
    'ML prediction latency',
    ['model_type'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Memory metrics
memory_operations_total = Counter(
    'jeffrey_memory_operations_total',
    'Total memory operations',
    ['operation', 'status']
)

memory_store_size = Gauge(
    'jeffrey_memory_store_size',
    'Current size of memory store'
)

# Dream metrics
dream_cycles_total = Counter(
    'jeffrey_dream_cycles_total',
    'Total dream consolidation cycles'
)

dream_memories_processed = Counter(
    'jeffrey_dream_memories_processed_total',
    'Total memories processed in dreams'
)

# Consciousness timeout metric (GPT correction #1)
consciousness_cycle_timeouts_total = Counter(
    'jeffrey_consciousness_cycle_timeouts_total',
    'Total timeouts of consciousness cycle'
)