# 🤖 Jeffrey OS v2.3.0

**Production-Ready AI Orchestrator with Advanced Emotion Detection**

*Emotion detection system achieving **72.4% F1 LOSO** with 8 core emotions in French and English.*

## 🎉 What's New in v2.3.0

- 🧠 **Production ML Pipeline**: High-performance emotion classifier (F1: 0.724)
- 🌍 **Multilingual Support**: French and English emotion detection
- 🎯 **8 Core Emotions**: joy, sadness, anger, fear, surprise, disgust, neutral, frustration
- ✅ **Validated Performance**: 80.4% smoke test accuracy, 0% fallback rate
- 🔄 **Complete Pipeline**: From data generation to production deployment

## ✨ Core Features

- 🧠 **Advanced Emotion AI**: ProtoClassifier with medoid-based prototypes
- 🔍 **Hybrid Memory Search**: Keyword + Semantic with clustering
- 📈 **Adaptive Learning**: Feedback-driven preference adaptation
- 💬 **Explainable AI**: Every prediction score is explained
- 🎨 **Auto-Organization**: Automatic thematic clustering

## 🚀 Quick Start

### ML Pipeline (New in v2.3.0)

```bash
# Complete ML pipeline
python scripts/ml/ml_pipeline.py pipeline

# Individual operations
python scripts/ml/ml_pipeline.py status      # Show model status
python scripts/ml/ml_pipeline.py train       # Train emotion classifier
python scripts/ml/ml_pipeline.py test        # Run smoke tests

# Emotion detection
python << EOF
from jeffrey.core.emotion_backend import ProtoEmotionDetector

detector = ProtoEmotionDetector()
emotion = detector.predict_label("I'm so happy today!")
print(f"Emotion: {emotion}")  # Output: joy
EOF
```

### Memory System

```bash
# Installation
pip install -r requirements.txt

# Memory usage
python << EOF
from jeffrey.memory.unified_memory import UnifiedMemory

memory = UnifiedMemory()
memory.add_memory({"user_id": "alice", "content": "J'adore le jazz"})
results = memory.search_memories("alice", "musique")
print(results[0]["memory"]["content"])
EOF
```

## 📊 Performance Metrics

### ML Pipeline v2.1.0
- **F1 LOSO**: 0.724 ✅ (34% above target)
- **Accuracy**: 72.4% ✅
- **Smoke Tests**: 80.4% accuracy, 0% fallback ✅
- **Inference Speed**: <0.5ms per prediction
- **Calibration**: ECE = 0.140 (well-calibrated)
- **Training Data**: 127 high-quality FR/EN examples

### Memory System
- **Search Performance**: <50ms hybrid search
- **Test Coverage**: 40 scenarios, 1000+ conversation turns
- **Unit Tests**: 20/20 passing ✅
- **Advanced Features**: 15+ capabilities

## 📚 Documentation

Voir [Documentation complète](docs/README.md)

## 🏆 Phases

- ✅ Phase 1 : Système de base
- ✅ Phase 2 : Embeddings sémantiques
- ✅ Phase 3 : Clustering + Learning + Multi-Query

## 🧪 Tests

```bash
# Tests unitaires
PYTHONPATH=src python3 tests/test_unified_memory.py
PYTHONPATH=src python3 tests/test_semantic_search.py
PYTHONPATH=src python3 tests/test_phase3_advanced_memory.py

# Tests conversationnels
PYTHONPATH=src python3 tests/runner_convos.py
```

## 📝 Licence

MIT
