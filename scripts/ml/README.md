# Jeffrey OS v2.3.0 - ML Scripts

This directory contains the production-ready ML pipeline for Jeffrey OS emotion detection.

## Quick Start

Use the unified ML pipeline manager:

```bash
# Show current model status
python scripts/ml/ml_pipeline.py status

# Run complete pipeline from scratch
python scripts/ml/ml_pipeline.py pipeline

# Individual operations
python scripts/ml/ml_pipeline.py generate-data
python scripts/ml/ml_pipeline.py train
python scripts/ml/ml_pipeline.py test
python scripts/ml/ml_pipeline.py verify
```

## Scripts Overview

### 🚀 `ml_pipeline.py`
**Main entry point** - Production-ready unified interface for all ML operations.
- `status` - Show current model status
- `generate-data` - Generate training YAML dataset
- `train` - Train prototype classifier
- `test` - Run smoke tests (FR/EN)
- `verify` - Check training-inference parity
- `monitor` - Start emotion monitoring
- `pipeline` - Run complete pipeline

### 📊 Core ML Scripts

#### `generate_training_dataset.py`
- Generates 127 high-quality YAML training examples
- Covers 8 core emotions: joy, sadness, anger, fear, surprise, disgust, neutral, frustration
- Includes FR/EN multilingual support
- Contains challenging cases (negation, irony, mixed emotions)

#### `train_prototypes_optimized.py`
- Trains ProtoClassifier with advanced features:
  - L2 normalization + medoid computation
  - LOSO cross-validation by scenario
  - ECE calibration calculation
  - Confusion matrix generation
  - Hybrid data pipeline with quality boosting

#### `smoke_test_fr_en.py`
- Validates trained model with 56 realistic FR/EN test cases
- **Gates**: Accuracy ≥ 60%, Fallback rate ≤ 2%
- Provides detailed per-emotion breakdown
- Shows challenging cases for analysis

#### `verify_training_inference_parity.py`
- Ensures consistency between training and inference pipelines
- **Gates**: Embedding similarity ≥ 99.5%, Prediction consistency = 100%
- Critical for production deployment confidence

#### `monitor_emotions.py`
- Runtime monitoring of emotion detection performance
- Tracks prediction accuracy and confidence distributions

## Model Performance (v2.1.0)

- **F1 LOSO**: 0.724 (34% above target of 0.537)
- **Accuracy**: 72.4%
- **ECE**: 0.140 (good calibration)
- **Smoke Test**: 80.4% accuracy, 0% fallback

### Per-Emotion Performance
- Sadness: 100% (perfect)
- Surprise: 100% (perfect)
- Joy: 87.5% (excellent)
- Fear: 83.3% (very good)
- Frustration: 75.0% (good)
- Neutral: 66.7% (acceptable)
- Anger: 66.7% (acceptable)
- Disgust: 66.7% (acceptable)

## Data Files Generated

- `data/prototypes.npz` - Trained model weights
- `data/prototypes.meta.json` - Model metadata
- `data/confusion_matrix.png` - Performance visualization
- `data/conversations/*.yaml` - Training dataset (127 files)

## Requirements

- Python 3.11+
- Dependencies: `sentence-transformers`, `numpy`, `scikit-learn`, `yaml`, `matplotlib`
- Jeffrey OS source code in `src/` directory

## Architecture

The emotion detection system uses:

1. **Encoder**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim embeddings)
2. **Classifier**: ProtoClassifier with medoid-based prototypes
3. **Validation**: LOSO (Leave-One-Scenario-Out) cross-validation
4. **Normalization**: L2 normalization for cosine similarity
5. **Calibration**: Expected Calibration Error (ECE) monitoring

## Troubleshooting

### Common Issues

**ModuleNotFoundError**: Ensure you're running from Jeffrey OS root directory with `PYTHONPATH=src`

**Prototypes not found**: Run `python scripts/ml/ml_pipeline.py train` first

**Low accuracy**: Check that you have all 127 YAML training files in `data/conversations/`

### Debug Commands

```bash
# Check environment
ls src/jeffrey/  # Should show Jeffrey modules
ls data/conversations/  # Should show *.yaml files

# Validate training data
python -c "import yaml; print(len(list(Path('data/conversations').glob('*.yaml'))))"

# Test encoder
PYTHONPATH=src python -c "from jeffrey.ml.encoder import create_default_encoder; print('Encoder OK')"
```

## Version History

- **v2.1.0** (2025-10-13): Production-ready with 72.4% F1 LOSO
- **v2.0.0**: Initial prototype implementation
