#!/usr/bin/env python3
"""Script de test simple pour le linear head"""

import logging
import sys
from pathlib import Path

# Setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO)

import numpy as np

from jeffrey.ml.calibration import apply_temperature, fit_temperature
from jeffrey.ml.heads import LinearHead

# Test basique
print("🔍 Test LinearHead...")

# Données factices
np.random.seed(42)
X_train = np.random.randn(100, 384)
y_train = np.random.choice(['anger', 'joy', 'sadness', 'neutral'], 100)

print(f"X_train shape: {X_train.shape}")
print(f"y_train unique: {np.unique(y_train)}")

# Test LinearHead
head = LinearHead(n_classes=4, in_dim=384)
head.fit(X_train, y_train)

print(f"Classes: {head.classes_}")

# Test logits
X_test = np.random.randn(10, 384)
logits = head.decision_function(X_test)
print(f"Logits shape: {logits.shape}")

# Test calibration
temp = fit_temperature(logits, ['anger'] * 10, head.classes_)
print(f"Temperature: {temp}")

probs = apply_temperature(logits, temp)
print(f"Probs shape: {probs.shape}")

print("✅ LinearHead test passed")
