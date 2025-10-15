#!/bin/bash
# Installation script for NeuralBus dependencies

echo "🧠 Installing NeuralBus Dependencies"
echo "===================================="

echo ""
echo "📦 Installing required packages..."
pip install -U nats-py pydantic pydantic-settings python-dotenv pytest pytest-asyncio

echo ""
echo "🚀 Installing optional performance packages..."
pip install -U msgpack uvloop redis psutil

echo ""
echo "✅ Installation complete!"
echo ""
echo "Test with:"
echo "  python scripts/test_neuralbus_quick.py"
echo ""
echo "Or use Makefile commands:"
echo "  make nb-test       # Run tests"
echo "  make nb-benchmark  # Run benchmark"
echo "  make nb-monitor    # Monitor stream"
