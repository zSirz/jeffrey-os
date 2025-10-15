#!/bin/bash

echo "🚀 Starting LLM test with Ollama..."
echo "==================================="

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export LLM_BASE_URL="http://localhost:9010/v1"
export LLM_MODEL="mistral:7b-instruct"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running! Start it with: ollama serve"
    exit 1
fi

# Check if proxy is running
if ! curl -s http://localhost:9010/health > /dev/null 2>&1; then
    echo "⚠️  OpenAI proxy not running on port 9010"
    echo "Starting proxy..."
    python -m litellm --model ollama/mistral:7b-instruct --port 9010 &
    PROXY_PID=$!
    sleep 3
fi

echo "✅ Services ready"
echo ""

# Run the test
python tests/test_llm_adaptive.py

# Kill proxy if we started it
if [ ! -z "$PROXY_PID" ]; then
    kill $PROXY_PID 2>/dev/null
fi
