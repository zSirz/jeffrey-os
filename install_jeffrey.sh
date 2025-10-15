#!/bin/bash
# ============================================================
# Jeffrey OS - Installation Script
# Installe toutes les dépendances dans un environnement virtuel
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Installing Jeffrey OS dependencies..."
echo "============================================"

# Check Python version
echo -n "Checking Python version... "
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}❌ Error: Python 3.11+ is required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $python_version${NC}"

# Check if we're on Mac or Linux for uvloop
OS_TYPE="$(uname -s)"
echo "Detected OS: $OS_TYPE"

# Create virtual environment if it doesn't exist
VENV_DIR=".venv_prod"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv $VENV_DIR
else
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip, setuptools, wheel
echo -e "${YELLOW}📚 Upgrading pip and build tools...${NC}"
pip install --upgrade pip setuptools wheel

# Install requirements
echo -e "${YELLOW}📚 Installing production requirements...${NC}"

# Handle platform-specific dependencies
if [[ "$OS_TYPE" == "Linux" ]] || [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Installing with uvloop support (performance optimization)..."
    pip install -r requirements.txt
else
    echo "Installing without uvloop (Windows)..."
    grep -v uvloop requirements.txt > requirements_windows.txt
    pip install -r requirements_windows.txt
    rm requirements_windows.txt
fi

# Create necessary directories
echo -e "${YELLOW}📂 Creating project directories...${NC}"
mkdir -p data
mkdir -p logs
mkdir -p cache
mkdir -p keys
mkdir -p backups

# Verify core installations
echo -e "${YELLOW}✅ Verifying core installations...${NC}"
python3 -c "
import sys
import importlib

required_modules = [
    ('httpx', 'httpx'),
    ('networkx', 'networkx'),
    ('kivy', 'Kivy'),
    ('msgpack', 'msgpack'),
    ('numpy', 'numpy'),
    ('redis', 'redis'),
    ('pydantic', 'pydantic')
]

all_ok = True
for module, name in required_modules:
    try:
        importlib.import_module(module)
        print(f'✅ {name} installed')
    except ImportError:
        print(f'❌ {name} NOT installed')
        all_ok = False

if all_ok:
    print('\n✅ All core packages installed successfully!')
else:
    print('\n⚠️ Some packages are missing. Please check the errors above.')
    sys.exit(1)
"

# Check Ollama installation
echo -e "${YELLOW}🤖 Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama is installed${NC}"

    # Check if mistral model is available
    if ollama list | grep -q "mistral"; then
        echo -e "${GREEN}✅ Mistral model is available${NC}"
    else
        echo -e "${YELLOW}⚠️ Mistral model not found. To install:${NC}"
        echo "   ollama pull mistral:7b-instruct"
    fi
else
    echo -e "${YELLOW}⚠️ Ollama not installed. Jeffrey will work but without LLM capabilities.${NC}"
    echo "   To install Ollama: https://ollama.ai"
fi

# Optional: Install development dependencies
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}📚 Installing development requirements...${NC}"
    pip install -r requirements-dev.txt
    echo -e "${GREEN}✅ Development tools installed${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}📝 Creating .env file...${NC}"
    cat > .env << EOL
# Jeffrey OS Environment Configuration
JEFFREY_ENV=production
JEFFREY_LOG_LEVEL=INFO
JEFFREY_CACHE_DIR=./cache
JEFFREY_DATA_DIR=./data

# Ollama Configuration
OLLAMA_HOST=http://localhost:9010
OLLAMA_MODEL=mistral:7b-instruct

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Security
JEFFREY_ENCRYPTION_KEY=CHANGE_THIS_TO_A_SECURE_KEY
EOL
    echo -e "${GREEN}✅ .env file created (please update with your settings)${NC}"
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

# Final summary
echo
echo "============================================"
echo -e "${GREEN}🎉 Jeffrey OS installation complete!${NC}"
echo
echo "To activate the environment:"
echo -e "  ${YELLOW}source $VENV_DIR/bin/activate${NC}"
echo
echo "To run tests:"
echo -e "  ${YELLOW}python tests/test_bridge_v3_basic.py${NC}"
echo
echo "To start Jeffrey:"
echo -e "  ${YELLOW}python test_kivy_integration.py${NC}"
echo
echo "Configuration:"
echo -e "  ${YELLOW}Edit .env file for your settings${NC}"
echo "============================================"
