#!/bin/bash
# IGNISIA Environment Setup
# Run this script to configure your development environment

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         IGNISIA Environment Setup                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# Step 1: Create/activate conda environment
echo "[1/4] Setting up conda environment..."
if ! conda activate hc01 2>/dev/null; then
    echo "Creating new conda environment 'hc01'..."
    conda create -n hc01 python=3.11 -y
    conda activate hc01
fi
echo "✓ Conda environment ready (hc01)"
echo

# Step 2: Install pip dependencies
echo "[2/4] Installing Python dependencies..."
pip install -q \
    pandas numpy \
    httpx \
    openai \
    python-dotenv \
    aiofiles
echo "✓ Dependencies installed"
echo

# Step 3: Create .env file
echo "[3/4] Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file"
    echo "  ⚠ Configure your API keys in .env"
else
    echo "✓ .env file already exists"
fi
echo

# Step 4: Verify setup
echo "[4/4] Verifying installation..."
python3 << 'EOF'
import sys
try:
    import pandas
    import httpx
    import openai
    from dotenv import load_dotenv
    import os
    
    print("✓ All modules imported successfully")
    print(f"  • OpenAI SDK: {openai.__version__}")
    print(f"  • Python: {sys.version.split()[0]}")
    
    # Check .env
    load_dotenv()
    chief_key = os.getenv("NIM_API_KEY_CHIEF")
    fallback_key = os.getenv("NIM_API_KEY_FALLBACK")
    
    if chief_key:
        print(f"✓ NIM_API_KEY_CHIEF configured")
    else:
        print(f"⚠ NIM_API_KEY_CHIEF not set (add to .env)")
    
    if fallback_key:
        print(f"✓ NIM_API_KEY_FALLBACK configured")
    else:
        print(f"⚠ NIM_API_KEY_FALLBACK not set (add to .env)")
        
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
EOF
echo

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      Setup Complete!                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo
echo "Next steps:"
echo "1. Edit .env file with your NIM API keys"
echo "2. Start Ollama: ollama serve"
echo "3. Run the demo: python test_nim.py"
echo
