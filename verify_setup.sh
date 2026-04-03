#!/bin/bash
# IGNISIA Setup Verification
# Checks that all components are properly configured

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                 IGNISIA Setup Verification                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo

# Step 1: Python environment
echo "[1/6] Checking Python environment..."
if conda env list | grep -q "hc01"; then
    conda activate hc01 2>/dev/null || true
    python_version=$(python --version 2>&1)
    echo "✓ Conda environment 'hc01' ready ($python_version)"
else
    echo "✗ Conda environment 'hc01' not found"
    exit 1
fi
echo

# Step 2: Dependencies
echo "[2/6] Checking Python packages..."
python3 << 'EOF'
import sys
required = {
    'pandas': 'Data loading',
    'numpy': 'Numerical computing',
    'httpx': 'HTTP client',
    'openai': 'NIM API client',
    'dotenv': 'Environment variables',
    'asyncio': 'Async operations',
}

missing = []
for package, desc in required.items():
    try:
        __import__(package)
        print(f"  ✓ {package:15} - {desc}")
    except ImportError:
        print(f"  ✗ {package:15} - {desc} (MISSING)")
        missing.append(package)

if missing:
    print(f"\n✗ Missing packages: {', '.join(missing)}")
    print("Run: pip install " + " ".join(missing))
    sys.exit(1)
else:
    print("\n✓ All dependencies installed")
EOF
echo

# Step 3: Data files
echo "[3/6] Checking MIMIC data files..."
mimic_files=("NOTEEVENTS.csv" "LABEVENTS.csv" "ICUSTAYS.csv" "PATIENTS.csv" "PRESCRIPTIONS.csv" "D_LABITEMS.csv")
missing_files=()

for file in "${mimic_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✓ All MIMIC data files present"
else
    echo "✗ Missing MIMIC files: ${missing_files[@]}"
fi
echo

# Step 4: Configuration
echo "[4/6] Checking configuration..."
if [ -f ".env" ]; then
    echo "  ✓ .env file present"
    python3 << 'EOF'
import os

with open(".env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value

chief = os.getenv("NIM_API_KEY_CHIEF")
fallback = os.getenv("NIM_API_KEY_FALLBACK")

if chief:
    print(f"  ✓ NIM_API_KEY_CHIEF set ({chief[:20]}...)")
else:
    print(f"  ⚠ NIM_API_KEY_CHIEF not configured")

if fallback:
    print(f"  ✓ NIM_API_KEY_FALLBACK set ({fallback[:20]}...)")
else:
    print(f"  ⚠ NIM_API_KEY_FALLBACK not configured")
EOF
else
    echo "  ⚠ .env file not found (use .env.example as template)"
fi
echo

# Step 5: Source files
echo "[5/6] Checking source files..."
source_files=(
    "app/config.py"
    "app/data_loader.py"
    "app/nim_client.py"
    "app/clients.py"
    "app/agents.py"
    "app/data.py"
)

for file in "${source_files[@]}"; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "  ✓ $file ($lines lines)"
    else
        echo "  ✗ $file (MISSING)"
    fi
done
echo

# Step 6: Test files
echo "[6/6] Checking test and demo files..."
test_files=("test_nim_quick.py" "test_nim.py" "demo.py")
for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
    fi
done
echo

# Summary
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                        Verification Complete                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo
echo "Next steps:"
echo "1. Verify NIM API keys are set in .env"
echo "2. Run: python test_nim_quick.py (to test NIM APIs)"
echo "3. Run: python demo.py (full clinical pipeline)"
echo
