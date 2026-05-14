#!/bin/bash
# Run all visualization scripts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Generating Visualizations ==="

echo ""
echo "[1/3] LP Results"
python scripts/visualize_lp.py

echo ""
echo "[2/3] QP Results"
python scripts/visualize_qp.py

echo ""
echo "[3/3] Constraint Shift Results"
python scripts/visualize_constraint_shift.py

echo ""
echo "=== All Visualizations Complete ==="
echo "Check the 'figures' directory for output."
