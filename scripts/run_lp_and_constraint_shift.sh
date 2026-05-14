#!/bin/bash
# LP Training (deg=2,4,6) + Constraint Shift Experiments (all degrees)
# deg=8 models already exist

set -e

cd "$(dirname "$0")/.."

METHODS="mse spo dbb pfyl projection lava"
SEEDS="0 1 2 3 4"
SAVE_DIR=results/lp

mkdir -p $SAVE_DIR

echo "=============================================="
echo "Phase 1: LP Training (deg=2,4,6)"
echo "=============================================="

# Train LP models for degrees 2, 4, 6 (deg=8 already done)
for deg in 2 4 6; do
    echo ""
    echo "=== Degree $deg ==="

    # Shortest Path
    echo "--- Shortest Path (deg=$deg) ---"
    for method in $METHODS; do
        for seed in $SEEDS; do
            echo "[SP] Method: $method, Deg: $deg, Seed: $seed"
            python -m experiments.lp.run \
                --prob sp \
                --method $method \
                --deg $deg \
                --seed $seed \
                --save_dir $SAVE_DIR
        done
    done

    # Knapsack
    echo "--- Knapsack (deg=$deg) ---"
    for method in $METHODS; do
        for seed in $SEEDS; do
            echo "[KS] Method: $method, Deg: $deg, Seed: $seed"
            python -m experiments.lp.run \
                --prob ks \
                --method $method \
                --deg $deg \
                --seed $seed \
                --save_dir $SAVE_DIR
        done
    done
done

echo ""
echo "=============================================="
echo "Phase 2: Constraint Shift Experiments (all degrees)"
echo "=============================================="

DEGREES="2 4 6 8"

# Knapsack Capacity Generalization
echo ""
echo "--- Capacity Generalization (KS) ---"
for deg in $DEGREES; do
    for method in $METHODS; do
        for seed in $SEEDS; do
            echo "[Capacity] Method: $method, Deg: $deg, Seed: $seed"
            python -m experiments.constraint_shift.run_capacity_generalization \
                --method $method --deg $deg --seed $seed
        done
    done
done

# Shortest Path Direction Generalization
echo ""
echo "--- Direction Generalization (SP) ---"
for deg in $DEGREES; do
    for method in $METHODS; do
        for seed in $SEEDS; do
            echo "[Direction] Method: $method, Deg: $deg, Seed: $seed"
            python -m experiments.constraint_shift.run_direction_generalization \
                --method $method --deg $deg --seed $seed
        done
    done
done

echo ""
echo "=============================================="
echo "All Experiments Done!"
echo "=============================================="
