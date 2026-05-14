#!/bin/bash
# LP Experiments (Shortest Path, Knapsack)
# All methods with degrees 2,4,6,8 and seeds 0,1,2,3,4

set -e

cd "$(dirname "$0")/.."

METHODS="mse spo dbb pfyl projection lava"
SEEDS="0 1 2 3 4"
DEGREES="2 4 6 8"
SAVE_DIR=results/lp

mkdir -p $SAVE_DIR

echo "=== LP Experiments ==="

# Shortest Path
echo ""
echo "--- Shortest Path ---"
for deg in $DEGREES; do
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
done

# Knapsack
echo ""
echo "--- Knapsack ---"
for deg in $DEGREES; do
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
echo "=== LP Experiments Done ==="
