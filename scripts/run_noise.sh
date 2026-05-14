#!/bin/bash
# Noise Experiments for LP (Shortest Path, Knapsack)
# Degree 8 only, noise levels: 0.1, 0.3, 0.5

set -e

cd "$(dirname "$0")/.."

METHODS="mse spo dbb pfyl projection lava"
SEEDS="0 1 2 3 4"
NOISES="0.1 0.3 0.5"
SAVE_DIR=results/lp_noise

mkdir -p $SAVE_DIR

echo "=== Noise Experiments ==="
echo "Problems: sp, ks"
echo "Degree: 8"
echo "Noise: $NOISES"
echo "Methods: $METHODS"
echo "Seeds: $SEEDS"
echo ""

# Shortest Path
echo "--- Shortest Path ---"
for noise in $NOISES; do
    for method in $METHODS; do
        for seed in $SEEDS; do
            echo "[SP] Method: $method, Noise: $noise, Seed: $seed"
            python -m experiments.lp.run \
                --prob sp \
                --method $method \
                --deg 8 \
                --noise $noise \
                --seed $seed \
                --save_dir $SAVE_DIR
        done
    done
done

# Knapsack
echo ""
echo "--- Knapsack ---"
for noise in $NOISES; do
    for method in $METHODS; do
        for seed in $SEEDS; do
            echo "[KS] Method: $method, Noise: $noise, Seed: $seed"
            python -m experiments.lp.run \
                --prob ks \
                --method $method \
                --deg 8 \
                --noise $noise \
                --seed $seed \
                --save_dir $SAVE_DIR
        done
    done
done

echo ""
echo "=== Noise Experiments Done ==="
