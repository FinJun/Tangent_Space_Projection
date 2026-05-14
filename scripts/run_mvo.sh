#!/bin/bash
# MVO experiments: Fair comparison
# All methods use batch_size=16

cd "$(dirname "$0")/.."

SEEDS=(0 1 2 3 4)
LR=0.001
LAMBDA=1.0
PATIENCE=3
BATCH_SIZE=16
SAVE_DIR=results/qp

# MSE baseline
for seed in "${SEEDS[@]}"; do
    echo "Running: method=mse, lr=$LR, seed=$seed, lambda=$LAMBDA, batch=$BATCH_SIZE"
    python -m experiments.qp.run \
        --method mse \
        --seed $seed \
        --lr $LR \
        --risk_aversion $LAMBDA \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --save_dir $SAVE_DIR
done

# Projection with OSQP backend
for seed in "${SEEDS[@]}"; do
    echo "Running: method=projection (osqp), lr=$LR, seed=$seed, lambda=$LAMBDA, batch=$BATCH_SIZE"
    python -m experiments.qp.run \
        --method projection \
        --projection_backend osqp \
        --seed $seed \
        --lr $LR \
        --risk_aversion $LAMBDA \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --save_dir $SAVE_DIR
done

# Projection Batch (OSQP forward, batched torch backward)
for seed in "${SEEDS[@]}"; do
    echo "Running: method=projection_batch, lr=$LR, seed=$seed, lambda=$LAMBDA, batch=$BATCH_SIZE"
    python -m experiments.qp.run \
        --method projection_batch \
        --seed $seed \
        --lr $LR \
        --risk_aversion $LAMBDA \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --save_dir $SAVE_DIR
done

# BPQP
for seed in "${SEEDS[@]}"; do
    echo "Running: method=bpqp, lr=$LR, seed=$seed, lambda=$LAMBDA, batch=$BATCH_SIZE"
    python -m experiments.qp.run \
        --method bpqp \
        --seed $seed \
        --lr $LR \
        --risk_aversion $LAMBDA \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --save_dir $SAVE_DIR
done

# QPTH
for seed in "${SEEDS[@]}"; do
    echo "Running: method=qpth, lr=$LR, seed=$seed, lambda=$LAMBDA, batch=$BATCH_SIZE"
    python -m experiments.qp.run \
        --method qpth \
        --seed $seed \
        --lr $LR \
        --risk_aversion $LAMBDA \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --save_dir $SAVE_DIR
done

# CVXPYLayers
for seed in "${SEEDS[@]}"; do
    echo "Running: method=cvxpy, lr=$LR, seed=$seed, lambda=$LAMBDA, batch=$BATCH_SIZE"
    python -m experiments.qp.run \
        --method cvxpy \
        --seed $seed \
        --lr $LR \
        --risk_aversion $LAMBDA \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --save_dir $SAVE_DIR
done

# QPTH Sequential (model batch=16, solver batch=1)
for seed in "${SEEDS[@]}"; do
    echo "Running: method=qpth_seq, lr=$LR, seed=$seed, lambda=$LAMBDA, batch=$BATCH_SIZE"
    python -m experiments.qp.run \
        --method qpth_seq \
        --seed $seed \
        --lr $LR \
        --risk_aversion $LAMBDA \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --save_dir $SAVE_DIR
done

# CVXPYLayers Sequential (model batch=16, solver batch=1)
for seed in "${SEEDS[@]}"; do
    echo "Running: method=cvxpy_seq, lr=$LR, seed=$seed, lambda=$LAMBDA, batch=$BATCH_SIZE"
    python -m experiments.qp.run \
        --method cvxpy_seq \
        --seed $seed \
        --lr $LR \
        --risk_aversion $LAMBDA \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --save_dir $SAVE_DIR
done

echo "All experiments completed!"
