#!/bin/bash
# Michel Physics Training Script
cd /home/msingh1/nugraph
# Navigate to repository root


# Set paths dynamically
export PYTHONPATH="$(pwd)/nugraph:$PYTHONPATH"
DATA_PATH="/nugraph/NG2-paper.gnn.h5"  # path need user adjustment
EPOCHS=15
BATCH_SIZE=4

if [ "$1" == "with_michel" ]; then
    echo "Training WITH Michel physics regularization..."
    python scripts/train.py \
        --data $DATA_PATH --name michel_experiment --name michel_experiment \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --semantic \
        --filter \
        --instance \
        --enable-michel-reg \
        --michel-reg-lambda 0.1

elif [ "$1" == "without_michel" ]; then
    echo "Training WITHOUT Michel physics (baseline)..."
    python scripts/train.py \
        --data $DATA_PATH --name michel_experiment --name michel_experiment \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --semantic \
        --filter \
        --instance

elif [ "$1" == "strong_michel" ]; then
    echo "Training with STRONG Michel physics regularization..."
    python scripts/train.py \
        --data $DATA_PATH --name michel_experiment --name michel_experiment \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --semantic \
        --filter \
        --instance \
        --enable-michel-reg \
        --michel-reg-lambda 1.0

else
    echo "Usage: ./train.sh [with_michel|without_michel|strong_michel]"
    echo ""
    echo "Run this to see available training options:"
    echo "python -m nugraph.train --help"
    echo ""
    echo "Options:"
    echo "  with_michel     - Enable 30 MeV physics constraint (λ=0.1)"
    echo "  without_michel  - Standard training"
    echo "  strong_michel   - Strong physics constraint (λ=1.0)"
fi
