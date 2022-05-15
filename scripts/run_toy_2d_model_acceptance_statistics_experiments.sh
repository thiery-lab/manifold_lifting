#!/usr/bin/env bash

usage() {
cat <<- HELP_USAGE
Run experiments to compute grid of oaverage acceptance rate statistics for toy 2D model
with varying (fixed) observation noise scales and step sizes.

Usage: $0 [--help]

The script does not accept arguments other than the flag --help to display this message

Instead environment variables can be set to override the default settings (default or
current values if set are shown in parentheses)

PYTHON_BIN: Path to Python binary to use to run experiments with (${PYTHON_BIN})
MODEL_DIR: Path to directory containing model .py scripts (${MODEL_DIR})
OUTPUT_DIR: Path to output results to (${OUTPUT_DIR})
ALGORITHMS: MCMC algorithms to run experiments for (${ALGORITHMS})
SEEDS: Integer value(s) to use to seed random number generator (${SEEDS})
NUM_CHAIN: Number of chains to run for each experiment (${NUM_CHAIN})
NUM_ITER: Number of sampling iterations per chain (${NUM_MAIN_ITER})

For example to run with 2 chains / experiment and remaining arguments as defaults run

NUM_CHAIN=2 $0
HELP_USAGE
}

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-mlift/example_models}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
ALGORITHMS="${ALGORITHMS:-rwm mala hmc fisher_rmhmc softabs_rmhmc chmc pd_rwm simple_pd_mala xifara_pd_mala}"
SEEDS="${SEEDS:-202101}"
NUM_CHAIN="${NUM_CHAIN:-4}"
NUM_ITER="${NUM_ITER:-1000}"


SEP_1="================================================================================"
SEP_2="--------------------------------------------------------------------------------"

if [ "$1" = "--help" ]; then
    usage
    exit 0
fi

echo ${SEP_1}
echo "Running toy 2D model acceptance statistic experiments"

echo ${SEP_1}
for ALGORITHM in ${ALGORITHMS[@]}; do
    for SEED in ${SEEDS[@]}; do
        echo "algorithm=${ALGORITHM} seed=${SEED}"
        echo ${SEP_2}
        ${PYTHON_BIN} ${MODEL_DIR}/acceptance_statistics_grid.py \
            --output-root-dir ${OUTPUT_DIR} \
            --seed ${SEED} \
            --algorithm ${ALGORITHM} \
            --num-chain ${NUM_CHAIN} \
            --num-iter ${NUM_ITER}
        echo ${SEP_2}
    done
done
