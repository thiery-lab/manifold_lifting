#!/usr/bin/env bash

usage() {
cat <<- HELP_USAGE
Run experiments with toy 2D model with varying (fixed) observation noise scale.

Usage: $0 [--help]

The script does not accept arguments other than the flag --help to display this message

Instead environment variables can be set to override the default settings (default or
current values if set are shown in parentheses)

PYTHON_BIN: Path to Python binary to use to run experiments with (${PYTHON_BIN})
MODEL_DIR: Path to directory containing model .py scripts (${MODEL_DIR})
OUTPUT_DIR: Path to output results to (${OUTPUT_DIR})
MODELS: Model(s) (script names in ${MODEL_DIR}) to run experiments for (${MODELS})
OBS_NOISE_STDS: Observation noise scale(s) to run experiments for (${OBS_NOISE_STDS})
SOLVERS: Solver(s) for projection step to use for C-HMC chains (${SOLVERS})
HMC_METRICS: Metric type(s) to use for HMC chains (${HMC_METRICS})
RMHMC_METRICS: Metric type(s) to use for RM-HMC chains (${RMHMC_METRICS})
SEEDS: Integer value(s) to use to seed random number generator (${SEEDS})
NUM_CHAIN: Number of chains to run for each experiment (${NUM_CHAIN})
NUM_WARM_UP_ITER: Number of adaptive warm-up iterations per chain (${NUM_WARM_UP_ITER})
NUM_MAIN_ITER: Number of main sampling iterations per chain (${NUM_MAIN_ITER})
TARGET_ACCEPT: Target acceptance statistic for step size adaptation (${TARGET_ACCEPT})
MAX_TREE_DEPTH: Maximum depth of binary trajectory tree (${MAX_TREE_DEPTH})

For example to run with 2 chains / experiment and remaining arguments as defaults run

NUM_CHAIN=2 $0
HELP_USAGE
}

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-mlift/example_models}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
OBS_NOISE_STDS="${OBS_NOISE_STDS:-0.01 0.03162 0.1 0.3162 1.}"
SOLVERS="${SOLVERS:-newton quasi-newton}"
HMC_METRICS="${HMC_METRICS:-diagonal dense}"
RMHMC_METRICS="${RMHMC_METRICS:-fisher softabs}"
SEEDS="${SEEDS:-202101 202102 202103}"
NUM_CHAIN="${NUM_CHAIN:-4}"
NUM_WARM_UP_ITER="${NUM_WARM_UP_ITER:-1000}"
NUM_MAIN_ITER="${NUM_MAIN_ITER:-2500}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.9}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-15}"

SEP_1="================================================================================"
SEP_2="--------------------------------------------------------------------------------"

if [ "$1" = "--help" ]; then
    usage
    exit 0
fi

echo ${SEP_1}
echo "Running toy 2D model experiments"

echo ${SEP_1}
for SEED in ${SEEDS[@]}; do
    for OBS_NOISE_STD in ${OBS_NOISE_STDS[@]}; do
        for SOLVER in ${SOLVERS[@]}; do
            echo "algorithm=chmc seed=${SEED} σ=${OBS_NOISE_STD} solver=${SOLVER}"
            echo ${SEP_2}
            ${PYTHON_BIN} ${MODEL_DIR}/toy_2d_model.py \
                --seed ${SEED} \
                --algorithm chmc \
                --num-chain ${NUM_CHAIN} \
                --num-warm-up-iter ${NUM_WARM_UP_ITER} \
                --num-main-iter ${NUM_MAIN_ITER} \
                --step-size-adaptation-target ${TARGET_ACCEPT} \
                --max-tree-depth ${MAX_TREE_DEPTH} \
                --obs-noise-std ${OBS_NOISE_STD} \
                --projection-solver ${SOLVER} \
                --output-root-dir ${OUTPUT_DIR} \
                --run-chmc-to-initialise
            echo ${SEP_2}
        done
    done
done

echo ${SEP_1}
for SEED in ${SEEDS[@]}; do
    for OBS_NOISE_STD in ${OBS_NOISE_STDS[@]}; do
        for RMHMC_METRIC in ${RMHMC_METRICS[@]}; do
            echo "algorithm=rmhmc seed=${SEED} σ=${OBS_NOISE_STD} metric=${RMHMC_METRIC}"
            echo ${SEP_2}
            ${PYTHON_BIN} ${MODEL_DIR}/toy_2d_model.py \
                --seed ${SEED} \
                --algorithm rmhmc \
                --num-chain ${NUM_CHAIN} \
                --num-warm-up-iter ${NUM_WARM_UP_ITER} \
                --num-main-iter ${NUM_MAIN_ITER} \
                --step-size-adaptation-target ${TARGET_ACCEPT} \
                --max-tree-depth ${MAX_TREE_DEPTH} \
                --obs-noise-std ${OBS_NOISE_STD} \
                --rmhmc-metric-type ${RMHMC_METRIC} \
                --output-root-dir ${OUTPUT_DIR} \
                --run-chmc-to-initialise
            echo ${SEP_2}
        done
    done
done

echo ${SEP_1}
for SEED in ${SEEDS[@]}; do
    for OBS_NOISE_STD in ${OBS_NOISE_STDS[@]}; do
        for HMC_METRIC in ${HMC_METRICS[@]}; do
            echo "algorithm=hmc seed=${SEED} σ=${OBS_NOISE_STD} metric=${HMC_METRIC}"
            echo ${SEP_2}
            ${PYTHON_BIN} ${MODEL_DIR}/toy_2d_model.py \
                --seed ${SEED} \
                --algorithm hmc \
                --num-chain ${NUM_CHAIN} \
                --num-warm-up-iter ${NUM_WARM_UP_ITER} \
                --num-main-iter ${NUM_MAIN_ITER} \
                --step-size-adaptation-target ${TARGET_ACCEPT} \
                --max-tree-depth ${MAX_TREE_DEPTH} \
                --obs-noise-std ${OBS_NOISE_STD} \
                --hmc-metric-type ${HMC_METRIC} \
                --output-root-dir ${OUTPUT_DIR} \
                --run-chmc-to-initialise
            echo ${SEP_2}
        done
    done
done
