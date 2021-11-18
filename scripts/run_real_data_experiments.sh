#!/usr/bin/env bash

usage() {
cat <<- HELP_USAGE
Run experiments with real data

Usage: $0 [--help]

The script does not accept arguments other than the flag --help to display this message

Instead environment variables can be set to override the default settings (default or
current values if set are shown in parentheses)

PYTHON_BIN: Path to Python binary to use to run experiments with (${PYTHON_BIN})
MODEL_DIR: Path to directory containing model .py scripts (${MODEL_DIR})
DATA_DIR: Path to directory containing observed data files (${DATA_DIR})
OUTPUT_DIR: Path to output results to (${OUTPUT_DIR})
MODELS: Model(s) (script names in ${MODEL_DIR}) to run experiments for (${MODELS})
SOLVERS: Solver(s) for projection step to use for C-HMC chains (${SOLVERS})
METRICS: Metric type(s_ to use for HMC chains (${METRICS})
SEEDS: Integer value(s) to use to seed random number generator (${SEEDS})
NUM_CHAIN: Number of chains to run for each experiment (${NUM_CHAIN})
NUM_WARM_UP_ITER: Number of adaptive warm-up iterations per chain (${NUM_WARM_UP_ITER})
NUM_MAIN_ITER: Number of main sampling iterations per chain (${NUM_MAIN_ITER})
TARGET_ACCEPT: Target acceptance statistic for step size adaptation (${TARGET_ACCEPT})
MAX_TREE_DEPTH: Maximum depth of binary trajectory tree (${MAX_TREE_DEPTH})
PARAMETRIZATION: Parameterization to use for model variables (${PARAMETRIZATION})

For example to run with 2 chains / experiment on the fitzhugh_nagumo and poisson models

NUM_CHAIN=2 MODELS="lotka_volterra soil_incubation" $0
HELP_USAGE
}

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-mlift/example_models}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
MODELS="${MODELS:-lotka_volterra soil_incubation hh_voltage_clamp_potassium hh_voltage_clamp_sodium}"
SOLVERS="${SOLVERS:-newton-line-search quasi-newton}"
METRICS="${METRICS:-diagonal dense}"
SEEDS="${SEEDS:-202101 202102 202103}"
NUM_CHAIN="${NUM_CHAIN:-4}"
NUM_WARM_UP_ITER="${NUM_WARM_UP_ITER:-1000}"
NUM_MAIN_ITER="${NUM_MAIN_ITER:-2500}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.9}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-10}"
PARAMETRIZATION="${PARAMETRIZATION:-unbounded}"

SEP_1="================================================================================"
SEP_2="--------------------------------------------------------------------------------"

if [ "$1" = "--help" ]; then
    usage
    exit 0
fi

for MODEL in ${MODELS[@]}; do
    echo ${SEP_1}
    echo "Running ${MODEL} model real data experiments"
    echo ${SEP_1}
    for SEED in ${SEEDS[@]}; do
        for SOLVER in ${SOLVERS[@]}; do
            echo "algorithm=chmc seed=${SEED} solver=${SOLVER}"
            echo ${SEP_2}
            ${PYTHON_BIN} ${MODEL_DIR}/${MODEL}.py \
                --seed ${SEED} \
                --algorithm chmc \
                --num-chain ${NUM_CHAIN} \
                --num-warm-up-iter ${NUM_WARM_UP_ITER} \
                --num-main-iter ${NUM_MAIN_ITER} \
                --step-size-adaptation-target ${TARGET_ACCEPT} \
                --max-tree-depth ${MAX_TREE_DEPTH} \
                --projection-solver ${SOLVER} \
                --prior-parametrization ${PARAMETRIZATION} \
                --output-root-dir ${OUTPUT_DIR} \
                --data-dir ${DATA_DIR}
            echo ${SEP_2}
        done
    done
    echo ${SEP_1}
    for SEED in ${SEEDS[@]}; do
        for METRIC in ${METRICS[@]}; do
            echo "algorithm=hmc seed=${SEED} metric=${METRIC}"
            echo ${SEP_2}
            ${PYTHON_BIN} ${MODEL_DIR}/${MODEL}.py \
                --seed ${SEED} \
                --algorithm hmc \
                --num-chain ${NUM_CHAIN} \
                --num-warm-up-iter ${NUM_WARM_UP_ITER} \
                --num-main-iter ${NUM_MAIN_ITER} \
                --step-size-adaptation-target ${TARGET_ACCEPT} \
                --max-tree-depth ${MAX_TREE_DEPTH} \
                --metric-type ${METRIC} \
                --prior-parametrization ${PARAMETRIZATION} \
                --output-root-dir ${OUTPUT_DIR} \
                --data-dir ${DATA_DIR}
            echo ${SEP_2}
        done
    done
done
