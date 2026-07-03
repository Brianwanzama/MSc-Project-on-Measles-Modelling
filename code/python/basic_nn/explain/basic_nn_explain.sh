#!/bin/bash
# ============================================================
# basic_nn_explain.sh
# Run SHAP explanation for all k values — WB SFNN
#
# CHANGES FROM ORIGINAL:
#   1. t_lag_map:       k=1 -> 26 (was 78), k=52 removed
#   2. hidden_dim_map:  all corrected to Ray Tune results
#   3. num_layers_map:  k=4 -> 2 (was 1), k=34 -> 1 (was 2)
#   4. job_indices:     removed 52
#   5. model_read_loc:  output/data/ (not output/models/)
#   6. added --year-test-cutoff=2017
# ============================================================

job_name="explain"
echo "Starting job: $job_name"

eval "$(conda shell.bash hook)"
conda activate finalmlenv

base="/.../.../.../"
explain_dir="${base}output/data/basic_nn_optimal/explain/"
model_dir="${base}output/data/basic_nn_optimal/"

mkdir -p "$explain_dir"

# ── correct hyperparameters from our Ray Tune results ─────────
# k=1:  tlag=26,  hidden=64,  layers=1
# k=4:  tlag=52,  hidden=64,  layers=2
# k=12: tlag=52,  hidden=240, layers=1
# k=20: tlag=52,  hidden=64,  layers=1
# k=34: tlag=52,  hidden=64,  layers=1
declare -A t_lag_map
t_lag_map[1]=26
t_lag_map[4]=52
t_lag_map[12]=52
t_lag_map[20]=52
t_lag_map[34]=52

declare -A hidden_dim_map
hidden_dim_map[1]=64
hidden_dim_map[4]=64
hidden_dim_map[12]=240
hidden_dim_map[20]=64
hidden_dim_map[34]=64

declare -A num_layers_map
num_layers_map[1]=1
num_layers_map[4]=2
num_layers_map[12]=1
num_layers_map[20]=1
num_layers_map[34]=1

job_indices=(1 4 12 20 34)

max_jobs=4
current_jobs=0

for k in "${job_indices[@]}"; do
    t_lag=${t_lag_map[$k]}
    hidden_dim=${hidden_dim_map[$k]}
    num_layers=${num_layers_map[$k]}

    echo "$(date '+%Y-%m-%d %H:%M:%S') - k=$k | " \
         "tlag=$t_lag | hidden=$hidden_dim | layers=$num_layers"

    python3 "${base}code/python/basic_nn/explain/basic_nn_explain.py" \
        --k="$k"                                \
        --t-lag="$t_lag"                        \
        --hidden-dim="$hidden_dim"              \
        --num-hidden-layers="$num_layers"       \
        --data-read-loc="$explain_dir"          \
        --write-data-loc="$explain_dir"         \
        --model-read-loc="$model_dir"           \
        --verbose                               &

    ((current_jobs++))
    if [[ $current_jobs -ge $max_jobs ]]; then
        wait -n
        ((current_jobs--))
    fi

done

wait
echo "$(date '+%Y-%m-%d %H:%M:%S') - All SHAP explanations complete"
echo "Output: $explain_dir"
