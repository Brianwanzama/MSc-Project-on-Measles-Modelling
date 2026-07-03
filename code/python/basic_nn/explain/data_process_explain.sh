#!/bin/bash
# ============================================================
# data_process_explain.sh
# Run data_process_explain.py for all k values
# West Bengal SFNN — aligned with final Ray Tune results
#
# CHANGES FROM ORIGINAL:
#   1. t_lag_map: k=1 -> 26 (was 78), k=52 removed
#   2. year-test-cutoff: 2016 -> 2017
#   3. job_indices: removed 52
# ============================================================

job_name="dataprocessexplain"
echo "Starting job: $job_name"

eval "$(conda shell.bash hook)"
conda activate finalmlenv

base="/.../.../.../"
save_data_loc="${base}output/data/basic_nn_optimal/explain/"
cases_data_loc="${base}output/data/prefit_cases1/"

mkdir -p "$save_data_loc"

# ── correct t_lag per k from our Ray Tune results ─────────────
# k=1  -> tlag=26  (best: hidden=64,  layers=1)
# k=4  -> tlag=52  (best: hidden=64,  layers=2)
# k=12 -> tlag=52  (best: hidden=240, layers=1)
# k=20 -> tlag=52  (best: hidden=64,  layers=1)
# k=34 -> tlag=52  (best: hidden=64,  layers=1)
declare -A t_lag_map
t_lag_map[1]=26
t_lag_map[4]=52
t_lag_map[12]=52
t_lag_map[20]=52
t_lag_map[34]=52

job_indices=(1 4 12 20 34)

max_jobs=4
current_jobs=0

for k in "${job_indices[@]}"; do
    t_lag=${t_lag_map[$k]}
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Running k=$k tlag=$t_lag"

    python3 "${base}code/python/basic_nn/explain/data_process_explain.py" \
        --k="$k"                              \
        --t-lag="$t_lag"                      \
        --year-test-cutoff=2017               \
        --save-data-loc="$save_data_loc"      \
        --cases-data-loc="$cases_data_loc"    &

    ((current_jobs++))
    if [[ $current_jobs -ge $max_jobs ]]; then
        wait -n
        ((current_jobs--))
    fi

done

wait
echo "$(date '+%Y-%m-%d %H:%M:%S') - All data processing complete"
echo "Output: $save_data_loc"
