#!/bin/bash
# Please run this script at the root dir of cords using "bash ./scripts/run_tabular_dataloader_batch.sh"

start=$(date +%s)
log_path="./scripts/BATCH_PROCESS_$start"
echo "Strating batch $start... "
script_path="./scripts/run_tabular_dataloader.py"
#dataset="airline"
mkdir -p $log_path
datasets=("airline" "loan" "olympic")
#strategies=("glister" "random-ol" "full" "random" "facloc" "graphcut" "sumredun" "satcov" "CRAIG")
#strategies=("glister" "gradmatch" "random-ol" "full" "random" "facloc" "graphcut" "sumredun" "satcov" "CRAIG")
strategies=("random" "facloc" "graphcut" "sumredun" "satcov")
select_ratios=("0.1" "0.3" "0.5")
#strategies=("glister" "random-ol")
device="cuda"
#device="cpu"

pid=()
for select_ratio in "${select_ratios[@]}"; do
  for dataset in "${datasets[@]}"; do
    for strategy in "${strategies[@]}"; do
      echo "Running dataset: $dataset with strategy: $strategy... "
      python3 $script_path --dataset $dataset --dss_strategy $strategy --device $device --select_ratio $select_ratio
        1>$log_path/${dataset}_$strategy.log 2>$log_path/${dataset}_$strategy.err &
      _pid=$!
      pid+=($_pid)
    done
    for _pid in "${pid[@]}"; do
      wait $_pid
    done
    pid=()
  done
done

kill_scripts() {
  echo "Killing processes... "
  for _pid in "${pid[@]}"; do
    kill -s SIGTERM $_pid >/dev/null 2>&1 || (sleep 10 && kill -9 $_pid >/dev/null 2>&1 &)
  done
}

trap_quit() {
  kill_scripts
}

trap trap_quit EXIT
