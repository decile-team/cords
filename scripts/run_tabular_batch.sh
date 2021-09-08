#!/bin/bash
# Please run this script at the root dir of cords using "sh ./scripts/run_tabular_batch.sh"

start=$(date +%s)
log_path="./scripts/BATCH_PROCESS_$start"
script_path="./scripts/run_tabular.py"
dataset="airline"
mkdir -p $log_path
strategies=("glister" "random-ol" "full" "random" "facloc" "graphcut" "sumredun" "satcov")
pid=()
for strategy in "${strategies[@]}"; do
  echo "Running strategy: $strategy... "
  python3 $script_path --dataset $dataset --dss_strategy $strategy 1>$log_path/$strategy.log 2>$log_path/$strategy.err &
  _pid=$!
  pid+=($_pid)
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

for _pid in "${pid[@]}"; do
  wait $_pid
done
