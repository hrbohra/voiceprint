#!/usr/bin/env bash
# Low-cost progress bar for the tweet-attribution queue: reads logs and counts checkpoint files
# every 30 s. No Python, no GPU. Ctrl+C to quit.
cd "$(dirname "$0")/.."
# stage weights, roughly proportional to measured run time (NLI social acts dominate)
declare -A W=([surface]=1 [syntax]=8 [dialogue]=1 [sentiment]=6 [emotions]=6 [dialogue_acts]=10 [social_acts]=40)
TOTAL=72; BLOCKS=8

run_pct() {  # percent of one run, from its log plus the newest checkpoint dir
  local log=$1
  grep -q "^EXIT" "$log" 2>/dev/null && { echo 100; return; }
  [ -f "$log" ] || { echo 0; return; }
  local dir; dir=$(ls -td .voiceprint/features/*/ 2>/dev/null | head -1)
  local done=0
  for s in surface syntax dialogue sentiment emotions dialogue_acts social_acts; do
    local n=0 max=$BLOCKS
    [ "$s" = surface ] || [ "$s" = dialogue ] && max=1
    [ -n "$dir" ] && n=$(ls "$dir" 2>/dev/null | grep -c "^$s\.")
    grep -q "    $s" "$log" && n=$max
    done=$(( done + W[$s] * n / max ))
  done
  echo $(( done * 100 / TOTAL ))
}

bar() {  # bar <label> <pct>
  local w=40 f=$(( $2 * 40 / 100 )) full="" empty="" i
  for ((i = 0; i < f; i++)); do full+="#"; done
  for ((i = f; i < w; i++)); do empty+="."; done
  printf "  %-26s [%s%s] %3d%%\n" "$1" "$full" "$empty" "$2"
}

while true; do
  a=$(run_pct scratch/twcs.log)
  b=0; [ "$a" = 100 ] && b=$(run_pct scratch/twcs_nosig.log)
  clear
  echo "Voiceprint: tweet attribution queue   $(date +%H:%M:%S)"
  echo
  bar "1. corrected attribution" "$a"
  bar "2. signature ablation" "$b"
  echo
  bar "overall" $(( (a + b) / 2 ))
  gpu=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null)
  echo; echo "  GPU temp, load: ${gpu:-n/a}"
  [ "$b" = 100 ] && { echo; echo "  Done."; break; }
  sleep 30
done
