#!/bin/bash
BASE="p142503002-swapnilh@madhava.iitpkd.ac.in:/scratch/p142503002-swapnilh/wmt26"
LOCAL="$HOME/Research/WMT 26"
rsync -avz --progress -e "ssh -p 49122" $BASE/src/     "$LOCAL/src/"
rsync -avz --progress -e "ssh -p 49122" $BASE/outputs/ "$LOCAL/outputs/"
rsync -avz --progress -e "ssh -p 49122" $BASE/logs/    "$LOCAL/logs/"
rsync -avz --progress -e "ssh -p 49122" \
  --include="*/final/***" --include="*/" --exclude="checkpoint-*" \
  $BASE/ckpts/ "$LOCAL/ckpts/"
echo "Sync done: $(date)"
