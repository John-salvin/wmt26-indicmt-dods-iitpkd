#!/bin/bash
set -e
source /scratch/p142503002-swapnilh/nllb_venv/bin/activate
cd /scratch/p142503002-swapnilh/wmt26

TESTFILE=$1
DIR=$2

case $DIR in
  en-kha)
    sbatch run.sbatch run_nllb.py translate \
      --infile $TESTFILE --src en --tgt kha --src-col en \
      --adapter ckpts/nllb_en-kha/final \
      --out outputs/DoDS-IITPKD_primary_en_kha.txt ;;
  kha-en)
    sbatch run.sbatch run_nllb.py translate \
      --infile $TESTFILE --src kha --tgt en --src-col kha \
      --adapter ckpts/nllb_kha-en/final \
      --out outputs/DoDS-IITPKD_primary_kha_en.txt ;;
  en-lus)
    sbatch run.sbatch run_nllb.py translate \
      --infile $TESTFILE --src en --tgt lus --src-col en \
      --adapter ckpts/nllb_en-lus/final \
      --out outputs/DoDS-IITPKD_primary_en_lus.txt ;;
  lus-en)
    sbatch run.sbatch run_nllb.py translate \
      --infile $TESTFILE --src lus --tgt en --src-col lus \
      --adapter ckpts/nllb_lus-en/final \
      --out outputs/DoDS-IITPKD_primary_lus_en.txt ;;
  all)
    bash $0 $TESTFILE en-kha
    bash $0 $TESTFILE kha-en
    bash $0 $TESTFILE en-lus
    bash $0 $TESTFILE lus-en ;;
  *)
    echo "Use: en-kha | kha-en | en-lus | lus-en | all" ;;
esac
echo "Submitted: $DIR"
