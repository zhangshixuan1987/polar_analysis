#!/bin/bash
# Running on chrysalis
#SBATCH  --job-name=mov_diag
#SBATCH  --account=e3sm
#SBATCH  --nodes=1
#SBATCH  --output=mov_diag.o%j
#SBATCH  --exclusive
#SBATCH  --time=04:00:00
#SBATCH  --partition=debug #compute

source /lcrc/soft/climate/e3sm-unified/load_latest_e3sm_unified_chrysalis.sh
conda activate pcmdi_metrics_master

for file in *.py;do
 python $file & 
done 
wait 
exit
