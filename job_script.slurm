#!/bin/bash
#SBATCH --job-name=job_as_test    # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=00:10:00               # Time limit hrs:min:sec
#SBATCH --output=%j.log   # Standard output and error log
#SBATCH --partition=standard
#SBATCH --qos=standard
export OMP_NUM_THREADS=1
pwd; hostname; date
source /work/tc046/tc046/eilishnewmark/get_data/datavenv/bin/activate
echo "Running python script"
python ./get_training_data.py
date