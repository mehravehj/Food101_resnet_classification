#!/usr/bin/env bash

#SBATCH --account=def-mpederso       
#SBATCH --cpus-per-task=6                
#SBATCH --gpus-per-node=p100l:1                   
#SBATCH --mem=32G                        
#SBATCH --time=2:00:00               

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "SLURM_ARRAYID="$SLURM_ARRAYID
echo "SLURM_ARRAYID"=$SLURM_ARRAYID
echo "SLURM_ARRAY_JOB_ID"=$SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID
echo "working directory = "$SLURM_SUBMIT_DIR



module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
pip install pandas numpy --no-index
pip install torchmetrics==0.10.3
pip install opencv-python-headless
pip install numba
pip install cupy
pip install ffcv

cp train_256_0.10_90.ffcv $SLURM_TMPDIR
cp val_256_0.10_90.ffcv $SLURM_TMPDIR

python train_resisc_initial.py -t $SLURM_TMPDIR/train_256_0.10_90.ffcv -v $SLURM_TMPDIR/val_256_0.10_90.ffcv -b 256
