#!/usr/bin/env bash

#SBATCH --account=def-mpederso       
#SBATCH --cpus-per-task=6                
#SBATCH --gres=gpu:1                     
#SBATCH --mem=25G                        
#SBATCH --time=1:00:00               

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "SLURM_ARRAYID="$SLURM_ARRAYID
echo "SLURM_ARRAYID"=$SLURM_ARRAYID
echo "SLURM_ARRAY_JOB_ID"=$SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID
echo "working directory = "$SLURM_SUBMIT_DIR

cp food101_processed.tar $SLURM_TMPDIR
tar -xf $SLURM_TMPDIR/food101_processed.tar -C $SLURM_TMPDIR
rm $SLURM_TMPDIR/food101_processed.tar
cd $SLURM_TMPDIR/food-101/train
ls

cd $SLURM_SUBMIT_DIR

export IMAGENET_DIR=$SLURM_TMPDIR/food-101
export WRITE_DIR=$SLURM_TMPDIR

./write_imagenet.sh 500 0.5 90
cp $SLURM_TMPDIR/train_256_0.5_90.ffcv $SLURM_SUBMIT_DIR
