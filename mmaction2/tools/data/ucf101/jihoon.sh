#!/bin/sh
sbatch <<EOT
#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=1     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory 
#SBATCH -o "sbatchout/outfile"$1            # send stdout to outfile
#SBATCH -e "sbatchout/errfile"$1            # send stderr to errfile
#SBATCH -t 100:00:00           # time requested in hour:minute:second
#SBATCH --exclude=node718,node028,node903

squeue -u jc5933
nvidia-smi

source ~/.bashrc

export ZZROOT=$HOME/jcproject/app
export PATH=$ZZROOT/bin:$PATH
export LD_LIBRARY_PATH=$ZZROOT/lib:$ZZROOT/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/n/fs/jc-project/app/src/lib64:$LD_LIBRARY_PATH

conda activate 3090

cd /u/jc5933/vaiscr/metric1/mmaction2/tools/data
python build_rawframes_jihoon.py ../../data/ucf101/videos/ ../../data/ucf101/rawframes/ --task rgb --level 2  --ext avi --jihoon $1
echo "Genearte raw frames (RGB only)"

cd ucf101/

EOT