#!/bin/bash
###  replace the .py file in the horovodrun call with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err

#SBATCH -J biggan_d64_hdf5
#SBATCH -o biggan_d64_hdf5_%j.out
#SBATCH -e biggan_d64_hdf5_%j.err

#SBATCH --mail-user=seungwook.han@ibm.com
#SBATCH --mail-type=ALL
###  the following parameters will get you two nodes with 4 V100's each
###  note the run time limit is currently only 12 hours, it will be increased
###  next week
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=500g
#SBATCH --time=24:00:00
#SBATCH -p sched_system_all 

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=wmlce-ea
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date


## Horovod execution
###  replace the .py with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err 

python train_pixel.py \
--model BigGANdeep \
--dataset D64_hdf5 --shuffle --num_workers 8 --batch_size 256 --load_in_mem \
--base_root $HOME2/BigGAN-PyTorch/results_deep_pixel/ --data_root $HOME2/BigGAN-PyTorch/data/ILSVRC_D64.hdf5 \
--num_G_accumulations 2 --num_D_accumulations 2 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_ch 128 --D_ch 128 \
--G_depth 2 --D_depth 2 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 128 --shared_dim 128 \
--ema --use_ema --ema_start 20000 --G_eval_mode \
--test_every 2000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler --num_epochs=100 \
--wandb_project biggan_deep_bs512_pixel_hdf5

### --resume --load_weights copy1

echo "Run completed at:- "
date
