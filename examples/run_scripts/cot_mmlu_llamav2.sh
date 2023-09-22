#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=cot_mmlu_llamav2.out

torchrun --nproc_per_node 1 examples/ChainOfThought/ChainOfThought.py -c configs/MMLU/llamav2.yml -b Test