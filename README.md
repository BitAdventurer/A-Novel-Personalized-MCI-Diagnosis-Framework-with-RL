# A-Novel-Personalized-MCI-Diagnosis-Framework-with-RL

## Sparse Graph Representation Learning based on Reinforcement Learning for Personalized Mild Cognitive Impairment (MCI) Diagnosis

## Abstract

Resting-state functional magnetic resonance imaging (rs-fMRI) has gained attention as a reliable tech- nique for investigating the intrinsic function patterns of the brain. It facilitates the extraction of functional connectivity networks (FCNs) that capture synchronized activity pat- terns among regions of interest (ROIs). Analyzing FCNs en- ables the identification of distinctive connectivity patterns associated with mild cognitive impairment (MCI). For MCI diagnosis, various sparse representation techniques have been introduced, including statistical- and deep learning- based methods. However, these methods face limitations due to their reliance on supervised learning schemes, which restrict the exploration necessary for probing novel solutions. To overcome such limitation, prior work has incorporated reinforcement learning (RL) to dynamically select ROIs, but effective exploration remains challenging due to the vast search space during training. To tackle this issue, in this study, we propose an advanced RL-based framework that utilizes a divide-and-conquer approach to decompose the FCN construction task into smaller sub- problems in a subject-specific manner, enabling efficient exploration under each sub-problem condition. Addition- ally, we leverage the learned value function to determine the sparsity level of FCNs, considering individual characteris- tics of FCNs. We validate the effectiveness of our proposed framework by demonstrating its superior performance in MCI diagnosis on publicly available cohort datasets.

## Installation

```
# create a new environment
conda create --name RL python=3.11
conda activate RL

# install requirements
pip install -r requirments.txt

```

## Usage

1. Build
    
    ```python
    pyhon setup.py build_ext --inplace
    
    ```
    
2. Execute [main.py](http://main.py/)
    
    ```python
    python train_cycle.py
    
    ```
