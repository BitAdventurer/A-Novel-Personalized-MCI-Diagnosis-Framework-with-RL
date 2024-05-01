# **Sparse Graph Representation Learning based on Reinforcement Learning for Personalized Mild Cognitive Impairment (MCI) Diagnosis**
This is the **official** PyTorch implementation of "[Sparse Graph Representation Learning based on Reinforcement Learning for Personalized Mild Cognitive Impairment (MCI) Diagnosis](https://https://ieeexplore.ieee.org/document/10509746)", which has been accepted for publication in Biomedical and Health Informatics(JBHI) in 2024.



## Abstract
Resting-state functional magnetic resonance imaging (rs-fMRI) has gained attention as a reliable tech- nique for investigating the intrinsic function patterns of the brain. It facilitates the extraction of functional connectivity networks (FCNs) that capture synchronized activity pat- terns among regions of interest (ROIs). 
Analyzing FCNs enables the identification of distinctive connectivity patterns associated with mild cognitive impairment (MCI). For MCI diagnosis, various sparse representation techniques have been introduced, including statistical and deep learning-based methods. 
However, these methods face limitations due to their reliance on supervised learning schemes, which restrict the exploration necessary for probing novel solutions. 
To overcome such limitation, prior work has incorporated reinforcement learning (RL) to dynamically select ROIs, but effective exploration remains challenging due to the vast search space during training. 
To tackle this issue, in this study, we propose an advanced RL-based framework that utilizes a divide-and-conquer approach to decompose the FCN construction task into smaller sub- problems in a subject-specific manner, enabling efficient exploration under each sub-problem condition. 
Additionally, we leverage the learned value function to determine the sparsity level of FCNs, considering individual characteris- tics of FCNs. 
We validate the effectiveness of our proposed framework by demonstrating its superior performance in MCI diagnosis on publicly available cohort datasets.
### Overview
<p align="center">
    <img src="/[JBHI]OVERVIEW.png" alt="drawing" width="800"/>
</p>

---
## Requirments 
- python >= 3.11
- pytorch >= 2.2.1

---
## Preparation
Due to storage limitations, we have not attached dataset files in the ```.npy`` format within the library and code files. The datasets used in this study are the ADNI2, ADNI3 and ADNI GO datasets, which can be downloaded from the links below:
- [Alzheimer’s Disease Neuroimaging Initiative(ADNI)](http://adni.loni.usc.edu)


## Usage
Our code can be executed with the configuration file 'train.yaml.' An example run for training the main code is as follows:
<br>
```bash
pip install -r package.txt
```
---
  
## Citation 
If you discover value in this content, kindly reference the following article:
```bibtex
  author={Ji, Chang-Hoon and Shin, Dong-Hee and Son, Young-Han and Kam, Tae-Eui},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Sparse Graph Representation Learning based on Reinforcement Learning for Personalized Mild Cognitive Impairment (MCI) Diagnosis}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Task analysis;Topology;Brain modeling;Bioinformatics;Supervised learning;Reinforcement learning;Network topology;Reinforcement Learning;Brain Disease Diagnosis;MCI;Functional Connectivity Network},
  doi={10.1109/JBHI.2024.3393625}}
```
---
  
  
## Acknowledgements
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant (No. 2019-0-00079, Artificial Intelligence Graduate School Program(Korea University), No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration), and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No.RS202300212498).
## Contact Us
This post was published through the official MAILAB account, and the issues tab has been disabled. If you encounter any issues while running the code or spot errors, please reach out to the code authors below:

- Chang-Hoon Ji: ckdgns0611@korea.ac.kr
- Dong-Hee Shin: dongheeshin@korea.ac.kr
- Young-Han Son: yhson135@korea.ac.kr

You can also locate their official GitHub accounts in the contributors list.
