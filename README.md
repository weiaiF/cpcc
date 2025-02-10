# Consistency Policy with Categorical Critic for Autonomous Driving

Code for "Consistency Policy with Categorical Critic for Autonomous Driving", AAMAS 2025 paper.

# Introduction
！[framework](./figures/method_framework.png)

# Quick Start

1. Setting up repo

```bash
git clone https://github.com/weiaiF/cpcc
```

2. Install Dependencies

```bash
conda create -n cpcc python=3.7
conda activate cpcc
cd cpcc
pip install -r requirements.txt
```
3. Train

```bash
sh train_cpcc.sh
```

# Acknowledgement
CPCC is greatly inspired by the following outstanding contributions to the open-source community:
[CPQL](https://github.com/cccedric/cpql), [MetaDrive](https://github.com/metadriverse/metadrive)
