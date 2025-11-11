# Multi-Agent Reinforcement Learning for V2V Communications with Inverse Reinforcement Learning (IRLD3QN)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%2B-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0%2B-orange.svg)]()

This repository contains the official implementation of the paper "[IRL-D3QN: An intelligent multi-agent learning framework for dynamic spectrum management in vehicular networks]" (under review). The code implements an **Inverse Reinforcement Learning with Dueling Double Deep Q-Network (IRLD3QN)** framework for resource allocation in Vehicle-to-Vehicle (V2V) communications.

#  Paper Highlight
We propose IRL-D3QN, a novel multi-agent reinforcement learning framework for dynamic spectrum allocation in vehicular networks.
The framework integrates inverse reinforcement learning to automatically infer reward functions, eliminating the need for manual reward design.
A dueling double Q-network is used to enhance policy stability and reduce overestimation bias in dynamic multi-agent environments.
The model supports distributed spectrum and power control via centralized training and decentralized execution.


#  Key Features
- **IRLD3QN Algorithm**: Inverse Reinforcement Learning combined with Dueling Double DQN
- **Multi-Agent Framework** with centralized training and decentralized execution
- **Advanced Neural Architecture**: Dueling networks with separate value and advantage streams
- **Double Q-Learning**: Reduces overestimation bias in value estimation
- **Realistic V2V/V2I channel modeling** including path loss and shadowing
- **Multiple baselines** for comparison (SARL, Random, DPRA)
- **Comprehensive testing framework** with success rate and throughput metrics

#  Project Structure
```
.
├── Environment_marl.py          # Training environment definition
├── Environment_marl_test.py     # Testing environment with multiple baselines
├── irl_d3qn_marltrain.py        # IRLD3QN training script
├── irl_d3qn_marltest.py         # IRLD3QN testing script
├── replay_memory.py             # Experience replay buffer
├── model/                       # Directory for saved models
│   ├── marl_irl_d3qn_model/
│   └── sarl_irl_d3qn_model/
└── results/                     # Evaluation results and plots
```

# Prerequisites
- Python 3.6+
- PyTorch 1.0+
- NumPy, SciPy

### Install Dependencies
```bash
pip install torch numpy scipy
```

## 🎯 Usage

### Training IRLD3QN
```bash
python irl_d3qn_marltrain.py
```

### Testing IRLD3QN
```bash
python irl_d3qn_marltest.py
```

# IRLD3QN Architecture
- **Dueling Network**: Separates value and advantage streams for better policy learning
- **Double Q-Learning**: Uses target network to reduce overestimation bias
- **IRL Reward Network**: Learns reward function from expert demonstrations
- **Centralized Training**: Uses global information for training
- **Decentralized Execution**: Each agent acts based on local observations

# Environment
- Realistic urban scenario with multiple lanes
- V2V and V2I channel models with fast/slow fading
- Dynamic vehicle movement with lane changing

# 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*Note: This code is released for academic research purposes. For commercial use, please contact the authors.*# IRL_D3QN-marl
