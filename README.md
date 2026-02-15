# Reinforcement Learning PPO for BipedalWalker

This project implements the Proximal Policy Optimization (PPO) algorithm from scratch to solve the **BipedalWalker-v3** environment using the Gymnasium library.

## Requirements

To run this project, you need to have the following dependencies installed:
- Python 3.8+
- PyTorch (with MPS or CUDA support)
- Gymnasium
- TensorBoard

### Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv .envRL
   source .envRL/bin/activate  # On macOS/Linux
   .envRL\Scripts\activate     # On Windows

2. **Install dependencies:**
pip install -r requirements.txt

3. **Run training:**
python -m scripts.train

4. **Run evaluation (deterministic):**
python -m scripts.eval --ckpt checkpoints/PPO_squashed_BipedalWalker-v3_seed0_1625811779_update100.pt

### File structure
.
├── requirements.txt       # Python dependencies
├── README.md              # Project description
├── .gitignore             # Git ignore file
├── scripts/               # Contains the training and evaluation scripts
│   ├── train.py           # Main training script
│   └── eval.py            # Evaluation script (for deterministic policy)
├── src/                   # Source code for PPO implementation
│   └── ppo/               # PPO components (model, buffer, training)
│       ├── _init_.py
│       ├── buffer.py
│       ├──model.py
│       ├──normalize.py
│       ├──train_core.py
│       ├──update.py
│       └──utils.py
├── checkpoints/           # Folder for saving model checkpoints
├── runs/                  # Folder for TensorBoard logs
└── videos/                # Folder for recorded videos of training episodes