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
python -m scripts.eval --ckpt checkpoints/PPO_squashed_BipedalWalker-v3_seed0_1771188272_update1625.pt # My personal higher update but you can put yours

