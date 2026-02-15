import os
import glob
import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch

from src.ppo.model import ActorCritic
from src.ppo.normalize import RunningMeanStd


def pick_checkpoint(checkpoints_dir="checkpoints", prefer_best=True):
    ckpts = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {checkpoints_dir}/")

    if prefer_best:
        best = [p for p in ckpts if p.endswith("_best.pt")]
        if best:
            # si plusieurs best (plusieurs runs), prends le plus récent
            return max(best, key=os.path.getmtime)

    # sinon: le plus récent
    return max(ckpts, key=os.path.getmtime)


@torch.no_grad()
def eval_policy(
    ckpt_path: str,
    env_id="BipedalWalker-v3",
    n_episodes=10,
    device="cpu",
    render=False,
    record_video=False,
    video_dir="videos_eval",
    deterministic=True,
):
    device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    # si tu veux forcer l'env_id du checkpoint, tu peux :
    env_id = config.get("env_id", env_id)

    # Env
    render_mode = "human" if render else ("rgb_array" if record_video else None)
    env = gym.make(env_id, render_mode=render_mode)

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        run_stamp = os.path.basename(ckpt_path).replace(".pt", "")
        out_dir = os.path.join(video_dir, f"{env_id}_{run_stamp}_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)
        env = RecordVideo(env, out_dir, episode_trigger=lambda ep: True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Model
    model = ActorCritic(obs_dim, act_dim, hidden=64).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Obs normalization (CRUCIAL)
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.mean = ckpt["obs_rms_mean"]
    obs_rms.var = ckpt["obs_rms_var"]
    obs_rms.count = ckpt["obs_rms_count"]

    returns = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        obs_n = obs_rms.normalize(obs)

        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)

            mu, std, v = model.mu_std_value(obs_t)

            if deterministic:
                act_t = torch.tanh(mu)  # policy déterministe pour "squashed Gaussian"
            else:
                # stochastique: sample comme en training
                eps = torch.randn_like(std)
                act_t = torch.tanh(mu + std * eps)

            act = act_t.squeeze(0).cpu().numpy()

            next_obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            ep_ret += float(rew)
            ep_len += 1

            obs = next_obs
            obs_n = obs_rms.normalize(obs)

        returns.append(ep_ret)
        lengths.append(ep_len)
        print(f"Episode {ep+1}/{n_episodes} | return={ep_ret:.1f} | len={ep_len}")

    env.close()

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)

    print("\n=== Eval summary ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Mean return: {returns.mean():.1f} ± {returns.std():.1f}")
    print(f"Min/Max return: {returns.min():.1f} / {returns.max():.1f}")
    print(f"Mean length: {lengths.mean():.1f}")

    return returns, lengths


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt_path = pick_checkpoint("checkpoints", prefer_best=True)
    print(f"Using checkpoint: {ckpt_path}")

    eval_policy(
        ckpt_path=ckpt_path,
        env_id="BipedalWalker-v3",
        n_episodes=10,
        device=device,
        render=False,          # mets True si tu veux voir en live
        record_video=True,     # mets True si tu veux des vidéos
        deterministic=True,
    )
