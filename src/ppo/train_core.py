import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
from torch.utils.tensorboard import SummaryWriter

from .utils import set_seed
from .normalize import RunningMeanStd
from .model import ActorCritic, sample_action_and_logp
from .buffer import RolloutBuffer
from .update import ppo_update

def save_learning_curves(log_dir, save_dir="plots"):
    """Sauvegarde les courbes d'apprentissage en images PNG."""
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # Définir les métriques à visualiser
        metrics = {
            'charts/episode_return': 'Episode Return',
            'charts/episode_length': 'Episode Length',
            'eval/avg_return': 'Evaluation Average Return',
            'loss/policy': 'Policy Loss',
            'loss/value': 'Value Loss',
            'loss/entropy': 'Entropy',
            'diagnostics/approx_kl': 'Approximate KL Divergence',
            'diagnostics/clipfrac': 'Clip Fraction',
            'diagnostics/explained_var': 'Explained Variance',
        }
        
        # Créer un graphique pour chaque métrique
        for tag, title in metrics.items():
            if tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                plt.figure(figsize=(10, 6))
                plt.plot(steps, values, linewidth=2)
                plt.xlabel('Update' if 'loss' in tag or 'diagnostics' in tag or 'eval' in tag else 'Global Step')
                plt.ylabel(title)
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                filename = tag.replace('/', '_') + '.png'
                plt.savefig(os.path.join(save_dir, filename), dpi=150)
                plt.close()
                
        print(f"[Plots] Learning curves saved to: {save_dir}/")
        
    except ImportError:
        print(f"[Warning] tensorboard not installed. Skipping learning curves export.")
    except Exception as e:
        print(f"[Warning] Could not save learning curves: {e}")

@torch.no_grad()
def evaluate_agent(model, obs_rms, env_id="BipedalWalker-v3", n_episodes=5, device="cpu", save_video=False, video_folder=None):
    model.eval()
    device = torch.device(device)

    env = gym.make(env_id, render_mode="rgb_array" if save_video else None)
    if save_video and video_folder:
        env = RecordVideo(env, video_folder, episode_trigger=lambda ep: True, disable_logger=True)
    
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        obs_n = obs_rms.normalize(obs)

        done = False
        ep_ret = 0.0

        while not done:
            obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)
            mu, std, v = model.mu_std_value(obs_t)

            # déterministe (squashed)
            act_t = torch.tanh(mu)
            act = act_t.squeeze(0).cpu().numpy()

            obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            ep_ret += float(rew)

            obs_n = obs_rms.normalize(obs)

        returns.append(ep_ret)

    env.close()
    model.train()
    return float(np.mean(returns))

def train(
    env_id="BipedalWalker-v3",
    total_steps=2_000_000,
    rollout_len=2048,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    lr=3e-4,
    epochs=10,
    minibatch_size=64,
    vf_coef=0.5,
    ent_coef=0.005,
    max_grad_norm=0.5,
    target_kl=None,
    seed=0,
    device="cpu",
    value_clip=True,
    save_video=True,
    video_every=25,
    save_ckpt_every=25,  # en updates
    evaluation_interval=25,  # Intervalle d'évaluation (ajouté)
):
    set_seed(seed)
    device = torch.device(device)

    # Création des répertoires nécessaires
    os.makedirs("runs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initialisation de l'environnement
    run_name = f"PPO_squashed_{env_id}_seed{seed}_{int(time.time())}"
    env = gym.make(env_id, render_mode="rgb_array")
    
    if save_video:
        vid_dir = f"videos/{run_name}"
        # RecordVideo avec disable_logger=True pour éviter les warnings et name_prefix unique
        env = RecordVideo(
            env, 
            vid_dir, 
            episode_trigger=lambda ep: ep % video_every == 0,
            disable_logger=True  # Évite les warnings qui peuvent causer des blocages
        )

    # Dimensions des observations et actions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Initialisation du modèle et de l'optimiseur
    model = ActorCritic(obs_dim, act_dim, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # Initialisation de TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # Normalisation des observations
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    # Réinitialisation de l'environnement
    obs, _ = env.reset(seed=seed)
    obs_rms.update(obs)
    obs_n = obs_rms.normalize(obs)

    # Variables pour suivre les métriques pendant l'entraînement
    ep_ret, ep_len, ep_id = 0.0, 0, 0
    num_updates = total_steps // rollout_len
    global_step = 0

    # Suivi du meilleur modèle
    best_return = -float('inf')  # Initialiser le meilleur retour à une valeur très basse

    for update in range(1, num_updates + 1):
        buffer = RolloutBuffer(rollout_len, obs_dim, act_dim, device)

        # Ajustement du learning rate (décroissance)
        frac = 1.0 - (update - 1.0) / num_updates
        for pg in optim.param_groups:
            pg["lr"] = lr * frac

        action_abs_acc = 0.0

        for _ in range(rollout_len):
            global_step += 1

            obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                mu, std, v = model.mu_std_value(obs_t)
                act_t, logp_t = sample_action_and_logp(mu, std)

            act_np = act_t.squeeze(0).cpu().numpy()
            next_obs, rew, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated

            obs_rms.update(next_obs)
            next_obs_n = obs_rms.normalize(next_obs)

            buffer.add(
                obs=obs_n,
                act=act_np,
                logp=float(logp_t.item()),
                rew=float(rew),
                done=float(done),
                val=float(v.item()),
            )

            ep_ret += rew
            ep_len += 1
            action_abs_acc += float(np.mean(np.abs(act_np)))

            obs = next_obs
            obs_n = next_obs_n

            if done:
                ep_id += 1
                writer.add_scalar("charts/episode_return", ep_ret, global_step)  # Episode Return
                writer.add_scalar("charts/episode_length", ep_len, global_step)  # Episode Length
                writer.add_scalar(
                    "diagnostics/mean_abs_action_episode",
                    action_abs_acc / max(1, ep_len),
                    global_step,
                )  # Mean absolute action

                obs, _ = env.reset()
                obs_rms.update(obs)
                obs_n = obs_rms.normalize(obs)

                ep_ret, ep_len = 0.0, 0
                action_abs_acc = 0.0

        with torch.no_grad():
            obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, last_val = model.mu_std_value(obs_t)
            last_val = float(last_val.item())

        buffer.compute_gae(last_val=last_val, gamma=gamma, lam=lam)
        data = buffer.get()

        stats = ppo_update(
            model=model,
            optim=optim,
            data=data,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            minibatch_size=minibatch_size,
            target_kl=target_kl,
            value_clip=value_clip,
        )

        writer.add_scalar("loss/policy", stats["loss_pi"], update)  # Policy Loss
        writer.add_scalar("loss/value", stats["loss_v"], update)    # Value Loss
        writer.add_scalar("loss/entropy", stats["entropy"], update) # Entropy Loss
        writer.add_scalar("diagnostics/approx_kl", stats["approx_kl"], update)
        writer.add_scalar("diagnostics/clipfrac", stats["clipfrac"], update)
        writer.add_scalar("diagnostics/explained_var", stats["explained_var"], update)
        writer.add_scalar("diagnostics/log_std_mean", stats["log_std_mean"], update)
        writer.add_scalar("charts/learning_rate", optim.param_groups[0]["lr"], update)  # Learning Rate

        if update % 10 == 0:
            print(
                f"Update {update}/{num_updates} | "
                f"pi={stats['loss_pi']:.3f} v={stats['loss_v']:.3f} "
                f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
                f"clip={stats['clipfrac']:.2f} ev={stats['explained_var']:.2f} "
                f"logstd={stats['log_std_mean']:.2f}"
            )

        # Sauvegarde régulière du modèle tous les `save_ckpt_every` updates
        if save_ckpt_every is not None and (update % save_ckpt_every == 0 or update == num_updates):
            ckpt_path = f"checkpoints/{run_name}_update{update}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_rms_mean": obs_rms.mean,
                    "obs_rms_var": obs_rms.var,
                    "obs_rms_count": obs_rms.count,
                    "config": dict(
                        env_id=env_id,
                        seed=seed,
                        total_steps=total_steps,
                        rollout_len=rollout_len,
                        gamma=gamma,
                        lam=lam,
                        clip_eps=clip_eps,
                        lr=lr,
                        epochs=epochs,
                        minibatch_size=minibatch_size,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        max_grad_norm=max_grad_norm,
                        target_kl=target_kl,
                        value_clip=value_clip,
                    ),
                },
                ckpt_path,
            )
            print(f"[checkpoint] saved: {ckpt_path}")

        # Sauvegarder le meilleur modèle basé sur la performance (retour moyen)
        if update % evaluation_interval == 0:
            avg_return = evaluate_agent(model, obs_rms, env_id=env_id, n_episodes=5, device=device)
            writer.add_scalar("eval/avg_return", avg_return, update)
            print(f"[Eval] Update {update} | avg_return: {avg_return:.2f}")
            
            if avg_return > best_return:
                best_return = avg_return
                best_ckpt_path = f"checkpoints/{run_name}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "obs_rms_mean": obs_rms.mean,
                        "obs_rms_var": obs_rms.var,
                        "obs_rms_count": obs_rms.count,
                        "config": dict(
                            env_id=env_id,
                            seed=seed,
                            total_steps=total_steps,
                            rollout_len=rollout_len,
                            gamma=gamma,
                            lam=lam,
                            clip_eps=clip_eps,
                            lr=lr,
                            epochs=epochs,
                            minibatch_size=minibatch_size,
                            vf_coef=vf_coef,
                            ent_coef=ent_coef,
                            max_grad_norm=max_grad_norm,
                            target_kl=target_kl,
                            value_clip=value_clip,
                        ),
                    },
                    best_ckpt_path,
                )
                print(f"[Best] Model saved at update {update} with avg_return: {avg_return:.2f}")

    env.close()
    
    # Sauvegarder les courbes d'apprentissage en images
    plot_dir = f"plots/{run_name}"
    save_learning_curves(f"runs/{run_name}", save_dir=plot_dir)
    
    writer.close()

