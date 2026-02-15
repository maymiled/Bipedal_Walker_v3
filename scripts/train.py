from src.ppo.train_core import train
import torch

if __name__ == "__main__":
    train(
        env_id="BipedalWalker-v3",
        total_steps=4_000_000,  # Doublé pour voir si le modèle continue à s'améliorer
        rollout_len=2048,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        lr=3e-4,
        epochs=10,
        minibatch_size=64,
        vf_coef=0.5,
        ent_coef=0.005,  # Légère exploration pour casser le plateau
        max_grad_norm=0.5,
        target_kl=None,
        value_clip=True,
        seed=0,
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        save_video=True,
        video_every=25,
        save_ckpt_every=25,
    )

    


