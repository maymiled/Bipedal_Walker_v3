import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import explained_variance
from .model import logp_from_action, gaussian_entropy, ActorCritic


def ppo_update(
    model: ActorCritic,
    optim: torch.optim.Optimizer,
    data: dict,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    max_grad_norm=0.5,
    epochs=10,
    minibatch_size=64,
    target_kl=None,
    value_clip=True,
):
    obs = data["obs"]
    act = data["act"]
    old_logp = data["logp"]
    adv = data["adv"]
    ret = data["ret"]
    old_val = data["val"]

    n = obs.shape[0]
    idxs = np.arange(n)

    loss_pi_acc, loss_v_acc = 0.0, 0.0
    ent_acc, kl_acc, clipfrac_acc = 0.0, 0.0, 0.0
    n_updates = 0
    stop = False

    for _ in range(epochs):
        np.random.shuffle(idxs)

        for start in range(0, n, minibatch_size):
            mb_idx = idxs[start : start + minibatch_size]

            mu, std, v = model.mu_std_value(obs[mb_idx])

            logp = logp_from_action(act[mb_idx], mu, std)
            entropy = gaussian_entropy(std).mean()

            ratio = torch.exp(logp - old_logp[mb_idx])

            unclipped = ratio * adv[mb_idx]
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv[mb_idx]
            loss_pi = -torch.mean(torch.min(unclipped, clipped))

            if value_clip:
                v_clipped = old_val[mb_idx] + torch.clamp(
                    v - old_val[mb_idx], -clip_eps, clip_eps
                )
                v_loss1 = (v - ret[mb_idx]).pow(2)
                v_loss2 = (v_clipped - ret[mb_idx]).pow(2)
                loss_v = 0.5 * torch.mean(torch.max(v_loss1, v_loss2))
            else:
                loss_v = 0.5 * F.mse_loss(v, ret[mb_idx])

            loss = loss_pi + vf_coef * loss_v - ent_coef * entropy

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()

            with torch.no_grad():
                approx_kl = torch.mean(old_logp[mb_idx] - logp).item()
                clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()

            loss_pi_acc += loss_pi.item()
            loss_v_acc += loss_v.item()
            ent_acc += entropy.item()
            kl_acc += approx_kl
            clipfrac_acc += clipfrac
            n_updates += 1

            if target_kl is not None and approx_kl > 1.5 * target_kl:
                stop = True
                break

        if stop:
            break

    denom = max(1, n_updates)
    stats = dict(
        loss_pi=loss_pi_acc / denom,
        loss_v=loss_v_acc / denom,
        entropy=ent_acc / denom,
        approx_kl=kl_acc / denom,
        clipfrac=clipfrac_acc / denom,
        early_stop_kl=float(stop),
    )

    with torch.no_grad():
        mu_all, std_all, v_all = model.mu_std_value(obs)
        stats["explained_var"] = float(explained_variance(v_all, ret))
        stats["log_std_mean"] = float(model.log_std.mean().item())

    return stats
