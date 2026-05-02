import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Dict, Any

from model.MTS2.profiler import TimeProfiler


# =========================
# utils（支持 batch flatten）
# =========================

def shift_right(x, pad_token_id):
    y = x.clone()
    y[:, 1:, :] = x[:, :-1, :]
    y[:, 0, :] = pad_token_id
    return y


def compute_log_probs_batch(
    model,
    G: int,
    input_ids,          # [G*B, L, H]
    encoder_outputs,    # [B, T, D]
    pad_token_id,
):
    _, L, H = input_ids.shape
    # repeat encoder
    encoder_outputs = encoder_outputs.repeat_interleave(G, dim=0)  # [G*B, T, D]

    decoder_input_ids = shift_right(input_ids, pad_token_id)

    outputs = model(
        input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        return_dict=True,
    )

    logits = outputs.logits  # [G*B, L, H, V]
    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(
        -1, input_ids.unsqueeze(-1)
    ).squeeze(-1)  # [G*B, L, H]

    mask = (input_ids != pad_token_id).float()
    token_log_probs = token_log_probs * mask

    # ----- 关键修复：reshape 为 [G, B, L, H] -----
    token_log_probs = token_log_probs.view(G, -1, L, H)   # [G, B, L, H]
    mask = mask.view(G, -1, L, H)

    lengths = mask.sum(dim=2).sum(dim=-1).clamp_min(1)   # [G, B]
    seq_log_probs = token_log_probs.sum(dim=2).sum(dim=-1) / lengths  # [G, B]

    return seq_log_probs, token_log_probs, mask

def compute_log_probs_with_velocity(
    model,
    G: int,
    responses,          # dict or Tensor [G*B, L, H]
    encoder_outputs,    # [B, T, D]
    pad_token_id: int,
    velocity_pad_token_id: int = 0,
    **kwargs
):
    if isinstance(responses, dict):
        token_ids = responses["token_ids"]
        velocity_ids = responses.get("velocity_tokens")
    else:
        token_ids = responses
        velocity_ids = None

    GB, L, H = token_ids.shape
    enc_rep = encoder_outputs.repeat_interleave(G, dim=0)

    token_dec = shift_right(token_ids, pad_token_id)

    outputs = model(input_ids=token_dec, encoder_outputs=enc_rep, return_dict=True)
    logits = outputs.logits  # [GB, L, H, V]

    token_lp = F.log_softmax(logits, dim=-1).gather(
        -1, token_ids.unsqueeze(-1)
    ).squeeze(-1)

    mask = (token_ids != pad_token_id).float()

    if velocity_ids is not None:
        vel_dec = shift_right(velocity_ids, velocity_pad_token_id)
        vel_logits = outputs.velocity_logits
        vel_lp = F.log_softmax(vel_logits, dim=-1).gather(
            -1, velocity_ids.unsqueeze(-1)
        ).squeeze(-1)

        total_lp = token_lp + vel_lp
        vel_mask = (velocity_ids != velocity_pad_token_id).float()
        mask = mask * vel_mask
    else:
        total_lp = token_lp

    total_lp = total_lp * mask

    # ----- 关键修复：reshape 为 [G, B, L, H] -----
    total_lp = total_lp.view(G, -1, L, H)   # [G, B, L, H]
    mask = mask.view(G, -1, L, H)

    lengths = mask.sum(dim=2).sum(dim=-1).clamp_min(1)   # [G, B]
    seq_log_probs = total_lp.sum(dim=2).sum(dim=-1) / lengths  # [G, B]

    return seq_log_probs, total_lp, mask
# =========================
# GRPO Trainer（优化版）
# =========================

class GRPOTrainer(Trainer):

    def __init__(
        self,
        *args,
        ref_model=None,
        num_generations=8,
        kl_coef=0.1,
        clip_range=0.2,
        pad_token_id=0,
        generation_kwargs=None,
        reward_function=None,
        compute_log_probs_fn=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.num_generations = num_generations
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.pad_token_id = pad_token_id

        self.generation_kwargs = generation_kwargs or {
            "max_length": 200,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
        }

        self.reward_function = reward_function or self._default_reward

        self.time_profiler = TimeProfiler()
        self.debug = False
        self.compute_log_probs_fn = compute_log_probs_fn or compute_log_probs_batch

    def _default_reward(self, responses, waveform):
        return torch.randn(responses.size(0), device=responses.device)

    def training_step(self, model, inputs, num_items_in_batch=None):

        device = self.accelerator.device

        waveform = inputs["waveform"].to(device)
        mids = inputs["mid"]  # ⚠️ 不要 .to(device)，保持 python list

        model.eval()

        # =========================
        # 1. encoder
        # =========================
        if self.debug:
            self.time_profiler.start('encode')
        with torch.no_grad():
            features = model.extract_features(waveform)
            encoder_outputs = model.get_encoder_outputs(features)
        if self.debug:
            self.time_profiler.stop('encode')

        # =========================
        # 2. rollout
        # =========================
        G = self.num_generations
        B = encoder_outputs.size(0)

        # ---------- expand encoder ----------
        encoder_outputs_expanded = encoder_outputs.repeat_interleave(G, dim=0)

        # ---------- generate ----------
        generated = model.generate(
            encoder_outputs=encoder_outputs_expanded,
            **self.generation_kwargs
        )
        # [G*B, L, H]
        if isinstance(generated, dict):
            # 新模型返回字典：{"token_ids": ..., "velocity_tokens": ...}
            responses_dict = generated
            token_ids = generated["token_ids"]
        else:
            # 旧模型返回纯张量
            responses_dict = None
            token_ids = generated

        # ---------- reshape ----------
        L, H = token_ids.shape[1], token_ids.shape[2]

        # =========================
        # 3. reward
        # =========================
        if self.debug:
            self.time_profiler.start('reward')

        rewards_list = self.reward_function(
            generated_sequence=responses_dict if responses_dict else token_ids,
            G = G,
            waveform=waveform,
            target_midi=mids
        ).to(device)

        rewards = rewards_list
        if self.debug:
            self.time_profiler.stop('reward')

        # =========================
        # 4. advantage（不变）
        # =========================
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True).clamp_min(1e-6)
        advantages = (rewards - mean) / std

        # =========================
        # 4. old log prob（无 deepcopy！）
        # =========================
        if self.debug:
            self.time_profiler.start('log_prob')
        with torch.no_grad():
            old_log_probs, _, _ = self.compute_log_probs_fn(
                model,
                G,
                responses_dict if responses_dict else token_ids,
                encoder_outputs,
                self.pad_token_id
            )

        # =========================
        # 5. new log prob（一次算完）
        # =========================
        model.train()

        new_log_probs, token_log_probs, mask = self.compute_log_probs_fn(
            model,
            G,
            responses_dict if responses_dict else token_ids,
            encoder_outputs,
            self.pad_token_id
        )
        if self.debug:
            self.time_profiler.stop('log_prob')

        # =========================
        # 6. PPO
        # =========================
        ratio = torch.exp(new_log_probs - old_log_probs)

        clipped = torch.clamp(
            ratio,
            1 - self.clip_range,
            1 + self.clip_range
        )

        policy_loss = -torch.min(
            ratio * advantages,
            clipped * advantages
        ).mean()

        # =========================
        # 7. KL（token-level）
        # =========================

        if self.ref_model is not None:
            with torch.no_grad():
                _, ref_token_log_probs, _ = self.compute_log_probs_fn(
                    self.ref_model,
                    G,
                    responses_dict if responses_dict else token_ids,
                    encoder_outputs,
                    self.pad_token_id
                )

            kl = (token_log_probs - ref_token_log_probs) * mask
            kl = kl.sum(dim=[2, 3])  # [G, B]
            kl_loss = kl.mean()
        else:
            kl_loss = torch.tensor(0.0, device=device)

        loss = policy_loss + self.kl_coef * kl_loss

        self.accelerator.backward(loss)
        if self.debug:
            self.time_profiler.report(self.state.global_step)

        return loss.detach()