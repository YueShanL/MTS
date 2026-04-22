import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Dict, Any

from model.MTS2.utils import TimeProfiler


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
    input_ids,          # [G, B, L, H]
    encoder_outputs,    # [B, T, D]
    pad_token_id,
):
    """
    一次性计算所有 rollout 的 log prob
    """

    G, B, L, H = input_ids.shape

    # ---------- flatten ----------
    input_ids = input_ids.view(G * B, L, H)

    # repeat encoder
    encoder_outputs = encoder_outputs.repeat_interleave(G, dim=0)

    # ---------- shift ----------
    decoder_input_ids = shift_right(input_ids, pad_token_id)

    outputs = model(
        input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        return_dict=True,
    )

    logits = outputs.logits  # [GB, L, H, V]
    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(
        -1, input_ids.unsqueeze(-1)
    ).squeeze(-1)  # [GB, L, H]

    mask = (input_ids != pad_token_id).float()
    token_log_probs = token_log_probs * mask

    # reshape back
    token_log_probs = token_log_probs.view(G, B, L, H)
    mask = mask.view(G, B, L, H)

    lengths = mask.sum(dim=2).sum(dim=-1)
    # seq log prob
    seq_log_probs = token_log_probs.sum(dim=2).sum(dim=-1)  # [G, B]
    seq_log_probs = seq_log_probs / lengths.clamp_min(1)

    return seq_log_probs, token_log_probs, mask


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

        # ---------- reshape ----------
        L, H = generated.shape[1], generated.shape[2]
        responses = generated.view(G, B, L, H)

        # =========================
        # 3. reward（逐样本！）
        # =========================
        if self.debug:
            self.time_profiler.start('reward')

        rewards_list = self.reward_function(
            generated_sequence=responses,
            waveform=waveform,
            target_midi=mids
            ).to(device=device)

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
            old_log_probs, _, _ = compute_log_probs_batch(
                model,
                responses,
                encoder_outputs,
                self.pad_token_id
            )

        # =========================
        # 5. new log prob（一次算完）
        # =========================
        model.train()

        new_log_probs, token_log_probs, mask = compute_log_probs_batch(
            model,
            responses,
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
                _, ref_token_log_probs, _ = compute_log_probs_batch(
                    self.ref_model,
                    responses,
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