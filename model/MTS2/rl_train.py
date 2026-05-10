import logging
import math
import os
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, vmap
from transformers import TrainingArguments

from model.MTS2.RLTrainer import GRPOTrainer, compute_log_probs_batch, compute_log_probs_with_velocity
from model.MTS2.data import decode_token, DataCollatorForMTSGen2ReinforceLearning
from model.MTS2.module import MTSGen2
from model.MTS2.profiler import TimeProfiler
from model.dataset import decode
from model.rl.mid_comparitor import MidiVersionComparator, midi_to_pretty_midi
from model.rl.simulator import GuitarSequenceAnalyzer, PresetConfigs
from model.rl.trainer import RLConfig, RLTrainer
from utils.gp2mid import gp5_to_midi

logger = logging.getLogger(__name__)

# =========================
# reward function（你可以替换）
# =========================
class MTSReward:
    def __init__(self,
                 difficulty_system = GuitarSequenceAnalyzer(PresetConfigs.get_default()),
                 similarity_system = MidiVersionComparator(),
                 config = RLConfig(),
                 num_worker = 16,
                 similarity_weight = 1.0,
                 difficulty_weight = 1.0,
                 ):
        self.similarity_weight = similarity_weight
        self.difficulty_weight = difficulty_weight
        self.difficulty_system = difficulty_system
        self.similarity_system = similarity_system
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.profiler = TimeProfiler()
        self.debug = False
        self.executor = ThreadPoolExecutor(max_workers=num_worker)

    def calculate_rewards_parallel(self, generated_sequence, target_midi, **kwargs):
        """
        responses: [G, B, L, H]
        waveform: [B, ...]
        mids: list of length B

        return:
            rewards: [G, B]
        """

        G, B = generated_sequence.shape[:2]

        tasks = []
        indices = []

        # -------- flatten tasks --------
        for g in range(G):
            for b in range(B):
                tasks.append((
                    generated_sequence[g, b].cpu(),
                    target_midi[b]
                ))
                indices.append((g, b))

        # -------- worker --------
        def worker(args):
            resp, mid = args
            return float(self._calculate_reward(generated_sequence = resp, target_midi=mid))

        # -------- parallel --------
        results = []
        results = list(self.executor.map(worker, tasks))

        # -------- reshape --------
        rewards = torch.zeros(G, B)

        for (g, b), r in zip(indices, results):
            rewards[g, b] = r

        return rewards

    def calculate_reward(self, generated_sequence, G, target_midi, **kwargs):
        GB = generated_sequence.shape[0]

        B = int(GB/G)

        rewards_list = []
        # -------- flatten tasks --------
        for g in range(G):
            row = []
            for b in range(B):
                r = self._calculate_reward(
                    generated_sequence=generated_sequence[g*B+b],
                    target_midi=target_midi[b]
                )
                row.append(float(r))
            rewards_list.append(row)

        return Tensor(rewards_list)
    def calculate_reward_with_velocity(self, generated_sequence, G, target_midi, **kwargs):
        token_ids = generated_sequence["token_ids"]
        velocity = generated_sequence["velocity_tokens"]
        GB = token_ids.shape[0]

        B = int(GB/G)

        rewards_list = []
        # -------- flatten tasks --------
        for g in range(G):
            row = []
            for b in range(B):
                r = self._calculate_reward(
                    generated_sequence=token_ids[g*B + b],
                    target_midi=target_midi[b],
                    velocity_tokens=velocity[g*B + b]
                )
                row.append(float(r))
            rewards_list.append(row)

        return Tensor(rewards_list)

    def _calculate_reward(self, generated_sequence, target_midi, **kwargs):
        fret, technique, duration = vmap(decode_token)(generated_sequence.to('cpu'))
        duration = duration.to(float).mean(dim=-1, keepdim=False).to(int)

        song = {'fret': fret.squeeze(), 'technique': technique.squeeze(), 'duration': duration.squeeze()}

        rewards = {}
        if self.debug:
            self.profiler.start('difficulty')
        # 1. 难度奖励
        difficulty_reward = self.difficulty_system.evaluate(song) * self.difficulty_weight
        rewards["difficulty"] = float(difficulty_reward)
        if self.debug:
            self.profiler.stop('difficulty')
        # 2. 相似度奖励
        if self.debug:
            self.profiler.start('similarity')
        if target_midi is not None:
            if self.debug:
                self.profiler.start('transcript')
            if kwargs.get("velocity_tokens") is not None:
                song = {'fret': fret.squeeze(), 'technique': technique.squeeze(), 'duration': duration.squeeze(),
                               'velocity': kwargs.get("velocity_tokens").squeeze().cpu()}
            generated_midi = midi_to_pretty_midi(gp5_to_midi(decode(song, post_process=False), None))
            # 检查是否有音符
            source_has_notes = any(len(instr.notes) > 0 for instr in target_midi.instruments)
            target_has_notes = any(len(instr.notes) > 0 for instr in generated_midi.instruments)
            if self.debug:
                self.profiler.stop('transcript')

            if source_has_notes and target_has_notes:
                similarity_reward = self.similarity_system.evaluate(generated_midi, target_midi, False) * self.similarity_weight
            elif not source_has_notes and not target_has_notes:
                similarity_reward = 1.0
            else:
                similarity_reward = 0.0
                if not source_has_notes: rewards["difficulty"] = math.inf
                self.logger.warning(
                    f'invalid midi format: expect notes > 0 but get 0 at {"source" if not source_has_notes else "target"}')
            rewards["similarity"] = float(similarity_reward)
        else:
            rewards["similarity"] = 0.0
        if self.debug:
            self.profiler.stop('similarity')
        # 综合奖励（应用变换函数和权重）
        total_reward = Tensor([
            self.config.difficulty_fn(rewards["difficulty"]) * self.config.reward_weights.get("difficulty", 1.0),
            self.config.similarity_fn(rewards["similarity"]) * self.config.reward_weights.get("similarity", 1.0),
        ])

        if self.debug:
            self.profiler.report(total_reward)

        return float(total_reward.sum())

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# =========================
# GRPO train
# =========================

def train(
    train_dataset,
    eval_dataset,
    output_dir: str,
    model,
    training_args: Optional[TrainingArguments] = None,
    datasize = 0,
    freeze_basic_pitch: bool = True,
    seed: int = 42,
):


    model_config = model.config

    if training_args is None:
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=os.path.join(output_dir, "runs"),
            max_steps=3 * datasize,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            learning_rate=5e-6,
            logging_steps=10,
            save_steps=200,
            eval_steps=20,
            save_total_limit=2,
            report_to="tensorboard",
            seed=seed,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            fp16=True,  # 推荐开启
        )

    model = model

    reward = MTSReward()

    collator = DataCollatorForMTSGen2ReinforceLearning(
        pad_token_id=model_config.decoder_start_token_id,
        label_pad_token_id=-100
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,

        # ===== GRPO参数 =====
        num_generations=8,
        kl_coef=0.05,
        clip_range=0.2,
        pad_token_id=model_config.decoder_start_token_id,

        generation_kwargs={
            "max_length": 64,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.9,
        },

        reward_function=reward.calculate_reward_with_velocity,
        compute_log_probs_fn=compute_log_probs_with_velocity,
    )

    trainer.train()
    trainer.save_model(output_dir)

    logger.info(f"GRPO 模型已保存至 {output_dir}")