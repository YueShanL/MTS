import math
from collections import deque
import datetime
import json
import logging
import os
import random
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Dict, List
import guitarpro as gp

import numpy as np
import torch
from gradio.monitoring_dashboard import total_requests
from matplotlib import pyplot as plt
from sympy import total_degree
from torch.utils.data import DataLoader

from model.dataset import decode
from model.mts_generate import MTSGen
from model.rl.simulator import GuitarSequenceAnalyzer
from model.rl.mid_comparitor import MidiVersionComparator, midi_to_pretty_midi

from torch import optim, Tensor

from utils.gp2mid import gp5_to_midi


@dataclass
class RLConfig:
    """强化学习训练配置"""
    # 训练参数
    num_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4

    # 生成与温度参数
    generate_length: int = 64
    initial_temp: float = 0.6
    min_temp: float = 0.1
    temp_decay: float = 1 - 1e-4
    exploration_temp_factor: float = 3.0  # 探索温度倍数

    # 经验回放与探索
    replay_buffer_size: int = 10000
    reward_threshold: float = 0.7
    exploration_interval: int = 10  # 每N步进行一次高温探索
    exploration_reward_threshold: float = 0.6  # 探索经验的最低奖励阈值

    # 奖励权重与函数
    reward_weights: Dict[str, float] = field(default_factory=lambda: {"difficulty": 0, "similarity": 1.0})
    complexity_weight: float = 0.1  # 复杂性奖励权重

    # 训练频率
    collect_freq: int = 2
    eval_freq: int = 5
    checkpoint_freq: int = 10
    log_interval: int = 10
    update_frequency: int = 1

    # 梯度裁剪与数据加载
    grad_clip: float = 1.0
    num_workers: int = 4

    # 路径配置
    save_dir: str = "./rl_training_results"

    dataset_limit: int = None

    def difficulty_fn(self, param):
        return (1.0 - torch.sigmoid(torch.tensor(param - 5.0))).float()

    def similarity_fn(self, param):
        return param - 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'RLConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class TestRLConfig(RLConfig):
    """测试配置"""
    def __init__(self):
        super().__init__(**self.to_dict())
        self.num_epochs: int = 100
        self.batch_size: int = 1
        self.num_workers: int = 1

        self.replay_buffer_size: int = 50
        self.exploration_interval: int = 5

        self.dataset_limit = 10


class RLTrainer:
    """强化学习训练器（集成FIFO经验池与高温探索）"""

    def __init__(self, model: MTSGen, difficulty_system: GuitarSequenceAnalyzer,
                 similarity_system: MidiVersionComparator, config: RLConfig):
        self.logger = Logger("RLTrainer")
        self.loss_history = []
        self.device = model.device
        self.model = model
        self.difficulty_system = difficulty_system
        self.similarity_system = similarity_system
        self.config = config

        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        # FIFO经验回放缓冲区
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)

        # 训练状态
        self.temperature = config.initial_temp
        self.exploration_temp = config.initial_temp * config.exploration_temp_factor
        self.step_count = 0
        self.reward_history = []

        # 定义输出头权重（可以根据任务重要性调整）
        self.output_weights = {
            'fret': 1.0,
            'technique': 0.5,
            'duration': 1,
        }

        # 日志
        self.setup_logging()
        self.logger.info(f"初始化RL训练器，设备: {self.device}")
        self.logger.info(f"输出头权重: {self.output_weights}")

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """强化学习训练步骤 - 多头输出版本"""

        # 1. 从经验池采样
        buffer_list = list(self.replay_buffer)
        batch_experiences = random.sample(buffer_list, min(batch_size, len(buffer_list)))

        # 准备输入数据和logits
        audio_inputs = []
        sequences = []
        rewards_list = []
        behavior_logits_list = []  # 存储行为策略logits
        temperatures_list = []  # 存储采样温度

        for exp in batch_experiences:
            audio_inputs.append(exp["audio_features"])
            sequences.append(exp["sequence"])
            rewards_list.append(exp["rewards"]["total"])
            behavior_logits_list.append(exp.get("behavior_logits", {}))
            temperatures_list.append(exp.get("temperature", self.temperature))

        # 转换为张量
        audio_tensor = torch.stack(audio_inputs).to(self.device)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(self.device)
        temperatures_tensor = torch.tensor(temperatures_list, dtype=torch.float16).to(self.device)

        # 2. 准备输入（使用经验中的序列作为目标）
        old_sequence = self._stack_dict_list(sequences)
        behavior_output = self._stack_dict_list(behavior_logits_list)

        #new_output, new_logits = (old_sequence, behavior_output)

        self.model.train()
        self.optimizer.zero_grad()

        # 5. 计算行为策略的对数概率
        behavior_log_probs = self._compute_log_probs_dict(behavior_output, old_sequence, temperatures_tensor)

        # 6. 计算目标策略（当前模型）的对数概率
        '''new_output, new_logits = self.model(
            audio_tensor,
            generate_length=self.config.generate_length,
            do_sample=True,
            return_logits=True,
            temperature=self.temperature,
            #teacher_forcing=True,
            #target_notes=sampled_actions,  # 使用采样得到的动作
        )
        target_log_probs = self._compute_log_probs_dict(new_logits, new_output, Tensor([self.temperature]))

        # 7. 计算重要性采样比率（用于Off-policy）
        log_ratios = {}
        for key in behavior_log_probs.keys():
            if key in target_log_probs:
                log_ratios[key] = target_log_probs[key] - behavior_log_probs[key].detach()

        # 8. 计算优势函数（标准化奖励）
        advantages = self._compute_advantages(rewards_tensor)'''

        total_log_prob = torch.zeros(audio_tensor.shape[0], device=self.device)

        for key, probs in behavior_log_probs.items():
            total_log_prob += probs

        # 损失 = -平均(奖励 * 对数概率)
        # 梯度下降时，这会最大化奖励高的动作的概率
        total_loss = -(rewards_tensor * total_log_prob).mean()

        # 7. 反向传播
        total_loss.backward()

        # 非常保守的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

        # 8. 更新
        self.optimizer.step()

        '''# 9. 记录
        avg_reward = rewards_tensor.mean().item()
        avg_log_prob = total_log_prob.mean().item()

        # 9. 计算PPO风格的损失
        total_loss = torch.tensor(0.0, device=self.device)
        policy_losses = {}
        value_losses = {}

        for key in log_ratios.keys():
            if key in self.output_weights:
                weight = self.output_weights[key]
                log_ratio = log_ratios[key]
                ratio = torch.exp(log_ratio)

                # PPO Clip损失
                clip_epsilon = 0.2
                surrogate1 = ratio * advantages.expand_as(ratio)
                surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages.expand_as(
                    ratio)

                # 加权策略损失
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                policy_losses[key] = policy_loss.item()
                total_loss += weight * policy_loss

        # 10. 添加熵正则化
        entropy_loss = self._compute_entropy_dict(new_logits)
        entropy_coef = 0.01
        total_loss -= entropy_coef * entropy_loss

        # 11. 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()'''

        # 12. 评估新策略并更新经验池
        #self._evaluate_and_update_buffer(audio_tensor, new_logits, batch_experiences)

        # 13. 更新状态
        self.loss_history.append(total_loss.item())
        self.step_count += 1
        self.temperature = max(self.config.min_temp, self.temperature * self.config.temp_decay)

        # 14. 返回统计
        stats = {
            "loss": total_loss.item(),
            "avg_reward": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "buffer_size": len(self.replay_buffer),
            "temperature": self.temperature,
            #"entropy": entropy_loss.item(),
        }

        # 添加各头的损失
        #for key, loss_val in policy_losses.items():
            #stats[f"{key}_policy_loss"] = loss_val
        #for key, loss_val in value_losses.items():
            #stats[f"{key}_value_loss"] = loss_val

        return stats

    def _stack_dict_list(self, sequences: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """准备Teacher Forcing输入（多头版本）"""
        # 收集所有可能的关键字
        all_keys = set()
        for seq in sequences:
            if isinstance(seq, dict):
                all_keys.update(seq.keys())

        # 为每个关键字创建张量列表
        tensor_dict = {}
        for key in all_keys:
            tensor_dict[key] = []

        # 填充数据
        for seq in sequences:
            if isinstance(seq, dict):
                for key in all_keys:
                    tensor_dict[key].append(seq[key].to(self.device))

        # 堆叠并移动到设备
        result = {}
        for key, tensor_list in tensor_dict.items():
            if tensor_list:
                result[key] = torch.stack(tensor_list).to(self.device)

        return result

    def _compute_log_probs_dict(self, logits_dict: Dict[str, torch.Tensor],
                                actions: Dict[str, torch.Tensor], temperature = None) -> Dict[str, torch.Tensor]:
        """计算字典形式logits的对数概率 - 确保梯度"""
        log_probs = {}
        temp = Tensor([1.0]) if temperature is None else temperature
        temp = temp.to(self.device)

        for key, logits in logits_dict.items():
            action_key = key

            action = actions[action_key]

            # 确保logits需要梯度
            if isinstance(logits, torch.Tensor) and not logits.requires_grad:
                logits = logits.requires_grad_(True)

            # 计算对数概率
            log_probs_tensor = torch.log_softmax(logits / temp.view([temp.shape[0]] + [1] * (logits.dim() - 1)), dim=-1)

            # 收集动作对应的对数概率
            if logits.dim() == 4 and action.dim() == 3:
                # logits: [B, T, 6, N], action: [B, T, 6]
                action_expanded = action.unsqueeze(-1).long()  # [B, T, 6, 1]
                log_prob = torch.gather(log_probs_tensor, -1, action_expanded).squeeze(-1)  # [B, T, 6]

                # 在弦和序列维度取平均
                log_probs[action_key] = log_prob.mean(dim=[1, 2])  # [B]

            elif logits.dim() == 3 and action.dim() == 2:
                # logits: [B, T, N], action: [B, T]
                action_expanded = action.unsqueeze(-1).long()  # [B, T, 1]
                log_prob = torch.gather(log_probs_tensor, -1, action_expanded).squeeze(-1)  # [B, T]

                # 在序列维度取平均
                log_probs[action_key] = log_prob.mean(dim=-1)  # [B]

        return log_probs

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算优势函数（标准化奖励）"""
        if rewards.std() > 0:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards - rewards.mean()
        return advantages

    def _compute_entropy_dict(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算字典形式logits的熵（多头版本）"""
        total_entropy = torch.tensor(0.0, device=self.device)
        count = 0

        for key, logits in logits_dict.items():
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            total_entropy += entropy
            count += 1

        if count > 0:
            return total_entropy / count
        else:
            return total_entropy

    def _evaluate_and_update_buffer(self, audio_tensor: torch.Tensor,
                                    logits: Dict[str, torch.Tensor],
                                    old_experiences: List[Dict[str, Any]]):
        """评估新策略并更新经验池"""
        with torch.no_grad():
            self.model.eval()

            batch_size = audio_tensor.shape[0]
            sequence = self.model._sample_next_note(logits, self.temperature, do_sample=True)
            for i in range(batch_size):
                # 构建序列字典
                seq_dict = {}
                for key, tensor in sequence.items():
                    seq_dict[key] = tensor[i].cpu()
                logits_dict = {}
                for key, tensor in logits.items():
                    logits_dict[key] = tensor[i].cpu()

                # 计算新奖励
                target_midi = old_experiences[i].get("target_midi")
                new_rewards = self.calculate_reward(seq_dict, target_midi)
                new_reward = new_rewards["total"]
                old_reward = old_experiences[i]["rewards"]["total"]

                # 保留更好的经验
                improvement_threshold = 0.05  # 至少提升5%
                if new_reward > old_reward * (1 + improvement_threshold):
                    self.logger.debug(f'update data with rewards: {new_rewards}')

                    experience = {
                        "sequence": seq_dict,
                        "rewards": new_rewards,
                        "audio_features": old_experiences[i]["audio_features"].clone(),
                        "target_midi": target_midi,
                        "step": self.step_count,
                        "source": "policy_update",
                        "temperature": self.temperature,
                        "behavior_logits": logits_dict,  # 存储新策略的logits
                        "improvement": new_reward - old_reward
                    }

                    # 添加到FIFO缓冲区
                    self.replay_buffer.append(experience)

            self.model.train()

    def collect_experience(self, batch: Dict[str, Any]):
        """
        收集经验 - 多头输出版本
        """
        self.model.eval()
        self.logger.debug(f'collecting experience')

        with torch.no_grad():
            audio_input = batch['audio_input'].to(self.device)
            batch_size = audio_input.shape[0]

            # 决定使用探索温度还是标准温度
            use_exploration_temp = (self.step_count % self.config.exploration_interval == 0)
            current_temp = self.exploration_temp if use_exploration_temp else self.temperature

            if use_exploration_temp: self.logger.debug(f'current exploration temp: {current_temp}')

            # 生成序列
            generated_output, logits_dict = self.model(
                audio_input,
                teacher_forcing=False,
                generate_length=self.config.generate_length,
                temperature=current_temp,
                do_sample=True,
                return_logits=True
            )

            # 处理多头输出
            sequences = self._process_multihead_output(generated_output, batch_size)

            # 处理logits字典
            behavior_logits_dict = {}
            for key, logits in logits_dict.items():
                behavior_logits_dict[key] = logits.cpu()  # 存储到CPU


            # 评估并存储经验
            for i in range(batch_size):
                seq = sequences[i] if i < len(sequences) else sequences[0]
                target_midi = batch['mid_input'][i] if i < len(batch['mid_input']) else None

                # 计算奖励
                rewards = self.calculate_reward(seq, target_midi)

                # 准备序列数据
                if not isinstance(seq, dict):
                    seq_dict = {
                        'fret': seq,
                        'technique': torch.zeros_like(seq),
                        'duration': torch.zeros(seq.shape[0], dtype=torch.float32)
                    }
                else:
                    seq_dict = seq

                # 提取该样本的logits
                sample_logits = {}
                for key, logits_tensor in behavior_logits_dict.items():
                    if logits_tensor.dim() > 0 and logits_tensor.shape[0] == batch_size:
                        sample_logits[key] = logits_tensor[i].clone()
                    else:
                        # 如果没有batch维度，直接使用
                        sample_logits[key] = logits_tensor.clone()

                # 创建经验（包含logits）
                experience = {
                    "sequence": seq_dict,
                    "rewards": rewards,
                    "audio_features": batch['audio_input'][i].cpu().clone(),
                    "target_midi": target_midi,
                    "step": self.step_count,
                    "temperature": current_temp,
                    "behavior_logits": sample_logits,  # 新增：存储logits
                    "source": "exploration" if use_exploration_temp else "standard"
                }

                # 添加到FIFO缓冲区
                self.replay_buffer.append(experience)

                # 记录高质量经验
                if rewards["total"] > self.config.reward_threshold:
                    self.logger.debug(f"高质量经验: 奖励={rewards['total']:.3f}, 来源={experience['source']}")


        # 更新温度（退火）
        self.temperature = max(
            self.config.min_temp,
            self.temperature * self.config.temp_decay
        )
        self.exploration_temp = max(
            self.config.min_temp * 1.5,
            self.exploration_temp * self.config.temp_decay
        )

        self.step_count += 1

    def _process_multihead_output(self, output, batch_size: int) -> List[Any]:
        """处理多头输出，返回序列列表"""
        if isinstance(output, dict):
            # 如果是字典，提取每个样本
            sequences = []
            for i in range(batch_size):
                sample_dict = {}
                for key, value in output.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 0:
                        if value.shape[0] == batch_size:
                            sample_value = value[i]
                        else:
                            sample_value = value
                    else:
                        sample_value = value

                    # 去掉_logits后缀（如果是logits）
                    if key.endswith('_logits'):
                        key = key.replace('_logits', '')

                    sample_dict[key] = sample_value
                sequences.append(sample_dict)
            return sequences
        else:
            # 如果是张量或其他格式
            return [output[i] for i in range(batch_size)]

    def setup_logging(self):
        """设置日志"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        log_file = os.path.join(
            self.config.save_dir,
            f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_reward(self, generated_sequence, target_midi) -> Dict[str, float]:
        rewards = {}

        # 1. 难度奖励
        difficulty_reward = self.difficulty_system.evaluate(generated_sequence)
        rewards["difficulty"] = float(difficulty_reward)

        # 2. 相似度奖励
        if target_midi is not None:
            generated_midi = midi_to_pretty_midi(gp5_to_midi(decode(generated_sequence, post_process=False), None))
            # 检查是否有音符
            source_has_notes = any(len(instr.notes) > 0 for instr in target_midi.instruments)
            target_has_notes = any(len(instr.notes) > 0 for instr in generated_midi.instruments)

            if source_has_notes and target_has_notes:
                similarity_reward = self.similarity_system.evaluate(generated_midi, target_midi)
            elif not source_has_notes and not target_has_notes:
                similarity_reward = 1.0
            else:
                similarity_reward = 0.0
                if not source_has_notes: rewards["difficulty"] = math.inf
                self.logger.warning(f'invalid midi format: expect notes > 0 but get 0 at {"source" if not source_has_notes else "target"}')
            rewards["similarity"] = float(similarity_reward)
        else:
            rewards["similarity"] = 0.0

        # 综合奖励（应用变换函数和权重）
        total_reward = Tensor([
            self.config.difficulty_fn(rewards["difficulty"]) * self.config.reward_weights.get("difficulty", 1.0),
            self.config.similarity_fn(rewards["similarity"]) * self.config.reward_weights.get("similarity", 1.0),
        ])

        rewards["total"] = float(total_reward.sum())


        # 记录奖励历史
        self.reward_history.append({
            "step": self.step_count,
            "rewards": rewards,
            "source": "exploration" if self.step_count % self.config.exploration_interval == 0 else "standard"
        })

        return rewards

    def evaluate(self, dataloader: DataLoader, num_batches: int = 5) -> Dict[str, Any]:
        """评估当前模型"""
        self.model.eval()

        all_rewards = []
        difficulty_scores = []
        similarity_scores = []
        complexity_scores = []
        best_reward = -float('inf')
        best_sequence = None

        with torch.no_grad():
            batch_count = 0
            for batch in dataloader:
                if batch_count >= num_batches:
                    break

                # 生成序列（评估使用较低温度）
                if hasattr(self.model, 'generate_from_audio'):
                    generated_sequences = self.model.generate_from_audio(
                        batch['audio_input'].to(self.device),
                        generate_length=self.config.generate_length,
                        temperature=0.5
                    )
                else:
                    generated_output = self.model(
                        batch['audio_input'].to(self.device),
                        teacher_forcing=False,
                        generate_length=self.config.generate_length,
                        temperature=0.5,
                        do_sample=True
                    )

                    if isinstance(generated_output, dict):
                        batch_size = batch['audio_input'].shape[0]
                        generated_sequences = self._process_multihead_output(generated_output, batch_size)
                    else:
                        generated_sequences = generated_output

                # 计算奖励
                for i, (seq, target_midi) in enumerate(zip(generated_sequences, batch['mid_input'])):
                    rewards = self.calculate_reward(seq, target_midi)

                    all_rewards.append(rewards["total"])
                    difficulty_scores.append(rewards["difficulty"])
                    similarity_scores.append(rewards["similarity"])

                    # 更新最佳序列
                    if rewards["total"] > best_reward:
                        best_reward = rewards["total"]
                        best_sequence = seq

                batch_count += 1

        # 计算统计
        if all_rewards:
            eval_stats = {
                "avg_total_reward": np.mean(all_rewards),
                "std_total_reward": np.std(all_rewards),
                "avg_difficulty": np.mean(difficulty_scores),
                "avg_similarity": np.mean(similarity_scores),
                "avg_complexity": np.mean(complexity_scores),
                "best_sequence": best_sequence,
                "best_reward": best_reward,
                "num_samples": len(all_rewards)
            }
        else:
            eval_stats = {
                "avg_total_reward": 0.0,
                "std_total_reward": 0.0,
                "avg_difficulty": 0.0,
                "avg_similarity": 0.0,
                "avg_complexity": 0.0,
                "best_sequence": None,
                "best_reward": 0.0,
                "num_samples": 0
            }

        return eval_stats

    def _compute_multi_log_probs(self, output, fret_target, technique_target, duration_target):
        """计算多任务对数概率"""
        batch_size = fret_target.shape[0]
        log_probs = torch.zeros(batch_size, device=self.device)

        # fret部分
        if 'fret_logits' in output:
            fret_logits = output['fret_logits']
            fret_probs = torch.log_softmax(fret_logits, dim=-1)
            fret_idx = fret_target.unsqueeze(-1)
            fret_log_probs = torch.gather(fret_probs, -1, fret_idx).squeeze(-1)
            log_probs += fret_log_probs.mean(dim=[1, 2])

        # technique部分
        if 'technique_logits' in output:
            technique_logits = output['technique_logits']
            technique_probs = torch.log_softmax(technique_logits, dim=-1)
            technique_idx = technique_target.unsqueeze(-1)
            technique_log_probs = torch.gather(technique_probs, -1, technique_idx).squeeze(-1)
            log_probs += technique_log_probs.mean(dim=[1, 2])

        return log_probs

    def save_checkpoint(self, epoch: int, stats: Dict[str, Any]):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.config.save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "stats": stats,
            "temperature": self.temperature,
            "exploration_temp": self.exploration_temp,
            "step_count": self.step_count,
            "reward_history": self.reward_history[-1000:],
            "loss_history": self.loss_history[-1000:],
            "buffer_size": len(self.replay_buffer)
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.temperature = checkpoint.get("temperature", self.config.initial_temp)
        self.exploration_temp = checkpoint.get("exploration_temp",
                                               self.config.initial_temp * self.config.exploration_temp_factor)
        self.step_count = checkpoint.get("step_count", 0)

        if "reward_history" in checkpoint:
            self.reward_history = checkpoint["reward_history"]
        if "loss_history" in checkpoint:
            self.loss_history = checkpoint["loss_history"]

        self.logger.info(f"从检查点加载: {checkpoint_path}")
        self.logger.info(f"恢复的训练步数: {self.step_count}")

    def train(self, train_dataset, val_dataset=None):
        """主训练循环"""
        self.logger.info("开始强化学习训练")
        self.logger.info(f"使用设备: {self.device}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn
        )

        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn
            )

        # 保存配置
        config_path = os.path.join(self.config.save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        rewards = []

        # 训练循环
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            epoch_rewards = []

            self.logger.info(f"开始第 {epoch + 1}/{self.config.num_epochs} 轮训练")

            for batch_idx, batch in enumerate(train_loader):
                if self.config.dataset_limit is not None and batch_idx * self.config.batch_size >= self.config.dataset_limit:
                    break
                # 收集经验（集成了高温探索）
                if batch_idx % self.config.collect_freq == 0:
                    self.logger.debug(f'collecting {batch_idx // self.config.collect_freq} data')
                    self.collect_experience(batch)

                # 训练步骤
                if len(self.replay_buffer) >= self.config.batch_size:
                    train_stats = self.train_step(batch_size=self.config.batch_size)
                    epoch_losses.append(train_stats["loss"])
                    epoch_rewards.append(train_stats["avg_reward"])

                    # 定期记录
                    if batch_idx % self.config.log_interval == 0:
                        self.logger.info(
                            f"Epoch {epoch + 1}, Batch {batch_idx}: "
                            f"Loss: {train_stats['loss']:.4f}, "
                            f"Avg Reward: {train_stats['avg_reward']:.4f}, "
                            f"Reward std: {train_stats['reward_std']:.4f}, "
                            f"Buffer: {train_stats['buffer_size']}, "
                            f"Temp: {self.temperature:.3f}"
                        )
            rewards.extend(epoch_rewards)

            # 计算轮次统计
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0

            # 评估
            if (epoch + 1) % self.config.eval_freq == 0:
                eval_stats = self.evaluate(val_loader if val_dataset else train_loader, num_batches=3)

                self.logger.info(
                    f"Epoch {epoch + 1} 评估结果: "
                    f"Train Loss: {avg_epoch_loss:.4f}, "
                    f"Train Reward: {avg_epoch_reward:.4f}, "
                    f"Eval Reward: {eval_stats['avg_total_reward']:.4f}, "
                    f"Difficulty: {eval_stats['avg_difficulty']:.3f}, "
                    f"Similarity: {eval_stats['avg_similarity']:.3f}, "
                )

                # 保存最佳序列
                if eval_stats['best_sequence'] is not None:
                    seq_path = os.path.join(self.config.save_dir, f"best_sequence_epoch_{epoch + 1:04d}.json")
                    with open(seq_path, 'w') as f:
                        json.dump({
                            "reward": eval_stats['best_reward'],
                            "epoch": epoch + 1
                        }, f, indent=2)

            # 保存检查点
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self.save_checkpoint(epoch + 1, {
                    "avg_loss": avg_epoch_loss,
                    "avg_reward": avg_epoch_reward,
                    "eval_stats": eval_stats if (epoch + 1) % self.config.eval_freq == 0 else None
                })
                if eval_stats['best_sequence'] is not None:
                    checkpoint_dir = os.path.join(self.config.save_dir, f"checkpoints_epoch{epoch + 1}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    gp.write(decode(eval_stats['best_sequence']), f'{checkpoint_dir}/out.gp5')
                    _generate_loss_plot(rewards, checkpoint_dir)


        self.logger.info("训练完成")

        # 保存最终模型
        final_path = os.path.join(self.config.save_dir, "final_model.pt")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"最终模型已保存: {final_path}")

    def _move_to_device(self, batch):
        """移动批次数据到设备"""
        device = next(self.model.parameters()).device

        if 'audio_input' in batch:
            batch['audio_input'] = batch['audio_input'].to(device)

        for key in ['context_notes', 'target_notes']:
            if key in batch:
                for subkey in batch[key]:
                    batch[key][subkey] = batch[key][subkey].to(device)

        return batch


def collate_fn(batch):
    """处理数据加载器的批次"""
    if not batch:
        return {}

    audio_list = []
    mid_list = []

    for sample in batch:
        if isinstance(sample, tuple) and len(sample) == 2:
            audio_tensor, midi = sample
        else:
            continue

        # 确保音频张量是3D的: [batch,1,time]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        audio_list.append(audio_tensor)
        mid_list.append(midi)

    if not audio_list:
        return {}

    # 堆叠音频张量
    try:
        audio_batch = torch.stack(audio_list, dim=0)
    except:
        max_len = max(a.shape[-1] for a in audio_list)
        padded_audio = []
        for a in audio_list:
            pad_size = max_len - a.shape[-1]
            if pad_size > 0:
                a = torch.nn.functional.pad(a, (0, pad_size))
            padded_audio.append(a)
        audio_batch = torch.stack(padded_audio, dim=0)

    return {
        'audio_input': audio_batch,
        'mid_input': mid_list,
    }

def _generate_loss_plot(all_losses, output_path):
    """生成loss图表"""
    plt.figure(figsize=(10, 6))

    # 绘制loss曲线
    plt.plot(all_losses, 'b-', linewidth=1.5, alpha=0.8)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Training Reward', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 保存图表
    plot_path = os.path.join(output_path, 'reward_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss plot saved to: {plot_path}")