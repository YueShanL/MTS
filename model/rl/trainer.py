from collections import deque
import datetime
import json
import logging
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from mts_generate import MTSGen
from simulator import GuitarSequenceAnalyzer
from mid_comparitor import MidiVersionComparator

from torch import optim

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, model: MTSGen, difficulty_system: GuitarSequenceAnalyzer,
                 similarity_system: MidiVersionComparator, config: RLConfig):
        self.model = model
        self.difficulty_system = difficulty_system
        self.similarity_system = similarity_system
        self.config = config
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # 目标序列（用于相似度计算）
        self.target_sequence = None
        
        # 训练状态
        self.temperature = config.initial_temp
        self.step_count = 0
        self.reward_history = []
        
        # 日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        log_file = os.path.join(self.config.save_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def set_target_sequence(self, sequence: List[List[int]]):
        """设置目标序列（用于相似度计算）"""
        self.target_sequence = sequence
        self.logger.info(f"设置目标序列，长度: {len(sequence)}")
    
    def calculate_reward(self, sequence: List[List[int]]) -> Dict[str, float]:
        """计算综合奖励"""
        rewards = {}
        
        # 1. 难度奖励
        difficulty_result = self.difficulty_system.evaluate(sequence)
        difficulty_reward = difficulty_result["total"]
        rewards["difficulty"] = difficulty_reward
        
        # 2. 相似度奖励（如果有目标序列）
        if self.target_sequence is not None:
            similarity_result = self.similarity_system.evaluate(sequence, self.target_sequence)
            similarity_reward = similarity_result["total"]
            rewards["similarity"] = similarity_reward
        else:
            rewards["similarity"] = 0.0
        
        # 3. 多样性奖励（鼓励探索）
        diversity_reward = self.calculate_diversity_reward(sequence)
        rewards["diversity"] = diversity_reward
        
        # 4. 音乐性奖励（基于一些音乐规则）
        musical_reward = self.calculate_musical_reward(sequence)
        rewards["musicality"] = musical_reward
        
        # 综合奖励
        total_reward = (
            rewards["difficulty"] * self.config.reward_weights["difficulty"] +
            rewards["similarity"] * self.config.reward_weights["similarity"] +
            rewards["diversity"] * self.config.reward_weights["diversity"] +
            rewards["musicality"] * 0.1  # 额外的小权重
        )
        rewards["total"] = total_reward
        
        # 记录奖励历史
        self.reward_history.append({
            "step": self.step_count,
            "rewards": rewards,
            "sequence_length": len(sequence)
        })
        
        return rewards
    
    def calculate_diversity_reward(self, sequence: List[List[int]]) -> float:
        """计算多样性奖励"""
        if len(self.replay_buffer) < 2:
            return 0.5
        
        # 计算当前序列与回放缓冲区中序列的相似度
        similarities = []
        for buffer_seq in list(self.replay_buffer)[-10:]:  # 检查最近10个
            sim_result = self.similarity_system.evaluate(sequence, buffer_seq)
            similarities.append(sim_result["total"])
        
        if not similarities:
            return 0.5
        
        avg_similarity = np.mean(similarities)
        
        # 多样性奖励：与已有序列越不相似，奖励越高
        diversity_reward = 1.0 - avg_similarity
        return diversity_reward
    
    def calculate_musical_reward(self, sequence: List[List[int]]) -> float:
        """计算音乐性奖励（基于简单规则）"""
        if not sequence:
            return 0.0
        
        musical_score = 0.0
        
        for i, chord in enumerate(sequence):
            # 检查空弦（0品）使用
            if 0 in chord:
                musical_score += 0.1
            
            # 检查是否有演奏的弦
            active_strings = [f for f in chord if f < 25]
            if 1 <= len(active_strings) <= 3:
                musical_score += 0.2  # 鼓励1-3个音的简洁演奏
            
            # 检查和弦是否合理（相邻品位差不太大）
            if len(active_strings) >= 2:
                active_strings.sort()
                max_diff = max(active_strings) - min(active_strings)
                if max_diff <= 5:  # 合理的手指跨度
                    musical_score += 0.3
        
        # 检查序列变化
        if len(sequence) > 1:
            changes = 0
            for i in range(1, len(sequence)):
                if sequence[i] != sequence[i-1]:
                    changes += 1
            change_ratio = changes / (len(sequence) - 1)
            # 鼓励适度的变化（既不太单调也不太大变化）
            if 0.3 <= change_ratio <= 0.7:
                musical_score += 0.5
        
        return musical_score / (len(sequence) + 1)
    
    def generate_sequence(self, batch_size: int = 1) -> List[List[List[int]]]:
        """生成序列"""
        # 随机起始标记
        start_tokens = torch.randint(0, 26, (batch_size, 1, 6))
        
        # 生成序列
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens, 
                self.config.sequence_length,
                temperature=self.temperature
            )
        
        # 转换为Python列表
        sequences = []
        for i in range(batch_size):
            seq = generated[i].cpu().numpy().tolist()  # (seq_len, 6)
            sequences.append(seq)
        
        return sequences
    
    def collect_experience(self, num_sequences: int = 16):
        """收集经验"""
        sequences = self.generate_sequence(num_sequences)
        
        for seq in sequences:
            # 计算奖励
            rewards = self.calculate_reward(seq)
            
            # 转换为张量
            seq_tensor = torch.tensor(seq, dtype=torch.long)  # (seq_len, 6)
            
            # 存储到回放缓冲区
            experience = {
                "sequence": seq_tensor,
                "rewards": rewards,
                "step": self.step_count
            }
            self.replay_buffer.append(experience)
        
        self.step_count += 1
        
        # 更新温度（退火）
        self.temperature = max(
            self.config.min_temp,
            self.temperature * self.config.temp_decay
        )
    
    def compute_policy_gradient_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算策略梯度损失"""
        sequences = batch["sequence"]  # (batch_size, seq_len, 6)
        rewards = batch["reward"]  # (batch_size,)
        
        # 获取模型预测
        logits = self.model(sequences)  # (batch_size, seq_len, 6, 26)
        
        # 计算对数概率
        log_probs = []
        for b in range(sequences.shape[0]):
            seq_log_probs = []
            for t in range(sequences.shape[1]):
                for s in range(6):
                    token = sequences[b, t, s]
                    log_prob = torch.log_softmax(logits[b, t, s, :], dim=0)[token]
                    seq_log_probs.append(log_prob)
            seq_total_log_prob = torch.stack(seq_log_probs).mean()
            log_probs.append(seq_total_log_prob)
        
        log_probs = torch.stack(log_probs)
        
        # 归一化奖励
        if rewards.std() > 0:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            normalized_rewards = rewards - rewards.mean()
        
        # 策略梯度损失
        loss = -(log_probs * normalized_rewards).mean()
        
        return loss
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """单步训练"""
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0, "avg_reward": 0.0}
        
        # 从回放缓冲区采样
        batch_experiences = random.sample(self.replay_buffer, batch_size)
        
        # 准备批次数据
        sequences = []
        rewards = []
        
        for exp in batch_experiences:
            sequences.append(exp["sequence"])
            rewards.append(exp["rewards"]["total"])
        
        sequences_tensor = torch.stack(sequences)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        batch = {
            "sequence": sequences_tensor,
            "reward": rewards_tensor
        }
        
        # 训练
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_policy_gradient_loss(batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step()
        
        # 记录统计
        stats = {
            "loss": loss.item(),
            "avg_reward": rewards_tensor.mean().item(),
            "std_reward": rewards_tensor.std().item(),
            "temperature": self.temperature,
            "buffer_size": len(self.replay_buffer)
        }
        
        return stats
    
    def evaluate(self, num_sequences: int = 10) -> Dict[str, Any]:
        """评估当前模型"""
        self.model.eval()
        
        # 生成测试序列
        test_sequences = self.generate_sequence(num_sequences)
        
        # 计算奖励统计
        all_rewards = []
        difficulty_scores = []
        similarity_scores = []
        
        for seq in test_sequences:
            rewards = self.calculate_reward(seq)
            all_rewards.append(rewards["total"])
            difficulty_scores.append(rewards["difficulty"])
            similarity_scores.append(rewards["similarity"])
        
        # 保存最佳序列
        best_idx = np.argmax(all_rewards)
        best_sequence = test_sequences[best_idx]
        best_rewards = self.calculate_reward(best_sequence)
        
        eval_stats = {
            "avg_total_reward": np.mean(all_rewards),
            "std_total_reward": np.std(all_rewards),
            "avg_difficulty": np.mean(difficulty_scores),
            "avg_similarity": np.mean(similarity_scores),
            "best_sequence": best_sequence,
            "best_rewards": best_rewards
        }
        
        return eval_stats
    
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
            "step_count": self.step_count,
            "reward_history": self.reward_history[-100:]  # 保存最近100条
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
        self.step_count = checkpoint.get("step_count", 0)
        
        # 加载奖励历史
        if "reward_history" in checkpoint:
            self.reward_history = checkpoint["reward_history"]
        
        self.logger.info(f"从检查点加载: {checkpoint_path}")
        self.logger.info(f"恢复的训练步数: {self.step_count}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始强化学习训练")
        self.logger.info(f"配置: {json.dumps(self.config.to_dict(), indent=2)}")
        
        # 保存配置
        config_path = os.path.join(self.config.save_dir, "config.json")
        self.config.save(config_path)
        
        for epoch in range(self.config.num_epochs):
            # 收集经验
            if epoch % self.config.update_frequency == 0:
                self.collect_experience(num_sequences=16)
            
            # 训练步骤
            if self.config.use_replay_buffer and len(self.replay_buffer) >= self.config.batch_size:
                train_stats = self.train_step(batch_size=self.config.batch_size)
            else:
                train_stats = {"loss": 0.0, "avg_reward": 0.0}
            
            # 评估
            if epoch % self.config.eval_freq == 0:
                eval_stats = self.evaluate(num_sequences=5)
                
                # 记录日志
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Loss: {train_stats['loss']:.4f}, "
                    f"Avg Reward: {train_stats['avg_reward']:.4f}, "
                    f"Eval Reward: {eval_stats['avg_total_reward']:.4f}, "
                    f"Temp: {self.temperature:.3f}"
                )
                
                # 保存最佳序列
                if epoch % (self.config.eval_freq * 5) == 0:
                    best_seq = eval_stats["best_sequence"]
                    best_rewards = eval_stats["best_rewards"]
                    
                    seq_path = os.path.join(self.config.save_dir, f"best_sequence_epoch_{epoch}.json")
                    with open(seq_path, 'w') as f:
                        json.dump({
                            "sequence": best_seq,
                            "rewards": best_rewards,
                            "epoch": epoch
                        }, f, indent=2)
            
            # 保存检查点
            if epoch % self.config.checkpoint_freq == 0 and epoch > 0:
                self.save_checkpoint(epoch, train_stats)
        
        self.logger.info("训练完成")
        
        # 保存最终模型
        final_path = os.path.join(self.config.save_dir, "final_model.pt")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"最终模型已保存: {final_path}")