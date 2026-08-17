#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microbiome-Metabolite Cross-Modal Contrastive Learning Training Script
基于 RoBERTa 的双塔对比学习微调，支持小样本和分层学习率。
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
from sklearn.metrics import accuracy_score
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 数据加载模块
# ==============================================================================

class MicrobiomeMetaboliteDataset(Dataset):
    """微生物 - 代谢组配对数据集"""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 max_length: int = 512,
                 augment_metab: bool = True,
                 metab_noise_std: float = 0.01):
        """
        data_path: .pt 文件路径，包含 [{'text': str, 'metab_features': tensor, 'label': int}, ...]
        tokenizer: RoBERTa tokenizer
        max_length: 最大序列长度
        augment_metab: 是否增强代谢数据
        metab_noise_std: 代谢数据增强的高斯噪声标准差
        """
        logger.info(f"Loading dataset from {data_path}...")
        
        # 使用 map_location 确保加载兼容
        try:
            data = torch.load(data_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
        
        if not isinstance(data, list):
            raise ValueError("数据集必须是列表格式")
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_metab = augment_metab
        self.metab_noise_std = metab_noise_std
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 处理代谢数据特征
        metab_features = item['metab_features']
        if isinstance(metab_features, np.ndarray):
            metab_features = torch.from_numpy(metab_features)
        else:
            metab_features = metab_features.clone()
        
        label = item.get('label', -1)
        if not isinstance(label, int):
            label = int(label) if label != -1 else -1
        
        # 文本编码
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        
        input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(enc['attention_mask'], dtype=torch.long)
        
        # 代谢数据增强
        if self.augment_metab and self.metab_noise_std > 0:
            noise = torch.randn_like(metab_features) * self.metab_noise_std
            metab_features = metab_features + noise
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'metab_features': metab_features.float(),
            'label': label
        }

@dataclass
class MicrobiomeMetaboliteDataCollator:
    """自定义 Collator，处理变长序列和代谢数据"""
    
    pad_token_id: int = 1
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        if not examples:
            return {}
            
        input_ids = [e['input_ids'] for e in examples]
        attention_masks = [e['attention_mask'] for e in examples]
        
        # 确保 metab_features 是 tensor 且维度一致
        try:
            metab_features = torch.stack([e['metab_features'] for e in examples])
        except RuntimeError:
            # 如果长度不一致，尝试转换为 list 或 pad (这里假设长度一致)
            raise ValueError("Metabolite features must have the same length")
        
        labels = torch.tensor([e['label'] for e in examples], dtype=torch.long)
        
        # Pad 文本序列
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'metab_features': metab_features.float(),
            'labels': labels
        }

# ==============================================================================
# 模型模块
# ==============================================================================

class MicrobiomeMetaboliteContrastiveModel(nn.Module):
    """微生物 - 代谢组双塔对比学习模型"""
    
    def __init__(self,
                 roberta_path: str,
                 metab_input_dim: int,
                 hidden_size: int = 768,
                 proj_dim: int = 256,
                 freeze_layers: int = 8,
                 temperature: float = 0.07):
        super().__init__()
        
        logger.info(f"Loading RoBERTa from {roberta_path}...")
        self.roberta = RobertaModel.from_pretrained(roberta_path)
        self.metab_input_dim = metab_input_dim
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.temperature = temperature
        
        # 冻结底层 (保留通用语义)
        logger.info(f"Freezing first {freeze_layers} layers of RoBERTa...")
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 代谢编码器
        self.metabolite_encoder = nn.Sequential(
            nn.Linear(metab_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_size),
            nn.ReLU(),
        )
        
        # 投影头 (两个模态共享维度)
        self.text_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )
        
        self.metab_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )
        
        logger.info(f"Model initialized: proj_dim={proj_dim}, temperature={temperature}")
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """编码文本，返回归一化的投影向量"""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        proj = self.text_projector(cls_repr)
        return nn.functional.normalize(proj, dim=1)
    
    def encode_metabolite(self, metab_features: torch.Tensor) -> torch.Tensor:
        """编码代谢数据，返回归一化的投影向量"""
        repr = self.metabolite_encoder(metab_features)
        proj = self.metab_projector(repr)
        return nn.functional.normalize(proj, dim=1)
    
    def forward(self, input_ids, attention_mask, metab_features, labels=None):
        """前向传播，计算对比损失"""
        z_text = self.encode_text(input_ids, attention_mask)
        z_metab = self.encode_metabolite(metab_features)
        
        # 计算相似度矩阵 [B, B]
        logits = torch.matmul(z_text, z_metab.T) / self.temperature
        
        # InfoNCE Loss (双向)
        batch_size = z_text.size(0)
        labels_pos = torch.arange(batch_size, device=z_text.device)
        
        loss_t2m = nn.CrossEntropyLoss()(logits, labels_pos)
        loss_m2t = nn.CrossEntropyLoss()(logits.T, labels_pos)
        loss = (loss_t2m + loss_m2t) / 2
        
        return {
            'loss': loss,
            'logits': logits,
            'z_text': z_text,
            'z_metab': z_metab
        }

# ==============================================================================
# 评估指标
# ==============================================================================

def compute_retrieval_metrics(logits, labels=None):
    """计算检索指标 (Recall@K)"""
    batch_size = logits.size(0)
    
    # 如果 labels 为 None，使用对角线作为正样本
    if labels is None:
        labels = torch.arange(batch_size, device=logits.device)
    
    # 计算 Recall@K
    recalls = {}
    for k in [1, 5, 10]:
        if k <= batch_size:
            _, indices = torch.topk(logits, k, dim=1)
            hits = (indices == labels.unsqueeze(1)).any(dim=1).float()
            recalls[f'Recall@{k}'] = hits.mean().item()
    
    return recalls

# ==============================================================================
# 自定义 Trainer
# ==============================================================================

class CrossModalContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 解包输入 (确保 pop 掉不需要的)
        input_ids = inputs.pop('input_ids', None)
        attention_mask = inputs.pop('attention_mask', None)
        metab_features = inputs.pop('metab_features', None)
        labels = inputs.pop('labels', None)
        
        if input_ids is None or attention_mask is None or metab_features is None:
            raise ValueError("Missing required inputs: input_ids, attention_mask, or metab_features")
        
        # 前向传播
        outputs = model(input_ids, attention_mask, metab_features, labels)
        loss = outputs['loss']
        
        # 计算评估指标 (用于日志)
        if self.args.logging_steps and self.state.global_step % self.args.logging_steps == 0:
            metrics = compute_retrieval_metrics(outputs['logits'])
            self.log(metrics)
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """自定义评估步骤"""
        input_ids = inputs.pop('input_ids', None)
        attention_mask = inputs.pop('attention_mask', None)
        metab_features = inputs.pop('metab_features', None)
        labels = inputs.pop('labels', None)
        
        if input_ids is None or attention_mask is None or metab_features is None:
            raise ValueError("Missing required inputs in prediction_step")
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, metab_features, labels)
            loss = outputs['loss']
            logits = outputs['logits']
        
        # 计算评估指标
        metrics = compute_retrieval_metrics(logits, labels)
        
        return loss, None, metrics

# ==============================================================================
# 主函数
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Microbiome-Metabolite Cross-Modal Contrastive Learning")
    
    # 数据路径
    parser.add_argument("--train-data", type=Path, required=True, help="训练数据 .pt 文件路径")
    parser.add_argument("--eval-data", type=Path, required=True, help="验证数据 .pt 文件路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--roberta-path", type=str, required=True, help="预训练 RoBERTa 模型路径")
    
    # 模型参数
    parser.add_argument("--metab-input-dim", type=int, required=True, help="代谢数据特征维度")
    parser.add_argument("--proj-dim", type=int, default=256, help="投影维度")
    parser.add_argument("--freeze-layers", type=int, default=8, help="冻结 RoBERTa 前 N 层")
    parser.add_argument("--temperature", type=float, default=0.07, help="对比学习温度参数")
    
    # 训练参数
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--metab-noise-std", type=float, default=0.01, help="代谢数据增强噪声")
    
    # 评估与保存
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--local-rank", type=int, default=-1, help="DDP rank")
    
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # 1. 加载 tokenizer
    logger.info(f"Loading tokenizer from {args.roberta_path}...")
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_path)
    
    # 2. 加载数据集
    logger.info("Loading datasets...")
    train_dataset = MicrobiomeMetaboliteDataset(
        args.train_data,
        tokenizer,
        max_length=args.max_length,
        augment_metab=True,
        metab_noise_std=args.metab_noise_std
    )
    
    eval_dataset = MicrobiomeMetaboliteDataset(
        args.eval_data,
        tokenizer,
        max_length=args.max_length,
        augment_metab=False  # 验证集不增强
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # 3. 初始化模型
    logger.info("Initializing model...")
    model = MicrobiomeMetaboliteContrastiveModel(
        roberta_path=args.roberta_path,
        metab_input_dim=args.metab_input_dim,
        proj_dim=args.proj_dim,
        freeze_layers=args.freeze_layers,
        temperature=args.temperature
    )
    
    # 4. 分层学习率分组
    logger.info("Setting up optimizer with layer-wise learning rates...")
    
    # 收集所有参数及其学习率
    optimizer_grouped_parameters = []
    
    # RoBERTa 顶层 (需要训练)
    roberta_top_params = [p for n, p in model.roberta.named_parameters() 
                          if p.requires_grad and any(f"layer.{i}" in n for i in range(args.freeze_layers, 12))]
    if roberta_top_params:
        optimizer_grouped_parameters.append({
            "params": roberta_top_params,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay
        })
    
    # RoBERTa 冻结层 (不训练，但为了完整性列出，lr=0)
    # 实际上不需要加到 optimizer 中，Trainer 会自动跳过 requires_grad=False 的
    
    # 新层 (代谢编码器、投影头)
    new_params = []
    new_params.extend(model.metabolite_encoder.parameters())
    new_params.extend(model.text_projector.parameters())
    new_params.extend(model.metab_projector.parameters())
    
    if new_params:
        optimizer_grouped_parameters.append({
            "params": new_params,
            "lr": 1e-4,
            "weight_decay": args.weight_decay
        })
    
    logger.info(f"Optimizer groups: {len(optimizer_grouped_parameters)}")
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=args.fp16,
        bf16=args.bf16,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        report_to="wandb",
        run_name=f"CrossModal-{args.proj_dim}d-{args.temperature}t",
        local_rank=args.local_rank,
    )
    
    # 6. 初始化 Trainer
    trainer = CrossModalContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MicrobiomeMetaboliteDataCollator(pad_token_id=tokenizer.pad_token_id),
    )
    
    # 7. 替换优化器 (关键：Trainer 初始化后设置)
    if optimizer_grouped_parameters:
        trainer.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate, # 这里的 lr 会被组内的 lr 覆盖
            weight_decay=args.weight_decay
        )
        logger.info("Custom optimizer with layer-wise learning rates set.")
    else:
        logger.warning("No trainable parameters found! Check freeze_layers setting.")
    
    # 8. 训练
    logger.info("Starting training...")
    trainer.train()
    
    # 9. 保存最佳模型
    logger.info(f"Saving best model to {args.output_dir}...")
    trainer.save_model()
    
    # 10. 最终评估
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
