import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler   


# 带有预热(warmup)的余弦退火学习率调度器
class CosineAnnealingSchedule(_LRScheduler):
    """
    余弦退火学习率调度器，带有线性预热阶段。
    
    学习率变化分为两个阶段：
    1. 预热阶段(Warmup): 学习率从 warmup_lr 线性增长到 base_lr
    2. 衰减阶段(Decay): 学习率按照余弦函数从 base_lr 平滑下降到 final_lr
    
    这种策略的优点：
    - 预热阶段避免训练初期因学习率过大导致的不稳定
    - 余弦退火提供平滑的学习率衰减，有助于模型收敛到更好的局部最优解
    """

    def __init__(self, opt, final_lr=0, n_epochs=1000, warmup_epochs=10, warmup_lr=0):
        """
        初始化调度器。

        参数:
            opt (Optimizer): PyTorch优化器对象（如Adam, SGD等）
            final_lr (float): 训练结束时的最终学习率，默认为0
            n_epochs (int): 总训练轮数，默认1000
            warmup_epochs (int): 预热阶段的轮数，默认10
            warmup_lr (float): 预热阶段的初始学习率，默认0
        """
        # 保存优化器引用（self.optimizer是为了兼容父类_LRScheduler的接口）
        self.opt = opt
        self.optimizer = self.opt
        
        # 从优化器中获取基础学习率（用户在创建优化器时设置的lr）
        self.base_lr = base_lr = opt.defaults["lr"]
        
        # 保存其他参数
        self.final_lr = final_lr          # 最终学习率
        self.n_epochs = n_epochs          # 总轮数
        self.warmup_epochs = warmup_epochs  # 预热轮数
        self.warmup_lr = warmup_lr        # 预热起始学习率

        # 计算衰减阶段的轮数
        # 注意：这里加1是为了确保在warmup结束后的第一个epoch也被包含
        decay_epochs = 1 + n_epochs - warmup_epochs
        self.decay_epochs = decay_epochs

        # ========== 构建学习率调度表 ==========
        
        # 预热调度：使用 np.linspace 创建从 warmup_lr 到 base_lr 的线性序列
        # 例如：warmup_lr=0, base_lr=0.1, warmup_epochs=5
        # 生成: [0.0, 0.025, 0.05, 0.075, 0.1]
        warmup_schedule = np.linspace(warmup_lr, base_lr, warmup_epochs)
        
        # 衰减调度：使用余弦退火公式
        # 公式: lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + cos(π * t / T))
        # 其中 t 是当前衰减epoch，T 是总衰减epochs
        # 
        # 数学原理：
        # - 当 t=0 时, cos(0)=1, lr = final_lr + 0.5*(base_lr-final_lr)*2 = base_lr
        # - 当 t=T 时, cos(π)=-1, lr = final_lr + 0.5*(base_lr-final_lr)*0 = final_lr
        # - 中间过程按余弦曲线平滑过渡
        decay_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_epochs) / decay_epochs)
        )
        
        # 将预热和衰减调度表拼接成完整的学习率计划
        # 结果是一个长度为 (warmup_epochs + decay_epochs) 的数组
        self.lr_schedule = np.hstack((warmup_schedule, decay_schedule))

        # 初始化状态变量
        self._last_lr = self.lr_schedule[0]  # 记录上一次的学习率（用于兼容父类接口）
        self.cur_epoch = 0                    # 当前epoch计数器

        # 初始化优化器的学习率
        self.init_opt()

    def init_opt(self):
        """
        初始化优化器的学习率。
        
        在创建调度器后立即调用，将优化器的学习率设置为调度表的第一个值。
        这确保了训练从正确的初始学习率开始。
        """
        self.step()  # 调用step()来设置第一个epoch的学习率

    def get_lr(self):
        """
        获取当前epoch对应的学习率。
        
        返回:
            float: 当前epoch应该使用的学习率
            
        注意：
            直接从预计算的lr_schedule数组中查表获取，时间复杂度O(1)
        """
        return self.lr_schedule[self.cur_epoch]

    def step(self):
        """
        执行一步学习率更新。
        
        工作流程:
            1. 获取当前epoch对应的学习率
            2. 更新优化器中所有参数组的学习率
            3. epoch计数器加1
            4. 记录当前学习率供后续查询
        
        返回:
            float: 更新后的学习率
            
        使用时机:
            通常在每个epoch结束后调用，用于更新下一个epoch的学习率
            
        示例:
            for epoch in range(n_epochs):
                train_one_epoch()
                scheduler.step()  # 更新学习率
        """
        # 获取当前epoch的学习率
        lr = self.get_lr()
        
        # 更新优化器中所有参数组的学习率
        # 注意：一个优化器可能有多个参数组（例如对不同层使用不同学习率）
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

        # 移动到下一个epoch
        self.cur_epoch += 1
        
        # 保存当前学习率（用于兼容_LRScheduler父类的接口）
        self._last_lr = lr
        
        return lr

    def set_epoch(self, epoch):
        """
        设置当前epoch（用于恢复训练）。
        
        参数:
            epoch (int): 要设置的epoch值
            
        使用场景:
            当从检查点(checkpoint)恢复训练时，需要将调度器的状态
            恢复到保存时的epoch，以确保学习率调度的连续性。
            
        示例:
            # 恢复训练时
            checkpoint = torch.load('checkpoint.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.set_epoch(checkpoint['epoch'])  # 恢复epoch状态
        """
        self.cur_epoch = epoch