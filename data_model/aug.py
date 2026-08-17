import numpy as np
from typing import Any, Dict, List, Callable, Optional, Tuple, Union

# =========================
# 注册表：便于扩展自定义增强
# =========================
_TRANSFORM_REGISTRY: Dict[str, Callable] = {}

def register_transform(name: str):
    def deco(cls):
        _TRANSFORM_REGISTRY[name] = cls
        return cls
    return deco

# =========================
# 基类：所有增强都继承它
# =========================
class BaseTransform:
    def __init__(self, p: float = 1.0, **kwargs):
        """
        p: 本变换被执行的概率（0~1）
        其他参数：各子类自定义
        """
        self.p = float(p)
        self.kwargs = kwargs

    def should_apply(self) -> bool:
        return np.random.rand() < self.p

    def __call__(self, x: np.ndarray, **context) -> np.ndarray:
        """
        x: 单条样本（1D 或 2D/ND 数组均可）
        context: 运行时上下文（index、m_neighbors、data_all、label 等）
        return: 增强后的 x
        """
        raise NotImplementedError

# =========================
# 内置增强 1：邻居混合（与你的 augment_base 对齐）
# =========================
@register_transform("NeighborMixup")
class NeighborMixup(BaseTransform):
    """
    从邻居中随机选一个样本，用 alpha 做线性插值：
    x_aug = x * alpha + x_neighbor * (1 - alpha)
    需要 context 里提供：
      - index: 当前样本索引
      - m_neighbors: 邻居索引列表/数组的集合，m_neighbors[i] -> i 的邻居们
      - data_all: 整个数据矩阵（可为 np.ndarray 或支持 __getitem__ 的对象）
    """
    def __init__(self, alpha_max: float = 0.5, p: float = 1.0):
        super().__init__(p=p, alpha_max=alpha_max)
        assert alpha_max > 0, "alpha_max 必须 > 0"
        self.alpha_max = alpha_max
        


    def __call__(self, x: np.ndarray, context) -> np.ndarray:
        # print('augment NeighborMixup')

        index = context["index"]
        m_neighbors = context["m_neighbors"]
        data_all = context["data_all"]

        neighbor_index_list = m_neighbors[index]

        # 选一个邻居
        sel_idx = np.random.choice(neighbor_index_list)
        x_neighbor = data_all[sel_idx]

        # 采样 alpha，保持与你的实现一致：uniform(0, alpha_max)
        alpha = np.random.uniform(0.0, self.alpha_max)
        x_aug = x * alpha + x_neighbor * (1.0 - alpha)
        return x_aug




@register_transform("TraceMixup")
class TraceMixup(BaseTransform):
    """
    从邻居中随机选一个样本，用 alpha 做线性插值：
    x_aug = x * alpha + x_neighbor * (1 - alpha)
    需要 context 里提供：
      - index: 当前样本索引
      - m_neighbors: 邻居索引列表/数组的集合，m_neighbors[i] -> i 的邻居们
      - data_all: 整个数据矩阵（可为 np.ndarray 或支持 __getitem__ 的对象）
    """
    def __init__(self, alpha_max: float = 0.5, p: float = 1.0, n_hop: int = 1, knn: int = 10):
        super().__init__(p=p, alpha_max=alpha_max)
        assert alpha_max > 0, "alpha_max 必须 > 0"
        self.alpha_max = alpha_max
        self.n_hop = n_hop
        self.knn = knn

    def random_select_neighbor(self, index: int, m_neighbors: np.ndarray) -> int:
        """
        从邻居中随机选择一个样本索引
        """
        neighbor_index_list = m_neighbors[index]
        return np.random.choice(neighbor_index_list[:self.knn])

    def __call__(self, x: np.ndarray, context) -> np.ndarray:
        # print('augment NeighborMixup')

        index = context["index"]
        m_neighbors = context["m_neighbors"]
        data_all = context["data_all"]

        aug_end_index = index
        n_hop_sample = np.random.randint(1, self.n_hop + 1)  # 随机选择跳跃次数
        for i in range(n_hop_sample):
            aug_start_index = aug_end_index
            aug_end_index = self.random_select_neighbor(aug_start_index, m_neighbors)
            # print(f"n_hop: {i+1}, aug_start_index: {aug_start_index}({label[aug_start_index]}), aug_end_index: {aug_end_index}({label[aug_end_index]})")

        x_start = data_all[aug_start_index]
        x_end = data_all[aug_end_index]

        alpha = np.random.uniform(low=0.05, high=self.alpha_max)
        x_aug = x_start * alpha + x_end * (1.0 - alpha)
        return x_aug


# =========================
# 内置增强 2：高斯噪声
# =========================
@register_transform("GaussianNoise")
class GaussianNoise(BaseTransform):
    def __init__(self, std: float = 0.1, mean: float = 0.0, p: float = 0.5):
        super().__init__(p=p, std=std, mean=mean)
        self.std = std
        self.mean = mean

    def __call__(self, x: np.ndarray, **context) -> np.ndarray:
        if not self.should_apply():
            return x
        noise = np.random.randn(*x.shape) * self.std + self.mean
        return x + noise

# =========================
# 内置增强 3：特征随机丢弃（像 dropout）
# =========================
@register_transform("FeatureDropout")
class FeatureDropout(BaseTransform):
    def __init__(self, drop_prob: float = 0.1, p: float = 0.5):
        super().__init__(p=p, drop_prob=drop_prob)
        assert 0.0 <= drop_prob < 1.0
        self.drop_prob = drop_prob

    def __call__(self, x: np.ndarray, **context) -> np.ndarray:
        if not self.should_apply():
            return x
        mask = (np.random.rand(*x.shape) > self.drop_prob).astype(x.dtype)
        return x * mask

# =========================
# Augmenter 主体
# =========================
class Augmenter:
    """
    顺序执行 transforms。默认返回 (orig, aug)：
      - 若未触发任何增强，aug == orig
      - 若触发增强，则在 orig 基础上依次应用得到 aug
    """
    def __init__(self,
                 transforms: List[Union[BaseTransform, Dict[str, Any]]],
                 seed: Optional[int] = None):
        """
        transforms:
          - 也可以传 dict 列表，如：
            [{"name":"NeighborMixup","alpha_max":0.4,"p":1.0}, {"name":"GaussianNoise","std":0.05,"p":0.5}]
        seed: 固定随机种子（可选）
        """
        if seed is not None:
            np.random.seed(seed)

        self.transforms: List[BaseTransform] = []
        for t in transforms:
            if isinstance(t, BaseTransform):
                self.transforms.append(t)
            elif isinstance(t, dict):
                name = t.get("name", None)
                if name is None or name not in _TRANSFORM_REGISTRY:
                    raise ValueError(f"未知变换或未注册：{name}")
                params = {k: v for k, v in t.items() if k != "name"}
                self.transforms.append(_TRANSFORM_REGISTRY[name](**params))
            else:
                raise TypeError("transforms 中的元素必须是 BaseTransform 或 dict")

    def __call__(self, x: np.ndarray, context, return_pair: bool = True ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        x: 原始样本（np.ndarray）
        context: 运行时上下文（index、m_neighbors、data_all、label 等）
        return_pair:
          - True: 返回 (orig, aug)
          - False: 只返回 aug（兼容部分 pipeline）
        """
        orig = x
        aug = x
        applied_any = False

        for t in self.transforms:
            new_aug = t(aug, context=context)
            # if not np.shares_memory(new_aug, aug) or new_aug is not aug:
            #     applied_any = True
            aug = new_aug

        if return_pair:
            # 若没有任何增强被应用，aug 就是 orig
            return orig, aug
        else:
            return aug

    @staticmethod
    def available_transforms() -> List[str]:
        return sorted(list(_TRANSFORM_REGISTRY.keys()))
# =========================
# Compatibility function for legacy code
# =========================
def augment_base(
    data_ori: np.ndarray,
    index_ori: int,
    m_neighbors: np.ndarray,
    data_all: np.ndarray,
    uniform_param: float = 1.0,
    augment_bool: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy augment function for backward compatibility.
    Performs neighbor-based mixup augmentation.

    Args:
        data_ori: Original data sample
        index_ori: Index of the original sample
        m_neighbors: Neighbor indices array
        data_all: Full dataset
        uniform_param: Maximum alpha value for mixup
        augment_bool: Whether to apply augmentation

    Returns:
        Tuple of (original_data, augmented_data)
    """
    if not augment_bool:
        return data_ori, data_ori

    neighbor_index_list = m_neighbors[index_ori]
    selected_index = np.random.choice(neighbor_index_list)
    alpha = np.random.uniform(0, uniform_param)

    try:
        selected_data = data_all[selected_index]
        data_aug = data_ori * alpha + selected_data * (1.0 - alpha)
    except:
        data_aug = data_ori

    return data_ori, data_aug