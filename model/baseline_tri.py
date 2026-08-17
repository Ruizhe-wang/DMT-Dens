"""
Baseline dimensionality reduction methods for comparison experiments.

Supported methods:
    - `_tsne`
    - `_umap`
    - `_pacmap`
    - `_densmap`
    - `_densne`
    - `_phate`
    - `_denssne`
"""

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
import torch
from lightning import LightningModule
import umap
from sklearn.manifold import TSNE as SklearnTSNE

try:
    from openTSNE import TSNE as OpenTSNE
except ImportError:
    OpenTSNE = None

try:
    import phate
except ImportError:
    phate = None

try:
    import pacmap
except ImportError:
    pacmap = None


def _load_repo_densne_backend() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    backend_dir = Path(__file__).resolve().parent / 'densne'
    backend_file = backend_dir / 'densne.py'
    binary_path = backend_dir / ('windows/den_sne.exe' if os.name == 'nt' else 'den_sne')

    if not backend_file.is_file():
        return None, None, None

    if not binary_path.is_file():
        return None, None, (
            f"Repository densNE wrapper was found at {backend_file}, but the compiled den_sne binary "
            f"is missing at {binary_path}."
        )

    try:
        module = importlib.import_module('model.densne.densne')
    except Exception as exc:
        return None, None, f"Failed to import repository densNE backend: {exc}"

    return module, 'model.densne.densne', None


def _resolve_backend(
    candidate_modules: Sequence[Tuple[str, Optional[str]]],
) -> Tuple[Optional[Any], Optional[str]]:
    for module_name, attr_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        candidate = getattr(module, attr_name, None) if attr_name is not None else module
        if candidate is None:
            continue

        if hasattr(candidate, 'run_densne') or hasattr(candidate, 'densne') or callable(candidate):
            backend_name = module_name if attr_name is None else f'{module_name}.{attr_name}'
            return candidate, backend_name

    return None, None


repo_densne_obj, repo_densne_backend_name, repo_densne_backend_error = _load_repo_densne_backend()


def _resolve_densne_backend() -> Tuple[Optional[Any], Optional[str]]:
    if repo_densne_obj is not None:
        return repo_densne_obj, repo_densne_backend_name

    candidate_modules = (
        ('densne', None),
        ('denssne', None),
    )
    return _resolve_backend(candidate_modules)


def _resolve_denssne_backend() -> Tuple[Optional[Any], Optional[str]]:
    if repo_densne_obj is not None:
        return repo_densne_obj, repo_densne_backend_name

    candidate_modules = (
        ('densne', None),
        ('denssne', None),
        ('openTSNE', 'denssne'),
        ('openTSNE', 'densSNE'),
        ('openTSNE', 'DensSNE'),
    )
    return _resolve_backend(candidate_modules)


densne_obj, densne_backend_name = _resolve_densne_backend()
denssne_obj, denssne_backend_name = _resolve_denssne_backend()


class DMTEVT_model(LightningModule):
    """
    Baseline DR model supporting multiple non-parametric DR baselines.

    Compatible with PyTorch Lightning training loop and ConsistencyCallback.

    Args:
        method: Baseline family identifier, e.g. '_tsne' or '_densmap'
        data_name: Dataset name for saving embeddings
        num_input_dim: Input feature dimension
        n_components: Output dimension (default: 2)
        perplexity: t-SNE perplexity (default: 30)
        early_exaggeration: t-SNE early exaggeration (default: 12)
        n_neighbors: UMAP/densMAP/PaCMAP n_neighbors (default: 15)
        min_dist: UMAP/densMAP min_dist (default: 0.1)
        mn_ratio: PaCMAP mid-near pair ratio (default: 0.5)
        fp_ratio: PaCMAP far pair ratio (default: 2.0)
        knn: PHATE knn graph size (default: 15)
        decay: PHATE decay parameter (default: 40)
        p1, p2: Hyperparameter indices for grid search
        p3: Set to 1 for grid search mode
    """

    # Hyperparameter grids
    TSNE_PERPLEXITY = [15, 30, 50, 80, 120, 200]
    TSNE_EXAGGERATION = [4, 8, 12, 18, 24, 32]
    UMAP_NEIGHBORS = [10, 15, 20, 40, 80, 120]
    UMAP_MIN_DIST = [0.001, 0.01, 0.05, 0.08, 0.15, 0.3]
    PACMAP_NEIGHBORS = [10, 15, 20, 40, 80, 120]
    PACMAP_PAIR_RATIOS = [
        (0.3, 1.0),
        (0.5, 1.0),
        (0.5, 2.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (3.0, 8.0),
    ]
    PHATE_KNN = [5, 10, 15, 20, 40, 80]
    PHATE_DECAY = [10, 20, 40, 80, 120, 160]

    METHOD_DISPLAY_NAMES = {
        '_tsne': 't-SNE',
        '_umap': 'UMAP',
        '_pacmap': 'PaCMAP',
        '_densmap': 'densMAP',
        '_densne': 'densNE',
        '_phate': 'PHATE',
        '_denssne': 'densSNE',
    }

    def __init__(
        self,
        method: str = '_tsne',
        data_name: str = 'mnist',
        num_input_dim: int = 784,
        n_components: int = 2,
        perplexity: Optional[int] = None,
        early_exaggeration: Optional[int] = None,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None,
        mn_ratio: Optional[float] = None,
        fp_ratio: Optional[float] = None,
        knn: Optional[int] = None,
        decay: Optional[int] = None,
        dens_lambda: Optional[float] = None,
        dens_frac: Optional[float] = None,
        p1: int = 2,
        p2: int = 2,
        p3: int = 0,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.method = method
        self.data_name = data_name
        self.n_components = n_components
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        # When random_state is left unset (null in the config), follow the global
        # seed so a single `seed_everything` sweep axis varies the baselines too.
        # Lightning's seed_everything exports PL_GLOBAL_SEED; resolving to an int
        # here keeps native backends happy (e.g. densNE's run_densne(randseed=int),
        # which would crash on None).
        if random_state is None:
            random_state = int(os.environ.get("PL_GLOBAL_SEED", 42))
        self.random_state = random_state

        # Storage
        self.reducer = None
        self.embedding = None
        self.visited = False

        # For ConsistencyCallback compatibility
        self.validation_step_outputs_high = None
        self.validation_step_outputs_vis = None
        self.validation_step_lat_vis_exp = None
        self.validation_origin_input = None
        self._labels = None

        # Save path for embeddings
        self.save_path = f"save_emb_data/{method}/{data_name}"
        os.makedirs(self.save_path, exist_ok=True)

    @classmethod
    def get_method_display_name(cls, method: str) -> str:
        return cls.METHOD_DISPLAY_NAMES.get(method, method.upper())

    @staticmethod
    def _clip_grid_index(index: int, values: Sequence[Any], name: str) -> int:
        if not values:
            raise ValueError(f"{name} grid is empty")
        return max(0, min(int(index), len(values) - 1))

    @staticmethod
    def _override_or_grid(overrides: Dict[str, Any], key: str, grid_value: Any) -> Any:
        value = overrides.get(key)
        return grid_value if value is None else value

    @classmethod
    def _get_tsne_params(
        cls,
        p1: int,
        p2: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        overrides = overrides or {}
        p1 = cls._clip_grid_index(p1, cls.TSNE_PERPLEXITY, 'TSNE_PERPLEXITY')
        p2 = cls._clip_grid_index(p2, cls.TSNE_EXAGGERATION, 'TSNE_EXAGGERATION')
        return (
            cls._override_or_grid(overrides, 'perplexity', cls.TSNE_PERPLEXITY[p1]),
            cls._override_or_grid(overrides, 'early_exaggeration', cls.TSNE_EXAGGERATION[p2]),
        )

    @classmethod
    def _get_umap_params(
        cls,
        p1: int,
        p2: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        overrides = overrides or {}
        p1 = cls._clip_grid_index(p1, cls.UMAP_NEIGHBORS, 'UMAP_NEIGHBORS')
        p2 = cls._clip_grid_index(p2, cls.UMAP_MIN_DIST, 'UMAP_MIN_DIST')
        return (
            cls._override_or_grid(overrides, 'n_neighbors', cls.UMAP_NEIGHBORS[p1]),
            cls._override_or_grid(overrides, 'min_dist', cls.UMAP_MIN_DIST[p2]),
        )

    @classmethod
    def _get_pacmap_params(
        cls,
        p1: int,
        p2: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Any, Any]:
        overrides = overrides or {}
        p1 = cls._clip_grid_index(p1, cls.PACMAP_NEIGHBORS, 'PACMAP_NEIGHBORS')
        p2 = cls._clip_grid_index(p2, cls.PACMAP_PAIR_RATIOS, 'PACMAP_PAIR_RATIOS')
        mn_ratio, fp_ratio = cls.PACMAP_PAIR_RATIOS[p2]
        return (
            cls._override_or_grid(overrides, 'n_neighbors', cls.PACMAP_NEIGHBORS[p1]),
            cls._override_or_grid(overrides, 'mn_ratio', mn_ratio),
            cls._override_or_grid(overrides, 'fp_ratio', fp_ratio),
        )

    @classmethod
    def _get_phate_params(
        cls,
        p1: int,
        p2: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        overrides = overrides or {}
        p1 = cls._clip_grid_index(p1, cls.PHATE_KNN, 'PHATE_KNN')
        p2 = cls._clip_grid_index(p2, cls.PHATE_DECAY, 'PHATE_DECAY')
        return (
            cls._override_or_grid(overrides, 'knn', cls.PHATE_KNN[p1]),
            cls._override_or_grid(overrides, 'decay', cls.PHATE_DECAY[p2]),
        )

    @classmethod
    def has_optional_dependency(cls, method: str) -> Tuple[bool, Optional[str]]:
        if method == '_pacmap' and pacmap is None:
            return False, "PaCMAP requires the `pacmap` package."
        if method == '_phate' and phate is None:
            return False, "PHATE requires the `phate` package."
        if method == '_densne' and densne_obj is None:
            if repo_densne_backend_error is not None:
                return False, repo_densne_backend_error
            return False, "densNE requires the repository `model/densne` backend or an installed `densne`/`denssne` package exposing a densNE API."
        if method == '_denssne':
            if denssne_obj is None and OpenTSNE is None:
                if repo_densne_backend_error is not None:
                    return False, (
                        f"{repo_densne_backend_error} openTSNE is also unavailable, so no densSNE fallback exists."
                    )
                return False, "densSNE requires the repository `model/densne` backend, an installed `densne`/`denssne` package, or `openTSNE` for FFT t-SNE fallback."
        return True, None

    @staticmethod
    def _extract_densne_embedding(result: Any) -> np.ndarray:
        if isinstance(result, tuple):
            if not result:
                raise ValueError("densNE backend returned an empty tuple.")
            result = result[0]
        return np.asarray(result, dtype=np.float64)

    def _run_native_densne_module(
        self,
        backend: Any,
        x: np.ndarray,
        perplexity: float,
        early_exaggeration: float,
        final_dens: bool,
    ) -> np.ndarray:
        dens_frac = self.hparams.get('dens_frac')
        dens_lambda = self.hparams.get('dens_lambda')
        kwargs = {
            'no_dims': self.n_components,
            'perplexity': perplexity,
            'early_exaggeration': early_exaggeration,
            'randseed': self.random_state,
            'verbose': False,
            'initial_dims': min(50, x.shape[1]),
            'use_pca': True,
            'max_iter': int(self.hparams.get('max_iter', 1000)),
            'dens_frac': float(dens_frac) if dens_frac is not None else 0.3,
            'dens_lambda': float(dens_lambda) if dens_lambda is not None else 0.1,
            'final_dens': final_dens,
        }
        result = backend.run_densne(x, **kwargs)
        return self._extract_densne_embedding(result)

    def _run_opentsne_fft_fallback(
        self,
        x: np.ndarray,
        perplexity: float,
        early_exaggeration: float,
    ) -> np.ndarray:
        if OpenTSNE is None:
            raise ImportError(
                "densSNE backend is unavailable and openTSNE fallback is not installed."
            )

        warnings.warn(
            "No dedicated densSNE API was found; falling back to openTSNE FFT t-SNE. "
            "This keeps the baseline runnable but is not an exact densSNE implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        reducer = OpenTSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            random_state=self.random_state,
            n_jobs=-1,
            negative_gradient_method='fft',
        )
        embedding = reducer.fit(x)
        return np.asarray(embedding)

    def _run_denssne(
        self,
        x: np.ndarray,
        perplexity: float,
        early_exaggeration: float,
    ) -> np.ndarray:
        available, message = self.has_optional_dependency('_denssne')
        if not available:
            raise ImportError(message)

        if denssne_obj is None:
            return self._run_opentsne_fft_fallback(
                x,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
            )

        if hasattr(denssne_obj, 'run_densne'):
            return self._run_native_densne_module(
                denssne_obj,
                x,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                final_dens=True,
            )

        if hasattr(denssne_obj, 'densne'):
            embedding = denssne_obj.densne(
                x,
                no_dims=self.n_components,
                perplexity=perplexity,
                randseed=self.random_state,
            )
            return np.asarray(embedding)

        if callable(denssne_obj):
            try:
                # If denssne is an estimator class
                reducer = denssne_obj(
                    n_components=self.n_components,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                if hasattr(reducer, 'fit'):
                    embedding = reducer.fit(x)
                    return np.asarray(embedding)
            except TypeError:
                pass

            # If denssne is a function directly
            try:
                embedding = denssne_obj(
                    x,
                    no_dims=self.n_components,
                    perplexity=perplexity,
                    early_exag_coeff=early_exaggeration,
                    randseed=self.random_state,
                )
                return self._extract_densne_embedding(embedding)
            except TypeError:
                return self._run_opentsne_fft_fallback(
                    x,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                )

        return self._run_opentsne_fft_fallback(
            x,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
        )

    def _run_densne(
        self,
        x: np.ndarray,
        perplexity: float,
        early_exaggeration: float,
    ) -> np.ndarray:
        available, message = self.has_optional_dependency('_densne')
        if not available:
            raise ImportError(message)

        if hasattr(densne_obj, 'run_densne'):
            return self._run_native_densne_module(
                densne_obj,
                x,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                final_dens=False,
            )

        if hasattr(densne_obj, 'densne'):
            embedding = densne_obj.densne(
                x,
                no_dims=self.n_components,
                perplexity=perplexity,
                randseed=self.random_state,
            )
            return np.asarray(embedding)

        if callable(densne_obj):
            try:
                reducer = densne_obj(
                    n_components=self.n_components,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                if hasattr(reducer, 'fit'):
                    embedding = reducer.fit(x)
                    return np.asarray(embedding)
            except TypeError:
                pass

            try:
                embedding = densne_obj(
                    x,
                    no_dims=self.n_components,
                    perplexity=perplexity,
                    early_exag_coeff=early_exaggeration,
                    randseed=self.random_state,
                )
                return self._extract_densne_embedding(embedding)
            except TypeError as exc:
                backend_name = densne_backend_name or 'densNE backend'
                raise TypeError(
                    f"{backend_name} was found, but its callable API is not supported by `_densne`."
                ) from exc

        raise ImportError(
            "Installed densNE backend does not expose a supported densNE API "
            "(expected `run_densne`, `densne`, or a callable estimator/function)."
        )

    def _create_reducer(self):
        """Create the DR reducer based on method and hyperparameters."""
        if self.method == '_tsne':
            perplexity, exaggeration = self._get_tsne_params(
                self.p1,
                self.p2,
                {
                    'perplexity': self.hparams.get('perplexity'),
                    'early_exaggeration': self.hparams.get('early_exaggeration'),
                },
            )
            print(f"[t-SNE] perplexity={perplexity}, early_exaggeration={exaggeration}")
            if OpenTSNE is not None:
                return OpenTSNE(
                    n_components=self.n_components,
                    perplexity=perplexity,
                    early_exaggeration=exaggeration,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            print("[t-SNE] openTSNE not available, falling back to sklearn TSNE")
            return SklearnTSNE(
                n_components=self.n_components,
                perplexity=perplexity,
                early_exaggeration=exaggeration,
                random_state=self.random_state,
                init='pca',
                learning_rate='auto',
            )
        elif self.method == '_umap':
            n_neighbors, min_dist = self._get_umap_params(
                self.p1,
                self.p2,
                {
                    'n_neighbors': self.hparams.get('n_neighbors'),
                    'min_dist': self.hparams.get('min_dist'),
                },
            )
            print(f"[UMAP] n_neighbors={n_neighbors}, min_dist={min_dist}")
            return umap.UMAP(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.method == '_pacmap':
            available, message = self.has_optional_dependency(self.method)
            if not available:
                raise ImportError(message)
            n_neighbors, mn_ratio, fp_ratio = self._get_pacmap_params(
                self.p1,
                self.p2,
                {
                    'n_neighbors': self.hparams.get('n_neighbors'),
                    'mn_ratio': self.hparams.get('mn_ratio'),
                    'fp_ratio': self.hparams.get('fp_ratio'),
                },
            )
            print(
                f"[PaCMAP] n_neighbors={n_neighbors}, mn_ratio={mn_ratio}, fp_ratio={fp_ratio}"
            )
            return pacmap.PaCMAP(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                MN_ratio=mn_ratio,
                FP_ratio=fp_ratio,
                random_state=self.random_state,
                apply_pca=True,
            )
        elif self.method == '_densmap':
            n_neighbors, min_dist = self._get_umap_params(
                self.p1,
                self.p2,
                {
                    'n_neighbors': self.hparams.get('n_neighbors'),
                    'min_dist': self.hparams.get('min_dist'),
                },
            )
            dens_lambda = self.hparams.get('dens_lambda')
            dens_frac = self.hparams.get('dens_frac')
            dens_kwargs = {}
            if dens_lambda is not None:
                dens_kwargs['dens_lambda'] = float(dens_lambda)
            if dens_frac is not None:
                dens_kwargs['dens_frac'] = float(dens_frac)
            print(
                f"[densMAP] n_neighbors={n_neighbors}, min_dist={min_dist}, "
                f"dens_lambda={dens_kwargs.get('dens_lambda', 'umap-default')}, "
                f"dens_frac={dens_kwargs.get('dens_frac', 'umap-default')}"
            )
            return umap.UMAP(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                densmap=True,
                output_dens=False,
                random_state=self.random_state,
                n_jobs=-1,
                **dens_kwargs,
            )
        elif self.method == '_densne':
            available, message = self.has_optional_dependency(self.method)
            if not available:
                raise ImportError(message)
            perplexity, exaggeration = self._get_tsne_params(
                self.p1,
                self.p2,
                {
                    'perplexity': self.hparams.get('perplexity'),
                    'early_exaggeration': self.hparams.get('early_exaggeration'),
                },
            )
            print(f"[densNE] perplexity={perplexity}, early_exaggeration={exaggeration}")
            return {
                'perplexity': perplexity,
                'early_exaggeration': exaggeration,
            }
        elif self.method == '_phate':
            available, message = self.has_optional_dependency(self.method)
            if not available:
                raise ImportError(message)
            knn, decay = self._get_phate_params(
                self.p1,
                self.p2,
                {
                    'knn': self.hparams.get('knn'),
                    'decay': self.hparams.get('decay'),
                },
            )
            print(f"[PHATE] knn={knn}, decay={decay}")
            return phate.PHATE(
                n_components=self.n_components,
                knn=knn,
                decay=decay,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
        elif self.method == '_denssne':
            available, message = self.has_optional_dependency(self.method)
            if not available:
                raise ImportError(message)
            perplexity, exaggeration = self._get_tsne_params(
                self.p1,
                self.p2,
                {
                    'perplexity': self.hparams.get('perplexity'),
                    'early_exaggeration': self.hparams.get('early_exaggeration'),
                },
            )
            print(f"[densSNE] perplexity={perplexity}, early_exaggeration={exaggeration}")
            return {
                'perplexity': perplexity,
                'early_exaggeration': exaggeration,
            }
        else:
            supported = ", ".join(sorted(self.METHOD_DISPLAY_NAMES))
            raise ValueError(f"Unknown method: {self.method}. Supported methods: {supported}")

    def _fit_transform(self, x: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            self.reducer = self._create_reducer()

        if self.method == '_pacmap':
            self.validation_origin_input = np.asarray(x).copy()

        if self.method == '_densne':
            return self._run_densne(
                x,
                perplexity=float(self.reducer['perplexity']),
                early_exaggeration=float(self.reducer['early_exaggeration']),
            )

        if self.method == '_denssne':
            return self._run_denssne(
                x,
                perplexity=float(self.reducer['perplexity']),
                early_exaggeration=float(self.reducer['early_exaggeration']),
            )

        if self.method == '_tsne' and OpenTSNE is not None:
            embedding = self.reducer.fit(x)
            return np.asarray(embedding)

        return np.asarray(self.reducer.fit_transform(x))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            return self._fit_transform(x)

        if self.method == '_pacmap':
            transform = getattr(self.reducer, 'transform', None)
            if callable(transform) and self.validation_origin_input is not None:
                transformed = transform(x, basis=self.validation_origin_input)
                return np.asarray(transformed)
            if self.embedding is not None and len(self.embedding) == len(x):
                return np.asarray(self.embedding)
            return self._fit_transform(x)

        transform = getattr(self.reducer, 'transform', None)
        if callable(transform):
            try:
                transformed = transform(x)
                return np.asarray(transformed)
            except NotImplementedError:
                # Some methods like densMAP do not support transform()
                pass

        if self.embedding is not None and len(self.embedding) == len(x):
            return np.asarray(self.embedding)

        return self._fit_transform(x)

    def forward(self, x):
        """Forward pass - fit and transform data."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)
        if self.method == "_pacmap":
            # PaCMAP caches sampled pairs on the reducer instance. A new fit
            # must not reuse row indices sampled from a prior input order.
            self.reducer = self._create_reducer()
        self.embedding = self._fit_transform(x)
        return self.embedding

    def configure_optimizers(self):
        """No optimizer needed for baseline methods."""
        return None

    def training_step(self, batch, batch_idx):
        """Fit the DR method on training data."""
        self.visited = True

        # Get input data
        data_input = batch['data_input_item']
        if isinstance(data_input, torch.Tensor):
            data_input = data_input.cpu().numpy()
        data_input = data_input.reshape(data_input.shape[0], -1).astype(np.float32)

        print(f"\n[{self.method}] Fitting on {data_input.shape[0]} samples...")

        # Create and fit reducer
        self.reducer = self._create_reducer()
        self.embedding = self._fit_transform(data_input)

        print(f"[{self.method}] Fitting complete. Embedding shape: {self.embedding.shape}")

        # Store for ConsistencyCallback (important!)
        self.validation_step_outputs_high = data_input
        self.validation_step_outputs_vis = np.array(self.embedding)
        self.validation_step_lat_vis_exp = np.array(self.embedding)
        self.validation_origin_input = data_input

        # Store labels for visualization to ensure alignment with shuffled data
        if 'label' in batch:
            self._labels = batch['label']
            if isinstance(self._labels, torch.Tensor):
                self._labels = self._labels.cpu().numpy()

        # Save embedding
        save_file = os.path.join(self.save_path, f"embedding_p{self.p1}_{self.p2}.npy")
        np.save(save_file, self.embedding)
        print(f"[{self.method}] Embedding saved to {save_file}")

        return None

    def validation_step(self, batch, batch_idx):
        """Transform validation data using fitted reducer."""
        if not self.visited or self.reducer is None:
            return None

        # Get input data
        data_input = batch['data_input_item']
        if isinstance(data_input, torch.Tensor):
            data_input = data_input.cpu().numpy()
        data_input = data_input.reshape(data_input.shape[0], -1).astype(np.float32)

        # Transform
        embedding = self._transform(data_input)

        # For baseline models, we use training data embedding from training_step
        # Do NOT override validation_step_outputs_* if already set
        # ConsistencyCallback should use the full training set embedding
        if self.validation_step_outputs_high is None:
            self.validation_step_outputs_high = data_input
        if self.validation_step_outputs_vis is None:
            self.validation_step_outputs_vis = np.array(embedding)
        if self.validation_step_lat_vis_exp is None:
            self.validation_step_lat_vis_exp = np.array(embedding)
        if self.validation_origin_input is None:
            self.validation_origin_input = data_input

        return None

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        pass
