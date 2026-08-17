import importlib


__all__ = []
_SYMBOL_TO_MODULE = {}
_IMPORT_ERRORS = {}


def _export_optional(module_name, symbol_names):
    module_basename = module_name.rsplit('.', 1)[-1]
    _SYMBOL_TO_MODULE[module_basename] = module_name
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        _IMPORT_ERRORS[module_basename] = exc
        for symbol_name in symbol_names:
            _SYMBOL_TO_MODULE[symbol_name] = module_name
            _IMPORT_ERRORS[symbol_name] = exc
        return

    globals()[module_basename] = module
    for symbol_name in symbol_names:
        _SYMBOL_TO_MODULE[symbol_name] = module_name
        globals()[symbol_name] = getattr(module, symbol_name)
        __all__.append(symbol_name)


def __getattr__(name):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'callbacks' has no attribute '{name}'")

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        previous_exc = _IMPORT_ERRORS.get(name)
        raise ImportError(
            f"Unable to import callbacks symbol '{name}' from '{module_name}'. "
            f"Missing dependency or module import failed: {exc}"
        ) from previous_exc or exc

    module_basename = module_name.rsplit('.', 1)[-1]
    if name == module_basename:
        globals()[module_basename] = module
        return module

    value = getattr(module, name)
    globals()[module_basename] = module
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)
    return value


_export_optional(
    'callbacks.local_global_consistency',
    ['LocalGlobalConsistencyVisualizer', 'quick_consistency_analysis', 'D3_COLORS'],
)
_export_optional(
    'callbacks.consistency_callback',
    ['ConsistencyCallback', 'DMTEVTConsistencyCallback', 'QuickConsistencyCallback'],
)
_export_optional(
    'callbacks.xc_difftree_density_branch',
    ['DiffTreeDensityBranchStructureCallback'],
)
_export_optional(
    'callbacks.xc_plot_callback',
    [],
)
_export_optional(
    'callbacks.xc_plot_heterogeneity',
    [],
)
_export_optional(
    'callbacks.xc_plot_marker_genes',
    ['MarkerGeneExpressionCallback'],
)
_export_optional(
    'callbacks.xc_save_consolidated_embeddings',
    ['SaveConsolidatedEmbeddingsCallback'],
)
_export_optional(
    'callbacks.bone_marrow_paper_plot_callback',
    ['BoneMarrowPaperPlotCallback'],
)
_export_optional(
    'callbacks.pancreas_paper_plot_callback',
    ['PancreasPaperPlotCallback'],
)
_export_optional(
    'callbacks.paper_embedding_plot_callback',
    [],
)
_export_optional(
    'callbacks.manifold_diagnostics',
    ['ManifoldDiagnosticsCallback'],
)
