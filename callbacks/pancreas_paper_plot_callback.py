from __future__ import annotations

from callbacks.bone_marrow_paper_plot_callback import BoneMarrowPaperPlotCallback


PANCREAS_LABEL_ORDER = [
    "Ngn3 low EP",
    "Ngn3 high EP",
    "Fev+",
    "Alpha",
    "Beta",
    "Delta",
    "Epsilon",
]

PANCREAS_LABEL_COLORS = {
    "Ngn3 low EP": "#8c564b",
    "Ngn3 high EP": "#bcbd22",
    "Fev+": "#ff7f0e",
    "Alpha": "#e15759",
    "Beta": "#4e79a7",
    "Delta": "#59a14f",
    "Epsilon": "#b07aa1",
}

PANCREAS_STATE_GROUPS = {
    "Ngn3 low EP": "early",
    "Ngn3 high EP": "transition",
    "Fev+": "transition",
    "Alpha": "terminal",
    "Beta": "terminal",
    "Delta": "terminal",
    "Epsilon": "terminal",
}

PANCREAS_MARKER_GENES = [
    "Neurog3",
    "Fev",
    "Gcg",
    "Ins1",
    "Ins2",
    "Sst",
    "Ghrl",
    "Pdx1",
    "Sox9",
    "Hes1",
    "Krt19",
]


class PancreasPaperPlotCallback(BoneMarrowPaperPlotCallback):
    """Publication-focused plots for pancreatic endocrinogenesis embeddings.

    This callback shares the proven paper-plotting implementation used for the
    bone marrow case study, but uses pancreatic endocrine labels, pseudotime,
    fine annotations, and canonical marker genes.
    """

    def __init__(
        self,
        output_dir: str = "outputs/paper_figures/pancreas",
        signal_keys: list[str] | None = None,
        marker_genes: list[str] | None = None,
        dataset_slug: str = "pancreas",
        label_order: list[str] | None = None,
        label_colors: dict[str, str] | None = None,
        branch_probability_keys: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            output_dir=output_dir,
            signal_keys=signal_keys
            or [
                "final_annotation",
                "state_group",
                "local_density",
                "clusters_fine",
                "clusters_coarse",
                "pseudotime",
            ],
            branch_probability_keys=branch_probability_keys or [],
            marker_genes=marker_genes or PANCREAS_MARKER_GENES,
            dataset_slug=dataset_slug,
            label_order=label_order or PANCREAS_LABEL_ORDER,
            label_colors=label_colors or PANCREAS_LABEL_COLORS,
            state_group_map=PANCREAS_STATE_GROUPS,
            **kwargs,
        )
