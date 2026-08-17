"""
Interactive MNIST image explorer for DiffTree and UMAP embeddings.

Launches a Dash web app where you can:
  - View 2D scatter plot of embeddings (colored by digit label)
  - Click any point to see the original 28x28 MNIST image
  - Switch DiffTree layers (t_mul scales) via dropdown

Usage:
    python scripts/visualize_subclusters.py --method difftree --ckpt PATH [--labels 1 7]
    python scripts/visualize_subclusters.py --method umap [--labels 1 7]
    Open http://127.0.0.1:8050 in browser
"""

import argparse
import sys
import os
import base64
from io import BytesIO
from datetime import datetime

import numpy as np
import torch
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar as numpy_scalar
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, no_update, callback
from PIL import Image
from torchvision.datasets import MNIST
from torchvision import transforms

# ---------------------------------------------------------------------------
# Add project root to path so we can import model
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

torch.set_float32_matmul_precision("medium")
add_safe_globals([numpy_scalar, np.dtype])

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
D3_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# ---------------------------------------------------------------------------
# Global state (populated in main)
# ---------------------------------------------------------------------------
data = None
labels = None
all_embeddings = None
VIEW_MAP = {}
EMBEDDING_METHOD = "difftree"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5000
PORT = 8050



def load_mnist_subset(label_subset):
    """Load MNIST train set, flatten to [N, 784], optionally filter by labels."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(root="data", train=True, download=True, transform=transform)

    data_list, label_list = [], []
    for img, label in train_data:
        data_list.append(img.numpy().squeeze())
        label_list.append(label)

    data = np.stack(data_list).reshape((-1, 784))
    labels = np.array(label_list)

    if label_subset is not None:
        mask = np.isin(labels, label_subset)
        data = data[mask]
        labels = labels[mask]

    return data, labels


def extract_all_layer_embeddings(checkpoint_path, input_data):
    """Load DiffTree from checkpoint, extract 2D embeddings for all layers."""
    global VIEW_MAP

    from model.DiffTreeVQ_v2 import DMTEVT_model

    mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Modified:   {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    model = DMTEVT_model.load_from_checkpoint(
        checkpoint_path, map_location=DEVICE, weights_only=False
    )
    model.eval()
    model = model.to(DEVICE)

    tau = getattr(model.hparams, "tau", 1.0)
    n_layers = len(model.hparams.num_use_mlevel_list)

    VIEW_MAP = {i: v for i, v in enumerate(model.hparams.num_use_mlevel_list)}
    n_samples = len(input_data)

    layer_results = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for start in range(0, n_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_samples)
            batch = torch.from_numpy(input_data[start:end]).float().to(DEVICE)
            doubled = torch.cat([batch, batch])
            output = model(doubled, tau=tau)
            lat_vis_list = output[3]

            for i in range(n_layers):
                emb = lat_vis_list[i][: len(batch)].cpu().numpy()
                layer_results[i].append(emb)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = {}
    for i in range(n_layers):
        result[i] = np.concatenate(layer_results[i], axis=0)

    print(f"  Extracted: {n_layers} layers, {n_samples} points each")
    print(f"  Scales: {list(VIEW_MAP.values())}")
    return result


def extract_umap_embedding(input_data, n_neighbors, min_dist, seed):
    """Fit a 2D UMAP embedding on flattened MNIST data."""
    global VIEW_MAP

    import umap

    print(f"  UMAP n_neighbors={n_neighbors}, min_dist={min_dist}, seed={seed}")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        n_jobs=-1,
    )
    embedding = reducer.fit_transform(input_data)
    VIEW_MAP = {0: f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})"}
    print(f"  Extracted: 1 view, {len(input_data)} points")
    return {0: embedding}


def numpy_to_base64(img_1d):
    """Convert a 784-dim flat array to a base64 PNG data URI."""
    img = img_1d.reshape(28, 28)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = img * 255
    img = img.astype(np.uint8)

    pil_img = Image.fromarray(img, mode="L")
    pil_img = pil_img.resize((196, 196), Image.NEAREST)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_placeholder_image():
    """Create a grey placeholder PNG with '?' text."""
    img = np.full((28, 28), 40, dtype=np.uint8)
    pil_img = Image.fromarray(img, mode="L")
    pil_img = pil_img.resize((196, 196), Image.NEAREST)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
app = Dash(__name__)


def build_layout():
    """Build the Dash layout with dropdown options from VIEW_MAP."""
    if EMBEDDING_METHOD == "difftree":
        dropdown_options = [
            {"label": f"Layer {i}  (t_mul = {VIEW_MAP[i]})", "value": i}
            for i in sorted(VIEW_MAP.keys())
        ] if VIEW_MAP else []
        default_value = max(VIEW_MAP.keys()) if VIEW_MAP else 0
        dropdown_label = "Layer:"
        heading = "DiffTree Sub-cluster Explorer"
        description = "Click any point in the scatter plot to view its original MNIST image."
    else:
        dropdown_options = [
            {"label": str(VIEW_MAP[i]), "value": i}
            for i in sorted(VIEW_MAP.keys())
        ] if VIEW_MAP else []
        default_value = 0
        dropdown_label = "Embedding:"
        heading = "UMAP MNIST Explorer"
        description = "Click any UMAP point to view its original MNIST image."
    placeholder = make_placeholder_image()

    return html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "20px", "maxWidth": "1400px", "margin": "0 auto"},
        children=[
            html.H2(heading, style={"marginBottom": "5px"}),
            html.P(
                description,
                style={"color": "#666", "marginBottom": "15px"},
            ),
            html.Div(
                style={"marginBottom": "15px", "display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Label(dropdown_label, style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="layer-dropdown",
                        options=dropdown_options,
                        value=default_value,
                        clearable=False,
                        style={"width": "250px"},
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
                children=[
                    dcc.Graph(
                        id="scatter-plot",
                        style={"flex": "3", "minHeight": "600px"},
                        config={"scrollZoom": True},
                    ),
                    html.Div(
                        style={
                            "flex": "1",
                            "textAlign": "center",
                            "padding": "20px",
                            "border": "1px solid #ddd",
                            "borderRadius": "8px",
                            "backgroundColor": "#fafafa",
                            "minWidth": "240px",
                        },
                        children=[
                            html.H4("Selected Point", style={"marginTop": "0"}),
                            html.Img(
                                id="digit-image",
                                src=placeholder,
                                style={
                                    "width": "196px",
                                    "height": "196px",
                                    "border": "2px solid #ccc",
                                    "borderRadius": "4px",
                                    "imageRendering": "pixelated",
                                },
                            ),
                            html.P(
                                id="point-info",
                                children="Click a point to view its image",
                                style={"marginTop": "10px", "color": "#555", "fontSize": "14px"},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


@callback(Output("scatter-plot", "figure"), Input("layer-dropdown", "value"))
def update_scatter(layer_idx):
    """Rebuild scatter plot when layer selection changes."""
    if all_embeddings is None or layer_idx is None:
        return go.Figure()

    emb = all_embeddings[layer_idx]
    fig = go.Figure()

    for digit in sorted(np.unique(labels)):
        mask = labels == digit
        indices = np.where(mask)[0]
        color = D3_COLORS[int(digit) % len(D3_COLORS)]

        # Use Scatter (SVG) instead of Scattergl for reliable click events
        fig.add_trace(
            go.Scatter(
                x=emb[mask, 0].tolist(),
                y=emb[mask, 1].tolist(),
                mode="markers",
                marker={"size": 3, "color": color, "opacity": 0.7},
                name=f"Digit {digit}",
                customdata=indices.tolist(),
                hovertemplate=(
                    f"Digit {digit}<br>"
                    "x: %{x:.3f}<br>"
                    "y: %{y:.3f}<br>"
                    "idx: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

    view_name = VIEW_MAP.get(layer_idx, "?")
    if EMBEDDING_METHOD == "difftree":
        title = f"Layer {layer_idx} (t_mul = {view_name}) - {len(emb)} points"
    else:
        title = f"{view_name} - {len(emb)} points"
    fig.update_layout(
        title=title,
        clickmode="event",
        dragmode="pan",
        plot_bgcolor="white",
        xaxis={"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
        yaxis={"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)", "scaleanchor": "x"},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
        height=620,
        margin={"t": 50, "b": 40, "l": 50, "r": 20},
    )

    return fig


@callback(
    [Output("digit-image", "src"), Output("point-info", "children")],
    Input("scatter-plot", "clickData"),
    prevent_initial_call=True,
)
def show_image(click_data):
    """Display the MNIST image for the clicked point."""
    if click_data is None:
        return no_update, no_update

    point = click_data["points"][0]
    idx = point.get("customdata")

    if idx is None:
        return no_update, no_update

    idx = int(idx)
    img_src = numpy_to_base64(data[idx])
    digit = int(labels[idx])
    x, y = point["x"], point["y"]

    info = f"Digit: {digit}  |  Index: {idx}  |  ({x:.3f}, {y:.3f})"

    return img_src, info


def parse_args():
    parser = argparse.ArgumentParser(description="DiffTree Sub-cluster Explorer")
    parser.add_argument(
        "--method",
        type=str,
        choices=["difftree", "umap"],
        default="difftree",
        help="Embedding method to visualize.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to .ckpt file (required when --method difftree).",
    )
    parser.add_argument("--labels", type=int, nargs="*", default=[1, 7],
                        help="Digit labels to include (default: 1 7, use --labels for all)")
    parser.add_argument("--umap-neighbors", type=int, default=15,
                        help="UMAP n_neighbors when --method umap.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                        help="UMAP min_dist when --method umap.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for UMAP when --method umap.")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


def main():
    global data, labels, all_embeddings, PORT, EMBEDDING_METHOD

    args = parse_args()
    PORT = args.port
    EMBEDDING_METHOD = args.method
    label_subset = args.labels if args.labels else None

    print("=" * 60)
    if args.method == "difftree":
        print("DiffTree Sub-cluster Explorer")
    else:
        print("UMAP MNIST Explorer")
    print("=" * 60)

    # Load data
    label_str = str(label_subset) if label_subset else "all"
    print(f"\nLoading MNIST (labels: {label_str})...")
    data, labels = load_mnist_subset(label_subset)
    print(f"  -> {len(data)} samples, unique labels: {np.unique(labels)}")

    # Extract embeddings
    if args.method == "difftree":
        if not args.ckpt:
            print("ERROR: --ckpt is required when --method difftree")
            sys.exit(1)
        ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJECT_ROOT, args.ckpt)
        if not os.path.exists(ckpt):
            print(f"ERROR: Checkpoint not found: {ckpt}")
            sys.exit(1)

        print(f"\nLoading model and extracting embeddings...")
        all_embeddings = extract_all_layer_embeddings(ckpt, data)
    else:
        print(f"\nRunning UMAP and extracting embedding...")
        all_embeddings = extract_umap_embedding(
            data,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
        )

    # Build layout now that VIEW_MAP is populated
    app.layout = build_layout()

    # Launch app
    print(f"\n{'=' * 60}")
    print(f"Open http://127.0.0.1:{PORT} in your browser")
    print(f"Drag to pan, scroll to zoom, CLICK point to view image")
    print(f"{'=' * 60}\n")

    app.run(debug=False, port=PORT)


if __name__ == "__main__":
    main()
