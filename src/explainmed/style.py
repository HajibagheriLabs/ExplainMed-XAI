"""Colours and matplotlib settings shared by every figure in reports/."""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")
# one-hue ramp from near-surface to dark, for magnitudes such as confusion rates
SEQUENTIAL = (
    "#f4f8fe",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)
FIGURE_STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelcolor": SECONDARY_INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "legend.frameon": False,
}
BAR_LABEL_STYLE = {"padding": 3, "color": SECONDARY_INK, "fontsize": 8}
# one colour per attribution method in every figure, with the random control in grey
METHOD_COLOURS = {
    "grad_cam": SERIES[0],
    "integrated_gradients": SERIES[1],
    "attention_rollout": "#1f9e89",
    "random_image": MUTED_INK,
    "random_text": MUTED_INK,
}
METHOD_LABELS = {
    "grad_cam": "Grad-CAM",
    "integrated_gradients": "Integrated gradients",
    "attention_rollout": "Attention rollout",
    "random_image": "Random attribution",
    "random_text": "Random attribution",
}
