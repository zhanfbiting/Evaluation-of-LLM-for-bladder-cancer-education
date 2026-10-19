import os

os.environ.setdefault("MPLCONFIGDIR", str(__import__("config").MPLCONFIG_DIR))

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import config


FONT_FAMILY = ["STIXGeneral", "Times New Roman", "DejaVu Serif"]


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_FAMILY,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 2,
            "axes.facecolor": "#FCFDFE",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "grid.color": "#D7DEE8",
            "grid.linewidth": 1.2,
            "grid.alpha": 0.65,
            "xtick.major.width": 3,
            "ytick.major.width": 3,
            "xtick.major.size": 10,
            "ytick.major.size": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "font.weight": "bold",
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#1F1F1F",
            "legend.facecolor": "#FFFFFF",
        }
    )


def _bold_ticklabels(ax) -> None:
    for tick in ax.get_xticklabels():
        tick.set_fontsize(config.TICK_SIZE)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(config.TICK_SIZE)
        tick.set_fontweight("bold")


def _bold_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor("#1F1F1F")
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(config.LEGEND_SIZE)
        legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontsize(config.LEGEND_SIZE)
        text.set_fontweight("bold")


def style_axes(ax, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
    if title is not None:
        ax.set_title(title, fontsize=config.TITLE_SIZE, fontweight="bold", pad=18)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=config.LABEL_SIZE, fontweight="bold", labelpad=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=config.LABEL_SIZE, fontweight="bold", labelpad=12)

    ax.tick_params(
        axis="both",
        labelsize=config.TICK_SIZE,
        width=3,
        length=10,
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )
    _bold_ticklabels(ax)
    _bold_legend(ax)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)


def style_colorbar(colorbar) -> None:
    colorbar.ax.tick_params(labelsize=config.TICK_SIZE, width=3, length=10, direction="in")
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontsize(config.TICK_SIZE)
        tick.set_fontweight("bold")
    if colorbar.ax.yaxis.label is not None:
        colorbar.ax.yaxis.label.set_fontsize(config.LABEL_SIZE)
        colorbar.ax.yaxis.label.set_fontweight("bold")
    colorbar.outline.set_linewidth(2)


def save_figure(fig, output_dir, filename: str) -> None:
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
