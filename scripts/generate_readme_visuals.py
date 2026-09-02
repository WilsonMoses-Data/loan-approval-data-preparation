"""Generate branded README visuals for the loan-approval project."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "loan_approval_train.csv"
IMAGE_DIR = ROOT / "images"

DARK = "#0d0d0d"
OFF_WHITE = "#f4f0e8"
GREY = "#a9a7a2"
GOLD = "#c69a4b"
GRID = "#333333"


def finish_chart(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(DARK)
    axis.tick_params(colors=OFF_WHITE, labelsize=10)
    axis.xaxis.label.set_color(GREY)
    axis.yaxis.label.set_color(GREY)
    axis.title.set_color(OFF_WHITE)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def create_social_preview() -> None:
    fig = plt.figure(figsize=(8, 4), dpi=160, facecolor=DARK)
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_axis_off()
    canvas.add_patch(
        plt.Rectangle((0.065, 0.12), 0.008, 0.76, transform=canvas.transAxes, color=GOLD)
    )
    fig.text(0.10, 0.77, "FINANCIAL ANALYTICS  |  WEEK 2", color=GOLD, fontsize=12, weight="bold")
    fig.text(0.10, 0.58, "LOAN APPROVAL", color=OFF_WHITE, fontsize=29, weight="bold")
    fig.text(0.10, 0.45, "DATA PREPARATION", color=OFF_WHITE, fontsize=23, weight="bold")
    fig.text(0.10, 0.30, "614 applications  •  cleaning  •  feature engineering", color=GREY, fontsize=10)
    fig.text(0.10, 0.17, "WILSON MOSES  |  DATA SCIENCE × AI ENGINEERING", color=GOLD, fontsize=10)
    logo_axis = fig.add_axes((0.78, 0.32, 0.18, 0.34))
    logo_axis.imshow(plt.imread(IMAGE_DIR / "wilson-moses-logo.png"))
    logo_axis.set_axis_off()
    fig.savefig(IMAGE_DIR / "social-preview.png", facecolor=DARK)
    plt.close(fig)


def create_outcome_chart(data: pd.DataFrame) -> None:
    labels = ["Approved (Y)", "Not approved (N)"]
    counts = data["Loan_Status"].value_counts().reindex(["Y", "N"])
    rates = counts.div(len(data)).mul(100)
    fig, axis = plt.subplots(figsize=(8.5, 5), facecolor=DARK)
    bars = axis.bar(labels, counts.values, color=[GOLD, GREY], width=0.55)
    style_axis(axis)
    axis.set_title("Historical loan-approval outcomes", fontsize=17, weight="bold", pad=18)
    axis.set_ylabel("Applications")
    axis.set_ylim(0, counts.max() * 1.22)
    for bar, count, rate in zip(bars, counts.values, rates.values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.03,
            f"{count:,}  ({rate:.2f}%)",
            ha="center",
            color=OFF_WHITE,
            fontsize=10,
            weight="bold",
        )
    finish_chart(fig, IMAGE_DIR / "approval-outcomes.png")


def create_quality_chart(data: pd.DataFrame) -> None:
    missing = data.isna().sum()
    missing = missing[missing.gt(0)].sort_values()
    labels = [label.replace("_", " ") for label in missing.index]
    fig, axis = plt.subplots(figsize=(9, 5.3), facecolor=DARK)
    bars = axis.barh(labels, missing.values, color=GOLD, height=0.6)
    style_axis(axis)
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.65)
    axis.grid(axis="y", visible=False)
    axis.set_title("Missing values in the raw dataset", fontsize=17, weight="bold", pad=18)
    axis.set_xlabel("Missing records")
    axis.set_ylabel("")
    axis.set_xlim(0, missing.max() * 1.18)
    for bar, value in zip(bars, missing.values):
        axis.text(value + 1, bar.get_y() + bar.get_height() / 2, str(value), va="center", color=OFF_WHITE)
    finish_chart(fig, IMAGE_DIR / "missing-values.png")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    create_social_preview()
    create_outcome_chart(data)
    create_quality_chart(data)


if __name__ == "__main__":
    main()
