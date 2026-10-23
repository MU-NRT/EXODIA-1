from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import ttest_ind

from fragment_molecules_to_dataframe_file_v3 import fragment_molecules_to_dataframe
import parity_with_train_density_bottom_file_2 as _parity_with_train_density_bottom_module
from plot_top_functional_groups_in_grid_file import plot_top_functional_groups_in_grid

_FONT_FAMILY = "Arial"
_FONT_FALLBACKS = ["Arial", "DejaVu Sans"]
_AXIS_LABEL_SIZE = 12
_TEXT_SIZE = 10
_LINE_WIDTH = 0.7
_MARKER_SIZE = 4
_SCATTER_MARKER_AREA = _MARKER_SIZE**2
_TRAIN_DENSITY_BLUE = "blue"
_TRAIN_SERIES_BLUE = "#4c78a8"
_VALIDATION_SERIES_RED = "#c65a45"
_ABS_ERROR_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "abs_error_publication",
    ["#dce8f2", "#9eb9d2", "#e3b29f", _VALIDATION_SERIES_RED],
)
_RESIDUAL_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "residual_publication",
    [_TRAIN_SERIES_BLUE, "#e8ecef", _VALIDATION_SERIES_RED],
)


def _apply_publication_style() -> None:
    sns.set_theme(style="ticks", rc={"axes.grid": False})
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _FONT_FALLBACKS,
            "axes.labelsize": _AXIS_LABEL_SIZE,
            "axes.labelweight": "normal",
            "axes.titlesize": _AXIS_LABEL_SIZE,
            "axes.titleweight": "normal",
            "xtick.labelsize": _TEXT_SIZE,
            "ytick.labelsize": _TEXT_SIZE,
            "legend.fontsize": _TEXT_SIZE,
            "lines.linewidth": _LINE_WIDTH,
            "lines.markersize": _MARKER_SIZE,
            "axes.grid": False,
            "axes.linewidth": _LINE_WIDTH,
            "xtick.major.width": _LINE_WIDTH,
            "ytick.major.width": _LINE_WIDTH,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "patch.linewidth": _LINE_WIDTH,
        }
    )


def _style_axis(
    ax,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    *,
    despine: bool = True,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
    if title is not None:
        ax.set_title(title, fontsize=_AXIS_LABEL_SIZE, fontweight="normal")

    ax.grid(False)
    ax.tick_params(axis="both", labelsize=_TEXT_SIZE, width=_LINE_WIDTH, length=3.5, direction="out")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("normal")
    if despine:
        sns.despine(ax=ax)


def _style_legend(ax, **kwargs):
    legend = ax.legend(
        frameon=False,
        prop={"family": _FONT_FAMILY, "size": _TEXT_SIZE, "weight": "normal"},
        **kwargs,
    )
    return legend


def _style_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(label, fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
    colorbar.ax.tick_params(labelsize=_TEXT_SIZE, width=_LINE_WIDTH)
    for tick in [*colorbar.ax.get_xticklabels(), *colorbar.ax.get_yticklabels()]:
        tick.set_fontweight("normal")
    colorbar.outline.set_linewidth(_LINE_WIDTH)


def _resolve_heatmap_cmap(error_value: str, cmap: str | None) -> str | LinearSegmentedColormap:
    if cmap is not None:
        return cmap
    if error_value == "absolute":
        return _ABS_ERROR_HEATMAP_CMAP
    return _RESIDUAL_HEATMAP_CMAP


def _resolve_property_token(
    property_col: str | None = None,
    property_code: str | None = None,
) -> str:
    if property_col is not None and property_code is not None and property_col != property_code:
        raise ValueError(
            f"Conflicting property identifiers provided: property_col={property_col!r}, "
            f"property_code={property_code!r}"
        )
    resolved = property_col if property_col is not None else property_code
    if resolved is None:
        raise ValueError("Provide either `property_col` or `property_code`.")
    return resolved


def _resolve_ranking_token(
    ranking: str | None = None,
    rank_mode: str | None = None,
    rank_kind: str | None = None,
) -> str:
    provided = [value for value in (ranking, rank_mode, rank_kind) if value is not None]
    if not provided:
        return "best"

    normalized = [value.lower() for value in provided]
    if len(set(normalized)) > 1:
        raise ValueError(
            f"Conflicting ranking identifiers provided: ranking={ranking!r}, "
            f"rank_mode={rank_mode!r}, rank_kind={rank_kind!r}"
        )

    resolved = normalized[0]
    if resolved not in {"best", "worst"}:
        raise ValueError("Ranking must be either 'best' or 'worst'.")
    return resolved


def plot_loss_curves(metrics_csv: Path, save_path: Path | None = None) -> None:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    _apply_publication_style()
    log_df = pd.read_csv(metrics_csv).dropna(how="all")
    train_loss_epoch = log_df[["epoch", "train_loss_epoch"]].dropna()
    val_loss = log_df[["epoch", "val_loss"]].dropna()

    fig, ax = plt.subplots(figsize=(6.00, 3), dpi=1000)
    ax.plot(
        train_loss_epoch["epoch"],
        train_loss_epoch["train_loss_epoch"],
        label="Train Loss",
        marker="o",
        color=_TRAIN_SERIES_BLUE,
        linewidth=_LINE_WIDTH,
        markersize=_MARKER_SIZE,
    )
    ax.plot(
        val_loss["epoch"],
        val_loss["val_loss"],
        label="Validation Loss",
        marker="s",
        color=_VALIDATION_SERIES_RED,
        linewidth=_LINE_WIDTH,
        markersize=_MARKER_SIZE,
    )
    ax.margins(x=0.01, y=0.08)
    _style_axis(ax, xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss")
    _style_legend(ax, loc="upper right", ncol=2, handlelength=2.2, columnspacing=1.2)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def plot_parity_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    split_name: str = "test",
    density_split: str = "train",
    density_scale: float = 1.0,
    save_path: Path | None = None,
    property_code: str | None = None,
) -> None:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    parity_module = importlib.reload(_parity_with_train_density_bottom_module)
    parity_module.parity_with_train_density_bottom(
        model_output_data_path=str(predictions_csv),
        property_col=resolved_property,
        split_col=split_col,
        split_name=split_name,
        density_split=density_split,
        density_scale=density_scale,
        output_file=str(save_path) if save_path is not None else None,
    )


def plot_parity_scatter_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    split_name: str = "test",
    density_split: str = "train",
    save_path: Path | None = None,
    property_code: str | None = None,
) -> None:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    parity_module = importlib.reload(_parity_with_train_density_bottom_module)
    parity_module.plot_parity_only(
        model_output_data_path=str(predictions_csv),
        property_col=resolved_property,
        split_col=split_col,
        split_name=split_name,
        density_split=density_split,
        output_file=str(save_path) if save_path is not None else None,
    )


def plot_parity_scatter_grid_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    split_names: tuple[str, ...] | list[str] = ("train", "val", "test"),
    density_split: str = "train",
    x_axis_range: tuple[float, float] | None = None,
    y_axis_range: tuple[float, float] | None = None,
    x_tick_values: list[float] | np.ndarray | None = None,
    y_tick_values: list[float] | np.ndarray | None = None,
    mae_location: str = "top_left",
    save_path: Path | None = None,
    property_code: str | None = None,
) -> None:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    _apply_publication_style()
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    pred_df = pd.read_csv(predictions_csv)
    true_col = f"{resolved_property}_true"
    pred_col = f"{resolved_property}_pred"
    required_cols = {split_col, true_col, pred_col}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for parity grid: {sorted(missing)}")

    split_names = tuple(split_names)
    property_key = {
        "BP": "Boiling Point",
        "MP": "Melting Point",
        "VP": "$log_{10}(P^{sat} [torr])$",
        "OP": "Oxidation Potential",
        "pka": "pKa",
    }
    units_key = {
        "BP": "[K]",
        "MP": "[K]",
        "VP": " ",
        "OP": "[V]",
        "pka": " ",
    }
    prop_name = property_key.get(resolved_property, resolved_property)
    prop_unit = units_key.get(resolved_property, "")
    prop_label = f"{prop_name} {prop_unit}" if prop_unit.strip() else prop_name

    df_density = pred_df[pred_df[split_col] == density_split].copy()
    if df_density.empty:
        raise ValueError(f"No rows found for density split='{density_split}' in {predictions_csv}")

    split_frames: list[tuple[str, pd.DataFrame]] = []
    all_x_parts = [df_density[true_col].to_numpy()]
    for split_name in split_names:
        split_df = pred_df[pred_df[split_col] == split_name].copy()
        if split_df.empty:
            raise ValueError(f"No rows found for plot split='{split_name}' in {predictions_csv}")
        split_frames.append((split_name, split_df))
        all_x_parts.append(split_df[true_col].to_numpy())
        all_x_parts.append(split_df[pred_col].to_numpy())

    all_x = np.concatenate(all_x_parts)
    axis_min = float(all_x.min())
    axis_max = float(all_x.max())
    axis_range = axis_max - axis_min
    padding = 1.0 if axis_range <= 0 else 0.02 * axis_range
    axis_min -= padding
    axis_max += padding
    if x_axis_range is None:
        x_axis_range = (axis_min, axis_max)
    if y_axis_range is None:
        y_axis_range = x_axis_range
    mae_location = mae_location.lower()
    mae_position_map = {
        "top_left": (0.01, 0.98, "left", "top"),
        "top_right": (0.99, 0.98, "right", "top"),
        "bottom_left": (0.01, 0.02, "left", "bottom"),
        "bottom_right": (0.99, 0.02, "right", "bottom"),
    }
    if mae_location not in mae_position_map:
        raise ValueError(f"Unsupported mae_location: {mae_location}")

    n_panels = len(split_frames)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.0, 1.65),
        dpi=600,
        sharex=True,
        sharey=True,
    )
    if n_panels == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    for idx, (ax, (split_name, split_df)) in enumerate(zip(axes_list, split_frames, strict=False)):
        split_title = split_name.replace("_", " ").title()
        ax.scatter(
            split_df[true_col],
            split_df[pred_col],
            c=_VALIDATION_SERIES_RED,
            s=_SCATTER_MARKER_AREA,
            alpha=0.7,
        )
        diag_min = max(x_axis_range[0], y_axis_range[0])
        diag_max = min(x_axis_range[1], y_axis_range[1])
        ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=_LINE_WIDTH)
        ax.set_xlim(*x_axis_range)
        ax.set_ylim(*y_axis_range)
        current_y_ticks = ax.get_yticks()
        default_x_ticks = current_y_ticks[
            (current_y_ticks >= x_axis_range[0]) & (current_y_ticks <= x_axis_range[1])
        ]
        if x_tick_values is not None:
            ax.set_xticks(np.asarray(x_tick_values, dtype=float))
        else:
            ax.set_xticks(default_x_ticks)
        if y_tick_values is not None:
            ax.set_yticks(np.asarray(y_tick_values, dtype=float))
        ax.margins(x=0.0, y=0.0)
        ax.set_aspect("auto")
        ylabel = f"Predicted {prop_label}" if idx == 0 else None
        _style_axis(ax, xlabel=f"True {prop_label}", ylabel=ylabel, title=split_title)
        ax.tick_params(axis="both", labelsize=12, width=_LINE_WIDTH, length=3.5, direction="out")
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontsize(12)
            label.set_fontweight("normal")
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(12)
        if idx > 0:
            ax.tick_params(labelleft=False)
        mae = float(np.mean(np.abs(split_df[true_col].to_numpy() - split_df[pred_col].to_numpy())))
        metrics_text = f"MAE: {mae:.1f} {prop_unit}".strip()
        mae_x, mae_y, mae_ha, mae_va = mae_position_map[mae_location]
        ax.text(
            mae_x,
            mae_y,
            metrics_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="normal",
            horizontalalignment=mae_ha,
            verticalalignment=mae_va,
        )

    fig.subplots_adjust(left=0.05, right=1.0, bottom=0.0, top=1.0, wspace=0.08)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.0)
    plt.show()


def _resolve_property_plot_label(resolved_property: str) -> tuple[str, str, str]:
    property_key = {
        "BP": "Boiling Point",
        "MP": "Melting Point",
        "VP": "$log_{10}(P^{sat} [torr])$",
        "OP": "Oxidation Potential",
        "pka": "pKa",
    }
    units_key = {
        "BP": "[K]",
        "MP": "[K]",
        "VP": " ",
        "OP": "[V]",
        "pka": " ",
    }
    prop_name = property_key.get(resolved_property, resolved_property)
    prop_unit = units_key.get(resolved_property, "")
    prop_label = f"{prop_name} {prop_unit}" if prop_unit.strip() else prop_name
    return prop_name, prop_unit, prop_label


def _resolve_parity_axis_limits(values: np.ndarray) -> tuple[float, float]:
    axis_min = float(np.min(values))
    axis_max = float(np.max(values))
    axis_range = axis_max - axis_min
    padding = 1.0 if axis_range <= 0 else 0.02 * axis_range
    return axis_min - padding, axis_max + padding


def _resolve_preferred_ticks(
    axis_min: float,
    axis_max: float,
    *,
    min_tick_count: int = 4,
    prefer_integer_ticks: bool = True,
) -> np.ndarray:
    axis_min = float(axis_min)
    axis_max = float(axis_max)
    if np.isclose(axis_min, axis_max):
        center = axis_min
        return np.asarray(
            [center - 1.5, center - 0.5, center + 0.5, center + 1.5],
            dtype=float,
        )

    if axis_max < axis_min:
        axis_min, axis_max = axis_max, axis_min

    axis_range = axis_max - axis_min
    target_tick_count = max(min_tick_count, 6)
    raw_step = axis_range / max(target_tick_count - 1, 1)

    nice_bases = [1, 2, 5, 10]
    base_exponent = int(np.floor(np.log10(raw_step))) if raw_step > 0 else 0
    candidate_steps: list[float] = []
    for exponent in range(base_exponent - 1, base_exponent + 3):
        scale = 10.0**exponent
        for base in nice_bases:
            step = float(base * scale)
            if step <= 0:
                continue
            if prefer_integer_ticks and step >= 1 and not np.isclose(step, round(step)):
                continue
            candidate_steps.append(step)

    best_ticks: np.ndarray | None = None
    best_score: tuple[float, float] | None = None
    for step in sorted(set(candidate_steps)):
        if prefer_integer_ticks and step < 1:
            continue
        start = np.floor(axis_min / step) * step
        end = np.ceil(axis_max / step) * step
        ticks = np.arange(start, end + (0.5 * step), step, dtype=float)
        ticks = np.round(ticks, 10)
        tick_count = len(ticks)
        if tick_count < min_tick_count:
            continue
        score = (abs(tick_count - target_tick_count), tick_count)
        if best_score is None or score < best_score:
            best_ticks = ticks
            best_score = score

    if best_ticks is not None:
        return best_ticks

    if prefer_integer_ticks:
        return _resolve_preferred_ticks(
            axis_min,
            axis_max,
            min_tick_count=min_tick_count,
            prefer_integer_ticks=False,
        )

    return np.linspace(axis_min, axis_max, num=min_tick_count, dtype=float)


def plot_stacked_test_parity_for_runs(
    plot_specs: list[dict],
    *,
    save_path: Path | None = None,
    default_split_col: str = "split",
    default_split_name: str = "test",
    min_tick_count: int = 4,
    prefer_integer_ticks: bool = True,
    figsize: tuple[float, float] = (3.35, 7.6),
) -> None:
    if not plot_specs:
        raise ValueError("Provide at least one plot specification.")

    _apply_publication_style()
    mae_position_map = {
        "top_left": (0.01, 0.98, "left", "top"),
        "top_right": (0.99, 0.98, "right", "top"),
        "bottom_left": (0.01, 0.02, "left", "bottom"),
        "bottom_right": (0.99, 0.02, "right", "bottom"),
    }

    fig, axes = plt.subplots(
        len(plot_specs),
        1,
        figsize=figsize,
        dpi=600,
        sharex=False,
        sharey=False,
    )
    axes_list = [axes] if len(plot_specs) == 1 else list(axes)

    for ax, spec in zip(axes_list, plot_specs, strict=False):
        predictions_csv = Path(spec["predictions_csv"])
        if not predictions_csv.exists():
            raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

        split_col = str(spec.get("split_col", default_split_col))
        split_name = str(spec.get("split_name", default_split_name))
        resolved_property = _resolve_property_token(
            property_col=spec.get("property_col"),
            property_code=spec.get("property_code"),
        )
        _, prop_unit, prop_label = _resolve_property_plot_label(resolved_property)

        pred_df = pd.read_csv(predictions_csv)
        true_col = f"{resolved_property}_true"
        pred_col = f"{resolved_property}_pred"
        required_cols = {split_col, true_col, pred_col}
        missing = required_cols - set(pred_df.columns)
        if missing:
            raise ValueError(f"Missing required columns for stacked parity plot: {sorted(missing)}")

        split_df = pred_df[pred_df[split_col] == split_name].copy()
        if split_df.empty:
            raise ValueError(f"No rows found for plot split='{split_name}' in {predictions_csv}")

        axis_values = np.concatenate(
            [
                split_df[true_col].to_numpy(dtype=float),
                split_df[pred_col].to_numpy(dtype=float),
            ]
        )
        default_axis_range = _resolve_parity_axis_limits(axis_values)
        x_axis_range = tuple(spec.get("x_axis_range", default_axis_range))
        y_axis_range = tuple(spec.get("y_axis_range", x_axis_range))

        x_tick_values = spec.get("x_tick_values")
        if x_tick_values is None:
            x_tick_values = _resolve_preferred_ticks(
                x_axis_range[0],
                x_axis_range[1],
                min_tick_count=min_tick_count,
                prefer_integer_ticks=prefer_integer_ticks,
            )
        y_tick_values = spec.get("y_tick_values")
        if y_tick_values is None:
            y_tick_values = _resolve_preferred_ticks(
                y_axis_range[0],
                y_axis_range[1],
                min_tick_count=min_tick_count,
                prefer_integer_ticks=prefer_integer_ticks,
            )

        ax.scatter(
            split_df[true_col],
            split_df[pred_col],
            c=_VALIDATION_SERIES_RED,
            s=_SCATTER_MARKER_AREA,
            alpha=0.7,
        )
        diag_min = max(x_axis_range[0], y_axis_range[0])
        diag_max = min(x_axis_range[1], y_axis_range[1])
        ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=_LINE_WIDTH)
        ax.set_xlim(*x_axis_range)
        ax.set_ylim(*y_axis_range)
        ax.set_xticks(np.asarray(x_tick_values, dtype=float))
        ax.set_yticks(np.asarray(y_tick_values, dtype=float))
        ax.margins(x=0.0, y=0.0)
        ax.set_aspect("auto")

        title = spec.get("title", resolved_property)
        xlabel = str(spec.get("xlabel", f"True {prop_label}"))
        linebreak_after_predicted = spec.get("ylabel_linebreak_after_predicted")
        if linebreak_after_predicted is None:
            linebreak_after_predicted = resolved_property in {"VP", "OP"}
        default_ylabel = (
            f"Predicted\n{prop_label}" if linebreak_after_predicted else f"Predicted {prop_label}"
        )
        ylabel = str(spec.get("ylabel", default_ylabel))
        _style_axis(ax, xlabel=xlabel, ylabel=ylabel, title=str(title))
        ax.tick_params(axis="both", labelsize=12, width=_LINE_WIDTH, length=3.5, direction="out")
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontsize(12)
            label.set_fontweight("normal")
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)

        mae = float(np.mean(np.abs(split_df[true_col].to_numpy() - split_df[pred_col].to_numpy())))
        metrics_text = f"MAE: {mae:.1f} {prop_unit}".strip()
        mae_location = str(spec.get("mae_location", "top_left")).lower()
        if mae_location not in mae_position_map:
            raise ValueError(f"Unsupported mae_location: {mae_location}")
        mae_x, mae_y, mae_ha, mae_va = mae_position_map[mae_location]
        ax.text(
            mae_x,
            mae_y,
            metrics_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="normal",
            horizontalalignment=mae_ha,
            verticalalignment=mae_va,
        )

    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.06, top=0.98, hspace=0.48)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.show()


def plot_training_density_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    density_split: str = "train",
    density_scale: float = 1.0,
    save_path: Path | None = None,
    property_code: str | None = None,
) -> None:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    parity_module = importlib.reload(_parity_with_train_density_bottom_module)
    parity_module.plot_training_density_only(
        model_output_data_path=str(predictions_csv),
        property_col=resolved_property,
        split_col=split_col,
        density_split=density_split,
        density_scale=density_scale,
        output_file=str(save_path) if save_path is not None else None,
    )


def plot_subgroups_for_run(
    predictions_csv: Path,
    split_col: str,
    smiles_col: str,
    split_name: str = "train",
    top_n: int = 18,
    max_per_row: int = 6,
    outfile: Path | None = None,
    title: str = "Top Molecular Subgroups",
):
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    pred_df = pd.read_csv(predictions_csv)
    df_unique = pred_df[pred_df[split_col] == split_name][[smiles_col]].drop_duplicates()
    return plot_top_functional_groups_in_grid(
        df_unique,
        smiles_col=smiles_col,
        max_per_row=max_per_row,
        top_n=top_n,
        outfile=str(outfile) if outfile is not None else None,
        title=title,
    )


def summarize_target_distribution(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    include_overall: bool = True,
    property_code: str | None = None,
) -> pd.DataFrame:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    pred_df = pd.read_csv(predictions_csv)
    true_col = f"{resolved_property}_true"
    if true_col not in pred_df.columns:
        raise ValueError(f"Missing required target column: {true_col}")

    frames = []
    if include_overall:
        overall_df = (
            pred_df[true_col]
            .agg(["min", "max", "mean", "median"])
            .to_frame()
            .T.assign(split="overall")
        )
        frames.append(overall_df)

    if split_col in pred_df.columns:
        split_df = (
            pred_df.groupby(split_col, dropna=False)[true_col]
            .agg(["min", "max", "mean", "median"])
            .reset_index()
            .rename(columns={split_col: "split"})
        )
        frames.append(split_df)

    if not frames:
        raise ValueError("Could not build target-distribution summary.")

    summary_df = pd.concat(frames, ignore_index=True)
    return summary_df[["split", "min", "max", "mean", "median"]]


def build_prediction_error_table(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    smiles_col: str = "SMILES",
    split_name: str | None = "test",
    outlier_z_threshold: float = 2.0,
    outlier_iqr_multiplier: float = 1.5,
    property_code: str | None = None,
) -> pd.DataFrame:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    pred_df = pd.read_csv(predictions_csv)
    true_col = f"{resolved_property}_true"
    pred_col = f"{resolved_property}_pred"
    required_cols = {smiles_col, split_col, true_col, pred_col}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for error table: {sorted(missing)}")

    if split_name is None or str(split_name).lower() == "all":
        split_df = pred_df.copy()
    else:
        split_df = pred_df[pred_df[split_col] == split_name].copy()
        if split_df.empty:
            raise ValueError(f"No rows found for split='{split_name}' in {predictions_csv}")

    split_df["residual"] = split_df[true_col] - split_df[pred_col]
    split_df["abs_error"] = split_df["residual"].abs()

    residual_std = split_df["residual"].std(ddof=0)
    if pd.isna(residual_std) or np.isclose(residual_std, 0.0):
        split_df["residual_z"] = 0.0
    else:
        split_df["residual_z"] = (split_df["residual"] - split_df["residual"].mean()) / residual_std

    q1 = split_df["abs_error"].quantile(0.25)
    q3 = split_df["abs_error"].quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr):
        abs_error_iqr_threshold = np.nan
    else:
        abs_error_iqr_threshold = q3 + (outlier_iqr_multiplier * iqr)
    split_df["abs_error_iqr_threshold"] = abs_error_iqr_threshold

    if pd.isna(abs_error_iqr_threshold):
        split_df["flag_large_abs_error"] = False
    else:
        split_df["flag_large_abs_error"] = split_df["abs_error"] > abs_error_iqr_threshold

    split_df["flag_large_residual_z"] = split_df["residual_z"].abs() > outlier_z_threshold
    split_df["flag_potential_outlier"] = split_df["flag_large_abs_error"] | split_df["flag_large_residual_z"]

    split_df["best_rank"] = split_df["abs_error"].rank(method="first", ascending=True).astype(int)
    split_df["worst_rank"] = split_df["abs_error"].rank(method="first", ascending=False).astype(int)
    split_df = split_df.sort_values("abs_error", ascending=False).reset_index(drop=True)
    split_df["error_rank"] = split_df.index + 1
    return split_df


def build_molecule_error_summary(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    smiles_col: str = "SMILES",
    split_name: str = "test",
    temperature_col: str = "T_C",
    outlier_z_threshold: float = 2.0,
    outlier_iqr_multiplier: float = 1.5,
    property_code: str | None = None,
) -> pd.DataFrame:
    error_df = build_prediction_error_table(
        predictions_csv=predictions_csv,
        property_col=property_col,
        split_col=split_col,
        smiles_col=smiles_col,
        split_name=split_name,
        outlier_z_threshold=outlier_z_threshold,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        property_code=property_code,
    )

    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    true_col = f"{resolved_property}_true"
    pred_col = f"{resolved_property}_pred"

    def _rmse(x: pd.Series) -> float:
        arr = x.to_numpy(dtype=float)
        return float(np.sqrt(np.mean(arr**2))) if len(arr) else float("nan")

    aggregations = {
        "n_entries": ("abs_error", "size"),
        "mean_true": (true_col, "mean"),
        "mean_pred": (pred_col, "mean"),
        "mae": ("abs_error", "mean"),
        "rmse": ("residual", _rmse),
        "mean_residual": ("residual", "mean"),
        "max_abs_error": ("abs_error", "max"),
    }
    if temperature_col in error_df.columns:
        aggregations.update(
            {
                "mean_temperature": (temperature_col, "mean"),
                "min_temperature": (temperature_col, "min"),
                "max_temperature": (temperature_col, "max"),
            }
        )

    molecule_df = (
        error_df.groupby(smiles_col, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("mae", ascending=True)
        .reset_index(drop=True)
    )
    molecule_df["best_molecule_rank"] = np.arange(1, len(molecule_df) + 1)
    molecule_df["worst_molecule_rank"] = molecule_df["mae"].rank(method="first", ascending=False).astype(int)
    molecule_df["split"] = split_name
    return molecule_df


def _resolve_bin_edges(
    values: pd.Series,
    bins: int | list[float] | np.ndarray,
    axis_name: str,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    if value_range is not None:
        vmin, vmax = value_range
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            raise ValueError(f"{axis_name} value_range must be finite and increasing.")
        if isinstance(bins, int):
            if bins < 1:
                raise ValueError(f"{axis_name} bins must be at least 1.")
            return np.linspace(vmin, vmax, bins + 1)

    if isinstance(bins, int):
        if bins < 1:
            raise ValueError(f"{axis_name} bins must be at least 1.")
        finite_values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(finite_values) == 0:
            raise ValueError(f"Cannot build {axis_name} bins because no finite values were found.")
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if np.isclose(vmin, vmax):
            padding = 0.5 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
            vmin -= padding
            vmax += padding
        return np.linspace(vmin, vmax, bins + 1)

    edges = np.asarray(bins, dtype=float)
    if len(edges) < 2:
        raise ValueError(f"{axis_name} bins must provide at least two bin edges.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"{axis_name} bin edges must be strictly increasing.")
    return edges


def plot_prediction_heatmaps_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    split_names: tuple[str, ...] | list[str] = ("train", "val", "test"),
    temperature_col: str = "T_C",
    pressure_source: str = "pred",
    temperature_bins: int | list[float] | np.ndarray = 50,
    pressure_bins: int | list[float] | np.ndarray = 50,
    pressure_range: tuple[float, float] | None = None,
    temperature_tick_values: list[float] | np.ndarray | None = None,
    pressure_tick_values: list[float] | np.ndarray | None = None,
    error_value: str = "absolute",
    cmap: str | None = None,
    show_split_titles: bool | None = None,
    show_figure_title: bool = True,
    save_path: Path | None = None,
    property_code: str | None = None,
) -> dict[str, pd.DataFrame]:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    _apply_publication_style()
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    pressure_source = pressure_source.lower()
    if pressure_source not in {"pred", "true"}:
        raise ValueError("pressure_source must be either 'pred' or 'true'.")
    error_value = error_value.lower()
    if error_value not in {"absolute", "residual"}:
        raise ValueError("error_value must be either 'absolute' or 'residual'.")
    resolved_cmap = _resolve_heatmap_cmap(error_value, cmap)

    pressure_col = f"{resolved_property}_{pressure_source}"
    true_col = f"{resolved_property}_true"
    pred_col = f"{resolved_property}_pred"
    pred_df = pd.read_csv(predictions_csv)
    required_cols = {split_col, temperature_col, pressure_col, true_col, pred_col}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for prediction heatmaps: {sorted(missing)}")

    split_names = tuple(split_names)
    property_key = {
        "BP": "Boiling Point",
        "MP": "Melting Point",
        "VP": "$log_{10}(P^{sat} [torr])$",
        "OP": "Oxidation Potential",
        "pka": "pKa",
    }
    units_key = {
        "BP": "[K]",
        "MP": "[K]",
        "VP": " ",
        "OP": "[V]",
        "pka": " ",
    }
    prop_name = property_key.get(resolved_property, resolved_property)
    prop_unit = units_key.get(resolved_property, "")
    prop_label = f"{prop_name} {prop_unit}" if prop_unit.strip() else prop_name

    heatmap_df = pred_df[pred_df[split_col].isin(split_names)].copy()
    heatmap_df = heatmap_df.dropna(subset=[temperature_col, pressure_col, true_col, pred_col])
    if heatmap_df.empty:
        raise ValueError(f"No rows found for requested splits: {split_names}")

    heatmap_df["residual"] = heatmap_df[true_col] - heatmap_df[pred_col]
    heatmap_df["abs_error"] = heatmap_df["residual"].abs()
    value_col = "abs_error" if error_value == "absolute" else "residual"

    temperature_edges = _resolve_bin_edges(heatmap_df[temperature_col], temperature_bins, "temperature")
    pressure_edges = _resolve_bin_edges(
        heatmap_df[pressure_col],
        pressure_bins,
        "pressure",
        value_range=pressure_range,
    )

    n_splits = len(split_names)
    if show_split_titles is None:
        show_split_titles = n_splits <= 2
    fig, axes = plt.subplots(
        1,
        n_splits,
        figsize=(6.0, 2.15),
        sharex=True,
        sharey=True,
        dpi=600,
    )
    if n_splits == 1:
        axes = [axes]

    heatmaps: dict[str, pd.DataFrame] = {}
    colorbar_label = "Mean absolute error" if error_value == "absolute" else "Mean residual (true - pred)"
    plot_payloads = []

    for split_name in split_names:
        split_df = heatmap_df[heatmap_df[split_col] == split_name]
        weighted_sum, _, _ = np.histogram2d(
            split_df[temperature_col].to_numpy(dtype=float),
            split_df[pressure_col].to_numpy(dtype=float),
            bins=[temperature_edges, pressure_edges],
            weights=split_df[value_col].to_numpy(dtype=float),
        )
        counts, _, _ = np.histogram2d(
            split_df[temperature_col].to_numpy(dtype=float),
            split_df[pressure_col].to_numpy(dtype=float),
            bins=[temperature_edges, pressure_edges],
        )
        values = np.divide(
            weighted_sum,
            counts,
            out=np.full_like(weighted_sum, np.nan, dtype=float),
            where=counts > 0,
        ).T

        heatmaps[split_name] = pd.DataFrame(
            values,
            index=pd.IntervalIndex.from_breaks(pressure_edges, closed="left"),
            columns=pd.IntervalIndex.from_breaks(temperature_edges, closed="left"),
        )

        plot_payloads.append((split_name, split_df, values))

    finite_chunks = [values[np.isfinite(values)] for _, _, values in plot_payloads if np.isfinite(values).any()]
    if finite_chunks:
        finite_values = np.concatenate(finite_chunks)
    else:
        finite_values = np.array([], dtype=float)

    shared_vmin = None
    shared_vmax = None
    if len(finite_values):
        if error_value == "absolute":
            shared_vmin = 0.0
            shared_vmax = float(np.nanquantile(finite_values, 0.90))
            data_max = float(np.nanmax(finite_values))
            if np.isclose(shared_vmax, 0.0):
                shared_vmax = data_max
            if np.isclose(shared_vmax, shared_vmin):
                padding = 0.5 if np.isclose(shared_vmax, 0.0) else abs(shared_vmax) * 0.05
                shared_vmax += padding
        else:
            abs_limit = float(np.nanquantile(np.abs(finite_values), 0.90))
            data_limit = float(np.nanmax(np.abs(finite_values)))
            if np.isclose(abs_limit, 0.0):
                abs_limit = data_limit
            if np.isclose(abs_limit, 0.0):
                abs_limit = 0.5
            shared_vmin = -abs_limit
            shared_vmax = abs_limit

    for ax, (split_name, split_df, values) in zip(axes, plot_payloads, strict=False):
        split_title = split_name.replace("_", " ").title() if show_split_titles else None
        im = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=[temperature_edges[0], temperature_edges[-1], pressure_edges[0], pressure_edges[-1]],
            cmap=resolved_cmap,
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
        ylabel = None
        if ax is axes[0]:
            label_prefix = "Predicted" if pressure_source == "pred" else "True"
            ylabel = f"{label_prefix} {prop_label}"
        _style_axis(
            ax,
            xlabel=r"Temperature ($^\circ$C)",
            ylabel=ylabel,
            title=split_title,
            despine=False,
        )
        ax.set_xlim(float(temperature_edges[0]), float(temperature_edges[-1]))
        ax.set_ylim(float(pressure_edges[0]), float(pressure_edges[-1]))
        if temperature_tick_values is not None:
            ax.set_xticks(np.asarray(temperature_tick_values, dtype=float))
        if pressure_tick_values is not None:
            ax.set_yticks(np.asarray(pressure_tick_values, dtype=float))
        ax.margins(x=0.0, y=0.0)
        ax.set_box_aspect(1)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.11, right=0.86, bottom=0.21, top=0.90, wspace=0.12)
    cax = fig.add_axes([0.885, 0.24, 0.018, 0.58])
    colorbar = fig.colorbar(im, cax=cax)
    _style_colorbar(colorbar, colorbar_label)

    if show_figure_title:
        pressure_label = "Predicted Pressure" if pressure_source == "pred" else "True Pressure"
        value_label = "Mean Absolute Error" if error_value == "absolute" else "Mean Residual"
        fig.suptitle(
            f"{resolved_property} {value_label} by Temperature and {pressure_label}",
            fontsize=_AXIS_LABEL_SIZE,
            fontweight="normal",
            y=0.98,
        )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

    return heatmaps


def _select_ranked_subset(error_df: pd.DataFrame, ranking: str, top_n: int) -> pd.DataFrame:
    ranking = ranking.lower()
    if ranking == "best":
        return error_df.sort_values("abs_error", ascending=True).head(top_n).copy()
    if ranking == "worst":
        return error_df.sort_values("abs_error", ascending=False).head(top_n).copy()
    raise ValueError("ranking must be either 'best' or 'worst'")


def _canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _normalize_fragment_for_counting(fragment_smiles: str) -> str | None:
    halogens = {"F", "Cl", "Br", "I"}
    if not isinstance(fragment_smiles, str):
        return None
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetSymbol() in halogens:
            atom.SetAtomicNum(17)
    return Chem.MolToSmiles(rw, canonical=True)


def _render_molecule_image(mol: Chem.Mol, size: tuple[int, int] = (620, 620), bond_line_width: float = 3.2):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = bond_line_width
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    import io
    from PIL import Image

    return Image.open(io.BytesIO(png))


def _build_fragment_presence_table(smiles_series: pd.Series) -> tuple[pd.DataFrame, dict]:
    canonical_smiles = sorted(
        {
            can
            for can in (_canonicalize_smiles(str(smi)) for smi in smiles_series.dropna().tolist())
            if can is not None
        }
    )
    if not canonical_smiles:
        return pd.DataFrame(), {}

    input_df = pd.DataFrame({"canonical_smiles": canonical_smiles})
    _, frag_df = fragment_molecules_to_dataframe(input_df, smiles_col="canonical_smiles", enforce_exactly_one=True)
    if frag_df is None or frag_df.empty:
        return pd.DataFrame(index=canonical_smiles), {}

    frag_df = frag_df.copy()
    frag_df["fragment_norm"] = frag_df["fragment_smiles"].map(_normalize_fragment_for_counting)
    frag_df = frag_df.dropna(subset=["fragment_norm", "fragment_label"])
    if frag_df.empty:
        return pd.DataFrame(index=canonical_smiles), {}

    frag_df["fragment_key"] = list(zip(frag_df["fragment_norm"], frag_df["fragment_label"]))
    display_smiles_by_key = (
        frag_df.drop_duplicates("fragment_key")
        .set_index("fragment_key")["fragment_norm"]
        .to_dict()
    )

    presence_df = (
        frag_df[["parent_smiles", "fragment_key"]]
        .drop_duplicates()
        .assign(present=1)
        .pivot(index="parent_smiles", columns="fragment_key", values="present")
        .fillna(0)
        .astype(int)
    )
    presence_df = presence_df.reindex(canonical_smiles, fill_value=0)
    return presence_df, display_smiles_by_key


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index, dtype=float)

    n = len(valid)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = valid.iloc[i] * n / rank
        running = min(running, candidate)
        adjusted[i] = running

    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    q_values.loc[valid.index] = np.clip(adjusted, 0.0, 1.0)
    return q_values


def _count_heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


_HETEROATOM_TO_CARBON_SYMBOLS = {"O", "N", "S", "I", "Br", "Cl", "F"}


def _count_carbons_and_selected_heteroatoms(smiles: str) -> dict[str, float | int]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            "carbon_count": np.nan,
            "heteroatom_count": np.nan,
            "heteroatom_to_carbon_ratio": np.nan,
        }

    atom_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    carbon_count = int(atom_counts.get("C", 0))
    heteroatom_count = int(sum(atom_counts.get(symbol, 0) for symbol in _HETEROATOM_TO_CARBON_SYMBOLS))
    ratio = np.nan if carbon_count == 0 else heteroatom_count / carbon_count
    return {
        "carbon_count": carbon_count,
        "heteroatom_count": heteroatom_count,
        "heteroatom_to_carbon_ratio": ratio,
    }


def plot_atom_count_hist(
    predictions_csv: Path,
    split_col: str,
    smiles_col: str,
    max_atoms: int = 40,
    hist_bins: int = 20,
    figsize: tuple[float, float] = (3.5, 3.0),
    save_path: Path | None = None,
) -> None:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    _apply_publication_style()
    pred_df = pd.read_csv(predictions_csv)
    df_train = pred_df[pred_df[split_col] == "train"]
    df_unique = df_train[[smiles_col]].drop_duplicates().copy()
    df_unique["atom_count"] = df_unique[smiles_col].apply(_count_heavy_atoms)
    df_unique = df_unique[df_unique["atom_count"] <= max_atoms]

    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    sns.histplot(
        data=df_unique,
        x="atom_count",
        bins=hist_bins,
        ax=ax,
        color=_TRAIN_DENSITY_BLUE,
        linewidth=_LINE_WIDTH,
    )
    for patch in ax.patches:
        patch.set_linewidth(_LINE_WIDTH)
    _style_axis(ax, xlabel="Number of Heavy Atoms", ylabel="Count")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

    print("Molecule size summary:")
    print(df_unique["atom_count"].describe())


def plot_heteroatom_pie(
    predictions_csv: Path,
    split_col: str,
    smiles_col: str,
    save_path: Path | None = None,
) -> None:
    _apply_publication_style()
    halogens = {"F", "Cl", "Br", "I"}
    atom_colors = {
        "O": "indianred",
        "N": "royalblue",
        "S": "goldenrod",
        "P": "darkorange",
        "Si": "sienna",
        "X": "seagreen",
    }

    pred_df = pd.read_csv(predictions_csv)
    df_train = pred_df[pred_df[split_col] == "train"]
    df_unique = df_train[[smiles_col]].drop_duplicates()

    element_counts = Counter()
    for smi in df_unique[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        elements = {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() not in ["C", "H"]}
        for el in elements:
            element_counts["X" if el in halogens else el] += 1

    element_df = pd.DataFrame.from_dict(element_counts, orient="index", columns=["Compounds"])
    element_df = element_df.sort_values("Compounds", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    sizes = element_df["Compounds"]
    labels = element_df.index
    colors = [atom_colors.get(lbl, "gray") for lbl in labels]

    total = sizes.sum()
    pct = sizes / total * 100

    wedges, _ = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=_LINE_WIDTH),
    )

    for i, wedge in enumerate(wedges):
        if pct.iloc[i] > 5:
            ang = (wedge.theta2 + wedge.theta1) / 2
            x = np.cos(np.deg2rad(ang))
            y = np.sin(np.deg2rad(ang))
            ax.text(
                x * 0.6,
                y * 0.6,
                f"{labels[i]}",
                ha="center",
                va="center",
                fontsize=40,
                fontweight="bold",
            )

    ax.set_title("Relative Heteroatom Frequency", fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def plot_best_worst_molecules_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    smiles_col: str = "SMILES",
    split_name: str = "test",
    top_n: int = 10,
    save_dir: Path | None = None,
    bottom_n: int | None = None,
    save_path: Path | None = None,
    temperature_col: str = "T_C",
    label_mode: str = "mean_true_pred",
    property_code: str | None = None,
) -> dict:
    """
    Plots the top-N best and worst molecules by mean absolute error (MAE),
    aggregating across all rows for a given molecule (grouped by `smiles_col`).
    """
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    _apply_publication_style()
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    grouped = build_molecule_error_summary(
        predictions_csv=predictions_csv,
        property_col=resolved_property,
        split_col=split_col,
        smiles_col=smiles_col,
        split_name=split_name,
        temperature_col=temperature_col,
    )

    # Backward-compatibility: some notebooks pass `save_path` instead of `save_dir`.
    if save_dir is None and save_path is not None:
        save_dir = save_path.parent

    label_mode = label_mode.lower()
    if label_mode not in {"mean_true_pred", "mae_count"}:
        raise ValueError("label_mode must be either 'mean_true_pred' or 'mae_count'.")

    worst_n = top_n if bottom_n is None else bottom_n

    grouped = grouped.sort_values("mae", ascending=True).reset_index(drop=True)
    best_df = grouped.head(top_n).copy()
    worst_df = grouped.sort_values("mae", ascending=False).head(worst_n).copy()

    def _plot_grid(title: str, df_rows: pd.DataFrame, outfile_name: str) -> None:
        mols = []
        legends = []
        for _, row in df_rows.iterrows():
            smi = str(row[smiles_col])
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mols.append(mol)
            if label_mode == "mae_count":
                legends.append(
                    f"MAE: {row['mae']:.3f}\n"
                    f"No. of Temperatures: {int(row['n_entries'])}"
                )
            else:
                legends.append(
                    f"True: {row['mean_true']:.3f}\n"
                    f"Predicted: {row['mean_pred']:.3f}"
                )

        if not mols:
            raise ValueError(f"No valid molecules to plot for: {title}")

        n_cols = min(5, len(mols))
        n_rows = int(np.ceil(len(mols) / n_cols))
        fig_width = max(3.5, 2.35 * n_cols)
        row_height = 3.0
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, row_height * n_rows),
            dpi=600,
        )

        if n_rows == 1 and n_cols == 1:
            axes_list = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes_list = list(axes)
        else:
            axes_list = [ax for row in axes for ax in row]

        for i, (mol, legend) in enumerate(zip(mols, legends, strict=False)):
            ax = axes_list[i]
            img = _render_molecule_image(mol, size=(700, 700), bond_line_width=3.4)
            ax.imshow(np.asarray(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(legend, fontsize=15, fontweight="normal", labelpad=10)

        for ax in axes_list[len(mols) :]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle(title, fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
        fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.6, w_pad=1.0)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / outfile_name
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.show()

    _plot_grid(
        title=f"Top {min(top_n, len(best_df))} Best Molecules (lowest mean |error|) - {split_name} split",
        df_rows=best_df,
        outfile_name=f"{resolved_property}_{split_name}_best_{top_n}_molecules.png",
    )
    _plot_grid(
        title=f"Top {min(worst_n, len(worst_df))} Worst Molecules (highest mean |error|) - {split_name} split",
        df_rows=worst_df,
        outfile_name=f"{resolved_property}_{split_name}_worst_{worst_n}_molecules.png",
    )

    return {
        "best": best_df,
        "worst": worst_df,
        "grouped": grouped,
    }


def plot_best_worst_target_histograms_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    smiles_col: str = "SMILES",
    split_name: str = "test",
    top_n: int = 50,
    bottom_n: int | None = None,
    target_source: str = "true",
    bins: int | list[float] | np.ndarray = 24,
    save_path: Path | None = None,
    temperature_col: str = "T_C",
    property_code: str | None = None,
) -> pd.DataFrame:
    """
    Plot the target-value distributions for the best and worst molecules.

    Molecules are ranked by mean absolute error across all rows in `split_name`.
    The histogram then includes every target row for the selected molecules, which
    is useful for temperature-dependent properties with multiple rows per molecule.
    """
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    _apply_publication_style()
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    target_source = target_source.lower()
    if target_source not in {"true", "pred"}:
        raise ValueError("target_source must be either 'true' or 'pred'.")

    target_col = f"{resolved_property}_{target_source}"
    pred_df = pd.read_csv(predictions_csv)
    required_cols = {smiles_col, split_col, target_col, f"{resolved_property}_true", f"{resolved_property}_pred"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for target histograms: {sorted(missing)}")

    split_df = pred_df[pred_df[split_col] == split_name].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split='{split_name}' in {predictions_csv}")

    worst_n = top_n if bottom_n is None else bottom_n
    grouped = build_molecule_error_summary(
        predictions_csv=predictions_csv,
        property_col=resolved_property,
        split_col=split_col,
        smiles_col=smiles_col,
        split_name=split_name,
        temperature_col=temperature_col,
    )
    best_molecules = grouped.sort_values("mae", ascending=True).head(top_n).copy()
    worst_molecules = grouped.sort_values("mae", ascending=False).head(worst_n).copy()

    frames = []
    selections = [
        ("Best", best_molecules, "#287c71"),
        ("Worst", worst_molecules, "#c65a45"),
    ]
    for rank_label, molecule_df, _ in selections:
        selected = split_df[split_df[smiles_col].isin(molecule_df[smiles_col])].copy()
        selected["rank_group"] = rank_label
        selected["target_value"] = pd.to_numeric(selected[target_col], errors="coerce")
        selected["target_value"] = selected["target_value"].replace([np.inf, -np.inf], np.nan)
        selected = selected.dropna(subset=["target_value"])
        selected["selected_molecule_count"] = len(molecule_df)
        frames.append(selected)

    hist_df = pd.concat(frames, ignore_index=True)
    if hist_df.empty:
        raise ValueError("No finite target values found for selected best/worst molecules.")

    bin_edges = _resolve_bin_edges(hist_df["target_value"], bins, "target")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=600, sharex=True, sharey=True)

    summary_records = []
    for ax, (rank_label, molecule_df, color) in zip(axes, selections, strict=False):
        plot_df = hist_df[hist_df["rank_group"] == rank_label]
        ax.hist(
            plot_df["target_value"],
            bins=bin_edges,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=_LINE_WIDTH,
        )
        median_value = plot_df["target_value"].median()
        ax.axvline(median_value, color="black", linestyle="--", linewidth=_LINE_WIDTH)
        _style_axis(
            ax,
            xlabel=f"{target_source.capitalize()} {resolved_property} (log10 pressure)",
            title=f"{rank_label} {len(molecule_df)} molecules\n{len(plot_df)} target points",
        )

        stats = plot_df["target_value"].agg(["min", "max", "mean", "median"])
        ax.text(
            0.98,
            0.96,
            f"min {stats['min']:.2f}\nmax {stats['max']:.2f}\nmed {stats['median']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_TEXT_SIZE,
            fontweight="normal",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"),
        )
        summary_records.append(
            {
                "rank_group": rank_label.lower(),
                "n_molecules": len(molecule_df),
                "n_target_points": len(plot_df),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "mean": float(stats["mean"]),
                "median": float(stats["median"]),
            }
        )

    axes[0].set_ylabel("Count", fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
    fig.suptitle(
        f"{resolved_property} Target Range for Best vs Worst Predictions ({split_name} split)",
        fontsize=_AXIS_LABEL_SIZE,
        fontweight="normal",
        y=1.04,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

    return pd.DataFrame(summary_records)


def plot_best_worst_heteroatom_carbon_ratio_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    smiles_col: str = "SMILES",
    split_name: str = "test",
    top_n: int = 50,
    bottom_n: int | None = None,
    bins: int | list[float] | np.ndarray = 16,
    save_path: Path | None = None,
    temperature_col: str = "T_C",
    remove_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    property_code: str | None = None,
) -> pd.DataFrame:
    """
    Plot heteroatom-to-carbon ratios for the best and worst molecules.

    Heteroatoms are counted as O, N, S, I, Br, Cl, and F. Molecules are ranked
    by mean absolute error across all rows in `split_name`, then counted once
    per selected molecule. Outliers are hidden from the figure by default, but
    kept in the returned table with a flag.
    """
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    _apply_publication_style()
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    worst_n = top_n if bottom_n is None else bottom_n
    grouped = build_molecule_error_summary(
        predictions_csv=predictions_csv,
        property_col=resolved_property,
        split_col=split_col,
        smiles_col=smiles_col,
        split_name=split_name,
        temperature_col=temperature_col,
    )

    selections = [
        ("Best", grouped.sort_values("mae", ascending=True).head(top_n).copy(), "#287c71"),
        ("Worst", grouped.sort_values("mae", ascending=False).head(worst_n).copy(), "#c65a45"),
    ]

    metric_frames = []
    for rank_label, molecule_df, _ in selections:
        metric_df = molecule_df.copy()
        count_df = pd.DataFrame(
            metric_df[smiles_col].map(_count_carbons_and_selected_heteroatoms).tolist(),
            index=metric_df.index,
        )
        metric_df = pd.concat([metric_df, count_df], axis=1)
        metric_df["rank_group"] = rank_label
        metric_frames.append(metric_df)

    ratio_df = pd.concat(metric_frames, ignore_index=True)
    ratio_df["ratio_outlier_for_histogram"] = False
    ratio_df["ratio_plotted_in_histogram"] = ratio_df["heteroatom_to_carbon_ratio"].notna()

    finite_df = ratio_df.dropna(subset=["heteroatom_to_carbon_ratio"]).copy()
    if finite_df.empty:
        raise ValueError("No finite heteroatom-to-carbon ratios found for selected molecules.")

    if remove_outliers:
        for rank_label in finite_df["rank_group"].dropna().unique():
            group_values = finite_df.loc[
                finite_df["rank_group"] == rank_label,
                "heteroatom_to_carbon_ratio",
            ]
            q1 = group_values.quantile(0.25)
            q3 = group_values.quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower_bound = q1 - (outlier_iqr_multiplier * iqr)
                upper_bound = q3 + (outlier_iqr_multiplier * iqr)
                outlier_mask = (group_values < lower_bound) | (group_values > upper_bound)
            else:
                median_value = group_values.median()
                deviation = (group_values - median_value).abs()
                threshold = deviation.quantile(0.95)
                outlier_mask = (deviation > threshold) & (deviation > 0)

            outlier_index = group_values.index[outlier_mask.fillna(False)]
            ratio_df.loc[outlier_index, "ratio_outlier_for_histogram"] = True
            ratio_df.loc[outlier_index, "ratio_plotted_in_histogram"] = False

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=600, sharex=False, sharey=False)

    for ax, (rank_label, molecule_df, color) in zip(axes, selections, strict=False):
        finite_group_df = ratio_df[
            (ratio_df["rank_group"] == rank_label)
            & ratio_df["heteroatom_to_carbon_ratio"].notna()
        ].copy()
        if finite_group_df.empty:
            ax.axis("off")
            ax.set_title(
                f"{rank_label} {len(molecule_df)} molecules\n0 finite ratios",
                fontsize=_AXIS_LABEL_SIZE,
                fontweight="normal",
            )
            continue

        group_df = finite_group_df[finite_group_df["ratio_plotted_in_histogram"]].copy()
        if group_df.empty:
            group_df = finite_group_df.copy()
            hidden_count = 0
        else:
            hidden_count = len(finite_group_df) - len(group_df)
        bin_edges = _resolve_bin_edges(
            group_df["heteroatom_to_carbon_ratio"],
            bins,
            "heteroatom-to-carbon ratio",
        )
        ax.hist(
            group_df["heteroatom_to_carbon_ratio"],
            bins=bin_edges,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=_LINE_WIDTH,
        )
        stats = finite_group_df["heteroatom_to_carbon_ratio"].agg(["min", "max", "mean", "median"])
        ax.axvline(
            stats["median"],
            color="black",
            linestyle="--",
            linewidth=_LINE_WIDTH,
            label=f"Median (all): {stats['median']:.2f}",
        )
        ax.axvline(
            stats["mean"],
            color="#1f4e79",
            linestyle="-.",
            linewidth=_LINE_WIDTH,
            label=f"Mean (all): {stats['mean']:.2f}",
        )
        hidden_label = f"{hidden_count} outlier hidden" if hidden_count == 1 else f"{hidden_count} outliers hidden"
        _style_axis(
            ax,
            xlabel="Heteroatoms / carbons",
            title=f"{rank_label} {len(molecule_df)} molecules\n{len(group_df)} plotted, {hidden_label}",
        )
        _style_legend(ax, loc="upper left", handlelength=2.4)
        ax.text(
            0.98,
            0.96,
            (
                f"all incl. outliers min {stats['min']:.2f}\n"
                f"all incl. outliers max {stats['max']:.2f}\n"
                f"all incl. outliers mean {stats['mean']:.2f}\n"
                f"all incl. outliers med {stats['median']:.2f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_TEXT_SIZE,
            fontweight="normal",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"),
        )

    axes[0].set_ylabel("Count", fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
    fig.suptitle(
        f"Heteroatom:Carbon Ratio for Best vs Worst Predictions ({split_name} split)",
        fontsize=_AXIS_LABEL_SIZE,
        fontweight="normal",
        y=1.04,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

    return ratio_df


def plot_train_best_worst_heteroatom_carbon_ratio_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    smiles_col: str = "SMILES",
    split_name: str = "test",
    train_split_name: str = "train",
    top_n: int = 50,
    bottom_n: int | None = None,
    bins: int | list[float] | np.ndarray = 16,
    save_path: Path | None = None,
    temperature_col: str = "T_C",
    remove_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    property_code: str | None = None,
) -> pd.DataFrame:
    """
    Plot heteroatom-to-carbon ratios for training molecules alongside the best
    and worst predicted molecules.

    The training distribution counts each unique training-set molecule once.
    Best and worst molecules are ranked by mean absolute error across all rows
    in `split_name`, then counted once per selected molecule.
    """
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    _apply_publication_style()
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    pred_df = pd.read_csv(predictions_csv)
    required_cols = {split_col, smiles_col}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for heteroatom/carbon ratio plot: {sorted(missing)}")

    worst_n = top_n if bottom_n is None else bottom_n
    grouped = build_molecule_error_summary(
        predictions_csv=predictions_csv,
        property_col=resolved_property,
        split_col=split_col,
        smiles_col=smiles_col,
        split_name=split_name,
        temperature_col=temperature_col,
    )

    train_df = pred_df[pred_df[split_col] == train_split_name][[smiles_col]].drop_duplicates().copy()
    if train_df.empty:
        raise ValueError(f"No molecules found for training split '{train_split_name}'.")

    selections = [
        ("Training", train_df, "#4c78a8"),
        ("Best", grouped.sort_values("mae", ascending=True).head(top_n).copy(), "#287c71"),
        ("Worst", grouped.sort_values("mae", ascending=False).head(worst_n).copy(), "#c65a45"),
    ]

    metric_frames = []
    for rank_label, molecule_df, _ in selections:
        metric_df = molecule_df.copy()
        count_df = pd.DataFrame(
            metric_df[smiles_col].map(_count_carbons_and_selected_heteroatoms).tolist(),
            index=metric_df.index,
        )
        metric_df = pd.concat([metric_df, count_df], axis=1)
        metric_df["rank_group"] = rank_label
        metric_df["selected_molecule_count"] = len(molecule_df)
        metric_frames.append(metric_df)

    ratio_df = pd.concat(metric_frames, ignore_index=True)
    ratio_df["ratio_outlier_for_histogram"] = False
    ratio_df["ratio_plotted_in_histogram"] = ratio_df["heteroatom_to_carbon_ratio"].notna()

    finite_df = ratio_df.dropna(subset=["heteroatom_to_carbon_ratio"]).copy()
    if finite_df.empty:
        raise ValueError("No finite heteroatom-to-carbon ratios found for selected molecules.")

    if remove_outliers:
        for rank_label in finite_df["rank_group"].dropna().unique():
            group_values = finite_df.loc[
                finite_df["rank_group"] == rank_label,
                "heteroatom_to_carbon_ratio",
            ]
            q1 = group_values.quantile(0.25)
            q3 = group_values.quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower_bound = q1 - (outlier_iqr_multiplier * iqr)
                upper_bound = q3 + (outlier_iqr_multiplier * iqr)
                outlier_mask = (group_values < lower_bound) | (group_values > upper_bound)
            else:
                median_value = group_values.median()
                deviation = (group_values - median_value).abs()
                threshold = deviation.quantile(0.95)
                outlier_mask = (deviation > threshold) & (deviation > 0)

            outlier_index = group_values.index[outlier_mask.fillna(False)]
            ratio_df.loc[outlier_index, "ratio_outlier_for_histogram"] = True
            ratio_df.loc[outlier_index, "ratio_plotted_in_histogram"] = False

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.4), dpi=600, sharex=False, sharey=False)

    for ax, (rank_label, molecule_df, color) in zip(axes, selections, strict=False):
        finite_group_df = ratio_df[
            (ratio_df["rank_group"] == rank_label)
            & ratio_df["heteroatom_to_carbon_ratio"].notna()
        ].copy()
        if finite_group_df.empty:
            ax.axis("off")
            ax.set_title(f"{rank_label}\n0 finite ratios", fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
            continue

        group_df = finite_group_df[finite_group_df["ratio_plotted_in_histogram"]].copy()
        if group_df.empty:
            group_df = finite_group_df.copy()
            hidden_count = 0
        else:
            hidden_count = len(finite_group_df) - len(group_df)

        bin_edges = _resolve_bin_edges(
            group_df["heteroatom_to_carbon_ratio"],
            bins,
            "heteroatom-to-carbon ratio",
        )
        ax.hist(
            group_df["heteroatom_to_carbon_ratio"],
            bins=bin_edges,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=_LINE_WIDTH,
        )
        stats = finite_group_df["heteroatom_to_carbon_ratio"].agg(["min", "max", "mean", "median"])
        ax.axvline(
            stats["median"],
            color="black",
            linestyle="--",
            linewidth=_LINE_WIDTH,
            label=f"Median (all): {stats['median']:.2f}",
        )
        ax.axvline(
            stats["mean"],
            color="#1f4e79",
            linestyle="-.",
            linewidth=_LINE_WIDTH,
            label=f"Mean (all): {stats['mean']:.2f}",
        )
        hidden_label = f"{hidden_count} outlier hidden" if hidden_count == 1 else f"{hidden_count} outliers hidden"
        _style_axis(
            ax,
            xlabel="Heteroatoms / carbons",
            title=f"{rank_label} {len(molecule_df)} molecules\n{len(group_df)} plotted, {hidden_label}",
        )
        _style_legend(ax, loc="upper left", handlelength=2.4)
        ax.text(
            0.98,
            0.96,
            (
                f"all incl. outliers min {stats['min']:.2f}\n"
                f"all incl. outliers max {stats['max']:.2f}\n"
                f"all incl. outliers mean {stats['mean']:.2f}\n"
                f"all incl. outliers med {stats['median']:.2f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_TEXT_SIZE,
            fontweight="normal",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"),
        )

    axes[0].set_ylabel("Count", fontsize=_AXIS_LABEL_SIZE, fontweight="normal")
    fig.suptitle(
        f"Heteroatom:Carbon Ratio in Training Data vs Best/Worst Predictions ({split_name} split)",
        fontsize=_AXIS_LABEL_SIZE,
        fontweight="normal",
        y=1.04,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

    return ratio_df


def plot_ranked_subgroups_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    smiles_col: str = "SMILES",
    split_name: str = "test",
    ranking: str | None = None,
    top_n_molecules: int = 25,
    top_n_subgroups: int = 18,
    max_per_row: int = 6,
    title: str | None = None,
    outfile: Path | None = None,
    rank_mode: str | None = None,
    rank_kind: str | None = None,
    property_code: str | None = None,
):
    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    resolved_ranking = _resolve_ranking_token(ranking=ranking, rank_mode=rank_mode, rank_kind=rank_kind)
    error_df = build_prediction_error_table(
        predictions_csv=predictions_csv,
        property_col=resolved_property,
        split_col=split_col,
        smiles_col=smiles_col,
        split_name=split_name,
    )
    subset_df = _select_ranked_subset(error_df, ranking=resolved_ranking, top_n=top_n_molecules)
    subset_unique = subset_df[[smiles_col]].drop_duplicates().copy()

    resolved_title = title
    if resolved_title is None:
        resolved_title = (
            f"Top Molecular Subgroups: {resolved_ranking.capitalize()} "
            f"{min(len(subset_unique), top_n_molecules)} {split_name.capitalize()} Molecules"
        )
    plot_result = plot_top_functional_groups_in_grid(
        subset_unique,
        smiles_col=smiles_col,
        max_per_row=max_per_row,
        top_n=top_n_subgroups,
        outfile=str(outfile) if outfile is not None else None,
        title=resolved_title,
    )
    return {
        "molecule_subset": subset_df,
        "unique_molecules": subset_unique,
        "plot_result": plot_result,
    }


def compare_ranked_subgroups_to_training_for_run(
    predictions_csv: Path,
    property_col: str | None = None,
    split_col: str = "split",
    smiles_col: str = "SMILES",
    split_name: str = "test",
    ranking: str | None = None,
    top_n_molecules: int = 50,
    baseline_split: str = "train",
    min_selected_count: int = 2,
    rank_mode: str | None = None,
    rank_kind: str | None = None,
    property_code: str | None = None,
) -> pd.DataFrame:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    resolved_property = _resolve_property_token(property_col=property_col, property_code=property_code)
    resolved_ranking = _resolve_ranking_token(ranking=ranking, rank_mode=rank_mode, rank_kind=rank_kind)

    pred_df = pd.read_csv(predictions_csv)
    if split_col not in pred_df.columns or smiles_col not in pred_df.columns:
        raise ValueError(f"Missing required columns: {split_col}, {smiles_col}")

    error_df = build_prediction_error_table(
        predictions_csv=predictions_csv,
        property_col=resolved_property,
        split_col=split_col,
        smiles_col=smiles_col,
        split_name=split_name,
    )
    selected_df = _select_ranked_subset(error_df, ranking=resolved_ranking, top_n=top_n_molecules)
    baseline_df = pred_df[pred_df[split_col] == baseline_split][[smiles_col]].drop_duplicates().copy()
    if baseline_df.empty:
        raise ValueError(f"No rows found for baseline split='{baseline_split}' in {predictions_csv}")

    selected_presence, selected_display = _build_fragment_presence_table(selected_df[smiles_col])
    baseline_presence, baseline_display = _build_fragment_presence_table(baseline_df[smiles_col])

    all_columns = sorted(set(selected_presence.columns).union(set(baseline_presence.columns)))
    if not all_columns:
        return pd.DataFrame(
            columns=[
                "fragment_smiles",
                "fragment_label",
                "selected_count",
                "selected_rate",
                "train_count",
                "train_rate",
                "delta_rate",
                "p_value",
                "q_value",
            ]
        )

    selected_presence = selected_presence.reindex(columns=all_columns, fill_value=0)
    baseline_presence = baseline_presence.reindex(columns=all_columns, fill_value=0)
    display_smiles_by_key = {}
    display_smiles_by_key.update(baseline_display)
    display_smiles_by_key.update(selected_display)

    records = []
    for key in all_columns:
        selected_vals = selected_presence[key].to_numpy(dtype=float)
        baseline_vals = baseline_presence[key].to_numpy(dtype=float)

        selected_mean = float(selected_vals.mean()) if len(selected_vals) else float("nan")
        baseline_mean = float(baseline_vals.mean()) if len(baseline_vals) else float("nan")

        if np.allclose(selected_vals, selected_vals[0]) and np.allclose(baseline_vals, baseline_vals[0]):
            p_value = 1.0 if np.isclose(selected_mean, baseline_mean) else 0.0
        else:
            test_result = ttest_ind(selected_vals, baseline_vals, equal_var=False, nan_policy="omit")
            p_value = float(test_result.pvalue) if pd.notna(test_result.pvalue) else np.nan

        records.append(
            {
                "fragment_smiles": display_smiles_by_key.get(key, key[0]),
                "fragment_label": key[1],
                "selected_count": int(selected_vals.sum()),
                "selected_rate": selected_mean,
                "train_count": int(baseline_vals.sum()),
                "train_rate": baseline_mean,
                "delta_rate": selected_mean - baseline_mean,
                "p_value": p_value,
            }
        )

    result_df = pd.DataFrame(records)
    result_df["q_value"] = _benjamini_hochberg(result_df["p_value"])
    result_df["ranking"] = resolved_ranking
    result_df["selected_split"] = split_name
    result_df["baseline_split"] = baseline_split
    result_df["selected_n_molecules"] = len(selected_presence)
    result_df["train_n_molecules"] = len(baseline_presence)

    result_df = result_df[result_df["selected_count"] >= min_selected_count].copy()
    result_df = result_df.sort_values(
        ["q_value", "delta_rate", "selected_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return result_df
