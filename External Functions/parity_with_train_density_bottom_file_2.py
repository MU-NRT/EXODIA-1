def _publication_style():
    font_fallbacks = ["Arial", "DejaVu Sans"]
    axis_label_size = 12
    text_size = 10
    line_width = 0.7
    marker_size = 4
    save_dpi = 600
    parity_point_color = "#c65a45"

    return {
        "font_fallbacks": font_fallbacks,
        "axis_label_size": axis_label_size,
        "text_size": text_size,
        "line_width": line_width,
        "marker_size": marker_size,
        "scatter_marker_area": marker_size**2,
        "save_dpi": save_dpi,
        "parity_point_color": parity_point_color,
    }


def _configure_matplotlib(plt):
    style = _publication_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": style["font_fallbacks"],
            "axes.labelsize": style["axis_label_size"],
            "axes.labelweight": "normal",
            "axes.titlesize": style["axis_label_size"],
            "axes.titleweight": "normal",
            "xtick.labelsize": style["text_size"],
            "ytick.labelsize": style["text_size"],
            "legend.fontsize": style["text_size"],
            "lines.linewidth": style["line_width"],
            "lines.markersize": style["marker_size"],
            "axes.grid": False,
            "axes.linewidth": style["line_width"],
            "xtick.major.width": style["line_width"],
            "ytick.major.width": style["line_width"],
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "patch.linewidth": style["line_width"],
        }
    )
    return style


def _style_axis(ax, axis_label_size, text_size, line_width, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_label_size, fontweight="normal")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=axis_label_size, fontweight="normal")
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=text_size, width=line_width, length=3.5, direction="out")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("normal")


def _resolve_property_token(property_col=None, property_code=None):
    if property_col is not None and property_code is not None and property_col != property_code:
        raise ValueError(
            f"Conflicting property identifiers provided: property_col={property_col!r}, "
            f"property_code={property_code!r}"
        )
    resolved = property_col if property_col is not None else property_code
    if resolved is None:
        raise ValueError("Provide either `property_col` or `property_code`.")
    return resolved


def _prepare_parity_inputs(
    model_output_data_path,
    property_col=None,
    split_col="split",
    split_name="test",
    density_split="train",
    property_key=None,
    units_key=None,
    property_code=None,
):
    import numpy as np
    import pandas as pd

    if property_key is None:
        property_key = {
            "BP": "Boiling Point",
            "MP": "Melting Point",
            "VP": "$log_{10}(P^{sat} [torr])$",
            "OP": "Oxidation Potential",
            "pka": "pKa",
        }
    if units_key is None:
        units_key = {
            "BP": "[K]",
            "MP": "[K]",
            "VP": " ",
            "OP": "[V]",
            "pka": " ",
        }

    property_col = _resolve_property_token(property_col=property_col, property_code=property_code)
    df = pd.read_csv(model_output_data_path)
    if split_col not in df.columns:
        raise ValueError(f"Missing split column '{split_col}' in {model_output_data_path}")

    df_density = df[df[split_col] == density_split].copy()
    df_eval = df[df[split_col] == split_name].copy()

    if df_density.empty:
        raise ValueError(f"No rows found for density split='{density_split}' in {model_output_data_path}")
    if df_eval.empty:
        raise ValueError(f"No rows found for plot split='{split_name}' in {model_output_data_path}")

    true_col = f"{property_col}_true"
    pred_col = f"{property_col}_pred"
    required_cols = {true_col, pred_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    prop_name = property_key.get(property_col, property_col)
    prop_unit = units_key.get(property_col, "")
    prop_label = f"{prop_name} {prop_unit}" if prop_unit.strip() else prop_name

    all_x = np.concatenate(
        [
            df_density[true_col].to_numpy(),
            df_eval[true_col].to_numpy(),
            df_eval[pred_col].to_numpy(),
        ]
    )
    axis_min = float(all_x.min())
    axis_max = float(all_x.max())
    axis_range = axis_max - axis_min
    if axis_range <= 0:
        padding = 1.0
    else:
        padding = 0.02 * axis_range
    axis_min -= padding
    axis_max += padding

    return {
        "property_col": property_col,
        "prop_label": prop_label,
        "prop_unit": prop_unit,
        "df_density": df_density,
        "df_eval": df_eval,
        "true_col": true_col,
        "pred_col": pred_col,
        "axis_min": axis_min,
        "axis_max": axis_max,
    }


def _resolve_output_file(model_output_data_path, output_file, default_name):
    import os

    if output_file is None:
        figures_dir = os.path.join(os.path.dirname(model_output_data_path), "figures")
        os.makedirs(figures_dir, exist_ok=True)
        return os.path.join(figures_dir, default_name)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_file


def _scale_density_curve_to_target_peak(y_vals, density_scale):
    import numpy as np

    y_vals = np.asarray(y_vals, dtype=float)
    target_peak = float(density_scale)
    current_peak = float(np.nanmax(y_vals)) if y_vals.size else 0.0
    if target_peak <= 0 or current_peak <= 0:
        return y_vals
    return y_vals * (target_peak / current_peak)


def _set_density_axis_ticks(ax, density_scale):
    target_peak = float(density_scale)
    if target_peak > 0:
        ax.set_yticks([0.0, target_peak])
        ax.set_yticklabels(["0", f"{target_peak:g}"])
    else:
        ax.set_yticks([0.0])
        ax.set_yticklabels(["0"])


def plot_parity_only(
    model_output_data_path,
    property_col=None,
    split_col="split",
    split_name="test",
    density_split="train",
    property_key=None,
    units_key=None,
    output_file=None,
    property_code=None,
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error

    style = _configure_matplotlib(plt)
    payload = _prepare_parity_inputs(
        model_output_data_path=model_output_data_path,
        property_col=property_col,
        split_col=split_col,
        split_name=split_name,
        density_split=density_split,
        property_key=property_key,
        units_key=units_key,
        property_code=property_code,
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    ax.scatter(
        payload["df_eval"][payload["true_col"]],
        payload["df_eval"][payload["pred_col"]],
        c=style["parity_point_color"],
        s=style["scatter_marker_area"],
        alpha=0.7,
    )
    ax.plot(
        [payload["axis_min"], payload["axis_max"]],
        [payload["axis_min"], payload["axis_max"]],
        "k--",
        lw=style["line_width"],
    )
    ax.set_xlim(payload["axis_min"], payload["axis_max"])
    ax.set_ylim(payload["axis_min"], payload["axis_max"])
    ax.set_aspect("equal", adjustable="box")
    _style_axis(
        ax,
        style["axis_label_size"],
        style["text_size"],
        style["line_width"],
        xlabel=f"True {payload['prop_label']}",
        ylabel=f"Predicted {payload['prop_label']}",
    )

    mae = mean_absolute_error(
        payload["df_eval"][payload["true_col"]].to_numpy(),
        payload["df_eval"][payload["pred_col"]].to_numpy(),
    )
    metrics_text = f"MAE: {mae:.1f} {payload['prop_unit']}".strip()
    ax.text(
        0.01,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=style["text_size"],
        fontweight="normal",
        verticalalignment="top",
    )

    fig.tight_layout(pad=0.5)
    resolved_output = _resolve_output_file(
        model_output_data_path,
        output_file,
        f"parity_plot_{split_name}.png",
    )
    fig.savefig(resolved_output, dpi=style["save_dpi"], bbox_inches="tight", pad_inches=0.08)
    plt.show()


def plot_training_density_only(
    model_output_data_path,
    property_col=None,
    split_col="split",
    density_split="train",
    density_scale=1.0,
    property_key=None,
    units_key=None,
    output_file=None,
    property_code=None,
):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    style = _configure_matplotlib(plt)
    payload = _prepare_parity_inputs(
        model_output_data_path=model_output_data_path,
        property_col=property_col,
        split_col=split_col,
        split_name=density_split,
        density_split=density_split,
        property_key=property_key,
        units_key=units_key,
        property_code=property_code,
    )

    x_vals = np.linspace(
        payload["df_density"][payload["true_col"]].min(),
        payload["df_density"][payload["true_col"]].max(),
        1000,
    )
    kde = gaussian_kde(payload["df_density"][payload["true_col"]])
    y_vals = _scale_density_curve_to_target_peak(kde(x_vals), density_scale)

    fig, ax = plt.subplots(figsize=(3.5, 1.75), dpi=300)
    ax.plot(x_vals, y_vals, color="blue", lw=style["line_width"])
    ax.fill_between(x_vals, 0, y_vals, color="blue", alpha=0.2)
    ax.set_xlim(payload["axis_min"], payload["axis_max"])
    y_max = float(y_vals.max()) if len(y_vals) else 0.0
    ax.set_ylim(0.0, y_max * 1.08 if y_max > 0 else 1.0)
    _style_axis(
        ax,
        style["axis_label_size"],
        style["text_size"],
        style["line_width"],
        xlabel=f"True {payload['prop_label']}",
        ylabel="Training Density",
    )
    _set_density_axis_ticks(ax, density_scale)
    ax.yaxis.labelpad = 8

    fig.tight_layout(pad=0.6)
    resolved_output = _resolve_output_file(
        model_output_data_path,
        output_file,
        f"{payload['property_col']}_{density_split}_density.png",
    )
    fig.savefig(resolved_output, dpi=style["save_dpi"], bbox_inches="tight", pad_inches=0.1)
    plt.show()


def parity_with_train_density_bottom(
    model_output_data_path,
    property_col=None,
    split_col="split",
    split_name="test",
    density_split="train",
    density_scale=1.0,
    property_key={
        "BP": "Boiling Point",
        "MP": "Melting Point",
        "VP": "$log_{10}(P^{sat} [torr])$",
        "OP": "Oxidation Potential",
        "pka": "pKa",
    },
    units_key={
        "BP": "[K]",
        "MP": "[K]",
        "VP": " ",
        "OP": "[V]",
        "pka": " ",
    },
    output_file=None,
    property_code=None,
):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    from sklearn.metrics import mean_absolute_error

    style = _configure_matplotlib(plt)
    payload = _prepare_parity_inputs(
        model_output_data_path=model_output_data_path,
        property_col=property_col,
        split_col=split_col,
        split_name=split_name,
        density_split=density_split,
        property_key=property_key,
        units_key=units_key,
        property_code=property_code,
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(3.5, 5),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    x_vals = np.linspace(
        payload["df_density"][payload["true_col"]].min(),
        payload["df_density"][payload["true_col"]].max(),
        1000,
    )
    kde = gaussian_kde(payload["df_density"][payload["true_col"]])
    y_vals = _scale_density_curve_to_target_peak(kde(x_vals), density_scale)

    ax1.scatter(
        payload["df_eval"][payload["true_col"]],
        payload["df_eval"][payload["pred_col"]],
        c=style["parity_point_color"],
        s=style["scatter_marker_area"],
        alpha=0.7,
        label=f"{split_name.capitalize()} Predictions",
    )
    ax1.plot(
        [payload["axis_min"], payload["axis_max"]],
        [payload["axis_min"], payload["axis_max"]],
        "k--",
        lw=style["line_width"],
    )
    ax1.set_xlim(payload["axis_min"], payload["axis_max"])
    ax1.set_ylim(payload["axis_min"], payload["axis_max"])
    ax1.set_aspect("equal", adjustable="box")
    _style_axis(
        ax1,
        style["axis_label_size"],
        style["text_size"],
        style["line_width"],
        ylabel=f"Predicted\n{payload['prop_label']}",
    )

    ax2.plot(x_vals, y_vals, color="blue", lw=style["line_width"], label=f"{density_split.capitalize()} Data Density")
    ax2.fill_between(x_vals, 0, y_vals, color="blue", alpha=0.2)
    ax2.set_xlim(payload["axis_min"], payload["axis_max"])
    y_max = float(y_vals.max()) if len(y_vals) else 0.0
    ax2.set_ylim(0.0, y_max * 1.08 if y_max > 0 else 1.0)
    _style_axis(
        ax2,
        style["axis_label_size"],
        style["text_size"],
        style["line_width"],
        xlabel=f"True {payload['prop_label']}",
        ylabel="Training\nDensity",
    )
    _set_density_axis_ticks(ax2, density_scale)

    mae = mean_absolute_error(
        payload["df_eval"][payload["true_col"]].to_numpy(),
        payload["df_eval"][payload["pred_col"]].to_numpy(),
    )
    metrics_text = f"MAE: {mae:.1f} {payload['prop_unit']}".strip()
    ax1.text(
        0.01,
        0.98,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=style["text_size"],
        fontweight="normal",
        verticalalignment="top",
    )

    fig.tight_layout(pad=0.5)
    resolved_output = _resolve_output_file(
        model_output_data_path,
        output_file,
        "parity_plot.png",
    )
    fig.savefig(resolved_output, dpi=style["save_dpi"], bbox_inches="tight", pad_inches=0.08)
    plt.show()
