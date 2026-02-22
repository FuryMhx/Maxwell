import math
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from log_function import (
    LogParseOptions,
    apply_action_grouping,
    collect_log_files_by_date,
    load_and_clean_logs_from_files,
)


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


@st.cache_data(show_spinner=False)
def _load_printer_range(
    printer_folder: str,
    start_date: date,
    end_date: date,
    dedupe_granularity: str,
    rules_csv_path: str,
) -> Dict[str, object]:
    files = collect_log_files_by_date(printer_folder, start_date, end_date)
    options = LogParseOptions(dedupe_granularity=dedupe_granularity, exclude_codes=None)

    df = load_and_clean_logs_from_files(files, options=options, rules_csv_path=rules_csv_path)
    df = apply_action_grouping(df, rules_csv_path=rules_csv_path)

    action_df = (
        df["ActionGrouped"].value_counts().rename_axis("Action").reset_index(name="count")
        if not df.empty and "ActionGrouped" in df.columns
        else pd.DataFrame(columns=["Action", "count"])
    )

    return {
        "files": files,
        "df": df,
        "action_df": action_df,
    }


@st.cache_data(show_spinner=False)
def _build_daily_counts(
    printer_folders: Dict[str, str],
    start_date: date,
    end_date: date,
    dedupe_granularity: str,
    rules_csv_path: str,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    current = start_date
    while current <= end_date:
        for printer_name, folder in printer_folders.items():
            res = _load_printer_range(
                folder,
                current,
                current,
                dedupe_granularity,
                rules_csv_path,
            )
            action_df = res["action_df"].copy()
            if action_df.empty:
                continue
            action_df["Date"] = current
            action_df["Printer"] = printer_name
            rows.append(action_df)
        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["Date", "Printer", "Action", "count"])

    return pd.concat(rows, ignore_index=True)


def _set_plot_fonts() -> None:
    # Best-effort fonts for Chinese labels; safe if a font isn't installed.
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_top_actions_grid(
    per_printer_action_df: Dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
    top_n: int,
) -> Optional[plt.Figure]:
    names = list(per_printer_action_df.keys())
    if not names:
        return None

    cols = 2
    rows = max(1, math.ceil(len(names) / cols))

    _set_plot_fonts()
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), constrained_layout=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, printer_name in enumerate(names):
        ax = axes_list[i]
        action_df = per_printer_action_df[printer_name]
        plot_df = action_df.sort_values("count", ascending=False).head(top_n)

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(f"{printer_name} (No data)")
            ax.axis("off")
            continue

        x = list(range(len(plot_df)))
        counts = plot_df["count"].tolist()
        bars = ax.bar(x, counts)
        ax.set_title(f"{printer_name} Top {top_n} actions {start_date:%m/%d}-{end_date:%m/%d}")
        ax.set_xlabel("Action")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["Action"], rotation=60, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for j in range(len(names), len(axes_list)):
        axes_list[j].axis("off")

    return fig


def _plot_daily_counts_for_printer(
    daily_df: pd.DataFrame,
    printer_name: str,
    top_k_actions: int,
) -> Optional[plt.Figure]:
    if daily_df.empty:
        return None

    printer_df = daily_df[daily_df["Printer"] == printer_name]
    if printer_df.empty:
        return None

    pivot_df = printer_df.pivot_table(
        index="Action",
        columns="Date",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    if pivot_df.empty:
        return None

    top_actions = pivot_df.sum(axis=1).sort_values(ascending=False).head(top_k_actions).index
    pivot_df = pivot_df.loc[top_actions]

    _set_plot_fonts()
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    pivot_df.plot(kind="bar", width=0.85, ax=ax)

    ax.set_title(f"Daily Action Counts: {printer_name}")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2, fontsize=8)

    return fig


def main() -> None:
    st.set_page_config(page_title="Maxwell Log Dashboard", layout="wide")

    st.title("Maxwell Log Dashboard")

    with st.sidebar:
        st.header("Inputs")

        default_root = r"\\10.1.100.102\SPT_Log"
        root_path = st.text_input("Log root path", value=default_root)

        available_printers = ["P1", "P2", "P3", "P4"]
        selected_printers = st.multiselect(
            "Printers",
            options=available_printers,
            default=available_printers,
        )

        today = datetime.today().date()
        date_range = st.date_input(
            "Date range",
            value=(today - timedelta(days=7), today),
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = today - timedelta(days=7)
            end_date = today

        dedupe_granularity = st.selectbox(
            "Deduplicate granularity",
            options=["exact", "second", "minute"],
            index=1,
            help="Controls how Time is floored before dropping duplicates.",
        )

        rules_csv_path = st.text_input("Grouping rules CSV path", value="action_grouping_rules.csv")

        top_n = st.slider("Top N actions (summary)", min_value=3, max_value=20, value=7)
        top_k_daily = st.slider("Top K actions (daily plot)", min_value=3, max_value=20, value=5)

        run = st.button("Run")

    if not run:
        st.info("Choose inputs on the left, then click Run.")
        return

    if not selected_printers:
        st.warning("Select at least one printer.")
        return

    if start_date > end_date:
        st.warning("Start date must be on/before end date.")
        return

    printer_folders: Dict[str, str] = {
        p: os.path.join(root_path, p, "log") for p in selected_printers
    }

    st.subheader("Per-printer results")

    per_printer_action_df: Dict[str, pd.DataFrame] = {}

    for printer_name, folder in printer_folders.items():
        with st.spinner(f"Loading {printer_name}..."):
            res = _load_printer_range(
                folder,
                start_date,
                end_date,
                dedupe_granularity,
                rules_csv_path,
            )

        files = res["files"]
        df = res["df"]
        action_df = res["action_df"]
        per_printer_action_df[printer_name] = action_df

        st.markdown(f"### {printer_name}")
        st.caption(f"Folder: {folder}")
        st.write(f"Files matched: {len(files)} | Rows: {len(df)}")

        with st.expander("Show action counts (grouped)", expanded=True):
            st.dataframe(action_df, use_container_width=True)

        with st.expander("Show full DataFrame", expanded=False):
            st.dataframe(df, use_container_width=True, height=320)
            st.download_button(
                label=f"Download {printer_name} df as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{printer_name}_{start_date}_{end_date}.csv",
                mime="text/csv",
            )

    st.subheader("Plots")

    fig = _plot_top_actions_grid(per_printer_action_df, start_date, end_date, top_n)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No data to plot for the selected range.")

    with st.spinner("Building daily counts (may take a bit the first time)..."):
        daily_df = _build_daily_counts(
            printer_folders,
            start_date,
            end_date,
            dedupe_granularity,
            rules_csv_path,
        )

    if daily_df.empty:
        st.info("No daily data found to plot.")
        return

    st.subheader("Daily action counts")
    for printer_name in selected_printers:
        daily_fig = _plot_daily_counts_for_printer(daily_df, printer_name, top_k_daily)
        if daily_fig is None:
            st.markdown(f"#### {printer_name}")
            st.caption("No daily data")
            continue

        st.markdown(f"#### {printer_name}")
        st.pyplot(daily_fig)


if __name__ == "__main__":
    main()
