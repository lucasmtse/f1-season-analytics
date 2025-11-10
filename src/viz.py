from __future__ import annotations
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
import pandas as pd
from matplotlib.patches import Patch
import streamlit as st
from src.openf1 import q as q_openf1 
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc


TEAM_COLORS = {
    "Ferrari": "#DC0000",
    "Mercedes": "#28302F",
    "McLaren": "#FF8700",
    "Red Bull": "#1E41FF",
    "Williams": "#0082FA",
    "Aston Martin": "#006F62",
    "Haas F1 Team": "#B6BABD",
    "Alpine F1 Team": "#A364AC",
    "RB F1 Team": "#2B4562",
    "Sauber": "#27BB14"
}

def darken(hex_color: str, factor: float = 0.7) -> str:
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = [x/255.0 for x in (r,g,b)]
    hls = colorsys.rgb_to_hls(r, g, b)
    darker = colorsys.hls_to_rgb(hls[0], max(0, hls[1]*factor), hls[2])
    return "#%02x%02x%02x" % tuple(int(x*255) for x in darker)

def cumulative_points_plot(df_year: pd.DataFrame, year: int, top_n: int = 10):
    totals = df_year.groupby(["driverId","driver","team"])["cum_points"].max().reset_index()
    totals = totals.sort_values("cum_points", ascending=False).reset_index(drop=True)
    teams = totals["team"].unique()
    palette = [TEAM_COLORS.get(t, "#999999") for t in teams]
    color_map = dict(zip(teams, palette))

    fig, ax = plt.subplots(figsize=(12,7))
    for _, row in totals.iterrows():
        driver = row["driver"]; team = row["team"]
        g = df_year[df_year["driver"] == driver]
        ax.plot(g["round"], g["cum_points"], color=color_map.get(team, "#999999"), lw=1.5, alpha=0.9)

    top = totals.head(top_n)
    for _, row in top.iterrows():
        driver = row["driver"]; team = row["team"]
        g = df_year[df_year["driver"] == driver]
        x = g["round"].max(); y = g["cum_points"].iloc[-1]
        ax.text(x + 0.2, y, f"{driver} ({int(row['cum_points'])})",
                va="center", fontsize=8, color=color_map.get(team, "#999999"), weight="bold")

    ax.set_title(f"Cumulative points per round – {year}", fontsize=14, weight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel("Points")
    plt.tight_layout()
    return fig, ax

    # Compute total cumulative points per driver
    import plotly.graph_objects as go

def cumulative_points_plot_plotly(df_year: pd.DataFrame, year: int, top_n: int = 10):
    # Compute total cumulative points per driver
    totals = (
        df_year.groupby(["driverId", "driver", "team"])["cum_points"]
        .max()
        .reset_index()
        .sort_values("cum_points", ascending=False)
        .reset_index(drop=True)
    )

    # Team color mapping
    teams = totals["team"].unique()
    palette = [TEAM_COLORS.get(t, "#999999") for t in teams]
    color_map = dict(zip(teams, palette))

    fig = go.Figure()

    # Add line per driver
    for _, row in totals.iterrows():
        driver = row["driver"]
        team = row["team"]
        driver_df = df_year[df_year["driver"] == driver]

        fig.add_trace(
            go.Scatter(
                x=driver_df["round"],
                y=driver_df["cum_points"],
                mode="lines",
                name=driver,
                line=dict(color=color_map.get(team, "#999999"), width=2),
                hovertemplate=(
                    f"<b>{driver}</b><br>"
                    + f"Team: {team}<br>"
                    + "Round: %{x}<br>"
                    + "Points: %{y}<extra></extra>"
                ),
            )
        )

    # Annotate top N drivers
    top = totals.head(top_n)
    max_round = df_year["round"].max()
    for _, row in top.iterrows():
        driver = row["driver"]
        team = row["team"]
        driver_df = df_year[df_year["driver"] == driver]
        x = driver_df["round"].max()
        y = driver_df["cum_points"].iloc[-1]

        # Use white for Mercedes, else normal team color
        text_color = "white" if "Mercedes" in team else color_map.get(team, "#999999")

        fig.add_annotation(
            x=x + 0.1,
            y=y,
            xref="x",
            yref="y",
            text=f"{driver} ({int(row['cum_points'])})",
            showarrow=False,
            font=dict(size=10, color=text_color, weight="bold"),
            align="left",
            xanchor="left"
        )

    # Layout
    fig.update_layout(
        title=f"Cumulative points per round – {year}",
        xaxis_title="Round",
        yaxis_title="Points",
        template="plotly_white",
        showlegend=False,
        height=600,
        width=1000,
        margin=dict(l=60, r=240, t=60, b=50),
    )

    return fig

def cumulative_driver_points_by_round(res_df: pd.DataFrame, year: int, sprint_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Per driver & round, sum race + sprint points, then compute cumulative points.
    """
    # Race points per (year, round, driver)
    race = (res_df.query("year == @year")
                  .groupby(["year","round","driverId","driver","team"], as_index=False)["points"]
                  .sum()
                  .rename(columns={"points": "race_points"}))

    # Sprint points per (year, round, driver)
    if sprint_df is not None and not sprint_df.empty:
        spr = (sprint_df.query("year == @year")
                        .groupby(["year","round","driverId","driver","team"], as_index=False)["points"]
                        .sum()
                        .rename(columns={"points": "sprint_points"}))
    else:
        spr = race.loc[:, ["year","round","driverId","driver","team"]].copy()
        spr["sprint_points"] = 0.0

    # Align & sum
    df = race.merge(spr, on=["year","round","driverId","driver","team"], how="outer").fillna(0.0)
    df["points"] = df["race_points"] + df["sprint_points"]

    # Cumulate by driver across rounds
    df = df.sort_values(["driverId","round"])
    df["cum_points"] = df.groupby("driverId")["points"].cumsum()

    # Keep tidy columns used by the plotter
    return df[["year","round","driverId","driver","team","points","cum_points"]].sort_values(["round","driver"])

def constructor_share_pie(con_points: pd.DataFrame, year: int, top_n: int = 5):
    d = con_points[con_points["year"] == year].copy()
    total = d["computed_points"].sum()
    d["share"] = d["computed_points"] / total * 100
    d_sorted = d.sort_values("share", ascending=False)
    top = d_sorted.head(top_n)
    if len(d_sorted) > top_n:
        others = pd.DataFrame([{
            "team": "Others",
            "computed_points": d_sorted["computed_points"].iloc[top_n:].sum(),
            "share": d_sorted["share"].iloc[top_n:].sum()
        }])
        plot_df = pd.concat([top, others], ignore_index=True)
    else:
        plot_df = top

    fig, ax = plt.subplots(figsize=(6,6))
    colors = [TEAM_COLORS.get(t, "#999999") if t in TEAM_COLORS else "#cccccc" for t in plot_df["team"]]
    wedges, texts, autotexts = ax.pie(
        plot_df["share"], labels=plot_df["team"],
        autopct="%1.1f%%", startangle=90, pctdistance=0.8,
        textprops={"fontsize": 10}, colors=colors
    )
    ax.set_title(f"Constructor points share – {year}", fontsize=14, weight="bold")
    plt.tight_layout()
    return fig, ax

def constructor_driver_stacked(points_year: pd.DataFrame, year: int, normalize: bool = False):
    piv = (points_year.pivot_table(index="team", columns="driver", values="total_points", aggfunc="sum")
                    .fillna(0.0))
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]

    if normalize:
        row_sums = piv.sum(axis=1)
        piv = piv.div(row_sums, axis=0) * 100.0

    driver_colors = {}
    for team in piv.index:
        base = TEAM_COLORS.get(team, "#999999")
        darker = darken(base, 0.8)
        drivers = piv.loc[team][piv.loc[team] > 0].index.tolist()
        if len(drivers) == 1:
            driver_colors[drivers[0]] = base
        elif len(drivers) >= 2:
            driver_colors[drivers[0]] = base
            driver_colors[drivers[1]] = darker
            for i, drv in enumerate(drivers[2:], start=2):
                factor = 0.6 - (i-1)*0.1
                driver_colors[drv] = darken(base, max(0.3, factor))

    colors = [driver_colors.get(driver, "#999999") for driver in piv.columns]

    ax = piv.plot(kind="bar", stacked=True, figsize=(12,7),
                  color=colors, edgecolor="black", linewidth=0.3)
    ax.set_title(f"Constructor {'share (%)' if normalize else 'points'} by driver — {year}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Constructor"); ax.set_ylabel("Share (%)" if normalize else "Points")

    if not normalize:
        totals = piv.sum(axis=1)
        for i, val in enumerate(totals.values):
            ax.text(i, val + (val * 0.015 if val > 0 else 0.5), f"{int(round(val))}",
                    ha="center", va="bottom", fontsize=9)


    handles = []
    for team in piv.index:
        handles.append(Patch(color="none", label=f"{team}"))
        drivers = piv.loc[team][piv.loc[team] > 0].index.tolist()
        for drv in drivers:
            handles.append(Patch(color=driver_colors.get(drv, "#999999"), label=f"   {drv}"))
    ax.legend(handles=handles, title="Teams & Drivers", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    plt.tight_layout()
    return ax.get_figure(), ax



def constructor_quali_race(pos_changes_df: pd.DataFrame, year: int, palette_driver: Dict[str,str]):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=pos_changes_df, x="driverId", y="pos_change",
        palette=palette_driver,
        ax=ax
    )

    ax.axhline(0, color="k", linestyle="--", lw=1)
    ax.set_title(f"Qualifying vs Race Position Changes – {year}", fontsize=14, weight="bold")
    ax.set_xlabel("Constructor")
    ax.set_ylabel("Position Change (Quali to Race)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig, ax

def driver_quali_boxplot(df_quali: pd.DataFrame, year:int,palette_quali: Dict[str,str], quali_order: list[str]):
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.boxplot(
            data=df_quali, x="driver", y="position",
            order=quali_order, palette=palette_quali, ax=ax2
        )
    ax2.invert_yaxis()
    ax2.set_title(f"Qualifying result distribution — {year}", fontsize=12, weight="bold")
    ax2.set_xlabel("Driver")
    ax2.set_ylabel("Qualifying position (lower = better)")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig2, ax2


def driver_race_boxplot(res_plot: pd.DataFrame, year:int, palette_race: Dict[str,str], median_order: list[str]):
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.boxplot(
            data=res_plot, x="driver", y="position",
            order=median_order, palette=palette_race, ax=ax1
        )
        # Invert y so P1 is at the top
    ax1.invert_yaxis()
    ax1.set_title(f"Race result distribution — {year}", fontsize=12, weight="bold")
    ax1.set_xlabel("Driver")
    ax1.set_ylabel("Race position (lower = better)")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig1, ax1

def constructor_quali_race_plotly(pos_changes_df: pd.DataFrame, year: int, palette_driver: dict[str, str], selected_driver: str | None = None):
    # Filter driver if selected
    if selected_driver:
        pos_changes_df = pos_changes_df[pos_changes_df['driverId'] == selected_driver]

    fig = px.bar(
        pos_changes_df,
        x="driverId",
        y="pos_change",
        color="driverId",
        color_discrete_map=palette_driver,
        title=f"Qualifying vs Race Position Changes – {year}",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        xaxis_title="Driver",
        yaxis_title="Position Change (Quali to Race)",
        xaxis_tickangle=45,
        template="plotly_white"
    )
    return fig

def driver_quali_boxplot_plotly(df_quali: pd.DataFrame, year: int, palette_quali: dict[str, str], quali_order: list[str], selected_driver: str | None = None):
    if selected_driver:
        df_quali = df_quali[df_quali['driver'] == selected_driver]

    fig = px.box(
        df_quali,
        x="driver",
        y="position",
        category_orders={"driver": quali_order},
        color="driver",
        color_discrete_map=palette_quali,
        title=f"Qualifying result distribution — {year}",
    )
    fig.update_yaxes(autorange="reversed", title="Qualifying position (lower = better)")
    fig.update_xaxes(title="Driver", tickangle=45)
    fig.update_layout(template="plotly_white")
    return fig

def driver_race_boxplot_plotly(res_plot: pd.DataFrame, year: int, palette_race: dict[str, str], median_order: list[str], selected_driver: str | None = None):
    if selected_driver:
        res_plot = res_plot[res_plot['driver'] == selected_driver]

    fig = px.box(
        res_plot,
        x="driver",
        y="position",
        category_orders={"driver": median_order},
        color="driver",
        color_discrete_map=palette_race,
        title=f"Race result distribution — {year}",
    )
    fig.update_yaxes(autorange="reversed", title="Race position (lower = better)")
    fig.update_xaxes(title="Driver", tickangle=45)
    fig.update_layout(template="plotly_white")
    return fig

def cumulative_points_period_plot_plotly(
    res_window: pd.DataFrame,
    sprint_window: pd.DataFrame,
    year: int,
    start_round: int,
    end_round: int,
    team_colors: Dict[str, str],
    top_n: int | None = None,
):
    """
    Plot cumulative points *within a selected period* (round range) for a set of drivers,
    with driver name and total points annotated to the right of each line.

    Parameters
    ----------
    res_window : pd.DataFrame
        Race results already filtered to the desired period & drivers.
        Must contain: ['round', 'driver', 'team', 'points'].
    sprint_window : pd.DataFrame
        Sprint results filtered to the same period & drivers.
        Same columns as res_window, or empty DataFrame if no sprints.
    year : int
        Season year (for the title).
    start_round : int
        First round in the selected period.
    end_round : int
        Last round in the selected period.
    team_colors : dict
        Mapping {team_name: hex_color}.
    top_n : int | None
        Number of drivers to annotate on the right (by points in the period).
        If None, all drivers are annotated.
    """
    points_pieces = []

    if not res_window.empty and "points" in res_window.columns:
        points_pieces.append(
            res_window[["round", "driver", "team", "points"]].copy()
        )

    if not sprint_window.empty and "points" in sprint_window.columns:
        points_pieces.append(
            sprint_window[["round", "driver", "team", "points"]].copy()
        )

    fig = go.Figure()

    if not points_pieces:
        # No data to plot
        fig.update_layout(
            title=f"Cumulative points in selected period — {year}",
            template="plotly_white",
            xaxis_title="Round",
            yaxis_title="Points (within period)",
        )
        return fig

    # Concatenate
    pts_period = pd.concat(points_pieces, ignore_index=True)

    # Sum points per (round, driver, team)
    pts_period = (
        pts_period.groupby(["round", "driver", "team"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "points_period"})
    )

    # Cumulative points *within the selected period only* (starts at 0 on start_round)
    pts_period = pts_period.sort_values(["driver", "round"])
    pts_period["cum_points_period"] = (
        pts_period.groupby("driver")["points_period"].cumsum()
    )

    #
    totals = (
        pts_period.groupby(["driver", "team"], as_index=False)["cum_points_period"]
        .max()
        .sort_values("cum_points_period", ascending=False)
        .reset_index(drop=True)
    )

    if top_n is None:
        top_n = len(totals)
    palette = [team_colors.get(t, "#999999") for t in totals["team"].unique()]
    color_map = dict(zip(totals["team"].unique(), palette))
    # Curves for each driver
    for _, row in totals.iterrows():
        driver = row["driver"]
        team = row["team"]
        g = pts_period[pts_period["driver"] == driver]

        fig.add_trace(
            go.Scatter(
                x=g["round"],
                y=g["cum_points_period"],
                mode="lines",
                name=driver,
                line=dict(color=color_map.get(team, "#999999"), width=2),
                hovertemplate=(
                    f"<b>{driver}</b><br>"
                    + f"Team: {team}<br>"
                    + "Round: %{x}<br>"
                    + "Points in period: %{y}<extra></extra>"
                ),
            )
        )

    # Annotations on the right for the top_n
    top = totals.head(top_n)
    for _, row in top.iterrows():
        driver = row["driver"]
        team = row["team"]
        total_pts = row["cum_points_period"]

        g = pts_period[pts_period["driver"] == driver]
        x_last = g["round"].max()
        y_last = g[g["round"] == x_last]["cum_points_period"].iloc[0]

        # Text color: white for Mercedes, otherwise team color
        text_color = "white" if "Mercedes" in str(team) else color_map.get(team, "#999999")

        fig.add_annotation(
            x=x_last + 0.1,             
            y=y_last,
            xref="x",
            yref="y",
            text=f"{driver} ({int(total_pts)})",
            showarrow=False,
            font=dict(size=10, color=text_color),
            align="left",
            xanchor="left",
        )

    # Layout global
    fig.update_layout(
        title=f"Cumulative points in selected period — {year}",
        xaxis_title="Round",
        yaxis_title="Points (within period)",
        template="plotly_white",
        legend_title="Driver",
        margin=dict(l=60, r=240, t=60, b=50),  
    )

    fig.update_xaxes(range=[start_round - 0.5, end_round + 0.5])

    return fig
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from itertools import cycle


def _fmt_lap_total(x: float):
    try:
        x = float(x)
        m = int(x // 60)
        s = x % 60
        return f"{m}:{s:06.3f}"
    except Exception:
        return None


def stint_tab(df_laps: pd.DataFrame, df_drv: pd.DataFrame, session_key):
    st.header("Stint analysis")

    required_lap_cols = {"driver_number", "lap_number", "lap_duration"}
    if df_laps.empty or not required_lap_cols.issubset(df_laps.columns):
        st.info("No laps available for this session.")
        return
    # --- Clean laps dataframe ---
    laps = df_laps.copy()

    laps["driver_number"] = pd.to_numeric(laps["driver_number"], errors="coerce").astype("Int64")
    laps["lap_number"]    = pd.to_numeric(laps["lap_number"], errors="coerce").astype("Int64")
    laps["lap_duration"]  = pd.to_numeric(laps["lap_duration"], errors="coerce")
    if "is_pit" not in laps.columns:
        laps["is_pit"] = False

    # Map driver_number -> name
    def _fix_hex(c):
        if pd.isna(c):
            return None
        c = str(c).strip()
        if not c:
            return None
        # Add leading '#' if missing
        if not c.startswith("#"):
            c = "#" + c
        return c

# Normalize the team colour column
    df_drv["team_colour"] = df_drv["team_colour"].apply(_fix_hex)
    id2name = dict(zip(df_drv["driver_number"], df_drv["full_name"]))
    id2color = dict(zip(df_drv["driver_number"], df_drv["team_colour"]))  # e.g. "#DC0000" for Ferrari
    laps["Driver"] = laps["driver_number"].map(id2name)
    laps["team_color"] = laps["driver_number"].map(id2color)
    # --- Get stints from OpenF1 ---
    try:
        stints_raw = q_openf1("/stints", {"session_key": session_key})
    except Exception as e:
        st.warning(f"Could not fetch stints from OpenF1: {e}")
        return

    df_st = pd.DataFrame(stints_raw)

    if df_st.empty:
        st.info("No stint information available from OpenF1 for this session.")
        return

    # Normalize columns
    rename_map = {}
    for a, b in [
        ("driver", "driver_number"),
        ("driverId", "driver_number"),
        ("stint", "stint_number"),
        ("stint_id", "stint_number"),
        ("compound_name", "compound"),
        ("compound_type", "compound"),
        ("tyre_compound", "compound"),
        ("tyre", "compound"),
        ("lap_start", "lap_start"),
        ("lap_end", "lap_end"),
        ("lap_number_start", "lap_start"),
        ("lap_number_end", "lap_end"),
    ]:
        if a in df_st.columns and b not in df_st.columns:
            rename_map[a] = b

    if rename_map:
        df_st = df_st.rename(columns=rename_map)

    needed_stint_cols = {"driver_number", "stint_number", "lap_start", "lap_end"}
    if not needed_stint_cols.issubset(df_st.columns):
        st.info("Stint endpoint did not return expected columns.")
        st.write("Columns found:", sorted(df_st.columns))
        return

    df_st["driver_number"] = pd.to_numeric(df_st["driver_number"], errors="coerce").astype("Int64")
    df_st["stint_number"]  = pd.to_numeric(df_st["stint_number"], errors="coerce").astype("Int64")
    df_st["lap_start"]     = pd.to_numeric(df_st["lap_start"], errors="coerce").astype("Int64")
    df_st["lap_end"]       = pd.to_numeric(df_st["lap_end"], errors="coerce").astype("Int64")

    df_st["Driver"] = df_st["driver_number"].map(id2name)

    #  Multi-driver selection 
    drivers_available = (
        df_st["Driver"].dropna().astype(str).sort_values().unique().tolist()
    )
    if not drivers_available:
        st.info("No drivers with stint data found for this session.")
        return

    selected_drivers = st.multiselect(
        "Select drivers",
        options=drivers_available,
        default=drivers_available[:2],
    )

    if not selected_drivers:
        st.info("Select at least one driver to analyse stints.")
        return

    stints_filtered = df_st[df_st["Driver"].isin(selected_drivers)].copy()
    laps_filtered   = laps[laps["Driver"].isin(selected_drivers)].copy()

    if stints_filtered.empty or laps_filtered.empty:
        st.info("No data for selected drivers.")
        return

    #  Build stint summary per driver 
    stint_rows = []
    for _, srow in stints_filtered.iterrows():
        drv_name = srow["Driver"]
        drv_no   = srow["driver_number"]
        s_no     = int(srow["stint_number"])
        ls, le   = int(srow["lap_start"]), int(srow["lap_end"])
        compound = srow.get("compound", None)

        mask = (
            (laps_filtered["driver_number"] == drv_no) &
            (laps_filtered["lap_number"] >= ls) &
            (laps_filtered["lap_number"] <= le)
        )

        stint_laps = laps_filtered[mask & (~laps_filtered["is_pit"].fillna(False))].copy()
        if stint_laps.empty:
            continue

        stint_rows.append({
            "Driver": drv_name,
            "driver_number": drv_no,
            "stint_number": s_no,
            "Lap start": ls,
            "Lap end": le,
            "Laps": len(stint_laps),
            "Avg lap (s)": stint_laps["lap_duration"].mean(),
            "Best lap (s)": stint_laps["lap_duration"].min(),
            "Compound": compound,
        })

    if not stint_rows:
        st.info("No usable stints (non-pit laps) for selected drivers.")
        return

    stint_summary = (
        pd.DataFrame(stint_rows)
        .sort_values(["Driver", "stint_number"])
        .reset_index(drop=True)
    )
    stint_summary["Avg lap"]  = stint_summary["Avg lap (s)"].apply(_fmt_lap_total)
    stint_summary["Best lap"] = stint_summary["Best lap (s)"].apply(_fmt_lap_total)

    st.subheader("Stint summary – selected drivers")
    cols_to_show = ["Driver", "stint_number", "Laps", "Lap start", "Lap end", "Compound", "Avg lap", "Best lap"]
    st.dataframe(
        stint_summary[cols_to_show],
        hide_index=True,
        use_container_width=True,
    )

    # --- Assign stint & compound to each lap ---
    laps_filtered["stint_number"]    = pd.NA
    laps_filtered["stint_compound"]  = pd.NA

    for _, srow in stints_filtered.iterrows():
        drv_no   = srow["driver_number"]
        s_no     = int(srow["stint_number"])
        ls, le   = int(srow["lap_start"]), int(srow["lap_end"])
        compound = srow.get("compound", None)

        mask = (
            (laps_filtered["driver_number"] == drv_no) &
            (laps_filtered["lap_number"] >= ls) &
            (laps_filtered["lap_number"] <= le)
        )
        laps_filtered.loc[mask, "stint_number"]   = s_no
        laps_filtered.loc[mask, "stint_compound"] = compound

    laps_filtered = laps_filtered.dropna(subset=["stint_number", "lap_duration"])
    if laps_filtered.empty:
        st.info("No laps with stint attribution to plot.")
        return

    laps_filtered["stint_number"] = laps_filtered["stint_number"].astype(int)
    laps_filtered["Lap"]          = laps_filtered["lap_number"]
    laps_filtered["lap_sec"]      = laps_filtered["lap_duration"]

    # --- Tyre colors by compound ---
    compound_colors = {
        "SOFT": "#DA291C",         # red
        "MEDIUM": "#FFD700",       # yellow
        "HARD": "#FFFFFF",         # white
        "INTERMEDIATE": "#43B02A", # green
        "WET": "#0067AD",          # blue
    }

    # --- Line style by driver ---
    dash_cycle = cycle(["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"])
    driver_dash = {}
    for drv in selected_drivers:
        driver_dash[drv] = next(dash_cycle)

    # --- Plot ---
    st.subheader("Lap time evolution by stint")

    show_pit_laps = st.checkbox("Show pit laps on plot", value=False)
    plot_df = laps_filtered.copy()
    if not show_pit_laps:
        plot_df = plot_df[~plot_df["is_pit"].fillna(False)].copy()

    if plot_df.empty:
        st.info("No non-pit laps to plot.")
        return
    plot_df["team_color"] = plot_df["driver_number"].map(id2color)
    fig = go.Figure()

    # Assign a fixed color per driver
    driver_colors = dict(zip(selected_drivers, pc.qualitative.Set2))
    # Find the final stint number per driver
    max_stint = plot_df.groupby("Driver")["stint_number"].max().to_dict()
    # One trace per (Driver, stint_number)
    for (drv_name, s_no), group in plot_df.groupby(["Driver", "stint_number"]):
        group = group.sort_values("Lap")

        comp_raw = group["stint_compound"].dropna().astype(str).iloc[0] if group["stint_compound"].notna().any() else None
        comp_norm = comp_raw.upper().strip() if isinstance(comp_raw, str) else None

        line_color= compound_colors.get(comp_norm, "#A0A0A0")   # marker = tyre color
        tyre_color = (
        group["team_color"].dropna().iloc[0]
        if "team_color" in group.columns
        else "#888888"
    )  


        label = f"{drv_name} – Stint {s_no}"
        if comp_norm:
            label += f" ({comp_norm})"

        # Line trace (driver color)
        fig.add_trace(
            go.Scatter(
                x=group["Lap"],
                y=group["lap_sec"],
                mode="lines+markers",
                name=label,
                line=dict(color=line_color, width=2, dash=driver_dash.get(drv_name, "solid")),
                marker=dict(color=tyre_color, size=7, line=dict(color=line_color, width=1.2)),
                hovertemplate=(
                    f"Driver: {drv_name}<br>"
                    f"Stint: {s_no}"
                    + (f" ({comp_norm})" if comp_norm else "")
                    + "<br>Lap: %{x}<br>Time: %{y:.3f} s<extra></extra>"
                ),
            )
        )
        if s_no == max_stint.get(drv_name):
            x_last = group["Lap"].iloc[-1]
            y_last = group["lap_sec"].iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[x_last + 0.3],
                    y=[y_last],
                    mode="text",
                    text=[drv_name],
                    textposition="middle right",
                    textfont=dict(color=tyre_color, size=11, family="Arial Black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        xaxis_title="Lap number",
        yaxis_title="Lap time (s)",
        hovermode="x unified",
        legend_title="Driver – Stint (compound)",
    )
    fig.update_xaxes(dtick=1)

    st.plotly_chart(fig, use_container_width=True)
