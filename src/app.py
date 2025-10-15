import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from src.data_jolpi import get_paged
from src.transformers import results_to_df, sprint_to_df, qualifying_to_df, driver_standings_to_df, constructor_standings_to_df
from src.standings import computed_driver_points, computed_constructor_points, cumulative_driver_points_by_round, teammate_split
from src.viz import cumulative_points_plot, constructor_share_pie, constructor_driver_stacked, constructor_quali_race, TEAM_COLORS, driver_race_boxplot, driver_quali_boxplot
import re
import numpy as np
from src.openf1 import q as q_openf1 
st.set_page_config(page_title="F1 Season Analytics", layout="wide")
st.title("🏁 F1 Season Analytics")
st.caption("Data: Ergast via Jolpi (cached locally).")

# Sidebar season picker
st.sidebar.header("⚙️ Settings")
seasons_payload = get_paged("/seasons.json")
season_years = sorted([int(s["season"]) for s in seasons_payload["MRData"]["SeasonTable"]["Seasons"]])
# restrict to years 2021 (before 2021 the data API are different)
season_years = [y for y in season_years if y >= 2021]
default_year = max(season_years)
year = st.sidebar.selectbox(
    "Season",
    sorted(season_years, reverse=True),  # tri décroissant
    index=sorted(season_years, reverse=True).index(default_year)
)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Standings", 
    "Cumulative Points by Driver", 
    "Quali vs Race", 
    "Constructors", 
    "Driver Consistency", 
    "Data about sessions",
    "🤖 AI GP Assistant"  # NEW
])

# Load data for the selected year
res_df    = results_to_df(get_paged(f"/{year}/results.json"))
sprint_df = sprint_to_df(get_paged(f"/{year}/sprint.json"))
quali_df  = qualifying_to_df(get_paged(f"/{year}/qualifying.json"))

drv_comp = computed_driver_points(res_df, sprint_df)
con_comp = computed_constructor_points(res_df, sprint_df)


with tab1:
    st.subheader("Official standings (race + sprint)")

    # Official standings for the selected year
    off_drv_df = driver_standings_to_df(get_paged(f"/{year}/driverstandings.json"))
    off_con_df = constructor_standings_to_df(get_paged(f"/{year}/constructorstandings.json"))

    # Helpers from race & quali data (for podiums/poles & season team)
    res_y = res_df[res_df["year"] == year].copy()
    quali_y = quali_df[quali_df["year"] == year].copy()

    # Driver's season team (mode over races)
    def mode_or_first(s):
        # return most frequent value; fallback to first if tie/empty
        return s.mode().iloc[0] if not s.mode().empty else (s.iloc[0] if len(s) else None)

    driver_team = (
        res_y.groupby(["driverId", "driver"], as_index=False)["team"]
        .agg(mode_or_first)
        .rename(columns={"team": "team"})
    )

    # Driver podiums (P1-P2-P3 in race results)
    drv_podiums = (
        res_y[res_y["position"].between(1, 3)]
        .groupby(["driverId", "driver"], as_index=False)
        .size()
        .rename(columns={"size": "podiums"})
    )

    # Driver poles (qualifying position == 1)
    drv_poles = (
        quali_y[quali_y["position"] == 1]
        .groupby(["driverId", "driver"], as_index=False)
        .size()
        .rename(columns={"size": "poles"})
    )

    # Build clean DRIVER table
    # off_drv_df has: year, position (official rank), points (official), wins, driverId, driver
    drivers_table = (
        off_drv_df.merge(driver_team, on=["driverId", "driver"], how="left")
                  .merge(drv_podiums, on=["driverId", "driver"], how="left")
                  .merge(drv_poles, on=["driverId", "driver"], how="left")
                  .fillna({"podiums": 0, "poles": 0})
                  .sort_values(["position"])
                  .reset_index(drop=True)
    )

    # Select & rename columns; drop technical ones
    drivers_table = drivers_table[[
        "position",      # official rank
        "driver",        # name
        "team",          # season team
        "points",        # official points
        "wins",          # official wins
        "podiums",       # computed from res_df
        "poles"          # computed from quali_df
    ]].rename(columns={
        "position": "Pos",
        "driver": "Driver",
        "team": "Team",
        "points": "Pts",
        "wins": "Wins",
        "podiums": "Podiums",
        "poles": "Poles"
    })

    st.markdown("**Drivers**")
    st.dataframe(drivers_table, use_container_width=True, hide_index=True)

    # CONSTRUCTORS table (clean)
    # Team podiums (any driver on podium counts for team)
    team_podiums = (
        res_y[res_y["position"].between(1, 3)]
        .groupby("team", as_index=False)
        .size()
        .rename(columns={"size": "podiums"})
    )

    # Team poles (any driver with P1 in quali counts for team)
    team_poles = (
        quali_y[quali_y["position"] == 1]
        .groupby("team", as_index=False)
        .size()
        .rename(columns={"size": "poles"})
    )

    constructors_table = (
        off_con_df.merge(team_podiums, on="team", how="left")
                  .merge(team_poles, on="team", how="left")
                  .fillna({"podiums": 0, "poles": 0})
                  .sort_values(["position"])
                  .reset_index(drop=True)
    )

    constructors_table = constructors_table[[
        "position",   # official rank
        "team",       # constructor name
        "points",     # official points
        "wins",       # official wins
        "podiums",    # computed from res_df
        "poles"       # computed from quali_df
    ]].rename(columns={
        "position": "Pos",
        "team": "Team",
        "points": "Pts",
        "wins": "Wins",
        "podiums": "Podiums",
        "poles": "Poles"
    })


    st.markdown("**Constructors**")
    st.dataframe(constructors_table, use_container_width=True, hide_index=True)


with tab2:
    st.subheader("Cumulative points (drivers)")

    # Build cumulative per round for the selected season
    cum_df = cumulative_driver_points_by_round(res_df, year, sprint_df=sprint_df)
    # Order drivers by final cumulative points (for nicer defaults)
    final_totals = (cum_df.groupby(["driverId","driver","team"])["cum_points"]
                          .max()
                          .reset_index()
                          .sort_values("cum_points", ascending=False))
    all_drivers = final_totals["driver"].tolist()
    default_drivers = final_totals["driver"].head(10).tolist()  # preselect top 10

    # Let the user choose drivers
    chosen = st.multiselect("Select drivers", options=all_drivers, default=default_drivers)

    if chosen:
        cum_sel = cum_df[cum_df["driver"].isin(chosen)].copy()
        # annotate up to the number selected
        fig, ax = cumulative_points_plot(cum_sel, year, top_n=len(chosen))
        st.pyplot(fig, use_container_width=False, clear_figure=True)
    else:
        st.info("Select at least one driver to display.")

with tab3:
    from adjustText import adjust_text


    st.subheader("Qualifying vs Race analysis")

    # Préparation des données
    q = quali_df[["driverId","driver","position"]].rename(columns={"position":"quali_pos"})
    r = res_df[["driverId","driver","team","position"]].rename(columns={"position":"race_pos"})
    merged = q.merge(r, on=["driverId","driver"], how="inner")
    merged["pos_change"] = merged["quali_pos"] - merged["race_pos"]

    # Ordre des pilotes par gain moyen
    order = (merged.groupby("driverId")["pos_change"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist())

    # Palette par pilote (via team)
    drv_team_map_id = dict(merged.drop_duplicates("driverId").set_index("driverId")["team"])
    palette_driver_id = [TEAM_COLORS.get(drv_team_map_id.get(d, ""), "#999999") for d in order]

    # ---------- Haut : boxplot et barplot côte à côte ----------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution of position change per driver (boxplot)**")
        fig_box, ax_box = plt.subplots(figsize=(6,5))
        sns.boxplot(
            data=merged, x="driverId", y="pos_change",
            order=order, palette=palette_driver_id, ax=ax_box
        )
        ax_box.axhline(0, color="k", linestyle="--", lw=1)
        ax_box.set_xlabel("Driver ID"); ax_box.set_ylabel("Pos change (Quali → Race)")
        plt.setp(ax_box.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig_box, use_container_width=False, clear_figure=True)

    with col2:
        st.markdown("**Average position change per driver (barplot)**")
        avg_changes = (merged.groupby("driverId")["pos_change"]
                            .mean()
                            .sort_values(ascending=True))
        fig_bar, ax_bar = plt.subplots(figsize=(6,5))
        avg_changes.plot.bar(color=[TEAM_COLORS.get(drv_team_map_id.get(d, ""), "#999999")
                                    for d in avg_changes.index], ax=ax_bar)
        ax_bar.axhline(0, color="k", linestyle="--", lw=1)
        ax_bar.set_xlabel("Driver")
        ax_bar.set_ylabel("Average Pos Change")
        st.pyplot(fig_bar, use_container_width=False, clear_figure=True)

    # ---------- Bas : scatter de corrélation ----------
    st.markdown("**Correlation: Avg qualifying vs avg race result**")

    avg = merged.groupby(["driverId","driver","team"])[["quali_pos","race_pos"]].mean().reset_index()

    fig_scatter, ax_scatter = plt.subplots(figsize=(7,6))
    teams = avg["team"].unique()
    palette = sns.color_palette("tab20", len(teams))
    color_map = dict(zip(teams, palette))

    ax_scatter.scatter(avg["quali_pos"], avg["race_pos"],
                    c=avg["team"].map(color_map), s=70,
                    edgecolor="k", alpha=0.85)

    texts = []
    for _, row in avg.iterrows():
        texts.append(ax_scatter.text(row["quali_pos"], row["race_pos"], row["driverId"], fontsize=8))
    adjust_text(texts, ax=ax_scatter, arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))

    maxv = avg[["quali_pos","race_pos"]].max().max()
    ax_scatter.plot([1, maxv],[1, maxv], "k--", lw=1)

    ax_scatter.set_xlabel("Avg Qualifying Position (lower = better)")
    ax_scatter.set_ylabel("Avg Race Result (lower = better)")
    ax_scatter.set_title(f"Avg Quali vs Race — {year}")
    ax_scatter.invert_xaxis(); ax_scatter.invert_yaxis()

    handles = [plt.Line2D([0],[0], marker='o', color='w', label=t,
                        markerfacecolor=color_map[t], markersize=8) for t in teams]
    ax_scatter.legend(handles=handles, title="Teams", bbox_to_anchor=(1.05,1), loc="upper left")

    st.pyplot(fig_scatter, use_container_width=False, clear_figure=True)

with tab4:
 col1, col2 = st.columns([2,1])  

with col1:
    st.subheader("Constructor points by driver (stacked)")
    pts = teammate_split(res_df, sprint_df, year)
    fig2, ax2 = constructor_driver_stacked(pts, year, normalize=False)
    st.pyplot(fig2, use_container_width=False, clear_figure=True)

with col2:
    st.subheader("Constructor points share (pie)")
    fig, ax = constructor_share_pie(con_comp, year, top_n=5)
    st.pyplot(fig, use_container_width=False, clear_figure=True)
with tab5:
    st.subheader("Driver consistency (boxplots)")

    # Controls
    colc1, colc2, colc3 = st.columns([1,1,2])
    with colc1:
        min_race_starts = st.number_input("Min race/qualifying starts", min_value=1, max_value=23, value=3, step=1)
        min_quali_starts = min_race_starts
    # Season data
    res_y = res_df[res_df["year"] == year].copy()
    quali_y = quali_df[quali_df["year"] == year].copy()

    # Driver to team map (mode over races)
    def mode_or_first(s):
        return s.mode().iloc[0] if not s.mode().empty else (s.iloc[0] if len(s) else None)
    drv_team = (res_y.groupby(["driverId","driver"], as_index=False)["team"]
                .agg(mode_or_first).rename(columns={"team":"team"}))

    # Filter by participation counts
    race_counts = res_y.groupby("driver")["position"].count()
    keep_race = set(race_counts[race_counts >= min_race_starts].index)

    quali_counts = quali_y.groupby("driver")["position"].count()
    keep_quali = set(quali_counts[quali_counts >= min_quali_starts].index)

    res_plot = res_y[res_y["driver"].isin(keep_race)].copy()
    quali_plot = quali_y[quali_y["driver"].isin(keep_quali)].copy()

    # Driver order by median race position (lower = better)
    median_order = (res_plot.groupby("driver")["position"]
                    .median().sort_values().index.tolist())

    # Build per-driver colors from their team (base color)
    drv_team_map = dict(zip(drv_team["driver"], drv_team["team"]))
    palette_race = [TEAM_COLORS.get(drv_team_map.get(drv, ""), "#999999") for drv in median_order]

    # Layout: two columns for the two boxplots
    c1, c2 = st.columns(2)

    #  Race results boxplot
    with c1:
        fig1, ax1 = driver_race_boxplot(res_plot, year, palette_race, median_order)
        st.pyplot(fig1, use_container_width=False, clear_figure=True)

    # Qualifying results boxplot
    with c2:
        # Use same driver order but keep only drivers with enough quali sessions
        quali_order = [d for d in median_order if d in set(quali_plot["driver"].unique())]
        palette_quali = [TEAM_COLORS.get(drv_team_map.get(d, ""), "#999999") for d in quali_order]

        fig2, ax2 = driver_quali_boxplot(quali_plot, year, palette_quali, quali_order)
        st.pyplot(fig2, use_container_width=False, clear_figure=True)

    st.caption("Tip: change the minimum starts to hide one-offs or replacements and focus on full-time drivers.")
with tab6:
    st.subheader("OpenF1 — Session browser & live timing")

    # --- Imports (local) ---
    from src.openf1 import q as q_openf1
    import pandas as pd
    import numpy as np

    # --- 1) Year comes from your sidebar 'year'
    year_sel = int(year)

    # --- 2) Fetch sessions for the year (robust to schema differences)
    try:
        sessions = q_openf1("/sessions", {"year": year_sel})
    except Exception as e:
        st.error(f"Cannot reach OpenF1: {e}")
        sessions = []

    if not sessions:
        st.info("No OpenF1 sessions for this year.")
        st.stop()

    ses = pd.DataFrame(sessions).drop_duplicates("session_key")

    # Keep only columns that exist
    wanted_cols = ["session_key","meeting_key","country_name","circuit_short_name",
             "session_name","session_type","date_start","date_end"]
    keep_cols = [c for c in wanted_cols if c in ses.columns]
    ses = ses[keep_cols]

    # Pick a sort column (most recent first)
    sort_candidates = [c for c in ["date_start","date_end"] if c in ses.columns]
    if sort_candidates:
        ses = ses.sort_values(sort_candidates[0], ascending=False)

    # --- 3) UI: choose circuit then session
    circuits = ses["circuit_short_name"].dropna().unique().tolist() if "circuit_short_name" in ses else []
    if not circuits:
        st.info("No circuit names in session payload.")
        st.stop()

    circuit = st.selectbox("Circuit", sorted(circuits))
    ses_circ = ses[ses["circuit_short_name"] == circuit] if "circuit_short_name" in ses else ses.copy()

    ses_circ = ses_circ.copy()
    if "session_status" in ses_circ:
        ses_circ["_label"] = ses_circ["session_name"].astype(str) + "  —  " + ses_circ["session_status"].astype(str)
    else:
        ses_circ["_label"] = ses_circ["session_name"].astype(str)

    pick_label = st.selectbox("Session", ses_circ["_label"].tolist(), index=0)
    session_key = int(ses_circ.loc[ses_circ["_label"] == pick_label, "session_key"].iloc[0])
    session_status = ses_circ.loc[ses_circ["_label"] == pick_label, "session_status"].iloc[0] if "session_status" in ses_circ else "Unknown"

    st.caption(f"Session key: {session_key} • Status: {session_status}")

    # --- 4) Drivers list (name, team) ---
    try:
        drivers = q_openf1("/drivers", {"session_key": session_key})
    except Exception:
        drivers = []
    df_drv = pd.DataFrame(drivers).drop_duplicates("driver_number") if drivers else pd.DataFrame(columns=["driver_number","full_name","team_name"])
    if "team_name" not in df_drv.columns:
        df_drv["team_name"] = np.nan
    # Normalize type for merge
    if not df_drv.empty:
        df_drv["driver_number"] = pd.to_numeric(df_drv["driver_number"], errors="coerce").astype("Int64")

    # --- 5) Positions (latest known) ---
    try:
        pos = q_openf1("/position", {"session_key": session_key, "order_by": "date"})
    except Exception:
        pos = []
    df_pos = pd.DataFrame(pos)
    if not df_pos.empty:
        keep = [c for c in ["driver_number","position","date"] if c in df_pos.columns]
        df_pos = df_pos[keep].copy()
        if "driver_number" in df_pos.columns and "position" in df_pos.columns:
            if "date" in df_pos.columns:
                df_pos = (df_pos.sort_values("date")
                                .dropna(subset=["driver_number"])
                                .groupby("driver_number", as_index=False)
                                .tail(1))
            df_pos["driver_number"] = pd.to_numeric(df_pos["driver_number"], errors="coerce").astype("Int64")
            df_pos["position"] = pd.to_numeric(df_pos["position"], errors="coerce").astype("Int64")
        else:
            df_pos = pd.DataFrame(columns=["driver_number","position"])
    else:
        df_pos = pd.DataFrame(columns=["driver_number","position"])

    # --- 6) Laps (for best lap per driver + global best) ---
    try:
        laps = q_openf1("/laps", {"session_key": session_key})
    except Exception:
        laps = []
    df_laps = pd.DataFrame(laps)

    if not df_laps.empty and {"lap_duration","driver_number"}.issubset(df_laps.columns):
        df_laps = df_laps.copy()
        df_laps["driver_number"] = pd.to_numeric(df_laps["driver_number"], errors="coerce").astype("Int64")
        df_laps["lap_duration"] = pd.to_numeric(df_laps["lap_duration"], errors="coerce")
        best_per_driver = (df_laps.dropna(subset=["lap_duration"])
                                    .sort_values(["driver_number","lap_duration"])
                                    .groupby("driver_number", as_index=False)
                                    .first()[["driver_number","lap_number","lap_duration"]])
        if not best_per_driver.empty:
            idx_best = best_per_driver["lap_duration"].idxmin()
            global_best_driver = int(best_per_driver.loc[idx_best, "driver_number"])
            global_best_time = float(best_per_driver.loc[idx_best, "lap_duration"])
        else:
            global_best_driver, global_best_time = None, None
    else:
        best_per_driver = pd.DataFrame(columns=["driver_number","lap_number","lap_duration"])
        global_best_driver, global_best_time = None, None

    # Formatter mm:ss.mmm
    def fmt(sec):
        try:
            sec = float(sec); m = int(sec // 60); s = sec % 60
            return f"{m}:{s:06.3f}"
        except Exception:
            return None

    if "lap_duration" in best_per_driver:
        best_per_driver["Best Lap (s)"] = best_per_driver["lap_duration"]

    # --- 7) Build classification table ---
    base_cols = ["driver_number","full_name","team_name"]
    table = df_drv[base_cols].copy() if not df_drv.empty else pd.DataFrame(columns=base_cols)
    table = table.merge(df_pos, on="driver_number", how="left")

    if not best_per_driver.empty:
        bp = best_per_driver.rename(columns={"lap_number": "Best Lap #"})
        table = table.merge(bp[["driver_number","Best Lap #","Best Lap (s)"]], on="driver_number", how="left")

    # Order: by live Pos if present
    if "position" in table.columns and table["position"].notna().any():
        table = table.sort_values("position", na_position="last")
    else:
        table = table.sort_values("full_name")

    # Format best lap time
    if "Best Lap (s)" in table.columns:
        table["Best Lap (s)"] = table["Best Lap (s)"].apply(lambda x: fmt(x) if pd.notna(x) else None)

    disp = table.rename(columns={
        "position": "Pos",
        "full_name": "Driver",
        "team_name": "Team"
    })

    st.markdown("### Classification (live / last known)")

    TEAM_COLORS = {
        "Ferrari": "#DC0000",
        "Mercedes": "#28302F",
        "McLaren": "#FF8700",
        "Red Bull Racing": "#1E41FF",
        "Williams": "#0082FA",
        "Aston Martin": "#006F62",
        "Haas F1 Team": "#B6BABD",
        "Alpine": "#A364AC",
        "Racing Bulls": "#2B4562",
        "Kick Sauber": "#27BB14"
    }

    def team_row_style(row):
        """Default style: shade full row in team color"""
        team = row.get("Team", None)
        color = TEAM_COLORS.get(team, "#FFFFFF")
        text_color = "white" if color.lower() in {
            "#dc0000","#28302f","#ff8700","#1e41ff","#006f62","#2b4562","#a364ac"
        } else "black"
        return [f"background-color: {color}; color: {text_color}"] * len(row)

    def best_lap_cell_style(row):
        """Override: if global best driver, color only Best Lap (s) cell in purple"""
        styles = [''] * len(row)
        if (global_best_driver is not None) and (row.get("driver_number") == global_best_driver):
            try:
                idx = row.index.get_loc("Best Lap (s)")
                styles[idx] = "background-color: #EDE7FF; color: #5B2EFF"
            except Exception:
                pass
        return styles

    # Prepare display dataframe
    # Prepare display dataframe (robust to missing cols)
    disp = table.rename(columns={
        "position": "Pos",
        "full_name": "Driver",
        "team_name": "Team"
    }).copy()

    desired_cols = ["Driver", "Team", "Best Lap #", "Best Lap (s)"]
    present_cols = [c for c in desired_cols if c in disp.columns]

    if not present_cols:
        # Nothing useful to show (e.g., no drivers yet for this session)
        st.info("No classification data available to display for this session.")
    else:
        # Include driver_number only for styling/hiding if present
        extra_helpers = ["driver_number"] if "driver_number" in disp.columns else []
        view = disp[present_cols + extra_helpers].copy()

        styler = (view.style
                .apply(team_row_style, axis=1)
                .apply(best_lap_cell_style, axis=1))

        # Hide helper columns (cross-version friendly)
        try:
            if extra_helpers:
                styler = styler.hide(axis="columns", subset=extra_helpers)  # pandas ≥1.4
        except Exception:
            try:
                if extra_helpers:
                    styler = styler.hide_columns(extra_helpers)             # older pandas
            except Exception:
                pass

        st.dataframe(styler, use_container_width=True, hide_index=True)


    st.markdown("---")

    # --- 8) Subtabs: Live refresh & Recent laps
    live_tab, laps_tab, pace_tab = st.tabs(["Live (auto-refresh)", "Recent laps", "Pace (boxplot)"])

    with live_tab:
        st.caption("Auto-refresh if the session is live (status: Started/Active).")
        refresh_sec = st.number_input("Refresh interval (sec)", min_value=3, max_value=30, value=6, step=1)
        if str(session_status).lower() in {"started","active","live","running"}:
            st.info("Session appears live — auto-refresh enabled.")
            # Trigger periodic refresh
            st.experimental_rerun() if st.autorefresh(interval=refresh_sec * 1000, key=f"openf1_{session_key}") else None
        else:
            st.info("Session is not live right now.")

    with laps_tab:
        n_show = st.slider("Show last N laps per driver", 5, 50, 20, 1)
        required = {"driver_number","lap_number","lap_duration"}
        if df_laps.empty or not required.issubset(df_laps.columns):
            st.info("No laps available for this session.")
        else:
            df_laps = df_laps.copy()
            df_laps["driver_number"] = pd.to_numeric(df_laps["driver_number"], errors="coerce").astype("Int64")
            df_laps["lap_duration"] = pd.to_numeric(df_laps["lap_duration"], errors="coerce")
            df_laps["lap_number"]   = pd.to_numeric(df_laps["lap_number"], errors="coerce").astype("Int64")
            df_laps = df_laps.sort_values(["driver_number","lap_number"])

            # Ensure optional cols exist
            if "is_pit" not in df_laps.columns:
                df_laps["is_pit"] = False

            # Keep only columns that exist (robust)
            keep_cols = [c for c in ["driver_number","lap_number","lap_duration","is_pit"] if c in df_laps.columns]
            lastN = (df_laps.groupby("driver_number", as_index=False)
                            .tail(n_show)[keep_cols])

            # ---------- NEW: fetch sectors & merge ----------
            def _fmt_sector(x: float):
                try:
                    x = float(x)
                    if x >= 60:
                        m = int(x // 60); s = x % 60
                        return f"{m}:{s:06.3f}"
                    else:
                        return f"{x:0.3f}"
                except Exception:
                    return None

            # Try /sectors endpoint
            try:
                sectors = q_openf1("/sectors", {"session_key": session_key})
            except Exception:
                sectors = []

            df_sec = pd.DataFrame(sectors)
            sec_pivot = pd.DataFrame()

            if not df_sec.empty:
                # Normalize possible column names
                rename_map = {}
                for a, b in [
                    ("driver", "driver_number"),
                    ("driverId", "driver_number"),
                    ("lap", "lap_number"),
                    ("time", "duration"),
                    ("value", "duration"),
                    ("sector_number", "sector"),
                    ("sectorId", "sector"),
                ]:
                    if a in df_sec.columns and b not in df_sec.columns:
                        rename_map[a] = b
                if rename_map:
                    df_sec = df_sec.rename(columns=rename_map)

                needed = {"driver_number","lap_number","sector","duration"}
                if needed.issubset(df_sec.columns):
                    df_sec["driver_number"] = pd.to_numeric(df_sec["driver_number"], errors="coerce").astype("Int64")
                    df_sec["lap_number"]   = pd.to_numeric(df_sec["lap_number"], errors="coerce").astype("Int64")
                    df_sec["sector"]       = pd.to_numeric(df_sec["sector"], errors="coerce").astype("Int64")
                    df_sec["duration"]     = pd.to_numeric(df_sec["duration"], errors="coerce")

                    # If duplicates per (driver,lap,sector), take min duration
                    df_sec = (df_sec.sort_values(["driver_number","lap_number","sector"])
                                    .groupby(["driver_number","lap_number","sector"], as_index=False)["duration"].min())

                    sec_pivot = (
                        df_sec.pivot_table(index=["driver_number","lap_number"],
                                        columns="sector", values="duration", aggfunc="min")
                            .rename(columns={1:"S1", 2:"S2", 3:"S3"})
                            .reset_index()
                    )

            # Fallback: try sector columns in df_laps
            if sec_pivot.empty:
                fallback_sets = [
                    ("duration_sector_1", "duration_sector_2", "duration_sector_3"),
                    ("sector1_time", "sector2_time", "sector3_time"),
                    ("s1", "s2", "s3"),
                ]
                for a, b, c in fallback_sets:
                    if {a, b, c}.issubset(df_laps.columns):
                        sec_pivot = (
                            df_laps[["driver_number","lap_number", a, b, c]]
                            .rename(columns={a:"S1", b:"S2", c:"S3"})
                            .copy()
                        )
                        for col in ["S1","S2","S3"]:
                            sec_pivot[col] = pd.to_numeric(sec_pivot[col], errors="coerce")
                        break

            # Merge sectors if available
            if not sec_pivot.empty:
                lastN = lastN.merge(sec_pivot, on=["driver_number","lap_number"], how="left")

            # ---------- Build view ----------
            # ---------- Build view (robust) ----------
            id2name = dict(zip(df_drv["driver_number"], df_drv["full_name"]))
            lastN["Driver"] = lastN["driver_number"].map(id2name)

            # --- Driver filter UI (after lastN["Driver"] is created) ---
            drivers_available = (
                lastN["Driver"].dropna().astype(str).sort_values().unique().tolist()
            )

            # Quick actions
            c1, c2 = st.columns([1,1])
            with c1:
                st.caption("Filter drivers")
            with c2:
                col_all, col_none = st.columns([1,1])
                select_all = col_all.button("Select all", key="laps_select_all")
                select_none = col_none.button("Clear", key="laps_select_none")

            # Default selection = all (or respond to quick buttons)
            default_selection = drivers_available if not drivers_available else drivers_available
            if select_all:
                default_selection = drivers_available
            elif select_none:
                default_selection = []

            selected_drivers = st.multiselect(
                "Drivers to display",
                options=drivers_available,
                default=default_selection,
                label_visibility="collapsed",
            )

            # Apply filter
            if selected_drivers:
                lastN = lastN[lastN["Driver"].isin(selected_drivers)].reset_index(drop=True)
            else:
                st.info("No drivers selected.")
                st.stop()

            # Keep numeric helpers BEFORE formatting; align indexes
            lastN = lastN.reset_index(drop=True)

            # Numeric helpers for styling
            lastN["lap_sec"] = pd.to_numeric(lastN.get("lap_duration"), errors="coerce")
            if "S1" in lastN.columns:
                lastN["S1_sec"] = pd.to_numeric(lastN["S1"], errors="coerce")
            if "S2" in lastN.columns:
                lastN["S2_sec"] = pd.to_numeric(lastN["S2"], errors="coerce")
            if "S3" in lastN.columns:
                lastN["S3_sec"] = pd.to_numeric(lastN["S3"], errors="coerce")
            # 🆕 Drop rows with NaN Lap Time
            lastN = lastN[lastN["lap_sec"].notna()].reset_index(drop=True)
            if lastN.empty:
                st.info("No valid lap times among the last N laps.")
                st.stop()
            # Pretty formatting (string columns for display)
            def _fmt_lap_total(x):
                try:
                    x = float(x); m = int(x // 60); s = x % 60
                    return f"{m}:{s:06.3f}"
                except Exception:
                    return None

            def _fmt_sector(x):
                try:
                    x = float(x)
                    if x >= 60:
                        m = int(x // 60); s = x % 60
                        return f"{m}:{s:06.3f}"
                    return f"{x:0.3f}"
                except Exception:
                    return None

            lastN["Lap Time"] = lastN["lap_sec"].apply(_fmt_lap_total)
            for col in ["S1","S2","S3"]:
                if col in lastN.columns:
                    lastN[col] = lastN[f"{col}_sec"].apply(_fmt_sector)

            # Columns to show
            display_cols = ["Driver", "lap_number", "Lap Time"]
            display_cols += [c for c in ["S1","S2","S3","is_pit"] if c in lastN.columns]

            # Build the view INCLUDING helper cols (we'll hide them)
            helper_cols = [c for c in ["lap_sec","S1_sec","S2_sec","S3_sec"] if c in lastN.columns]
            view = lastN[display_cols + helper_cols].rename(columns={"lap_number": "Lap", "is_pit": "Pit"}).copy()

            # ---------- Global bests (session-level) ----------
            import numpy as np
            best_lap_session_sec = pd.to_numeric(df_laps["lap_duration"], errors="coerce").min()

            best_S1_session_sec = best_S2_session_sec = best_S3_session_sec = np.nan
            if "sec_pivot" in locals() and isinstance(sec_pivot, pd.DataFrame) and not sec_pivot.empty:
                if "S1" in sec_pivot.columns: best_S1_session_sec = pd.to_numeric(sec_pivot["S1"], errors="coerce").min()
                if "S2" in sec_pivot.columns: best_S2_session_sec = pd.to_numeric(sec_pivot["S2"], errors="coerce").min()
                if "S3" in sec_pivot.columns: best_S3_session_sec = pd.to_numeric(sec_pivot["S3"], errors="coerce").min()
            else:
                for a,b,c in [("duration_sector_1","duration_sector_2","duration_sector_3"),
                            ("sector1_time","sector2_time","sector3_time"),
                            ("s1","s2","s3")]:
                    if {a,b,c}.issubset(df_laps.columns):
                        best_S1_session_sec = pd.to_numeric(df_laps[a], errors="coerce").min()
                        best_S2_session_sec = pd.to_numeric(df_laps[b], errors="coerce").min()
                        best_S3_session_sec = pd.to_numeric(df_laps[c], errors="coerce").min()
                        break

            # ---------- Styling: orange default, green = best per driver, purple = best session ----------
            GREEN  = "background-color: #2ECC71; color: white"
            ORANGE = "background-color: #F39C12; color: black"
            PURPLE = "background-color: #5B2EFF; color: #EDE7FF"

            def color_best_with_session(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                target_pairs = [("lap_sec","Lap Time")]
                if "S1_sec" in df.columns and "S1" in df.columns: target_pairs.append(("S1_sec","S1"))
                if "S2_sec" in df.columns and "S2" in df.columns: target_pairs.append(("S2_sec","S2"))
                if "S3_sec" in df.columns and "S3" in df.columns: target_pairs.append(("S3_sec","S3"))

                # Orange by default where values exist
                for num_col, disp_col in target_pairs:
                    mask = df[num_col].notna()
                    styles.loc[mask, disp_col] = ORANGE

                # Green = per-driver best (min within visible lastN)
                for drv, idxs in df.groupby("Driver").groups.items():
                    g = df.loc[idxs]
                    for num_col, disp_col in target_pairs:
                        valid = g[num_col].dropna()
                        if not valid.empty:
                            styles.loc[valid.idxmin(), disp_col] = GREEN

                # Purple = session best (override)
                atol = 1e-6
                if "lap_sec" in df.columns and np.isfinite(best_lap_session_sec):
                    styles.loc[np.isclose(df["lap_sec"], best_lap_session_sec, atol=atol), "Lap Time"] = PURPLE
                if "S1_sec" in df.columns and np.isfinite(best_S1_session_sec):
                    styles.loc[np.isclose(df["S1_sec"], best_S1_session_sec, atol=atol), "S1"] = PURPLE
                if "S2_sec" in df.columns and np.isfinite(best_S2_session_sec):
                    styles.loc[np.isclose(df["S2_sec"], best_S2_session_sec, atol=atol), "S2"] = PURPLE
                if "S3_sec" in df.columns and np.isfinite(best_S3_session_sec):
                    styles.loc[np.isclose(df["S3_sec"], best_S3_session_sec, atol=atol), "S3"] = PURPLE

                return styles



                        # --- show only selected columns, but keep helpers for styling ---
        desired = ["Driver", "Lap", "Lap Time", "S1", "S2", "S3"]
        helpers = [c for c in ["lap_sec","S1_sec","S2_sec","S3_sec"] if c in view.columns]

        present = [c for c in desired if c in view.columns]
        df_for_style = view[present + helpers].copy()

        styler = df_for_style.style.apply(color_best_with_session, axis=None)

            # hide helper columns (best-effort across pandas versions)
        try:
            styler = styler.hide(axis="columns", subset=helpers)          # pandas ≥ 1.4
        except Exception:
            try:
                styler = styler.hide_columns(helpers)                     # older pandas
            except Exception:
                df_for_style = df_for_style.drop(columns=helpers, errors="ignore")
                styler = df_for_style.style.apply(color_best_with_session, axis=None)

        st.dataframe(styler, hide_index=True, use_container_width=True)



    with pace_tab:
        import plotly.express as px

        st.caption("Boxplot of lap times per driver for this session (robust filters).")

        if df_laps.empty or not {"driver_number","lap_duration","lap_number"}.issubset(df_laps.columns):
            st.info("No laps available for this session.")
        else:
            pace = df_laps.copy()
            pace["driver_number"] = pd.to_numeric(pace["driver_number"], errors="coerce").astype("Int64")
            pace["lap_duration"] = pd.to_numeric(pace["lap_duration"], errors="coerce")
            pace["lap_number"]   = pd.to_numeric(pace["lap_number"], errors="coerce").astype("Int64")
            pace = pace.dropna(subset=["driver_number","lap_duration","lap_number"])

            # Optional columns
            if "is_pit" not in pace.columns:
                pace["is_pit"] = False
            if "stint" not in pace.columns:
                pace["stint"] = pd.NA

            # Map names & teams
            id2name = dict(zip(df_drv["driver_number"], df_drv["full_name"]))
            id2team = dict(zip(df_drv["driver_number"], df_drv["team_name"]))
            pace["Driver"] = pace["driver_number"].map(id2name)
            pace["Team"]   = pace["driver_number"].map(id2team)

            # === Controls
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                drop_pit = st.checkbox("Exclude pit laps (if known)", value=True)
            with c2:
                top_pct = st.slider("Keep fastest X% per driver", 40, 100, 100, 5,
                                    help="Keeps the fastest fraction of laps per driver (after basic cleaning).")
                min_laps = st.number_input("Min laps per driver", min_value=3, max_value=30, value=4, step=1)
            with c3:
                slow_thr = st.number_input("Slow-lap threshold vs baseline (sec)", min_value=2.0, max_value=20.0, value=10.0, step=0.5,
                                        help="Drops laps slower than (driver 20th percentile + threshold).")
                use_mad = st.checkbox("Apply robust MAD filter", value=True,
                                    help="Per driver, keep laps within median ± 3×MAD.")
            metric = st.selectbox("Metric", ["Lap time (s)", "Δ to session best (s)"])
            show_points = st.checkbox("Show outlier markers", value=False)

            # === Cleaning
            clean = pace.copy()

            # 1) Known pit laps
            if drop_pit:
                clean = clean[~clean["is_pit"].fillna(False)]

      
            # 3) Drop laps slower than (driver baseline + threshold), baseline = 20th pct
            def drop_slow_vs_baseline(g):
                if g.empty:
                    return g
                base = g["lap_duration"].quantile(0.20)
                return g[g["lap_duration"] <= base + slow_thr]
            clean = clean.groupby("driver_number", group_keys=False).apply(drop_slow_vs_baseline)

            # 4) Robust MAD filter (optional)
            if use_mad and not clean.empty:
                def mad_filter(g):
                    med = g["lap_duration"].median()
                    mad = (g["lap_duration"].sub(med).abs().median())
                    if pd.isna(mad) or mad == 0:
                        return g  # nothing to do
                    k = 3.0
                    lo, hi = med - k*mad, med + k*mad
                    return g[(g["lap_duration"] >= lo) & (g["lap_duration"] <= hi)]
                clean = clean.groupby("driver_number", group_keys=False).apply(mad_filter)

            # 5) Keep fastest X% per driver
            def keep_fastest(g):
                q = np.clip(top_pct/100.0, 0.0, 1.0)
                thr = g["lap_duration"].quantile(q)
                return g[g["lap_duration"] <= thr]
            clean = clean.groupby("driver_number", group_keys=False).apply(keep_fastest)

            # 6) Drop drivers with too few laps
            counts = clean.groupby("driver_number").size()
            keep_ids = counts[counts >= min_laps].index
            clean = clean[clean["driver_number"].isin(keep_ids)]

            # === Metric transform
            if clean.empty:
                st.info("No pace data to plot after filters.")
            else:
                if metric == "Δ to session best (s)":
                    session_best = clean["lap_duration"].min()
                    clean["y_val"] = clean["lap_duration"] - session_best
                    y_label = "Δ to session best (s)"
                else:
                    clean["y_val"] = clean["lap_duration"]
                    y_label = "Lap time (s)"

                TEAM_COLORS = {
                        "Ferrari": "#DC0000",
                        "Mercedes": "#28302F",
                        "McLaren": "#FF8700",
                        "Red Bull Racing": "#1E41FF",
                        "Williams": "#0082FA",
                        "Aston Martin": "#006F62",
                        "Haas F1 Team": "#B6BABD",
                        "Alpine": "#A364AC",
                        "Racing Bulls": "#2B4562",
                        "Kick Sauber": "#27BB14"
                    }
                # # Order drivers by median pace
                # med = clean.groupby("Driver")["y_val"].median().sort_values()
                # ordered_drivers = med.index.tolist()
                # Count laps considered per driver
                counts = clean.groupby("Driver").size()

                # Put n into the label shown on the x-axis
                label_map = {d: f"{d} (n={int(counts.loc[d])})" for d in counts.index}
                clean["DriverLabel"] = clean["Driver"].map(label_map)

                # Keep your median-based order but swap to the new labels
                med = clean.groupby("Driver")["y_val"].median().sort_values()
                ordered_drivers = med.index.tolist()

                # Also add n into hover via customdata
                clean["n_laps"] = clean["Driver"].map(counts)
                fig = px.box(
                    clean,
                    x="DriverLabel",
                    y="y_val",
                    color="Team",
                    points="suspectedoutliers" if show_points else False,
                    category_orders={"Driver": ordered_drivers},
                    labels={"y_val": y_label, "DriverLabel": "Driver (n laps)"},
                    hover_data={"n_laps": True, "Team": True},  # show n and team in hover
                    color_discrete_map=TEAM_COLORS
                )

                fig.update_traces(boxmean=False, width=0.6, line=dict(width=2))  
                fig.update_traces(customdata=np.stack([clean["n_laps"]], axis=-1),
                   hovertemplate="Driver: %{x}<br>n: %{customdata[0]}<br>Value: %{y:.3f}s<extra></extra>")

                fig.update_layout(
                    xaxis_title="Driver",
                    yaxis_title=y_label,
                    boxmode="group",
                    boxgap=0.2,       # spacing between boxes of different drivers
                    boxgroupgap=0.1,  # spacing between groups if multiple traces per category
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=520
                )
                fig.update_xaxes(tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)


# =========================
# === NEW: 🤖 AI GP Assistant (Conversational + Entities)
# =========================
with tab7:


    st.subheader("Chat with the GP Assistant")
    st.caption("Examples: 'latest race results', 'fastest lap Norris', 'points for Russell', '#1 pit stops', 'Singapore quali winner'")

    # ----------------- Intent patterns -----------------
    INTENTS = {
        "fastest_lap": re.compile(r"\b(fastest|best)\s+lap\b", re.I),
        "pits":        re.compile(r"\b(pit\s*stops?|pitstop|arrêts?)\b", re.I),
        "results":     re.compile(r"\b(result|classification|classement|winner|who\s+won)\b", re.I),
        "points":      re.compile(r"\b(point|points)\b", re.I),
        "tyres":       re.compile(r"\b(tyres?|pneus?|stints?|compound)\b", re.I),
        "status":      re.compile(r"\b(safety\s*car|vsc|virtual\s*safety\s*car|red\s*flag|track\s*status)\b", re.I),
    }
    def detect_intent(text: str) -> str:
        for k, rx in INTENTS.items():
            if rx.search(text or ""):
                return k
        return "results"  # fallback to something broadly useful

    # ------------- Session inference (same idea, robust) -------------
    SESSION_SYNONYMS = {
        "Race":       [r"\brace\b", r"\bgp\b", r"grand\s*prix", r"\bcourse\b", r"\blatest\b", r"\blast\b"],
        "Qualifying": [r"\bqualifying\b", r"\bquali\b", r"\bq\b"],
        "Sprint":     [r"\bsprint\b"],
        "FP1":        [r"\bfp1\b", r"practice\s*1", r"free\s*practice\s*1", r"\bp1\b"],
        "FP2":        [r"\bfp2\b", r"practice\s*2", r"free\s*practice\s*2", r"\bp2\b"],
        "FP3":        [r"\bfp3\b", r"practice\s*3", r"free\s*practice\s*3", r"\bp3\b"],
    }
    SESSION_ORDER = ["Race","Qualifying","Sprint","FP1","FP2","FP3"]

    def extract_year(q: str, default_year: int) -> int:
        m = re.search(r"\b(20\d{2})\b", q or "")
        return int(m.group(1)) if m else int(default_year)

    def extract_session_hint(q: str) -> str | None:
        ql = (q or "").lower()
        for canon, pats in SESSION_SYNONYMS.items():
            if any(re.search(p, ql) for p in pats):
                return canon
        return None

    def strip_session_words(q: str) -> str:
        q2 = (q or "")
        for pats in SESSION_SYNONYMS.values():
            for p in pats:
                q2 = re.sub(p, " ", q2, flags=re.I)
        q2 = re.sub(r"\b(grand|prix|race|gp|qualifying|quali|sprint|practice|free|latest|last)\b", " ", q2, flags=re.I)
        q2 = re.sub(r"[^A-Za-z0-9\s\-]", " ", q2)
        q2 = re.sub(r"\s+", " ", q2).strip()
        return q2

    def list_sessions_openf1(years: list[int]) -> pd.DataFrame:
        dfs = []
        for y in years:
            try:
                d = pd.DataFrame(q_openf1("/sessions", {"year": int(y)}))
                if not d.empty:
                    d["year"] = int(y)
                    dfs.append(d)
            except Exception:
                pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def build_search_str(row) -> str:
        return " | ".join([
            str(row.get("meeting_name","")).lower(),
            str(row.get("country_name","")).lower(),
            str(row.get("location","")).lower(),
            str(row.get("circuit_short_name","")).lower(),
        ])

    def find_best_session(nl_query: str, default_year: int) -> tuple[int | None, dict]:
        year_hint = extract_year(nl_query, default_year)
        sess_hint = extract_session_hint(nl_query)

        cache_key = f"sessions_idx_{year_hint}"
        if cache_key not in st.session_state:
            ses = list_sessions_openf1([year_hint])
            st.session_state[cache_key] = ses
        else:
            ses = st.session_state[cache_key]

        if ses.empty or "session_key" not in ses.columns:
            return None, {"reason": "no sessions", "year": year_hint}

        ses = ses.copy()
        for c in ["meeting_name","country_name","location","circuit_short_name","session_name","date_start"]:
            if c not in ses.columns: ses[c] = ""
        ses["_search"] = ses.apply(build_search_str, axis=1).astype(str)

        # tokens
        core = strip_session_words(nl_query)
        tokens = [t.lower() for t in core.split() if len(t) >= 3]
        tokens = list(dict.fromkeys(tokens))

        # filter by session type hint
        cand = ses
        if sess_hint:
            cand = cand[cand["session_name"].astype(str).str.contains(sess_hint, case=False, na=False)]

        # score
        if not tokens:
            cand = cand.assign(_score=0)
        else:
            s = cand["_search"].astype(str)
            score = np.zeros(len(cand), dtype=int)
            for t in tokens:
                score += s.str.contains(re.escape(t), case=False, na=False).astype(int).to_numpy()
            cand = cand.assign(_score=score)
            if (cand["_score"] > 0).any():
                cand = cand[cand["_score"] > 0]

        # sort by score & recency
        try:
            cand["_date_start"] = pd.to_datetime(cand["date_start"], errors="coerce", utc=True)
        except Exception:
            cand["_date_start"] = pd.NaT
        cand = cand.sort_values(["_score","_date_start","session_key"], ascending=[False, False, False])

        if cand.empty:
            return None, {"reason":"no candidates", "tokens": tokens, "sess_hint": sess_hint}

        # if user said latest (and no explicit type), prefer Race > Quali...
        if (not sess_hint) and re.search(r"\b(latest|last)\b", (nl_query or "").lower()):
            for name in SESSION_ORDER:
                sub = cand[cand["session_name"].astype(str).str.contains(name, case=False, na=False)]
                if not sub.empty:
                    row = sub.iloc[0]
                    return int(row["session_key"]), {"picked":"latest", "type":name}

        row = cand.iloc[0]
        return int(row["session_key"]), {
            "year": int(year_hint),
            "session_name": str(row.get("session_name","")),
            "meeting_name": str(row.get("meeting_name","")),
            "country_name": str(row.get("country_name","")),
            "score": int(row.get("_score",0))
        }

    # ------------- Driver directory + entity extraction -------------
    def driver_directory(session_key: int) -> pd.DataFrame:
        try:
            d = pd.DataFrame(q_openf1("/drivers", {"session_key": int(session_key)}))
        except Exception:
            d = pd.DataFrame()
        if d.empty:
            return pd.DataFrame(columns=["driver_number","full_name","broadcast_name"])
        for c in ["driver_number","full_name","broadcast_name","last_name","first_name","name_acronym","team_name"]:
            if c not in d.columns: d[c] = pd.NA
        d["driver_number"] = pd.to_numeric(d["driver_number"], errors="coerce").astype("Int64")
        return d

    def parse_driver_query(text: str, drv_dir: pd.DataFrame, last_driver: dict | None) -> dict | None:
        if drv_dir.empty:
            return None
        t = (text or "").lower()

        # #<number>
        m = re.search(r"#\s*(\d+)", t)
        if m:
            dn = int(m.group(1))
            row = drv_dir[drv_dir["driver_number"] == dn]
            if not row.empty:
                r = row.iloc[0]
                return {"driver_number": int(r["driver_number"]), "name": str(r.get("full_name") or r.get("broadcast_name") or "")}

        # last name (or full)
        # build maps
        names = []
        for _, r in drv_dir.iterrows():
            full = str(r.get("full_name") or r.get("broadcast_name") or "")
            last = str(r.get("last_name") or "").lower()
            first = str(r.get("first_name") or "").lower()
            names.append((int(r["driver_number"]), full, last, first))
        # try direct last-name match
        for dn, full, last, first in names:
            if last and re.search(rf"\b{re.escape(last)}\b", t):
                return {"driver_number": dn, "name": full}
            # full name
            fll = full.lower()
            if fll and re.search(rf"\b{re.escape(fll)}\b", t):
                return {"driver_number": dn, "name": full}
        # short follow-up like "and Norris?"
        if last_driver:
            if re.search(r"\b(and|also)\b", t) and not re.search(r"#\d+|\bpoint|fastest|pit|tyre|result|win|who\b", t):
                return last_driver
        return None

    # ------------- Data helpers -------------
    def name_map(session_key: int) -> dict[int,str]:
        d = driver_directory(session_key)
        mp = {}
        for _, r in d.iterrows():
            nm = r.get("full_name") or r.get("broadcast_name") or ""
            if pd.notna(r.get("driver_number")) and nm:
                mp[int(r["driver_number"])] = str(nm)
        return mp

    # ------------- Answer engines -------------
    def answer_results(session_key: int, top: int = 10) -> str:
        try:
            rs = pd.DataFrame(q_openf1("/results", {"session_key": int(session_key)}))
        except Exception:
            return "No results (endpoint error)."
        if rs.empty:
            return "No results for this session."
        for c in ["position","driver_number","team_name","status","points"]:
            if c not in rs.columns: rs[c] = pd.NA
        rs["position"] = pd.to_numeric(rs["position"], errors="coerce")
        rs = rs.dropna(subset=["position"]).sort_values("position")
        id2name = name_map(session_key)
        lines = []
        for _, r in rs.head(top).iterrows():
            nm = id2name.get(int(r["driver_number"]), f"#{int(r['driver_number'])}")
            pts = r.get("points")
            pts_str = f" — {float(pts):.0f} pts" if pd.notna(pts) else ""
            lines.append(f"P{int(r['position'])} {nm} ({str(r['team_name'])}){pts_str} [{str(r['status'])}]")
        return "\n".join(lines)

    def answer_fastest_lap(session_key: int, target_driver: dict | None) -> str:
        try:
            lp = pd.DataFrame(q_openf1("/laps", {"session_key": int(session_key)}))
        except Exception:
            return "No lap data."
        if lp.empty or "lap_duration" not in lp.columns:
            return "No lap data."
        for c in ["driver_number","lap_duration","lap_number","is_pit_out_lap","is_pit_in_lap"]:
            if c not in lp.columns: lp[c] = pd.NA
        lp["driver_number"] = pd.to_numeric(lp["driver_number"], errors="coerce").astype("Int64")
        lp = lp[~lp["is_pit_out_lap"].fillna(False) & ~lp["is_pit_in_lap"].fillna(False)]
        lp = lp.dropna(subset=["lap_duration","driver_number"])

        id2name = name_map(session_key)

        if target_driver:
            dn = target_driver["driver_number"]
            g = lp[lp["driver_number"] == dn]
            if g.empty: return f"No valid laps for {target_driver['name']}."
            r = g.sort_values("lap_duration").iloc[0]
            return f"Fastest lap for {id2name.get(dn, f'#{dn}')} — Lap {int(r['lap_number'])}, {float(r['lap_duration']):.3f}s"
        else:
            r = lp.sort_values("lap_duration").iloc[0]
            dn = int(r["driver_number"])
            return f"Session fastest lap — {id2name.get(dn, f'#{dn}')} Lap {int(r['lap_number'])}, {float(r['lap_duration']):.3f}s"

    def answer_pits(session_key: int, target_driver: dict | None) -> str:
        try:
            pt = pd.DataFrame(q_openf1("/pit", {"session_key": int(session_key)}))
        except Exception:
            return "No pit data."
        if pt.empty:
            return "No pit data."
        for c in ["driver_number"]: 
            if c not in pt.columns: pt[c] = pd.NA
        pt["driver_number"] = pd.to_numeric(pt["driver_number"], errors="coerce").astype("Int64")
        id2name = name_map(session_key)
        if target_driver:
            dn = target_driver["driver_number"]
            n = int((pt["driver_number"] == dn).sum())
            return f"{id2name.get(dn, f'#{dn}')} made {n} pit stop(s)."
        agg = (pt.groupby("driver_number").size().reset_index(name="stops").sort_values(["stops","driver_number"], ascending=[False, True]))
        lines = [f"{id2name.get(int(r.driver_number), f'#{int(r.driver_number)}')}: {int(r.stops)}" for _, r in agg.iterrows()]
        return "Pit stops:\n" + "\n".join(lines)

    def answer_points(session_key: int, target_driver: dict | None) -> str:
        """
        Uses OpenF1 /results.points if available. If not present, we can only say 'points not provided by API'.
        """
        try:
            rs = pd.DataFrame(q_openf1("/results", {"session_key": int(session_key)}))
        except Exception:
            return "No results to compute points."
        if rs.empty:
            return "No results to compute points."
        for c in ["driver_number","points"]:
            if c not in rs.columns: rs[c] = pd.NA
        rs["driver_number"] = pd.to_numeric(rs["driver_number"], errors="coerce").astype("Int64")
        if "points" not in rs.columns or rs["points"].isna().all():
            return "Points are not available in the OpenF1 results payload for this session."

        id2name = name_map(session_key)
        if target_driver:
            dn = target_driver["driver_number"]
            sub = rs[rs["driver_number"] == dn]
            if sub.empty or sub["points"].isna().all():
                return f"No points info for {id2name.get(dn, f'#{dn}')}"
            pts = float(sub["points"].iloc[0])
            return f"{id2name.get(dn, f'#{dn}')} scored {pts:.0f} point(s) in this session."
        else:
            top = (rs.dropna(subset=["points"])
                     .sort_values("points", ascending=False)[["driver_number","points"]]
                     .head(10))
            if top.empty: return "No points information available."
            lines = [f"{id2name.get(int(r.driver_number), f'#{int(r.driver_number)}')}: {float(r.points):.0f}"
                     for _, r in top.iterrows()]
            return "Top points (this session):\n" + "\n".join(lines)

    def answer_tyres(session_key: int, target_driver: dict | None) -> str:
        try:
            d = pd.DataFrame(q_openf1("/stints", {"session_key": int(session_key)}))
        except Exception:
            return "No tyre stint data."
        if d.empty:
            return "No tyre stint data."
        for c in ["driver_number","stint_number","lap_start","lap_end","compound"]:
            if c not in d.columns: d[c] = pd.NA
        d["driver_number"] = pd.to_numeric(d["driver_number"], errors="coerce").astype("Int64")
        d["stint_number"]  = pd.to_numeric(d["stint_number"], errors="coerce").astype("Int64")
        d["lap_start"]     = pd.to_numeric(d["lap_start"], errors="coerce")
        d["lap_end"]       = pd.to_numeric(d["lap_end"], errors="coerce")
        d["compound"]      = d["compound"].astype(str).str.upper()
        d["laps"] = (d["lap_end"] - d["lap_start"]).astype("Float64")
        d = d[(d["laps"].notna()) & (d["laps"] > 0)]
        id2name = name_map(session_key)

        def pack(g):
            parts = [f"{int(r.stint_number)}: {r.compound} ({int(r.laps)} laps)" for _, r in g.sort_values("stint_number").iterrows()]
            return " | ".join(parts)

        if target_driver:
            dn = target_driver["driver_number"]
            g = d[d["driver_number"] == dn]
            if g.empty: return f"No stints for {id2name.get(dn, f'#{dn}')}"
            return f"{id2name.get(dn, f'#{dn}')} → " + pack(g)
        lines = []
        for dn, g in d.groupby("driver_number"):
            lines.append(f"{id2name.get(int(dn), f'#{int(dn)}')} → " + pack(g))
        return "\n".join(lines[:20]) + ("\n… (truncated)" if len(lines) > 20 else "")

    def answer_status(session_key: int) -> str:
        try:
            ts = pd.DataFrame(q_openf1("/track_status", {"session_key": int(session_key)}))
        except Exception:
            return "No track status data."
        if ts.empty or "message" not in ts.columns:
            return "No track status events."
        vc = ts["message"].value_counts()
        return "Track status: " + " | ".join([f"{k} x{v}" for k, v in vc.items()])

    # ----------------- Conversation state -----------------
    if "ai_chat" not in st.session_state: st.session_state.ai_chat = []
    if "ai_ctx"  not in st.session_state: st.session_state.ai_ctx  = {"session_key": None, "year": int(year), "driver": None}

    # render history
    for role, msg in st.session_state.ai_chat:
        with st.chat_message(role): st.markdown(msg)

    user_msg = st.chat_input("Ask anything about this season/session, e.g. 'latest race results', 'fastest lap Norris', '#63 pits', 'points for Leclerc'")
    if user_msg:
        st.session_state.ai_chat.append(("user", user_msg))
        with st.chat_message("user"): st.markdown(user_msg)

        # 1) infer or reuse session
        default_year = int(year)
        skey = st.session_state.ai_ctx.get("session_key")
        dbg = {}
        if not skey or re.search(r"\b(20\d{2}|race|quali|sprint|fp\d|gp|grand|prix|latest|last)\b", user_msg.lower()):
            skey, dbg = find_best_session(user_msg, default_year)
            if not skey:
                msg = "I couldn't infer a session. Try adding a GP/country (e.g. 'Japan race', 'Monza quali', or say 'latest race')."
                with st.chat_message("assistant"):
                    st.markdown(msg)
                    if dbg: 
                        with st.expander("Debug"): st.json(dbg)
                st.session_state.ai_chat.append(("assistant", msg))
                st.stop()
            st.session_state.ai_ctx["session_key"] = int(skey)
            if "year" in dbg: st.session_state.ai_ctx["year"] = int(dbg["year"])

        # 2) driver directory & entity extraction (support follow-ups)
        drv_dir = driver_directory(int(skey))
        last_drv = st.session_state.ai_ctx.get("driver")
        target_driver = parse_driver_query(user_msg, drv_dir, last_drv)
        if target_driver: st.session_state.ai_ctx["driver"] = target_driver

        # 3) intent
        intent = detect_intent(user_msg)

        # 4) dispatch
        if intent == "results":
            ans = answer_results(int(skey))
        elif intent == "fastest_lap":
            ans = answer_fastest_lap(int(skey), target_driver)
        elif intent == "pits":
            ans = answer_pits(int(skey), target_driver)
        elif intent == "points":
            ans = answer_points(int(skey), target_driver)
        elif intent == "tyres":
            ans = answer_tyres(int(skey), target_driver)
        elif intent == "status":
            ans = answer_status(int(skey))
        else:
            ans = "I can help with **results/winner**, **fastest lap**, **points**, **pit stops**, **tyres**, **safety car**."

        with st.chat_message("assistant"):
            st.markdown(ans)
            with st.expander("Context"):
                st.json({
                    "intent": intent,
                    "session_key": int(skey),
                    "year": st.session_state.ai_ctx.get("year"),
                    "driver": st.session_state.ai_ctx.get("driver"),
                    **({ "session_debug": dbg } if dbg else {})
                })
        st.session_state.ai_chat.append(("assistant", ans))
