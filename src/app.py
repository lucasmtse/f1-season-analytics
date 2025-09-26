import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from src.data_jolpi import get_paged
from src.transformers import results_to_df, sprint_to_df, qualifying_to_df, driver_standings_to_df, constructor_standings_to_df
from src.standings import computed_driver_points, computed_constructor_points, cumulative_driver_points_by_round, teammate_split
from src.viz import cumulative_points_plot, constructor_share_pie, constructor_driver_stacked, constructor_quali_race, TEAM_COLORS, driver_race_boxplot, driver_quali_boxplot

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Standings", "Cumulative Points by Driver", "Quali vs Race", "Constructors", "Driver Consistency"])

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
    cum_df = cumulative_driver_points_by_round(res_df, year)

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
    st.subheader("Qualifying vs Race (per-driver averages)")
    q = quali_df[["driverId","driver","position"]].rename(columns={"position":"quali_pos"})
    r = res_df[["driverId","driver","team","position"]].rename(columns={"position":"race_pos"})
    merged = q.merge(r, on=["driverId","driver"], how="inner")
    avg = merged.groupby(["driverId","driver","team"])[["quali_pos","race_pos"]].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(6,5))
    teams = avg["team"].unique()
    palette = sns.color_palette("tab20", len(teams))
    color_map = dict(zip(teams, palette))
    ax.scatter(avg["quali_pos"], avg["race_pos"], c=avg["team"].map(color_map), s=70, edgecolor="k", alpha=0.85)
    texts = []
    for _, row in avg.iterrows():
        texts.append(ax.text(row["quali_pos"], row["race_pos"], row["driverId"], fontsize=8))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
    maxv = avg[["quali_pos","race_pos"]].max().max()
    ax.plot([1, maxv],[1, maxv], "k--", lw=1)
    ax.set_xlabel("Avg Qualifying Position (lower = better)")
    ax.set_ylabel("Avg Race Result (lower = better)")
    ax.set_title(f"Avg Quali vs Race — {year}")
    ax.invert_xaxis(); ax.invert_yaxis()
    handles = [plt.Line2D([0],[0], marker='o', color='w', label=t, 
                          markerfacecolor=color_map[t], markersize=8) for t in teams]
    ax.legend(handles=handles, title="Teams", bbox_to_anchor=(1.05,1), loc="upper left")
    st.pyplot(fig, use_container_width=False, clear_figure=True)

    st.subheader("Average position change (quali → race, therefore if >0 means average gain places)")
    merged["pos_change"] = merged["quali_pos"] - merged["race_pos"]
    avg_changes=round(merged.groupby(["driverId"])["pos_change"].mean().sort_values(ascending=True),2)
    pos_changes_df = pd.DataFrame(avg_changes).reset_index()
    # Prepare driver color palette (for later tabs)
    drv_team = (res_y.groupby(["driverId","driver"], as_index=False)["team"]
                    .agg(mode_or_first).rename(columns={"team":"team"}))
    drv_team_map_id = dict(zip(drv_team["driverId"], drv_team["team"]))
    palette_driver_id =[TEAM_COLORS.get(drv_team_map_id.get(d, ""), "#999999") for d in pos_changes_df.sort_values("pos_change", ascending=True)["driverId"]]
    fig, ax= constructor_quali_race(pos_changes_df, year, palette_driver_id)
    st.pyplot(fig, use_container_width=False, clear_figure=True)
with tab4:
    st.subheader("Constructor points by driver (stacked)")
    pts = teammate_split(res_df, sprint_df, year)
    fig2, ax2 = constructor_driver_stacked(pts, year, normalize=False)
    st.pyplot(fig2, use_container_width=False, clear_figure=True)

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
