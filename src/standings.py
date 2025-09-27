from __future__ import annotations
import pandas as pd

def computed_driver_points(res_df: pd.DataFrame, sprint_df: pd.DataFrame | None = None) -> pd.DataFrame:
    drv_race = (res_df.groupby(["year","driverId","driver"], as_index=False)["points"].sum()
                .rename(columns={"points":"race_points"}))
    if sprint_df is not None and not sprint_df.empty:
        drv_sprint = (sprint_df.groupby(["year","driverId","driver"], as_index=False)["points"].sum()
                      .rename(columns={"points":"sprint_points"}))
        drv = drv_race.merge(drv_sprint, on=["year","driverId","driver"], how="outer")
    else:
        drv = drv_race.copy()
        drv["sprint_points"] = 0.0
    drv = drv.fillna({"race_points":0.0, "sprint_points":0.0})
    drv["computed_points"] = drv["race_points"] + drv["sprint_points"]
    return drv

def computed_constructor_points(res_df: pd.DataFrame, sprint_df: pd.DataFrame | None = None) -> pd.DataFrame:
    con_race = (res_df.groupby(["year","constructorId","team"], as_index=False)["points"].sum()
                .rename(columns={"points":"race_points"}))
    if sprint_df is not None and not sprint_df.empty:
        con_sprint = (sprint_df.groupby(["year","constructorId","team"], as_index=False)["points"].sum()
                      .rename(columns={"points":"sprint_points"}))
        con = con_race.merge(con_sprint, on=["year","constructorId","team"], how="outer")
    else:
        con = con_race.copy()
        con["sprint_points"] = 0.0
    con = con.fillna({"race_points":0.0, "sprint_points":0.0})
    con["computed_points"] = con["race_points"] + con["sprint_points"]
    return con

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


def teammate_split(res_df: pd.DataFrame, sprint_df: pd.DataFrame | None, year: int) -> pd.DataFrame:
    race_pts = (res_df.query("year == @year")
                .groupby(["team","driver","driverId"], as_index=False)["points"].sum()
                .rename(columns={"points":"race_points"}))
    if sprint_df is not None and not sprint_df.empty:
        spr_pts = (sprint_df.query("year == @year")
                   .groupby(["team","driver","driverId"], as_index=False)["points"].sum()
                   .rename(columns={"points":"sprint_points"}))
        pts = race_pts.merge(spr_pts, on=["team","driver","driverId"], how="outer")
    else:
        pts = race_pts.copy()
        pts["sprint_points"] = 0.0
    pts = pts.fillna({"race_points":0.0, "sprint_points":0.0})
    pts["total_points"] = pts["race_points"] + pts["sprint_points"]
    return pts
