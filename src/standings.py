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

def cumulative_driver_points_by_round(res_df: pd.DataFrame, year: int) -> pd.DataFrame:
    sub = res_df[res_df["year"] == year].sort_values(["round","driver"])
    sub["cum_points"] = sub.groupby("driverId")["points"].cumsum()
    return sub

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
