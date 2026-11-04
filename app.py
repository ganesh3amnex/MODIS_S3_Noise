"""Crop Noise Explorer — Streamlit version.

Taluk-level land-cover noise in crop grids. All data processing and map
drawing run on the server; the browser only receives the rendered output.
"""
import io
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------- config
st.set_page_config(page_title="Crop Noise Explorer", layout="wide")

DATA_FILE = Path(__file__).parent / "data" / "crop_noise.json"
MIN_TALUK_HA = 5
NODATA = "#e6e5e1"
CONTEXT = "#f1f0ec"
SEASONS = ["Kharif", "Rabi", "Summer"]
YEARS = ["2024_25", "2025_26", "2026_27"]

ACRE = {"k": "Acreage", "name": "Crop Acreage (ha)", "unit": " ha",
        "ramp": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]}
VARS = [
    {"k": "Tree_Per", "name": "Tree_Noise(%)", "unit": "%",
     "ramp": ["#d9f0d3", "#a6dba0", "#5aae61", "#1b7837", "#0b4d1f"]},
    {"k": "OthVeg_Per", "name": "Other_Veg_Noise(%)", "unit": "%",
     "ramp": ["#d4f2e8", "#96dcc5", "#4dbd97", "#1b9670", "#0c5e45"]},
    {"k": "Build_Per", "name": "Buildup_Noise(%)", "unit": "%",
     "ramp": ["#fbdcd8", "#f4a49c", "#e3655f", "#bb3532", "#7a1b1a"]},
    {"k": "Bare_Per", "name": "Bare_Features_Noise(%)", "unit": "%",
     "ramp": ["#fbe6d0", "#f5bf8c", "#e58a45", "#b65e1f", "#723710"]},
]
MAPS = [ACRE] + VARS
K = math.cos(math.radians(22.5))  # longitude scale at ~22.5°N


# ---------------------------------------------------------------- helpers
def fmt(v, d=1):
    """Number with Indian digit grouping (12,34,567.8)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    neg = v < 0
    s = f"{abs(v):.{d}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + ("." + frac if frac else "")


def safe(s):
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def nice_max(m):
    if not m > 0:
        return 5
    p = 10 ** math.floor(math.log10(m))
    for f in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8, 10):
        if m <= f * p:
            return f * p
    return 10 * p


def classes(v, rows):
    mx = nice_max(max([r[v["k"]] for r in rows] + [0]))
    d = 0 if v["k"] == "Acreage" else (1 if mx < 5 else 0)
    br = [mx * i / 5 for i in range(6)]
    labels = [f"{fmt(br[k], d)}–{fmt(br[k + 1], d)}" for k in range(5)]

    def col(x):
        if x is None:
            return None
        return v["ramp"][min(4, int(math.floor(x / mx * 5)))]

    return labels, col


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    D = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    # rings stored as flat [lon*1000, lat*1000, ...]; convert once to (x, y) lists
    def to_xy(ring):
        return [(ring[j] / 1000 * K, ring[j + 1] / 1000) for j in range(0, len(ring), 2)]

    D["geo_xy"] = [[to_xy(r) for r in rings] for rings in D["geo"]]
    D["dist_xy"] = [[to_xy(r) for r in d[1]] for d in D["dists"]]
    del D["geo"]
    return D


D = load_data()
CROPS_AZ = sorted(D["crops"], key=str.lower)
DISTRICTS = sorted({t[0] for t in D["taluks"]})


def rows_for(crop, dist, th):
    src = D["agg"].get(crop, {}).get(th, [])
    out = []
    for r in src:
        t = D["taluks"][r[0]]
        if r[1] > MIN_TALUK_HA and (dist == "All_Districts" or t[0] == dist):
            out.append({
                "i": r[0], "UID": f"{t[0]},{t[1]}", "District": t[0], "Taluk": t[1],
                "LGD_T": t[2], "LGD_D": t[3], "Crop": crop, "Acreage": r[1],
                "Grids": r[2], "Area": r[3], "Tree_Per": r[4], "OthVeg_Per": r[5],
                "Build_Per": r[6], "Bare_Per": r[7],
            })
    return out


def totals(rs):
    ac = sum(r["Acreage"] for r in rs)
    t = {"Acreage": ac, "Grids": sum(r["Grids"] for r in rs), "Area": sum(r["Area"] for r in rs)}
    for v in VARS:
        t[v["k"]] = sum(r[v["k"]] * r["Acreage"] for r in rs) / ac if ac else None
    return t


def base_name(season, year, crop, dist):
    return "_".join([season, year, "M500", safe(crop), safe(dist), "Noise"])


# ---------------------------------------------------------------- maps
@st.cache_data(show_spinner="Drawing maps…")
def render_maps_png(crop, dist, th, season, year, dpi=150):
    rows = rows_for(crop, dist, th)
    total = totals(rows)
    by_idx = {r["i"]: r for r in rows}
    all_idx = range(len(D["taluks"]))
    if dist == "All_Districts":
        focus, ctx = list(all_idx), []
    else:
        focus = [i for i in all_idx if D["taluks"][i][0] == dist]
        ctx = [i for i in all_idx if D["taluks"][i][0] != dist]

    xs = [x for i in focus for ring in D["geo_xy"][i] for x, _ in ring]
    ys = [y for i in focus for ring in D["geo_xy"][i] for _, y in ring]
    padx, pady = (max(xs) - min(xs)) * 0.03, (max(ys) - min(ys)) * 0.03
    xlim, ylim = (min(xs) - padx, max(xs) + padx), (min(ys) - pady, max(ys) + pady)

    fig, axes = plt.subplots(1, 5, figsize=(26, 7.2))
    fig.patch.set_facecolor("white")
    title = base_name(season, year, crop, dist)
    fig.text(0.008, 0.975, title, fontsize=17, fontweight="bold", color="#0b0b0b", va="top")
    fig.text(0.008, 0.925,
             f"{crop} · {dist} · {season} {year} · {len(rows)} taluks (> {MIN_TALUK_HA} ha) · "
             f"{fmt(total['Acreage'], 0)} ha · {fmt(total['Grids'], 0)} grids · "
             "noise % weighted by crop acreage · bold lines = district boundaries",
             fontsize=11, color="#52514e", va="top")

    for ax, v in zip(axes, MAPS):
        labels, col = classes(v, rows)
        ax.set_facecolor("#fcfcfb")
        if ctx:
            polys = [ring for i in ctx for ring in D["geo_xy"][i]]
            ax.add_collection(PolyCollection(polys, facecolors=CONTEXT, edgecolors="#d6d5cf", linewidths=0.3))
        polys, fills = [], []
        for i in focus:
            r = by_idx.get(i)
            f = col(r[v["k"]]) if r else NODATA
            for ring in D["geo_xy"][i]:
                polys.append(ring)
                fills.append(f)
        ax.add_collection(PolyCollection(polys, facecolors=fills, edgecolors="white", linewidths=0.35))
        dpolys = [ring for d in D["dist_xy"] for ring in d]
        ax.add_collection(PolyCollection(dpolys, facecolors="none", edgecolors="#262624", linewidths=1.0))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e4e3de")
        sub = (f"total {fmt(total['Acreage'], 0)} ha" if v["k"] == "Acreage"
               else f"weighted avg {fmt(total[v['k']], 2)}%")
        ax.set_title(f"{v['name']}\n{sub}", fontsize=14, fontweight="bold", color="#0b0b0b", loc="left")
        pct = "%" if v["unit"] == "%" else ""
        handles = [Patch(facecolor=c, label=f"{labels[k]}{pct}") for k, c in enumerate(v["ramp"])]
        handles.append(Patch(facecolor=NODATA, label="No data / ≤5 ha"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
                  fontsize=10.5, frameon=False, handlelength=1.2, columnspacing=0.8)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.12, wspace=0.04)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- state
if "nav_idx" not in st.session_state:
    st.session_state.nav_idx = 0


def reset_nav():
    st.session_state.nav_idx = 0


# ---------------------------------------------------------------- UI
st.title("Crop Noise Explorer")
st.caption(
    "Taluk-level land-cover noise in crop grids. Grids are MODIS cells (~21.4 ha). For each taluk, "
    "the Tree, Other Veg, Buildup and Bare Features % of every grid that has the selected crop is "
    "averaged, weighted by that grid's crop acreage. Only taluks with crop acreage > 5 ha are shown and exported."
)

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.4, 1.2])
season = c1.selectbox("Season", SEASONS)
year = c2.selectbox("Year", YEARS, index=YEARS.index("2026_27"))
crop_options = ["All_Crops"] + CROPS_AZ
crop_sel = c3.selectbox("Crop", crop_options, index=crop_options.index("Cotton"), on_change=reset_nav)
dist = c4.selectbox("District", ["All_Districts"] + DISTRICTS)
th_values = D["th"]
th_label = {t: ("> 0 ha (any crop)" if t == 0 else f"> {t:g} ha") for t in th_values}
th_num = c5.selectbox("Grid crop acreage", th_values, format_func=lambda t: th_label[t])
th = str(th_num)  # data keys: "0", "0.5", "1", "2", "5"

is_all = crop_sel == "All_Crops"

# crop stepper for All_Crops
if is_all:
    n1, n2, n3 = st.columns([1, 4, 1])
    if n1.button("◀ Previous", disabled=st.session_state.nav_idx == 0, width="stretch"):
        st.session_state.nav_idx -= 1
        st.rerun()
    if n3.button("Next ▶", disabled=st.session_state.nav_idx == len(CROPS_AZ) - 1, width="stretch"):
        st.session_state.nav_idx += 1
        st.rerun()
    cur = CROPS_AZ[st.session_state.nav_idx]
    n2.markdown(
        f"<div style='text-align:center;font-size:18px;font-weight:650'>{cur}"
        f"<div style='font-size:12px;font-weight:400;opacity:.7'>Crop {st.session_state.nav_idx + 1} "
        f"of {len(CROPS_AZ)} (A→Z)</div></div>", unsafe_allow_html=True)
else:
    cur = crop_sel

rows = rows_for(cur, dist, th)
total = totals(rows)
csv_name = base_name(season, year, crop_sel, dist) + ".csv"
png_name = base_name(season, year, cur, dist) + "_Maps.png"
st.caption(f"Output name: `{csv_name}`  |  maps: `{png_name}`")

# KPIs
kpis = [("Crop", cur), ("Taluks", fmt(len(rows), 0)), ("Crop acreage", f"{fmt(total['Acreage'], 0)} ha"),
        ("Grids", fmt(total["Grids"], 0)), ("Grids area", f"{fmt(total['Area'], 0)} ha")]
kpis += [(v["name"], f"{fmt(total[v['k']], 2)} %") for v in VARS]
for chunk in (kpis[:5], kpis[5:]):
    cols = st.columns(5)
    for c, (label, val) in zip(cols, chunk):
        c.metric(label, val)

# maps
png = render_maps_png(cur, dist, th, season, year)
st.image(png, width="stretch")

# downloads
def build_csv():
    crops = CROPS_AZ if is_all else [cur]
    out = []
    for c in crops:
        out += sorted(rows_for(c, dist, th), key=lambda r: r["UID"])
    df = pd.DataFrame([{
        "Taluk_UID": r["UID"], "District": r["District"], "Taluk": r["Taluk"], "LGD_T": r["LGD_T"],
        "LGD_D": r["LGD_D"], "Season": season, "Year": year, "Crop": r["Crop"],
        "Acreage_ha": f"{r['Acreage']:.1f}", "Total_Grids": r["Grids"], "Grids_Area_ha": f"{r['Area']:.1f}",
        "Tree_Noise(%)": f"{r['Tree_Per']:.2f}", "Other_Veg_Noise(%)": f"{r['OthVeg_Per']:.2f}",
        "Buildup_Noise(%)": f"{r['Build_Per']:.2f}", "Bare_Features_Noise(%)": f"{r['Bare_Per']:.2f}",
    } for r in out], columns=["Taluk_UID", "District", "Taluk", "LGD_T", "LGD_D", "Season", "Year", "Crop",
                              "Acreage_ha", "Total_Grids", "Grids_Area_ha", "Tree_Noise(%)",
                              "Other_Veg_Noise(%)", "Buildup_Noise(%)", "Bare_Features_Noise(%)"])
    return ("﻿" + df.to_csv(index=False, lineterminator="\r\n")).encode("utf-8")


d1, d2, _ = st.columns([1, 1, 4])
d1.download_button("Download CSV", build_csv(), file_name=csv_name, mime="text/csv",
                   type="primary", width="stretch")
d2.download_button("Download maps (PNG)", render_maps_png(cur, dist, th, season, year, dpi=200),
                   file_name=png_name, mime="image/png", width="stretch")

# table
if rows:
    df = pd.DataFrame(rows).sort_values("Acreage", ascending=False)
    df = df[["UID", "District", "Taluk", "LGD_T", "Crop", "Acreage", "Grids", "Area",
             "Tree_Per", "OthVeg_Per", "Build_Per", "Bare_Per"]]
    colcfg = {
        "UID": "Taluk_UID",
        "LGD_T": st.column_config.TextColumn("LGD_T"),
        "Acreage": st.column_config.NumberColumn("Acreage (ha)", format="%.1f"),
        "Grids": st.column_config.NumberColumn("Total Grids", format="%d"),
        "Area": st.column_config.NumberColumn("Grids Area (ha)", format="%.1f"),
        "Tree_Per": st.column_config.NumberColumn("Tree_Noise(%)", format="%.2f"),
        "OthVeg_Per": st.column_config.NumberColumn("Other_Veg_Noise(%)", format="%.2f"),
        "Build_Per": st.column_config.NumberColumn("Buildup_Noise(%)", format="%.2f"),
        "Bare_Per": st.column_config.NumberColumn("Bare_Features_Noise(%)", format="%.2f"),
    }
    df["LGD_T"] = df["LGD_T"].astype(str)
    st.dataframe(df, column_config=colcfg, hide_index=True, width="stretch", height=560)
    tot = pd.DataFrame([{
        "UID": f"Total ({len(rows)} taluks)", "District": dist, "Taluk": "", "LGD_T": "", "Crop": cur,
        "Acreage": total["Acreage"], "Grids": total["Grids"], "Area": total["Area"],
        **{v["k"]: total[v["k"]] for v in VARS},
    }])
    st.dataframe(tot, column_config=colcfg, hide_index=True, width="stretch")
else:
    st.info(f"No taluk with more than {MIN_TALUK_HA} ha of {cur} for this selection.")

st.caption(
    "Taluks are kept only when their crop acreage is more than 5 ha. Total Grids = grids in the taluk whose "
    "crop acreage passes the grid threshold. Grids Area = sum of those grids' geometric area (equal-area "
    "projection). Noise % = Σ(grid % × grid crop acreage) ÷ Σ(grid crop acreage); the total row is the same "
    "weighted average over all listed taluks. With All_Crops, the arrows step through crops A→Z; the maps and "
    "table show the current crop, and Download CSV exports every crop. Season and Year only name the output "
    "files; the data loaded is Kharif 2026-27. Bold lines are district boundaries. Click a column header to sort."
)
