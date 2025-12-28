# app.py
import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ----------------------------
# Fix prefixek + jelmagyarázat
# ----------------------------
DOMAIN_PREFIXES = {
    "Matematika": ["Mat", "MA", "MD", "MG"],
    "Olvasás": ["Olv", "OA", "OD", "OG"],
    "Természettudomány": ["Term", "TA", "TD", "TG"],
}
EXTRA_PREFIXES = ["GH"]  # Géphaszn.

PREFIX_LABELS = {
    "Mat": "Mat. össz.",
    "MA": "Mat. alk.",
    "MD": "Mat. disz.",
    "MG": "Mat. gond.",
    "Olv": "Olv. össz.",
    "OA": "Olv. alk.",
    "OD": "Olv. disz.",
    "OG": "Olv. gond.",
    "Term": "Term. össz.",
    "TA": "Term. alk.",
    "TD": "Term. disz.",
    "TG": "Term. gond.",
    "GH": "Géphaszn.",
}

MISSING_NAME = "#HIÁNYZIK"

AVERAGE_LINE_VALUE = 500
AVERAGE_LINE_LABEL = "Sokévi átlag"

MAX_PERIODS_TO_SHOW = 6  # fixen 6
PERIOD_SHEET_REGEX = r"^\s*\d{4}_\d{4}_\d+\s*$"  # csak ezeket tekintjük mérésnek


# ----------------------------
# Segédfüggvények
# ----------------------------
def parse_period_sort_key(sheet_name: str) -> Tuple:
    s = str(sheet_name).strip()
    m = re.match(r"^\s*(\d{4})_(\d{4})_(\d+)\s*$", s)
    if m:
        y1, y2, p = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (y1, y2, p)
    return (9999, 9999, s)


def get_prefix(col_name: str) -> str:
    """
    Kinyeri a prefixet.
    Pl:
      'Mat: Mat. össz.' -> 'Mat'
      'Mat' -> 'Mat'
    """
    s = str(col_name).strip()
    if ":" in s:
        return s.split(":", 1)[0].strip()

    m = re.match(r"^([A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+)", s)
    if m:
        return m.group(1)

    return s


def clean_period_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Első 2 oszlop: ID, Név (bárhogy is hívják a fejlécben).
    #HIÁNYZIK sor kizárása.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    if df.shape[1] < 2:
        return df

    df = df.rename(columns={df.columns[0]: "TanulóID", df.columns[1]: "TanulóNév"})
    df["TanulóID"] = df["TanulóID"].astype(str).str.strip()
    df["TanulóNév"] = df["TanulóNév"].astype(str).str.strip()

    df = df[df["TanulóNév"].notna()]
    df = df[df["TanulóNév"] != ""]
    df = df[df["TanulóNév"] != MISSING_NAME]

    return df


@st.cache_data(show_spinner=False)
def read_measurement_sheets_xlsx(uploaded_file) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Csak a mérési időszak sheeteket olvassa be (YYYY_YYYY_N).
    """
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")

    all_sheets = xls.sheet_names
    period_sheets = [s for s in all_sheets if re.match(PERIOD_SHEET_REGEX, str(s).strip())]

    # rendezés név alapján
    period_sheets = sorted(period_sheets, key=parse_period_sort_key)

    dfs: Dict[str, pd.DataFrame] = {}
    for sh in period_sheets:
        try:
            dfs[sh] = xls.parse(sh)
        except Exception:
            continue

    # csak azok maradjanak, amik tényleg beolvashatók
    period_sheets = [s for s in period_sheets if s in dfs]
    return period_sheets, dfs


def build_long_table(period_order: List[str], period_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Long tábla: Mérés, TanulóID, TanulóNév, Prefix, Érték
    Csak a DOMAIN_PREFIXES + EXTRA_PREFIXES prefixekkel.
    """
    allowed_prefixes = set(sum(DOMAIN_PREFIXES.values(), [])) | set(EXTRA_PREFIXES)
    rows = []

    for period in period_order:
        df = clean_period_df(period_dfs.get(period))
        if df is None or df.empty:
            continue

        metric_cols = [c for c in df.columns if c not in ["TanulóID", "TanulóNév"]]

        for metric in metric_cols:
            prefix = get_prefix(metric)
            if prefix not in allowed_prefixes:
                continue

            sub = df[["TanulóID", "TanulóNév", metric]].copy()
            sub = sub.rename(columns={metric: "Érték"})
            sub["Mérés"] = period
            sub["Prefix"] = prefix
            sub["Érték"] = pd.to_numeric(sub["Érték"], errors="coerce")
            rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["Mérés", "TanulóID", "TanulóNév", "Prefix", "Érték"])

    return pd.concat(rows, ignore_index=True)


def pivot_for_domain(student_df: pd.DataFrame, periods: List[str], domain: str, include_gh: bool) -> pd.DataFrame:
    needed = DOMAIN_PREFIXES[domain].copy()
    if include_gh:
        needed += EXTRA_PREFIXES

    sub = student_df[student_df["Prefix"].isin(needed)].copy()
    pv = sub.pivot_table(index="Mérés", columns="Prefix", values="Érték", aggfunc="mean")
    pv = pv.reindex(index=periods, columns=needed)

    # jelmagyarázat nevei
    pv = pv.rename(columns={p: PREFIX_LABELS.get(p, p) for p in pv.columns})
    return pv


def fig_to_jpg_bytes(fig) -> bytes:
    """
    Biztos mentés: draw + savefig BytesIO-ba.
    """
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="jpg", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def plot_grouped_bars_with_labels(pivot_df: pd.DataFrame, title: str) -> plt.Figure:
    """
    Oszlopdiagram értékfeliratokkal + 500-as 'Sokévi átlag' vonallal.
    """
    periods = pivot_df.index.tolist()
    metrics = pivot_df.columns.tolist()
    values = pivot_df.values

    n_period = len(periods)
    n_metric = len(metrics)

    x = np.arange(n_period)
    fig, ax = plt.subplots(figsize=(min(14, 2 + n_period * 1.5), 5.2))

    bar_width = 0.8 / max(n_metric, 1)
    containers = []

    for i, m in enumerate(metrics):
        bars = ax.bar(
            x + (i - (n_metric - 1) / 2) * bar_width,
            values[:, i],
            width=bar_width,
            label=str(m),
        )
        containers.append(bars)

    # 500-as kiemelés
    ax.axhline(AVERAGE_LINE_VALUE, linewidth=2.5, linestyle="-", label=AVERAGE_LINE_LABEL)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=0)
    ax.set_ylabel("Érték")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # értékek oszlopokra
    for bars in containers:
        try:
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        except Exception:
            pass

    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="EDIA mérés eredmények", layout="wide")
st.title("EDIA mérés eredmények")

uploaded = st.file_uploader("Excel munkafüzet feltöltése (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Tölts fel egy .xlsx fájlt.")
    st.stop()

period_order, period_dfs = read_measurement_sheets_xlsx(uploaded)

if not period_order:
    st.error("Nem találtam mérési munkalapokat (várt minta: YYYY_YYYY_N, pl. 2026_2027_2).")
    st.stop()

# fixen utolsó 6 mérési időszak
period_order_last = period_order[-MAX_PERIODS_TO_SHOW:]
shown_n = len(period_order_last)

long_df = build_long_table(period_order_last, period_dfs)

students = (
    long_df[["TanulóID", "TanulóNév"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["TanulóNév", "TanulóID"])
)

if students.empty:
    st.error("Nem találtam tanuló adatokat (első 2 oszlop alapján), vagy nincs értelmezhető mérőoszlop.")
    st.stop()

# kereshető legördülő
student_labels = (students["TanulóNév"] + "  (" + students["TanulóID"] + ")").tolist()
label_to_id = dict(zip(student_labels, students["TanulóID"]))

st.sidebar.header("Beállítások")
selected_label = st.sidebar.selectbox("Tanuló (kereshető)", options=student_labels, index=0)
selected_id = label_to_id[selected_label]

# név + id külön a címhez
sel_row = students[students["TanulóID"] == selected_id].iloc[0]
sel_name = sel_row["TanulóNév"]
sel_id = sel_row["TanulóID"]

selected_domain = st.sidebar.radio(
    "Mérési terület (egyszerre csak egy)",
    options=["Matematika", "Olvasás", "Természettudomány"],
    index=0,
)
show_gh = st.sidebar.checkbox("GH (Géphaszn.) hozzáadása", value=False)

student_df = long_df[long_df["TanulóID"] == selected_id].copy()
pv = pivot_for_domain(student_df, period_order_last, selected_domain, include_gh=show_gh)

# CÍM: fix + dinamikus rész
chart_title = f"EDIA mérés eredmények – {selected_domain} – {sel_name} ({sel_id})"

st.subheader(f"{selected_domain} – utolsó {shown_n} mérési időszak – {sel_name} ({sel_id})")

if pv.empty or pv.dropna(how="all").empty:
    st.info("Nincs megjeleníthető adat ehhez a tanulóhoz / területhez az utolsó időszakokban.")
else:
    fig = plot_grouped_bars_with_labels(pv, chart_title)

    # JPG BYTES ELŐBB! (ne legyen fehér)
    jpg_bytes = fig_to_jpg_bytes(fig)

    # megjelenítés (NE clear_figure=True)
    st.pyplot(fig)

    safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", sel_name)[:60]
    safe_domain = re.sub(r"[^A-Za-z0-9_\-]+", "_", selected_domain)
    filename = f"EDIA_{safe_domain}_{safe_name}_{sel_id}.jpg"

    st.download_button(
        label="Diagram letöltése JPG-ben",
        data=jpg_bytes,
        file_name=filename,
        mime="image/jpeg",
        use_container_width=True,
    )

with st.expander("Nyers adatok (ellenőrzés)"):
    st.dataframe(student_df.sort_values(["Mérés", "Prefix"]))
