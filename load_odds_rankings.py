"""
Standalone script to load odds and rankings into Supabase.
Matches and stats are already loaded — this only does odds + rankings.
"""
import os
from collections import defaultdict
import pandas as pd
import requests
from io import BytesIO
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
DATA_DIR     = "./tennis_atp"
BATCH_SIZE   = 200
YEARS        = list(range(2015, 2025))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BOOKMAKERS = {
    "B365W": ("Bet365",   "winner"),
    "B365L": ("Bet365",   "loser"),
    "PSW":   ("Pinnacle", "winner"),
    "PSL":   ("Pinnacle", "loser"),
}


def si(v):
    try:
        return int(v) if pd.notna(v) else None
    except Exception:
        return None


def paginate_all(table, select, order=None):
    rows, offset = [], 0
    while True:
        q = supabase.table(table).select(select)
        if order:
            q = q.order(order)
        result = q.range(offset, offset + 999).execute()
        rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    return rows


def batch_insert(table, rows):
    if not rows:
        return
    for i in range(0, len(rows), BATCH_SIZE):
        supabase.table(table).insert(rows[i:i + BATCH_SIZE]).execute()


# ── Load player map ───────────────────────────────────────────────────────────
print("Loading players from DB ...")
all_players = paginate_all("players", "player_id,name")
player_map  = {r["name"]: r["player_id"] for r in all_players}
print(f"  {len(player_map)} players")

# Build last-name index for "Last F." -> player_id matching
lastname_index: dict = defaultdict(list)
for fullname, pid in player_map.items():
    parts = fullname.strip().split()
    if parts:
        lastname_index[parts[-1].lower()].append((pid, fullname))


def resolve_player(odds_name: str):
    odds_name = odds_name.strip()
    if not odds_name:
        return None
    parts = odds_name.split()
    if not parts:
        return None
    last = parts[0].rstrip(".").lower()
    candidates = lastname_index.get(last, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]
    if len(parts) > 1:
        initial = parts[1].replace(".", "").lower()
        for pid, fullname in candidates:
            fname_parts = fullname.strip().split()
            if fname_parts and fname_parts[0].lower().startswith(initial):
                return pid
    return candidates[0][0]


# ── Load match lookup ─────────────────────────────────────────────────────────
print("Loading matches from DB ...")
all_matches = paginate_all("matches", "match_id,match_date,winner_id,loser_id")
match_lookup = {}
for m in all_matches:
    key = (m["match_date"], m["winner_id"], m["loser_id"])
    match_lookup[key] = m["match_id"]
print(f"  {len(match_lookup)} matches in lookup")


# ── Odds ──────────────────────────────────────────────────────────────────────
print("\nDownloading odds from tennis-data.co.uk ...")
odds_rows = []
for year in YEARS:
    url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        odf = pd.read_excel(BytesIO(resp.content), engine="openpyxl")
    except Exception as e:
        print(f"  {year}: skipped ({e})")
        continue

    odf["Date"] = pd.to_datetime(odf["Date"], errors="coerce")
    matched = 0
    for _, row in odf.iterrows():
        raw_date = row.get("Date")
        if pd.isna(raw_date):
            continue
        date_str = raw_date.strftime("%Y-%m-%d")

        w_id = resolve_player(str(row.get("Winner", "")))
        l_id = resolve_player(str(row.get("Loser",  "")))
        if not w_id or not l_id:
            continue

        match_id = match_lookup.get((date_str, w_id, l_id))
        if not match_id:
            continue

        matched += 1
        for col, (bookmaker, side) in BOOKMAKERS.items():
            if col not in odf.columns:
                continue
            try:
                odds_val = float(row[col])
            except Exception:
                continue
            if pd.isna(odds_val) or odds_val <= 0:
                continue
            pid = w_id if side == "winner" else l_id
            odds_rows.append({
                "match_id":     match_id,
                "bookmaker":    bookmaker,
                "player_id":    pid,
                "closing_odds": round(odds_val, 4),
                "implied_prob": round(1.0 / odds_val, 4),
            })
    print(f"  {year}: {len(odf)} rows, {matched} linked")

print(f"\nInserting {len(odds_rows)} odds rows ...")
batch_insert("odds", odds_rows)
print(f"Done. {len(odds_rows)} odds rows loaded.")


# ── Rankings ──────────────────────────────────────────────────────────────────
print("\nLoading rankings ...")
RANKING_FILES = [
    os.path.join(DATA_DIR, "atp_rankings_10s.csv"),
    os.path.join(DATA_DIR, "atp_rankings_20s.csv"),
    os.path.join(DATA_DIR, "atp_rankings_current.csv"),
]
ranking_frames = []
for path in RANKING_FILES:
    if os.path.exists(path):
        rdf = pd.read_csv(path, header=None,
                          names=["rank_date", "atp_rank", "player_sackmann_id", "atp_points"],
                          low_memory=False)
        rdf = rdf[rdf["rank_date"] != "ranking_date"]
        ranking_frames.append(rdf)
        print(f"  {os.path.basename(path)}: {len(rdf)} rows")

if not ranking_frames:
    print("No ranking files found.")
else:
    rdf_all = pd.concat(ranking_frames, ignore_index=True)

    # Filter to 2015+ only to keep volume manageable
    rdf_all["rank_date"] = rdf_all["rank_date"].astype(str)
    rdf_all = rdf_all[rdf_all["rank_date"].str[:4] >= "2015"]
    print(f"  {len(rdf_all)} rows after filtering to 2015+")

    # Load Sackmann player ID -> full name mapping
    # atp_players.csv has header: player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id
    players_path = os.path.join(DATA_DIR, "atp_players.csv")
    pdf = pd.read_csv(players_path, low_memory=False)
    pdf["full_name"] = (pdf["name_first"].fillna("") + " " + pdf["name_last"].fillna("")).str.strip()
    sid_to_name = dict(zip(pdf["player_id"].astype(str), pdf["full_name"]))

    rdf_all["player_sackmann_id"] = rdf_all["player_sackmann_id"].astype(str)
    rdf_all["full_name"] = rdf_all["player_sackmann_id"].map(sid_to_name)
    rdf_all["player_id"] = rdf_all["full_name"].map(player_map)
    rdf_all = rdf_all.dropna(subset=["player_id"])

    rdf_all["rank_date_fmt"] = rdf_all["rank_date"].apply(
        lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}" if len(str(x)) == 8 else None
    )
    rdf_all = rdf_all.dropna(subset=["rank_date_fmt"])

    rank_rows = []
    for _, r in rdf_all.iterrows():
        rank_rows.append({
            "player_id":  int(r["player_id"]),
            "rank_date":  r["rank_date_fmt"],
            "atp_rank":   si(r["atp_rank"]),
            "atp_points": si(r["atp_points"]),
        })

    print(f"  Inserting {len(rank_rows)} ranking rows ...")
    batch_insert("rankings", rank_rows)
    print(f"  Done. {len(rank_rows)} ranking rows loaded.")

print("\nload_odds_rankings.py complete.")
