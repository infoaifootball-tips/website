"""
Football match prediction model (Dixon-Coles) + walk-forward backtest.

WHAT THIS IS
------------
A defensible, well-understood statistical model that turns historical results
into calibrated probabilities for Home / Draw / Away (and correct scores and
over/under). It is intentionally NOT a black-box ML model: in football the
bookmaker odds already encode most of the signal, so your edge comes from being
honest, calibrated, and transparent -- which this lets you publish.

DATA
----
Designed for the free historical CSVs from https://www.football-data.co.uk/
(one file per league per season). The columns this script uses:
    Date, HomeTeam, AwayTeam, FTHG (home goals), FTAG (away goals)
and, if present, bookmaker odds for de-vigged comparison:
    B365H, B365D, B365A   (Bet365 home/draw/away decimal odds)

If you don't have a CSV yet, run this file as-is: it generates synthetic data
so you can see the whole pipeline work, then swap in real data by passing a path.

USAGE
-----
    python predict.py                 # runs on synthetic data (demo)
    python predict.py path/to/E0.csv  # runs on a real football-data.co.uk file

Requires: numpy, pandas, scipy
"""

from __future__ import annotations
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# ---------------------------------------------------------------------------
# 1. THE MODEL
# ---------------------------------------------------------------------------
# Each team i has an attack strength a_i and a defensive weakness d_i.
# Expected goals:
#     home goals  ~ Poisson(exp(intercept + home_adv + a_home + d_away))
#     away goals  ~ Poisson(exp(intercept            + a_away + d_home))
# Dixon-Coles adds a correction tau() so low scores (0-0, 1-0, 0-1, 1-1) are
# modelled better than independent Poisson allows, and a time-decay weight so
# recent matches count more than old ones.

def _dc_tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score dependency correction."""
    out = np.ones_like(lam, dtype=float)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


@dataclass
class FittedModel:
    teams: list
    attack: dict
    defence: dict
    home_adv: float
    intercept: float
    rho: float

    def _rates(self, home, away):
        lam = np.exp(self.intercept + self.home_adv
                     + self.attack[home] + self.defence[away])
        mu = np.exp(self.intercept
                    + self.attack[away] + self.defence[home])
        return lam, mu

    def score_matrix(self, home, away, max_goals=10):
        """Probability matrix P[i, j] = P(home scores i, away scores j)."""
        lam, mu = self._rates(home, away)
        i = np.arange(max_goals + 1)
        home_p = poisson.pmf(i, lam)
        away_p = poisson.pmf(i, mu)
        mat = np.outer(home_p, away_p)
        # apply DC correction to the 2x2 low-score corner
        for x in (0, 1):
            for y in (0, 1):
                if x == 0 and y == 0:
                    mat[x, y] *= 1 - lam * mu * self.rho
                elif x == 0 and y == 1:
                    mat[x, y] *= 1 + lam * self.rho
                elif x == 1 and y == 0:
                    mat[x, y] *= 1 + mu * self.rho
                elif x == 1 and y == 1:
                    mat[x, y] *= 1 - self.rho
        mat = np.clip(mat, 0, None)
        mat /= mat.sum()
        return mat

    def predict(self, home, away, max_goals=10):
        """Return H/D/A probabilities and a couple of useful markets."""
        m = self.score_matrix(home, away, max_goals)
        p_home = np.tril(m, -1).sum()   # home goals > away goals
        p_away = np.triu(m, 1).sum()    # away goals > home goals
        p_draw = np.trace(m)
        # Over/Under 2.5 goals
        idx = np.arange(max_goals + 1)
        total = idx[:, None] + idx[None, :]
        p_over25 = m[total > 2.5].sum()
        # most likely correct score
        i, j = np.unravel_index(np.argmax(m), m.shape)
        return {
            "home": float(p_home), "draw": float(p_draw), "away": float(p_away),
            "over25": float(p_over25), "under25": float(1 - p_over25),
            "likely_score": (int(i), int(j)),
        }


def fit_model(df, as_of=None, xi=0.0018, max_goals=10):
    """
    Fit Dixon-Coles on all matches strictly BEFORE `as_of` (a Timestamp).
    `xi` is the time-decay rate per day (Dixon-Coles used ~half-life of months).
    """
    train = df if as_of is None else df[df["Date"] < as_of]
    train = train.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    if len(train) < 30:
        raise ValueError("Not enough matches to fit (need ~30+).")

    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    idx = {t: k for k, t in enumerate(teams)}
    n = len(teams)

    h = train["HomeTeam"].map(idx).to_numpy()
    a = train["AwayTeam"].map(idx).to_numpy()
    hg = train["FTHG"].to_numpy(dtype=float)
    ag = train["FTAG"].to_numpy(dtype=float)

    # time-decay weights
    ref = train["Date"].max() if as_of is None else as_of
    age_days = (ref - train["Date"]).dt.days.to_numpy()
    w = np.exp(-xi * age_days)

    # params: [attack(n), defence(n), home_adv, intercept, rho]
    # identifiability: mean(attack)=0, mean(defence)=0 enforced via penalty
    def unpack(p):
        att = p[:n]
        dfc = p[n:2 * n]
        home_adv = p[2 * n]
        intercept = p[2 * n + 1]
        rho = p[2 * n + 2]
        return att, dfc, home_adv, intercept, rho

    def neg_ll(p):
        att, dfc, home_adv, intercept, rho = unpack(p)
        lam = np.exp(intercept + home_adv + att[h] + dfc[a])
        mu = np.exp(intercept + att[a] + dfc[h])
        tau = _dc_tau(hg, ag, lam, mu, rho)
        tau = np.clip(tau, 1e-10, None)
        ll = (np.log(tau)
              + (-lam + hg * np.log(lam))
              + (-mu + ag * np.log(mu)))
        # soft identifiability penalty
        pen = 1000 * (att.mean() ** 2 + dfc.mean() ** 2)
        return -np.sum(w * ll) + pen

    x0 = np.concatenate([
        np.zeros(n), np.zeros(n), [0.25], [0.0], [-0.05]
    ])
    # keep rho in a sane range to avoid negative probabilities
    bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-2, 2), (-0.2, 0.2)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

    att, dfc, home_adv, intercept, rho = unpack(res.x)
    return FittedModel(
        teams=teams,
        attack={t: float(att[idx[t]]) for t in teams},
        defence={t: float(dfc[idx[t]]) for t in teams},
        home_adv=float(home_adv), intercept=float(intercept), rho=float(rho),
    )


# ---------------------------------------------------------------------------
# 2. SCORING METRICS  (how good are the probabilities, honestly?)
# ---------------------------------------------------------------------------
def rps(probs, outcome):
    """Ranked Probability Score for a 3-outcome event (lower is better).
    probs = (p_home, p_draw, p_away); outcome in {'H','D','A'}.
    RPS is the standard, ordering-aware metric for football forecasts."""
    order = ["H", "D", "A"]
    p = np.array(probs, dtype=float)
    o = np.array([1.0 if order[k] == outcome else 0.0 for k in range(3)])
    cum_p = np.cumsum(p)
    cum_o = np.cumsum(o)
    return np.sum((cum_p - cum_o) ** 2) / (len(p) - 1)


def log_loss_one(probs, outcome):
    order = ["H", "D", "A"]
    p = dict(zip(order, probs))
    return -np.log(max(p[outcome], 1e-12))


def devig(odds_h, odds_d, odds_a):
    """Turn bookmaker decimal odds into probabilities (remove the margin)."""
    inv = np.array([1 / odds_h, 1 / odds_d, 1 / odds_a])
    return inv / inv.sum()


# ---------------------------------------------------------------------------
# 3. WALK-FORWARD BACKTEST
# ---------------------------------------------------------------------------
def backtest(df, min_train=200, xi=0.0018, refit_every=10):
    """
    Walk forward in time. For each match (after an initial training window),
    fit on everything before it and predict it. Compare to actual result and,
    where available, to the de-vigged bookmaker probabilities.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    rows = []
    model = None
    for i in range(min_train, len(df)):
        match = df.iloc[i]
        # refit periodically (fitting every single match is slow and barely helps)
        if model is None or i % refit_every == 0:
            try:
                model = fit_model(df.iloc[:i], as_of=match["Date"], xi=xi)
            except ValueError:
                continue
        if match["HomeTeam"] not in model.attack or match["AwayTeam"] not in model.attack:
            continue  # newly-promoted team we haven't seen; skip
        pred = model.predict(match["HomeTeam"], match["AwayTeam"])
        probs = (pred["home"], pred["draw"], pred["away"])

        actual = ("H" if match["FTHG"] > match["FTAG"]
                  else "A" if match["FTHG"] < match["FTAG"] else "D")

        row = {
            "date": match["Date"], "home": match["HomeTeam"],
            "away": match["AwayTeam"], "actual": actual,
            "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2],
            "rps_model": rps(probs, actual),
            "logloss_model": log_loss_one(probs, actual),
        }
        # bookmaker comparison if odds columns exist
        if all(c in match and pd.notna(match[c]) for c in ("B365H", "B365D", "B365A")):
            bp = devig(match["B365H"], match["B365D"], match["B365A"])
            row["rps_book"] = rps(tuple(bp), actual)
            row["logloss_book"] = log_loss_one(tuple(bp), actual)
        rows.append(row)

    return pd.DataFrame(rows)


def calibration_report(bt, bins=5):
    """Are stated probabilities honest? Group predicted home-win prob into bins
    and check the actual home-win rate in each bin."""
    b = bt.copy()
    b["bin"] = pd.cut(b["p_home"], np.linspace(0, 1, bins + 1))
    out = b.groupby("bin", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "avg_predicted": g["p_home"].mean(),
            "actual_rate": (g["actual"] == "H").mean(),
        })
    )
    return out


# ---------------------------------------------------------------------------
# 4. DATA LOADING (+ synthetic fallback so the script always runs)
# ---------------------------------------------------------------------------
def load_football_data_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
            "B365H", "B365D", "B365A"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def make_synthetic(n_teams=16, seasons=3, seed=7):
    """Generate plausible league data with hidden true strengths, so the
    backtest has something to chew on when you have no CSV yet."""
    rng = np.random.default_rng(seed)
    teams = [f"Team_{k:02d}" for k in range(n_teams)]
    true_att = {t: rng.normal(0, 0.35) for t in teams}
    true_def = {t: rng.normal(0, 0.35) for t in teams}
    home_adv, intercept = 0.27, 0.10
    rows, day = [], pd.Timestamp("2022-08-01")
    for _ in range(seasons):
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        for k, (h, a) in enumerate(fixtures):
            lam = np.exp(intercept + home_adv + true_att[h] + true_def[a])
            mu = np.exp(intercept + true_att[a] + true_def[h])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            rows.append({"Date": day + pd.Timedelta(days=3 * (k // 8)),
                         "HomeTeam": h, "AwayTeam": a,
                         "FTHG": hg, "FTAG": ag})
        day += pd.Timedelta(days=300)
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. RUN
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        print(f"Loading real data: {sys.argv[1]}")
        df = load_football_data_csv(sys.argv[1])
    else:
        print("No CSV given -> using synthetic demo data.")
        print("Get real data free from https://www.football-data.co.uk/ "
              "and pass the path, e.g.  python predict.py E0.csv\n")
        df = make_synthetic()

    print(f"Matches: {len(df)}  |  "
          f"{df['Date'].min().date()} -> {df['Date'].max().date()}\n")

    # --- Backtest ---
    print("Running walk-forward backtest (this is the credibility step)...")
    bt = backtest(df, min_train=min(200, len(df) // 3))
    print(f"Evaluated {len(bt)} matches out of sample.\n")

    print("MODEL QUALITY (lower is better):")
    print(f"  Mean RPS      : {bt['rps_model'].mean():.4f}")
    print(f"  Mean log-loss : {bt['logloss_model'].mean():.4f}")
    if "rps_book" in bt.columns:
        print(f"  Bookmaker RPS : {bt['rps_book'].mean():.4f}  "
              f"(beating this is HARD and the real test)")
    print()

    print("CALIBRATION (predicted home-win % vs actual home-win %):")
    print("  If these two columns track each other, your probabilities are honest.")
    print(calibration_report(bt).to_string())
    print()

    # --- Fit final model on everything and show a sample prediction ---
    model = fit_model(df)
    t1, t2 = model.teams[0], model.teams[1]
    p = model.predict(t1, t2)
    print(f"SAMPLE PREDICTION  {t1} (H) vs {t2} (A):")
    print(f"  Home {p['home']*100:5.1f}%   Draw {p['draw']*100:5.1f}%   "
          f"Away {p['away']*100:5.1f}%")
    print(f"  Over 2.5: {p['over25']*100:.1f}%   "
          f"Most likely score: {p['likely_score'][0]}-{p['likely_score'][1]}")
    print("\nThis prints the numbers you'd write to your database each morning.")


if __name__ == "__main__":
    main()
