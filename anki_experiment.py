"""Kumbuka D2 on Anki: hand-written rule vs half-life regression (+ logistic regression stretch).
Token: read from the HF_TOKEN environment variable, or Colab's Secrets. Never hard-code it."""
import json, os, sys, time
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss

try:
    from google.colab import userdata
    TOK = userdata.get("HF_TOKEN")
except Exception:
    TOK = os.environ["HF_TOKEN"]

N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = 42
REPO = "open-spaced-repetition/anki-revlogs-10k"


def load(uid):
    p = hf_hub_download(REPO, f"revlogs/user_id={uid}/data.parquet", repo_type="dataset", token=TOK,
                        local_dir="anki_cache")
    d = pd.read_parquet(p).sort_values(["card_id", "day_offset"], kind="stable")
    d["user"] = uid
    d["remembered"] = (d.rating > 1).astype(int)                 # Again = forgot; Hard/Good/Easy = remembered
    g = d.groupby("card_id")
    d["n_right"] = g["remembered"].cumsum() - d["remembered"]    # right BEFORE this review
    d["n_wrong"] = g.cumcount() - d["n_right"]                   # wrong BEFORE this review
    d["last_rating"] = g["rating"].shift(1)                      # what they pressed last time (1..4)
    d["days_since"] = d["elapsed_days"]
    keep = (d.days_since >= 1) & (d.n_right + d.n_wrong >= 1) & d.last_rating.notna()
    return d.loc[keep, ["user", "days_since", "n_right", "n_wrong", "last_rating", "remembered"]]


t0 = time.time()
frames, failed = [], []
for u in range(1, N_USERS + 1):
    try:
        frames.append(load(u))
    except Exception as e:
        failed.append((u, type(e).__name__))
table = pd.concat(frames, ignore_index=True)
print(f"loaded {table.user.nunique()} users, {len(table):,} rows, {time.time()-t0:.0f}s; failed={failed}")
print("share remembered:", round(table.remembered.mean(), 3))

# ---- split by USER ----
rng = np.random.default_rng(SEED)
users = np.array(sorted(table.user.unique()))
rng.shuffle(users)
n_test = max(1, int(round(0.3 * len(users))))
test_users, train_users = set(users[:n_test]), set(users[n_test:])
assert not (test_users & train_users)
train = table[table.user.isin(train_users)].reset_index(drop=True)
test = table[table.user.isin(test_users)].reset_index(drop=True)
print(f"train: {len(train_users)} users / {len(train):,} rows | test: {len(test_users)} users / {len(test):,} rows (zero overlap checked)")

# ---- guesser 1: average ----
avg = train.remembered.mean()
pred_avg = np.full(len(test), avg)

# ---- guesser 2: hand-written rule (copy of the app's placeholder) ----
def rule(frame):
    rating03 = (frame.last_rating - 1).to_numpy()                 # Anki 1..4 -> app 0..3
    h = 4.0 * (1.0 + rating03 / 3.0)
    return np.power(0.5, frame.days_since.to_numpy() / h).clip(0, 1)

# ---- guesser 3: half-life regression, fitted straight to remembered/forgot ----
def feats(frame):
    return np.column_stack([np.ones(len(frame)), np.sqrt(1 + frame.n_right.to_numpy()), np.sqrt(1 + frame.n_wrong.to_numpy())])

def hlr(theta, frame):
    log2h = np.clip(feats(frame) @ theta, -4, 12)
    return np.power(2.0, -frame.days_since.to_numpy() / np.power(2.0, log2h))

Xtr = train
def loss(theta):
    p = np.clip(hlr(theta, Xtr), 1e-4, 1 - 1e-4)
    y = Xtr.remembered.to_numpy()
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 1e-3 * np.sum(theta[1:] ** 2)

best = None
for start in ([3.0, 0.0, 0.0], [4.0, 0.3, -0.3], [2.0, 0.5, -0.5]):
    r = minimize(loss, start, method="L-BFGS-B")
    if best is None or r.fun < best.fun:
        best = r
theta = best.x
print(f"HLR: log2(h) = {theta[0]:.3f} + {theta[1]:.3f}*sqrt(1+right) + {theta[2]:.3f}*sqrt(1+wrong)   (train log-loss {best.fun:.4f})")
pred_hlr = hlr(theta, test)

# ---- stretch: logistic regression (predicts yes/no directly) ----
def lfeats(frame):
    return np.column_stack([np.log1p(frame.days_since), np.sqrt(frame.n_right), np.sqrt(frame.n_wrong), frame.last_rating])
lr = LogisticRegression(max_iter=500).fit(lfeats(train), train.remembered)
pred_lr = lr.predict_proba(lfeats(test))[:, 1]

methods = {"Average guess": pred_avg, "Hand-written rule": rule(test), "Half-life regression": pred_hlr, "Logistic regression (stretch)": pred_lr}
y = test.remembered.to_numpy()
res = {}
print(f"\n{'Method':32s} {'MAE':>7s} {'AUC':>7s} {'LogLoss':>8s}")
for k, p in methods.items():
    auc = roc_auc_score(y, p) if np.std(p) > 0 else 0.5
    res[k] = dict(MAE=round(mean_absolute_error(y, p), 4), AUC=round(auc, 4), LogLoss=round(log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)), 4))
    print(f"{k:32s} {res[k]['MAE']:7.4f} {res[k]['AUC']:7.4f} {res[k]['LogLoss']:8.4f}")

# ---- is the win real? ----
wins = 0
for i in range(20):
    idx = rng.integers(0, len(test), len(test))
    wins += mean_absolute_error(y[idx], pred_hlr[idx]) < mean_absolute_error(y[idx], methods["Hand-written rule"][idx])
print(f"\nre-shuffles where HLR beats the rule on MAE: {wins}/20")
per_user = []
for u, g in test.groupby("user"):
    yy = g.remembered.to_numpy(); ix = g.index.to_numpy()
    per_user.append((u, len(g), mean_absolute_error(yy, methods["Hand-written rule"][ix]), mean_absolute_error(yy, pred_hlr[ix]), mean_absolute_error(yy, pred_avg[ix])))
pu = pd.DataFrame(per_user, columns=["user", "rows", "rule", "hlr", "avg"])
print(f"test users where HLR has smaller MAE than rule: {(pu.hlr < pu.rule).sum()}/{len(pu)}; than average guess: {(pu.hlr < pu.avg).sum()}/{len(pu)}")

# calibration of HLR and rule
for name in ("Hand-written rule", "Half-life regression"):
    p = methods[name]; b = np.digitize(p, np.linspace(0, 1, 6)[1:-1])
    print(name, "| predicted vs actual by fifth:", [(round(p[b == k].mean(), 2), round(y[b == k].mean(), 2), int((b == k).sum())) for k in range(5) if (b == k).sum() > 500])

json.dump(dict(users=int(len(users)), train_users=int(len(train_users)), test_users=int(len(test_users)), train_rows=int(len(train)),
               test_rows=int(len(test)), share_remembered=round(float(table.remembered.mean()), 4), theta=[round(float(t), 4) for t in theta],
               results=res, rows_win=int(wins), users_hlr_beats_rule=int((pu.hlr < pu.rule).sum()), users_tested=int(len(pu))),
          open("anki_results.json", "w"), indent=2)
