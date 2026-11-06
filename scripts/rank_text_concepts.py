"""scripts/rank_text_concepts.py — rank candidate word-concepts for Banking77.

Score per candidate f: max over classes of (mean(f | class) - mean(f)) / std(f).
Same spirit as the pixel ranking (PathMNIST): a concept is useful if it
separates at least one of the 77 intents well. Runs on TRAIN only; the
chosen set is then built via loop_lib.load_concept_set (normalised on train).

Usage: python3 scripts/rank_text_concepts.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from glassnet.text_data import load
from glassnet import text_concepts as TC

# --- documented candidate pool: fixed lexicons + structural formulas ----------
_LEX = {
    "card_family": ["card", "cards", "debit", "credit", "atm", "visa", "mastercard"],
    "money_family": ["money", "payment", "payments", "paid", "pay", "refund", "cash",
                     "deposit", "withdrawals", "withdrawal", "withdraw", "transfer",
                     "transfers", "transferred", "sent", "send", "topup", "top"],
    "account_family": ["account", "accounts", "balance", "bank", "banking", "statement",
                       "statements", "pin"],
    "fee_family": ["fee", "fees", "charge", "charges", "charged", "overcharged",
                   "chargeback"],
    "transfer_family": ["transfer", "transfers", "transferred", "transferring", "wire",
                        "send", "sent", "pay", "paid", "payment", "payments", "deposit",
                        "withdraw", "withdrawal", "withdrawals"],
    "delivery_family": ["deliver", "delivery", "delivered", "arrived", "arrive",
                        "receive", "received", "shipping", "shipment", "package",
                        "parcel", "post", "mail", "posted"],
    "lost_family": ["lost", "miss", "missing", "misplaced", "disappeared", "gone",
                    "stolen", "theft", "fraud", "hack", "scam", "lock", "stole"],
    "broken_family": ["broken", "error", "issue", "issues", "problem", "problems",
                      "fault", "faulty", "fail", "failed", "failing", "glitch",
                      "not working", "doesn't work", "doesn't work", "don't work"],
    "chargeback_family": ["chargeback", "reversal", "reverse", "dispute", "disputes",
                          "unauthorised", "unauthorized"],
    "refund_family": ["refund", "refunds", "refunded", "reimburse", "reimbursement",
                      "money back", "return", "returned", "returns"],
    "pin_family": ["pin", "passcode", "code", "codes", "password", "login", "otp",
                   "verification code", "security code"],
    "time_family": ["when", "day", "days", "week", "weeks", "hours", "hour", "schedule",
                    "appointment", "opening", "open", "time", "times", "long",
                    "waiting", "wait"],
    "polite": ["please", "kindly", "thanks", "thank"],
    "bank_family": ["bank", "branch", "banking", "atm", "cash point", "kiosk"],
    "app_family": ["app", "application", "online", "website", "site", "mobile",
                   "netbanking", "internet", "platform"],
    "contact_family": ["contact", "call", "called", "phone", "number", "numbers",
                       "reach", "speak", "talk", "agent", "representative", "human",
                       "chat", "email", "mail"],
    "status_family": ["status", "update", "updates", "progress", "tracking", "pending",
                      "process", "processed", "processing", "ongoing"],
    "transaction_family": ["transaction", "transactions", "history", "statement",
                           "statements", "record", "records", "receipt", "activity",
                           "recent"],
    "stop_family": ["stop", "cancel", "cancelled", "canceled", "cancellation", "block",
                    "blocked", "freeze", "frozen", "suspend", "suspended", "hold",
                    "held", "declined", "decline", "reject", "rejected", "refuse",
                    "refused", "unblock"],
    "address_family": ["address", "addresses", "mailing", "residential",
                       "change of address", "update address"],
    "negation": ["not", "don't", "dont", "can't", "cant", "cannot", "didn't", "didnt",
                 "isn't", "isnt", "won't", "wont", "never", "no", "doesn't", "doesnt"],
    "question_mid": ["what", "can", "which", "how", "why", "when", "where", "who",
                     "does", "do", "is", "are", "will", "would", "could"],
}
_STRUCT = ["word_count", "avg_word_len", "question_start", "punctuation_frac",
           "uppercase_frac"]

def _fleet(toks, lex):
    hits = 0
    for t in toks:
        if t in lex:
            hits += 1
    return hits

def _features(t_tr, names):
    """(N, len(names)) raw feature matrix for the candidate set."""
    base = TC._f(t_tr)                      # existing 13 computed concepts
    F = np.zeros((len(t_tr), len(names)), dtype=np.float64)
    ok = {}
    for j, n in enumerate(names):
        if n in base:
            F[:, j] = base[n]; ok[n] = True
        elif n in _LEX:
            lex = set(_LEX[n].split() if isinstance(_LEX[n], str) else _LEX[n])
            F[:, j] = np.array([_fleet(TC.tokenize(t), lex) for t in t_tr],
                               dtype=np.float64)
            ok[n] = True
        else:
            ok[n] = False
    return F, ok

def main():
    Xtr, ytr, Xval, yval, Xte, yte, t_tr, t_va, t_te, uniq = load()
    y = np.asarray(ytr)
    cands = sorted(set(list(TC.NAMES) + list(_LEX) + _STRUCT) - {"digit_frac"})
    F, ok = _features(t_tr, cands)
    scores = []
    for j, n in enumerate(cands):
        if not ok[n] or F[:, j].std() < 1e-9:
            continue
        f = F[:, j]
        mu, sd = f.mean(), f.std()
        per_class = np.array([f[y == c].mean() for c in range(len(uniq))])
        best = float(((per_class - mu) / sd).max())
        scores.append((best, n, float(per_class.max()), float(mu)))
    scores.sort(reverse=True)
    print(f"{'rank':>4} {'score':>7} {'mean':>6} {'maxclass':>8}  name")
    for i, (s, n, mc, m) in enumerate(scores, 1):
        print(f"{i:>4} {s:>7.3f} {m:>6.3f} {mc:>8.3f}  {n}")
    print("\ntop-16:", ", ".join(n for _, n, _, _ in scores[:16]))

if __name__ == "__main__":
    main()