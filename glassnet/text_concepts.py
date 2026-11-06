"""glassnet.text_concepts — named, documented word concepts for Banking77.

Each name has a FIXED lexicon/formula (see CONCEPTS.md, text section): the
neuron's meaning is exactly the fleet of words it counts. Min-max normalised
on TRAIN only, like the pixel concepts.
"""
import numpy as np

from .text_data import tokenize

NAMES = [
    "word_count", "avg_word_len", "digit_frac", "question_start",
    "polite", "card_family", "fee_family", "money_family",
    "account_family", "problem_family", "time_family",
    "punctuation_frac", "uppercase_frac",
    # ranked lexicon families (iter 18; single-token forms only — the bag
    # is unigram, so multi-word entries like "money back" can never match)
    "refund_family", "contact_family", "pin_family", "address_family",
    "lost_family", "bank_family", "status_family", "stop_family",
    "broken_family", "chargeback_family", "delivery_family",
    "transaction_family", "transfer_family",
]
DESC = {
    "word_count":       "number of tokens (length of the utterance)",
    "avg_word_len":     "mean token length in characters",
    "digit_frac":       "fraction of tokens containing a digit (amounts, card/account numbers)",
    "question_start":   "1 if first token is a question word (wh/do/can/docs/is/would/how)",
    "polite":           "count of polite words: please, kindly, thanks, thank",
    "card_family":      "count of {card, cards, debit, credit, atm, visa, mastercard}",
    "fee_family":       "count of {fee, fees, charge, charges, charged, overcharged, chargeback}",
    "money_family":     "count of {money, payment, payments, paid, pay, refund, cash, deposit, withdrawals, withdrawal, withdraw, transfer, transfers, transfered, sent, send, topup, top}",
    "account_family":   "count of {account, accounts, balance, bank, banking, statement, statements, card, cards, pin}",
    "problem_family":   "count of {problem, issue, wrong, broken, error, failed, failing, work, works, working, usable, unusable}",
    "time_family":      "count of {when, how long, days, day, week, weeks, hours, hour, schedule, appointment, opening, open}",
    "punctuation_frac": "non-alphanumeric characters / utterance length",
    "uppercase_frac":   "uppercase letters / alphabetic letters",
    "refund_family":    "count of {refund, refunds, refunded, reimburse, reimbursement, return, returned, returns}",
    "contact_family":   "count of {contact, call, called, phone, number, numbers, reach, speak, talk, agent, representative, human, chat, email, mail}",
    "pin_family":       "count of {pin, passcode, code, codes, password, login, otp}",
    "address_family":   "count of {address, addresses, mailing, residential}",
    "lost_family":      "count of {lost, miss, missing, misplaced, disappeared, gone, stolen, theft, fraud, hack, scam, lock, stole}",
    "bank_family":      "count of {bank, branch, banking, atm, kiosk}",
    "status_family":    "count of {status, update, updates, progress, tracking, pending, process, processed, processing, ongoing}",
    "stop_family":      "count of {stop, cancel, cancelled, canceled, cancellation, block, blocked, freeze, frozen, suspend, suspended, hold, held, declined, decline, reject, rejected, refuse, refused, unblock}",
    "broken_family":    "count of {broken, error, issue, issues, problem, problems, fault, faulty, fail, failed, failing, glitch}",
    "chargeback_family":"count of {chargeback, reversal, reverse, dispute, disputes, unauthorised, unauthorized}",
    "delivery_family":  "count of {deliver, delivery, delivered, arrived, arrive, receive, received, shipping, shipment, package, parcel, post, posted}",
    "transaction_family":"count of {transaction, transactions, history, statement, statements, record, records, receipt, activity, recent}",
    "transfer_family":  "count of {transfer, transfers, transferred, transferring, wire, send, sent, pay, paid, payment, payments, deposit, withdraw, withdrawal, withdrawals}",
    "negation":         "count of negation words {not, don't, dont, can't, cant, cannot, didn't, didnt, isn't, isnt, won't, wont, never, no, doesn't, doesnt}",
    "question_mid":     "count of question words anywhere {what, can, which, how, why, when, where, who, does, do, is, are, will, would, could} (contrast: question_start counts only the first token)",
    "app_family":       "count of {app, application, online, website, site, mobile, netbanking, internet, platform}",
}
LEXICONS = {
    "digit_frac": None,
    "question_start": {"when", "what", "where", "why", "how", "which", "who",
                       "do", "does", "did", "can", "could", "would", "should",
                       "is", "are", "am", "will"},
    "polite": {"please", "kindly", "thanks", "thank"},
    "card_family": {"card", "cards", "debit", "credit", "atm", "visa",
                    "mastercard"},
    "fee_family": {"fee", "fees", "charge", "charges", "charged",
                   "overcharged", "chargeback"},
    "money_family": {"money", "payment", "payments", "paid", "pay", "refund",
                     "cash", "deposit", "withdrawals", "withdrawal",
                     "withdraw", "transfer", "transfers", "transferred",
                     "sent", "send", "topup", "top"},
    "account_family": {"account", "accounts", "balance", "bank", "banking",
                       "statement", "statements", "card", "cards", "pin"},
    "problem_family": {"problem", "issue", "wrong", "broken", "error",
                       "failed", "failing", "work", "works", "working",
                       "usable", "unusable"},
    "time_family": {"when", "how long", "days", "day", "week", "weeks",
                    "hours", "hour", "schedule", "appointment", "opening",
                    "open"},
    "refund_family": {"refund", "refunds", "refunded", "reimburse",
                      "reimbursement", "return", "returned", "returns"},
    "contact_family": {"contact", "call", "called", "phone", "number",
                       "numbers", "reach", "speak", "talk", "agent",
                       "representative", "human", "chat", "email", "mail"},
    "pin_family": {"pin", "passcode", "code", "codes", "password", "login",
                   "otp"},
    "address_family": {"address", "addresses", "mailing", "residential"},
    "lost_family": {"lost", "miss", "missing", "misplaced", "disappeared",
                    "gone", "stolen", "theft", "fraud", "hack", "scam",
                    "lock", "stole"},
    "bank_family": {"bank", "branch", "banking", "atm", "kiosk"},
    "status_family": {"status", "update", "updates", "progress", "tracking",
                      "pending", "process", "processed", "processing",
                      "ongoing"},
    "stop_family": {"stop", "cancel", "cancelled", "canceled", "cancellation",
                    "block", "blocked", "freeze", "frozen", "suspend",
                    "suspended", "hold", "held", "declined", "decline",
                    "reject", "rejected", "refuse", "refused", "unblock"},
    "broken_family": {"broken", "error", "issue", "issues", "problem",
                      "problems", "fault", "faulty", "fail", "failed",
                      "failing", "glitch"},
    "chargeback_family": {"chargeback", "reversal", "reverse", "dispute",
                          "disputes", "unauthorised", "unauthorized"},
    "delivery_family": {"deliver", "delivery", "delivered", "arrived",
                        "arrive", "receive", "received", "shipping",
                        "shipment", "package", "parcel", "post", "posted"},
    "transaction_family": {"transaction", "transactions", "history",
                           "statement", "statements", "record", "records",
                           "receipt", "activity", "recent"},
    "transfer_family": {"transfer", "transfers", "transferred",
                        "transferring", "wire", "send", "sent", "pay",
                        "paid", "payment", "payments", "deposit",
                        "withdraw", "withdrawal", "withdrawals"},
    "negation": {"not", "don't", "dont", "can't", "cant", "cannot",
                 "didn't", "didnt", "isn't", "isnt", "won't", "wont",
                 "never", "no", "doesn't", "doesnt"},
    "question_mid": {"what", "can", "which", "how", "why", "when", "where",
                     "who", "does", "do", "is", "are", "will", "would",
                     "could"},
    "app_family": {"app", "application", "online", "website", "site",
                   "mobile", "netbanking", "internet", "platform"},
}


def _f(texts):
    """texts: list of raw strings. Returns dict name -> (N,) float array."""
    out = {}
    n = len(texts)
    wc = np.zeros(n, dtype=np.float32)
    awl = np.zeros(n, dtype=np.float32)
    dg = np.zeros(n, dtype=np.float32)
    qs = np.zeros(n, dtype=np.float32)
    punct = np.zeros(n, dtype=np.float32)
    upper = np.zeros(n, dtype=np.float32)
    fam = {name: np.zeros(n, dtype=np.float32) for name in LEXICONS}
    for i, raw in enumerate(texts):
        toks = tokenize(raw)
        wc[i] = len(toks)
        if toks:
            awl[i] = sum(len(t) for t in toks) / len(toks)
            dg[i] = sum(1 for t in toks if any(ch.isdigit() for ch in t)) / len(toks)
            qs[i] = 1.0 if toks[0] in LEXICONS["question_start"] else 0.0
        if raw:
            alnum = sum(c.isalnum() for c in raw)
            punct[i] = (len(raw) - alnum) / len(raw)
            alph = sum(c.isalpha() for c in raw)
            if alph:
                upper[i] = sum(c.isupper() for c in raw) / alph
        for name, lex in LEXICONS.items():
            if name in ("digit_frac", "question_start"):
                continue
            fam[name][i] = sum(1 for t in toks if t in lex)
    out["word_count"] = wc
    out["avg_word_len"] = awl
    out["digit_frac"] = dg
    out["question_start"] = qs
    out["punctuation_frac"] = punct
    out["uppercase_frac"] = upper
    for name in LEXICONS:
        if name not in ("digit_frac", "question_start"):
            out[name] = fam[name]
    return out


def _normalize(raw, lo, hi):
    return np.clip((raw - lo) / (hi - lo + 1e-9), 0.0, 1.0).astype(np.float32)


def compute_selected(texts, names, lo=None, hi=None):
    """texts: list of raw strings; names: list of concept names.
    Returns (C [N,K], lo, hi) with lo/hi from TRAIN when not provided."""
    names = list(names)
    f = _f(texts)
    raw = np.stack([f[n] for n in names], axis=1).astype(np.float32)
    if lo is None or hi is None:
        lo, hi = raw.min(axis=0), raw.max(axis=0)
    return _normalize(raw, lo, hi), np.asarray(lo, np.float32), np.asarray(hi, np.float32)