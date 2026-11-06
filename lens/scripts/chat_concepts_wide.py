"""chat_concepts_wide.py — ~48 conversational concepts, so the concept layer stops being a straw.

WHY
---
The optimisation diagnostic measured the bottleneck: the concept layer takes 1,024 numbers in and
puts 9 out (114:1). The scaling experiment showed the readable architecture needs a ratio near 50:1
-- the winning arm had 41 concepts for a 2,048-number input. Nine concepts for a 1,024-number input
is more than twice as compressed as every configuration that FAILED.

So the layer is widened. Every concept below is defined by EXACT token membership and its support is
measured per WINDOW on TRAIN text only. The definitions are written here, before any run, so the
target file is a fact about the data and not a choice made after seeing a result.

Groups are deliberately ordinary: these are the categories a person would name if asked what a
conversation is about. No concept was added because it looked like it would help a metric.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SPEC = os.path.join(PROJ, "lens", "runs", "chat", "concept_spec_wide.json")

DEFS = {
    # --- dialogue function (who is doing what in the conversation)
    "is_a_question":        ["?"],    "is_a_farewell":        ["bye", "goodbye", "farewell", "later", "cya"],
    "is_an_apology":        ["sorry", "apologize", "apologies", "apology"],
    "is_a_thanks":          ["thank", "thanks", "thankyou", "grateful"],
    "is_a_request":         ["please", "can", "could", "would", "help"],
    "is_an_offer":          ["let", "offer", "provide", "give", "suggest"],
    "is_a_denial":          ["no", "not", "never", "cannot", "don't", "won't"],
    "is_an_agreement":      ["yes", "yeah", "agree", "correct", "right", "exactly"],
    "is_uncertain":         ["maybe", "perhaps", "unsure", "possibly", "might"],
    "is_imperative":        ["do", "make", "take", "put", "keep", "stop", "start"],
    "is_explanatory":       ["because", "since", "therefore", "thus", "reason"],

    # --- participants
    "mentions_the_user":    ["you", "your", "yours", "yourself"],
    "mentions_self":        ["i", "me", "my", "mine", "myself"],
    "mentions_a_group":     ["we", "us", "our", "ours", "they", "them", "their"],
    "mentions_a_person":    ["he", "she", "him", "her", "person", "people", "someone"],
    "mentions_a_name":      ["john", "mary", "mike", "david", "sarah", "james", "anna"],

    # --- topics
    "mentions_feeling":     ["happy", "sad", "feel", "feeling", "angry", "afraid", "glad", "upset"],
    "mentions_food":        ["food", "eat", "eating", "cook", "cooking", "recipe", "meal", "dinner"],
    "mentions_a_number":    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "hundred"],
    "mentions_health":      ["health", "doctor", "medicine", "sick", "pain", "hurt", "hospital"],
    "mentions_time":        ["time", "today", "tomorrow", "yesterday", "now", "later", "week", "year"],
    "mentions_place":       ["place", "here", "there", "home", "house", "city", "country", "room"],
    "mentions_family":      ["family", "mother", "father", "mom", "dad", "child", "children", "son"],
    "mentions_work":        ["work", "job", "boss", "company", "office", "working", "career"],
    "mentions_money":       ["money", "pay", "cost", "price", "buy", "sell", "dollar", "cheap"],
    "mentions_travel":      ["travel", "trip", "car", "drive", "fly", "train", "bus", "road"],
    "mentions_weather":     ["weather", "rain", "snow", "sun", "cold", "hot", "warm", "wind"],
    "mentions_technology":  ["computer", "phone", "internet", "email", "online", "software", "data"],
    "mentions_education":   ["school", "learn", "teach", "study", "student", "teacher", "class"],
    "mentions_animals":     ["dog", "cat", "animal", "bird", "horse", "fish", "pet"],
    "mentions_sport":       ["game", "play", "team", "sport", "win", "lose", "ball", "score"],
    "mentions_music":       ["music", "song", "sing", "band", "guitar", "piano", "listen"],
    "mentions_art":         ["art", "paint", "draw", "picture", "photo", "movie", "film", "book"],
    "mentions_science":     ["science", "research", "study", "experiment", "theory", "evidence"],
    "mentions_law":         ["law", "legal", "court", "police", "crime", "right", "rule"],
    "mentions_politics":    ["government", "president", "vote", "law", "country", "state", "policy"],
    "mentions_body":        ["head", "hand", "eye", "face", "body", "arm", "leg", "hair"],
    "mentions_clothing":    ["wear", "clothes", "shirt", "dress", "shoes", "coat", "hat"],
    "mentions_a_colour":    ["red", "blue", "green", "black", "white", "yellow", "brown", "color"],
    "mentions_a_size":      ["big", "small", "large", "little", "long", "short", "tall", "huge"],
    "mentions_a_quantity":  ["many", "much", "few", "some", "all", "none", "more", "less", "most"],

    # --- register    "is_casual":            ["yeah", "gonna", "wanna", "kinda", "cool", "stuff", "ok"],
    "contains_profanity":   ["damn", "hell", "crap", "shit", "fuck"],
    "is_a_list":            ["first", "second", "third", "next", "finally", "also"],
}


def build(vocab, ids, train_mask, min_support=0.005, windows=None):
    """Return (concepts, spec). Support is measured PER WINDOW when windows are supplied."""
    import numpy as np
    vmap = {w: i for i, w in enumerate(vocab)}
    unit = windows if (windows is not None and len(windows)) else ids[train_mask]
    concepts, spec, dropped = [], [], []
    for name, words in DEFS.items():
        tid = [vmap[w] for w in words if w in vmap]
        if not tid:
            dropped.append({"name": name, "reason": "no token in vocab"})
            continue
        supp = float(np.isin(unit, tid).any(axis=1).mean())
        if supp < min_support:
            dropped.append({"name": name, "reason": f"support {supp:.5f} < floor {min_support}"})
            continue
        concepts.append({"name": name, "ids": tid, "support": supp, "words": words})
        spec.append({"name": name, "support": supp, "n_tokens": len(tid), "words": words})
    out = {"min_support": min_support, "kept": spec, "dropped": dropped,
           "n_kept": len(spec), "n_dropped": len(dropped),
           "unit": "window" if windows is not None else "token",
           "n_defined": len(DEFS)}
    os.makedirs(os.path.dirname(SPEC), exist_ok=True)
    json.dump(out, open(SPEC, "w"), indent=1)
    return concepts, out
