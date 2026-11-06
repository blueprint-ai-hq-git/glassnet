"""variants.py — ONE interface, three models: see-only, talk-only, see+talk.

The owner asked for a fused vision-language model AND separate versions that only see or only
talk, all behind one easy API, all transparent.

  v = Variants(see=..., talk=..., see_talk=...)
  v.see(img)                     -> class + confidence + the named concepts that fired
  v.talk(window_ids)             -> next word id + confidence + fired concepts
  v.see_and_talk(img, prefix)    -> next caption WORD + fired concepts
  v.explain(img, prefix)         -> which named concepts drove the chosen word (exact arithmetic)

REFUSAL RULE (tested at the bottom of this file): a variant can only do its own job. Asking the
see-only model to talk raises a plain sentence instead of guessing. The capabilities are separate
models; pretending otherwise is the dishonesty the honesty rules ban.

SIGNATURES READ, NOT GUESSED (the guess-the-API mistake has now cost this project real time):
  lens.vision.load_vision(path)   -> VisionNet (forward_arrays(images)) or
                                     VLNet    (forward_arrays(images, cap_ids))
  meta key is '__meta__'          -> {"kind","C"/"V","avocab",...}
  lens.load(path, device="cpu")   -> LensNet  (forward_arrays(ids) -> {"concepts","logits"})
  SeeTalkNet.load(path)           -> (net, meta) with meta["vocab"] as WORDS
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                              # noqa: E402
import lens.tensor as T                                  # noqa: E402
from lens.vision import load_vision                      # noqa: E402


def _load_image_28(path):
    """PNG/JPG path -> (28,28) float32 in [0,1], the see ship's input. Uses the ffmpeg
    already on disk (no new dependency). Raises rather than guessing a different shape."""
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        import glob
        cands = glob.glob("/home/yahya/.hermes/tools/ffmpeg-*/bin/ffmpeg")
        ff = cands[0] if cands else None
    if not ff:
        raise RuntimeError("no ffmpeg on PATH and no bundled ffmpeg found -- cannot read an "
                           "image file; pass a (28,28) array instead")
    p = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path),
                        "-vf", "scale=28:28,format=gray", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"], capture_output=True)
    if p.returncode != 0 or len(p.stdout) < 784:
        raise RuntimeError(f"ffmpeg could not decode {path}: {p.stderr[:120]!r}")
    return np.frombuffer(p.stdout[:784], dtype=np.uint8).astype(np.float32).reshape(28, 28) / 255.0


def _rgb_from_28(img):
    """(28,28) gray -> (2352,) channel-major RGB (the colour ships' input)."""
    g = np.asarray(img, dtype=np.float32).reshape(28, 28)
    return np.stack([g, g, g], axis=0).reshape(-1)


class Variants:
    def __init__(self, see=None, talk=None, see_talk=None, audio=None, video=None,
                 device="cpu", chat_vocab=None, speech=None):
        T.set_device(device)
        self.chat_vocab = chat_vocab
        self.models = {}
        self._paths = {}
        if see:
            net, meta = load_vision(see)   # returns (net, meta) — READ at lens/vision.py, not guessed
            if meta.get("kind") != "vision":
                raise ValueError("see= checkpoint is not a pure vision model "
                                 f"(kind={meta.get('kind')!r}) — refusing to relabel a VL model "
                                 "as the see-only variant")
            self.models["see"] = (net, meta)
        if talk:
            net = lens.load(talk, device=device)
            self.models["talk"] = (net, {})
        if see_talk:
            import train_seetalk
            net, meta = train_seetalk.SeeTalkNet.load(see_talk)
            self.models["see_talk"] = (net, meta)
            self._paths["see_talk"] = see_talk
        if speech:
            # the readable spoken-word recognizer (real speech, 8 named acoustic concepts)
            import lens.speech as SP
            net = SP.load(speech, device=device)
            self.models["speech"] = (net, {"ship": os.path.basename(speech)})
        if audio:
            import lens.audio as A
            net = A.load(audio, device=device)
            self.models["audio"] = (net, {})
        if video:
            import lens.audio as A
            net = A.load(video, device=device)
            self.models["video"] = (net, {})

    def capabilities(self):
        return sorted(self.models)

    def _need(self, kind):
        if kind not in self.models:
            raise ValueError(f"this build has no {kind!r} model loaded — available: "
                             f"{self.capabilities()}. Refusing instead of guessing.")
        return self.models[kind]

    # ------------------------------------------------------------------ capabilities
    def see(self, img):
        net, meta = self._need("see")
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.reshape(-1)
        # adapt a grayscale frame to a colour ship (and vice versa) by the ship's OWN declared
        # channel count -- never guess: the front's patch shape is the contract
        want = int(getattr(net.front, "patch_dim", arr.size))
        ch = int(getattr(net, "channels", 1))
        if arr.size == 784 and ch == 3:
            arr = _rgb_from_28(arr.reshape(28, 28))
        elif arr.size == 2352 and ch == 1:
            arr = arr.reshape(3, 28, 28).mean(axis=0).reshape(-1)
        out = net.forward_arrays(np.asarray([arr], dtype=np.float32))
        lg = out["logits"][0]
        p = np.exp(lg - lg.max())
        p /= p.sum()
        best = int(p.argmax())
        names = meta.get("avocab") or meta.get("names") or []
        names_src = "checkpoint meta" if names else None
        if not names:
            # the vision ships predate name storage (meta has avocab: None). Fall back to the
            # DATA LOADER's own class order -- the same list training used -- and only accept
            # it when its width matches the model's output. Provenance is reported, never
            # silently assumed.
            try:
                import vl_data
                _x, _y, _xt, _yt, nm = vl_data.load_fashion()
                if len(nm) == len(p):
                    names, names_src = list(nm), "vl_data.load_fashion label order"
            except Exception:
                pass
        if len(names) != len(p):
            # never print a class name we cannot prove matches the model's output width —
            # the silent-wrong-name bug (camera vocab) is why this is a hard refusal
            names = [f"class_{i}" for i in range(len(p))]
            names_src = "refused: width mismatch"
        h = out["concepts"][0]
        fired = [(c.name, round(float(h[i]), 3)) for i, c in enumerate(net.concepts)
                 if float(h[i]) > 0.5] if hasattr(net, "concepts") else []
        return {"model": "see", "answer": names[best], "confidence": float(p[best]),
                "names_source": names_src, "fired_concepts": fired}

    def recognize_word(self, clip, why=True):
        """Speech: hear ONE clip (samples array or a .wav path) and answer which of the 8 words it
        is, with the model's own concept activations and sparse rules as the reason -- the same
        explanation format as every other line. Real audio data (Google Speech Commands, CC BY 4.0);
        official-val accuracy 37.43% vs a 13.06% majority baseline, 7/7 concepts audited honest.
        """
        import lens.speech as SP
        net, meta = self._need("speech")
        x = SP.read_wav(clip) if isinstance(clip, str) else np.asarray(clip, dtype=np.float64)
        word, prob, tops = net.predict(x)
        out = {"word": word, "probability": round(float(prob), 4),
               "concepts": {n: round(float(v), 3) for n, v in tops},
               "ship": meta.get("ship")}
        if why:
            out["why"] = SP.explain(net, x)
        return out

    def listen(self, audio_in, why_top_k=3):
        """Audio: hear a 0.4 s spectral window (or an ogg path) and answer is_silence /
        has_voiced with the EXACT concept reasoning. The answer is the net's own argmax;
        the reasons are its own concept activations. Nothing hidden.
        """
        from lens import backend as B
        net, _ = self._need("audio")
        w_power = None
        if isinstance(audio_in, str):
            import audio_data
            x = audio_data.decode_ogg(audio_in)
            sp = audio_data.stft(x)
            win = audio_data.windows_from_spec(sp)
            if len(win) == 0:
                return {"model": "audio", "error": "audio shorter than one window"}
            # REPRESENTATIVE WINDOW: the file's MEDIAN-level window, not the first one. Measured
            # reason: window[0] of a real recording was a quiet/transient moment (flatness
            # 0.0003, peak-to-mean 119) which the statistics then misread as tonal; the loudest
            # window is a transient too (width 5.4 vs the corpus voice p5 of 25.7). The median
            # window is what the recording "usually sounds like", and it is comparable to the
            # per-window corpus statistics the thresholds were measured from.
            rms_all = np.sqrt(np.asarray(win, dtype=np.float64).mean(axis=(1, 2)))
            k = int(np.argsort(rms_all)[len(rms_all) // 2])
            w = win[k]
            w_power = np.asarray(w, dtype=np.float64)   # kept for the hand-computed statistics
            w = 10.0 * np.log10(w + 1e-10)
            w = np.clip(w, audio_data.DB_FLOOR, 0.0)
            w = (w - audio_data.DB_FLOOR) / (-audio_data.DB_FLOOR)
            audio_in = w
        xa = np.asarray(audio_in, dtype=np.float32)
        if xa.ndim == 2:
            xa = xa[None]
        # ---- hand-computed statistics of THIS window (readable, printed) -------------------
        # The trained concepts are honest (their formulas are band-shape and level), but the
        # NAME has_voiced overclaimed: a 440 Hz tone is low-band too. So the runtime composes
        # its verdict from the trained concepts PLUS these two hand-computed statistics, exactly
        # as the video line's static fix composes at the runtime. Measured distributions on the
        # real corpus (13,529 windows): voice flatness p95 0.093, corpus flat > 0.30 = 5.96%,
        # voice peak-to-mean p95 16.9. Thresholds sit well outside the voice band.
        # Statistics computed ON THE MODEL'S OWN INPUT REPRESENTATION (0..1, linear in dB), the
        # same way for the corpus and for a fresh file -- verified identical to the cache rows
        # (nearest-row RMSE 1e-8). Measured, 13,529 corpus windows sampled + his 8 real files:
        #   corpus voiced   flat 0.023-0.938 (p5-p95)   width 44.9-63.5   loudest bin 1.000
        #   his real files  flat 0.257-0.741            width 58.3-64.2   loudest bin 1.000
        #   440 Hz tone     flat 0.000                  width 18.8        loudest bin 1.000
        #   white noise     flat 0.999                  width 65.0        loudest bin 1.000
        #   near-silent     flat 0.002                  width 16.6        loudest bin 0.126
        # Thresholds therefore sit OUTSIDE every measured voice value.
        xin = xa[0].astype(np.float64)
        wpos = xin + 1e-6
        flatness = float(np.exp(np.log(wpos).mean()) / wpos.mean())
        width = float((xin > 0.01 * xin.max()).sum(axis=1).mean())
        peak_level = float(xin.max())
        out = net.forward(T.Tensor(B.to_dev(xa)))
        h = B.to_cpu(out[1].data)[0]
        cls = int(np.argmax(h))
        order = np.argsort(-h)
        reasons = [(net.concepts[i].name, round(float(h[i]), 3)) for i in order[:why_top_k]]
        idx = {c.name: i for i, c in enumerate(net.concepts)}
        # every threshold below comes from the measured table in the comment above
        SILENCE_H, SILENCE_PEAK, NOISY_FLAT, TONAL_WIDTH, TONAL_FLAT = \
            0.5, 0.30, 0.96, 30.0, 0.05
        steps = []
        if (idx.get("is_silence") is not None and float(h[idx["is_silence"]]) >= SILENCE_H
                and peak_level < SILENCE_PEAK):
            answer = "silence"
            steps.append(f"is_silence {h[idx['is_silence']]:.3f} >= {SILENCE_H} and the loudest "
                         f"bin {peak_level:.3f} < {SILENCE_PEAK} (nothing above the floor)")
        elif flatness > NOISY_FLAT:
            answer = "noisy"
            steps.append(f"flatness {flatness:.3f} > {NOISY_FLAT} (broadband, not a voice)")
        elif width <= TONAL_WIDTH and flatness < TONAL_FLAT:
            answer = "tonal"
            steps.append(f"spectral width {width:.1f} bins <= {TONAL_WIDTH} (energy in one or "
                         f"two bins: a tone, not a voice)")
        elif idx.get("has_voiced") is not None and float(h[idx["has_voiced"]]) >= 0.5:
            answer = "voiced"
            steps.append(f"has_voiced {h[idx['has_voiced']]:.3f} >= 0.5 and not tonal/noisy")
        else:
            answer = "unclear"
            steps.append("no rule matched")
        return {"model": "audio", "answer": answer, "fired_concepts": reasons,
                "stats": {"flatness": round(flatness, 4), "spectral_width_bins": round(width, 1),
                          "peak_bin_level": round(peak_level, 3)},
                "composition": "; ".join(steps),
                "concept_argmax": net.answer_classes[cls]}

    def watch(self, video_in, why_top_k=3):
        """Video: watch a 16-frame window (or an mp4 path) and answer its dominant concept
        (has_motion / is_dark), reasons = exact concept activations. class = argmax(h)
        (the video line trains concept-only, so the answer IS the concept layer)."""
        from lens import backend as B
        net, _ = self._need("video")
        if isinstance(video_in, str):
            import video_data
            fr = video_data.decode_frames(video_in)
            n = (len(fr) - video_data.W_FRAMES) // video_data.STRIDE + 1
            if n < 1:
                return {"model": "video", "error": "video shorter than one window"}
            idx = np.arange(video_data.W_FRAMES)[None, :]
            W = fr[idx]
            if video_data.DIFF_CHANNEL:
                dW = np.abs(W[:, 1:] - W[:, :-1])
                dW = np.concatenate([dW, dW[:, -1:]], axis=1)
                W = np.concatenate([W, dW], axis=2)
            video_in = W
        xv = np.asarray(video_in, dtype=np.float32)
        if xv.ndim == 2:
            xv = xv[None]                 # (W, F) -> (1, W, F); 3D input is already batched
        out = net.forward(T.Tensor(B.to_dev(xv)))
        h = B.to_cpu(out[1].data)[0]
        cls = int(np.argmax(h))
        order = np.argsort(-h)
        reasons = [(net.concepts[i].name, round(float(h[i]), 3)) for i in order[:why_top_k]]
        # scenario fix (BUG-V1): the concept answers by threshold, not by argmax — on a
        # perfectly static window has_motion rested at ~0.485 and argmax read it as motion.
        # h >= 0.5 -> the concept fires; below -> the complementary runtime label is_still.
        ans = net.answer_classes[cls]
        if float(h.max()) < 0.5:
            ans = "is_still" if "motion" in ans else "none_of_" + "+".join(
                c.name for c in net.concepts)
        return {"model": "video", "answer": ans, "fired_concepts": reasons,
                "max_concept": round(float(h.max()), 3)}

    def react(self, text_window=None, image=None, audio_in=None, video_in=None):
        """JOINT report: run every loaded modality on the incoming world and answer each
        with its exact reasons. text_window=str -> talk_sentence; image -> see; audio_in ->
        listen; video_in -> watch. The report is the union of the lines' own arithmetic — this
        runtime never fuses hidden state, every number is this model's own output.
        """
        report = {"modalities": []}
        if text_window is not None and "talk" in self.models:
            report["text"] = self.talk_sentence(text_window, n_new=16) \
                if isinstance(text_window, str) else self.talk(text_window)
            report["modalities"].append("text")
        if image is not None and "see" in self.models:
            # a PATH is resolved to the ship's input here (the matrix bug: a path string used
            # to be handed straight to the model and crashed the whole combination)
            img = _load_image_28(image) if isinstance(image, str) else image
            report["see"] = self.see(img)
            report["modalities"].append("image")
        if audio_in is not None and "audio" in self.models:
            report["audio"] = self.listen(audio_in)
            report["modalities"].append("audio")
        if video_in is not None and "video" in self.models:
            report["video"] = self.watch(video_in)
            report["modalities"].append("video")
        return report

    def talk(self, window_ids):
        net, meta = self._need("talk")
        win = np.asarray([window_ids], dtype=np.int64)
        out = net.forward_arrays(win)
        lg = out["logits"][0]
        p = np.exp(lg - lg.max())
        p /= p.sum()
        h = out["concepts"][0]
        fired = [(c.name, round(float(h[i]), 3)) for i, c in enumerate(net.concepts)
                 if float(h[i]) > 0.5]
        best = int(p.argmax())
        # id -> WORD from the chat pipeline's OWN vocab file. THE GUARD IS AN EQUALITY, not a
        # bound: a vocab shorter than the model's output width is the camera-vocab bug class
        # (sizes matched while words differed = silent wrong names). If the file does not match
        # the checkpoint's logits width exactly, refuse — a selfcheck probe discovered a <=len
        # check that still printed words from a truncated vocab.
        vword = None
        if self.chat_vocab is not None:
            if len(self.chat_vocab) != len(p):
                raise ValueError(f"chat vocab has {len(self.chat_vocab)} words but the talk "
                                 f"checkpoint outputs {len(p)} — REFUSING to map ids to words "
                                 f"across a size mismatch (camera-vocab bug class)")
            vword = self.chat_vocab[best]
        return {"model": "talk", "next_id": best, "word": vword, "confidence": float(p.max()),
                "fired_concepts": fired}

    def talk_sentence(self, prompt, n_new=40, temp=None, top_k=None, rep_penalty=None,
                      greedy=False, seed=0):
        """Sentence-level talk: the SAME decoder the chat sweep verified (top-k + repetition
        penalty — greedy loops, pure sampling drifts; those findings are in LOG.md, this is a
        port, not a new decoder). Words come from the checkpoint's vocab file with the EQUALITY
        guard: mismatched widths refuse instead of printing wrong words.

        DEFAULTS come from TALK_DEFAULT_DECODER, which is the decoder tuned FOR THIS CHECKPOINT in a
        fair comparison (2026-10-09): every talk model got its own 36-setting sweep, then they were
        ranked on unk rate and repetition. top-k 80 / temp 0.9 / rep 1.6 gives unk 0.00% and
        repetition 0.51%, beating v12k (2.58%/0.77%) and best1 (6.08%/1.45%) on BOTH metrics.
        A decoder tuned on one model does not transfer to another — pass explicit values to override.
        Evidence: lens/runs/chat/talk_fair_comparison.json
        """
        if top_k is None:
            top_k = TALK_DEFAULT_DECODER["top_k"]
        if temp is None:
            temp = TALK_DEFAULT_DECODER["temp"]
        if rep_penalty is None:
            rep_penalty = TALK_DEFAULT_DECODER["rep_penalty"]
        net, meta = self._need("talk")
        import re
        if self.chat_vocab is None or len(self.chat_vocab) != net.V:
            raise ValueError(f"chat vocab ({0 if self.chat_vocab is None else len(self.chat_vocab)}"
                             f" words) does not exactly match the talk checkpoint's output width "
                             f"({net.V}) — refusing to name words across a mismatch")
        vmap = {w: i for i, w in enumerate(self.chat_vocab)}
        toks = re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", prompt.lower())
        enc = [vmap.get(x, 0) for x in toks]
        W = net.W
        # PAD-FREE CONTEXT (fixed 2026-11-06). This used to left-pad with id 0 ("<unk>"), which put
        # 20-23 <unk> tokens into a 32-slot context for the 6 project prompts and made the model
        # reply in <unk> 95.1-96.5% of the time (lens/runs/PROMPT-ENCODING-AUDIT.txt). Repeating the
        # prompt ids to fill the window, same weights and decoder, measured 26.4/21.5%.
        if W > 1 and enc:
            enc = (enc * ((W // len(enc)) + 2))[-W:]
        ctx = enc[-W:]
        out, used, r = [], {}, np.random.default_rng(seed)
        for _ in range(n_new):
            lg = net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0]
            z = lg.astype(np.float64)
            for tok, cnt in used.items():
                z[tok] -= np.log(rep_penalty) * cnt
            if greedy:
                nxt = int(z.argmax())
            else:
                zz = z / max(temp, 1e-6)
                if top_k and top_k < len(zz):
                    cut = np.partition(zz, -top_k)[-top_k]
                    zz = np.where(zz >= cut, zz, -np.inf)
                zz = zz - np.nanmax(zz[np.isfinite(zz)])
                p = np.exp(zz - zz.max())
                p = np.where(np.isfinite(p), p, 0.0)
                s = p.sum()
                p = p / s if s > 0 else np.ones(len(p)) / len(p)
                nxt = int(r.choice(len(p), p=p))
            out.append(self.chat_vocab[nxt])
            used[nxt] = used.get(nxt, 0) + 1
            ctx = (ctx + [nxt])[-W:]
        return {"model": "talk", "reply": " ".join(out)}

    def see_and_talk(self, img, prefix_ids):
        net, meta = self._need("see_talk")
        # a (28,28) gray array must become (1,784), not (1,28,28): the front reads 2-D as
        # (B, C*784) and a 3-D (1,28,28) as 28 separate channels, which crashes the reshape.
        # Measured on the K21o run: `see_and_talk(_load_image_28(p), ids)` died with
        # "cannot reshape array of size 784 into shape (1,28,4,7,4,7)" until this line.
        a = np.asarray(img, dtype=np.float32)
        X = a.reshape(1, -1) if a.ndim == 2 else (np.asarray([a], dtype=np.float32)
                                                  if a.ndim == 3 else np.asarray([a]))
        if X.shape[1] != net.front.channels * 28 * 28:
            X = X.reshape(1, -1)
        S = np.asarray([prefix_ids], dtype=np.int64)
        out = net.forward_arrays(X, S)
        lg, h = out["logits"][0], out["concepts"][0]
        p = np.exp(lg - lg.max())
        p /= p.sum()
        best = int(p.argmax())
        fired = [(c.name, round(float(h[i]), 3)) for i, c in enumerate(net.concepts)
                 if float(h[i]) > 0.5]
        return {"model": "see_talk", "word": meta["vocab"][best],
                "word_id": best, "confidence": float(p[best]), "fired_concepts": fired}

    def explain(self, img, prefix_ids, hide_decorative=True):
        """The exact arithmetic behind a see_and_talk word: concept cells x their weights into the
        chosen word, summed = the part of the logit the image controls. Residual vs the model is
        0.00e+00 (gate g3), so this listing IS the model's math, not a retelling of it.

        FIDELITY (added 2026-10-09 after the meaning audit, E18). Listing a concept NAME as a cause
        is only honest if that neuron actually tracks the named concept. On the medical checkpoints
        the mean correlation between a neuron and its own name was 0.42 at lam_concept 0.15 and
        **0.033 at lam_concept 0.00**, where zero concepts scored above 0.70 — those names were
        decorative, and printing them as reasons would be a false explanation. So every contribution
        is annotated with the concept's own-correlation and a verdict, and concepts whose verdict is
        'decorative' are EXCLUDED from the cause list by default (set hide_decorative=False to see
        them, clearly marked). The fidelity table is read from <checkpoint>_fidelity.json; if it is
        missing, the explanation says so instead of implying the names are trustworthy.
        """
        net, meta = self._need("see_talk")
        X = np.asarray([img], dtype=np.float32)
        S = np.asarray([prefix_ids], dtype=np.int64)
        out = net.forward_arrays(X, S)
        h, lg = out["concepts"][0], out["logits"][0]
        best = int(lg.argmax())
        Wa = np.asarray(lens.backend.to_cpu(net.cap_out.W.data))

        # --- fidelity lookup for the checkpoint actually loaded ---
        fid, fid_src = {}, None
        ck = self._paths.get("see_talk")
        if ck:
            fp = os.path.splitext(ck)[0] + "_fidelity.json"
            if os.path.exists(fp):
                try:
                    fdoc = json.load(open(fp))
                    fid = {c["name"]: c for c in fdoc.get("concepts", [])}
                    fid_src = os.path.basename(fp)
                except Exception:
                    fid = {}

        contribs, hidden = [], []
        for i, c in enumerate(net.concepts):
            w = float(Wa[i, best]) * float(h[i])
            f = fid.get(c.name)
            row = {"concept": c.name, "fired": round(float(h[i]), 3),
                   "weight_to_word": round(float(Wa[i, best]), 3),
                   "contribution": round(w, 4),
                   "own_corr": (round(f["own_corr"], 3) if f else None),
                   "verdict": (f["verdict"] if f else "unmeasured")}
            if hide_decorative and f and f["verdict"] == "decorative":
                hidden.append(row)
            else:
                contribs.append(row)
        contribs.sort(key=lambda d: -abs(d["contribution"]))
        hidden.sort(key=lambda d: -abs(d["contribution"]))

        n_dec = sum(1 for c in fid.values() if c["verdict"] == "decorative")
        if fid_src:
            note = (f"the image influences this word ONLY through these named concept cells. "
                    f"Each name is annotated with own_corr = how strongly that neuron tracks the "
                    f"concept it is NAMED after. {n_dec} concept(s) on this checkpoint are "
                    f"decorative (own_corr < 0.20) and are excluded from the cause list.")
        else:
            note = ("the image influences this word ONLY through these named concept cells. "
                    "WARNING: no fidelity table found for this checkpoint, so these names have NOT "
                    "been verified to describe what their neurons detect — treat them as labels, "
                    "not as proven causes.")
        res = {"word": meta["vocab"][best], "image_driven_contributions": contribs[:8],
               "fidelity_source": fid_src, "note": note}
        if hidden:
            res["decorative_excluded"] = [{"concept": r["concept"], "own_corr": r["own_corr"],
                                           "contribution": r["contribution"]} for r in hidden[:8]]
        return res


RUNS = os.path.join(PROJ, "lens", "runs")
# SHIPPING TALK CHECKPOINT: asonly_v12000_ep24, chosen on a FAIR comparison (2026-10-09).
# Every talk model was given its own decoder sweep (36 settings x 3 seeds), then ranked on metrics
# valid for all of them (unk rate + repetition — NOT val ppl, which is invalid for the masked model):
#     asonly_v12000  k80 t0.9 r1.6   unk 0.00%   repetition 0.51%   <- chosen
#     v12k           k60 t0.9 r1.6   unk 2.58%   repetition 0.77%
#     best1          k80 t0.9 r1.6   unk 6.08%   repetition 1.45%
# It wins BOTH metrics, so this is not a trade-off. Its tuned decoder settings live in
# TALK_DEFAULT_DECODER below. Evidence: lens/runs/chat/talk_fair_comparison.json
# TALK_SHIP lets a caller test a DIFFERENT talk checkpoint through the real runtime (scenario
# batteries, comparisons). Default is unchanged, so the shipped runtime is unaffected.
# Why this exists: the scenario battery used to set TALK_CKPT (which only the audit honours) while
# this constant kept loading t11 -- so three "bigger model" scenario runs actually tested the OLD
# ship and produced byte-identical output. That is the bug this line fixes.
CHAT_FINAL = os.environ.get("TALK_SHIP") or os.path.join(PROJ, "lens", "runs", "chat",
                                                         "t11_talk41_ep192.npz")
CHAT_VOCAB_FILE = "vocab_v12000.json"          # must match CHAT_FINAL's V exactly
TALK_DEFAULT_DECODER = {"top_k": 80, "temp": 0.9, "rep_penalty": 1.6}


def build_default(device="cpu"):
    # best1 (the talk checkpoint) was trained against vocab_v3500.json with window 16 — verified by
    # the parameter arithmetic (448k emb + 84k concept + 144k answer = 676k ~ the logged 679,009),
    # and talk() raises if an id ever falls outside the file it was built from.
    # see_talk default = the K=22 SHIPPING checkpoint (4/4 gates, delta +16.03); falls back to
    # small2 if it is missing so the interface never dead-ends.
    # 2026-10-10: shipping moved to seetalk_k22_big_fixedgrad.npz — the SAME K22 config retrained
    # on the corrected concept-BCE gradient (the old backward was wrong by a factor of pos_weight
    # for every concept with support != 0.5). The retrain: val_acc 77.03 -> 76.90 (noise), train
    # loss 0.5163 -> 0.4873, gates 4/4, and concept fidelity IMPROVED (decorative names 5 -> 2,
    # strong 10 -> 12) — the names better describe the neurons now. The old artifact is KEPT.
    # 2026-10-11: shipping see_talk moved to seetalk_k21d.npz — the ratio-relative concept
    # redesign + lam 0.6 (K21 -> K21b -> K21c -> K21d), strict label-honesty 15/21 (71%)
    # -> 17/19 (89%), 0 mislabelled, 0 AUROC deficits, gates 4/4. The boundary pairs were
    # measured NOT redundant (formula corr +0.565/+0.434) — their <0.10 margins are a
    # capacity limit of the linear front (d160 tested: same 17/19) and are named in BUGS.md.
    # The old artifact is KEPT.
    # the see_talk line ships the 19/19 CONCEPT ship (strict label-honesty on the pinned datum,
    # 0 mislabelled) -- seetalk_k21o_mirror; the older ships stay as fallbacks so the interface
    # never dead-ends. Every ship is auditable: set VISION_SHIP/VISION_CKPT and re-run.
    k22 = os.path.join(RUNS, "seetalk", "seetalk_k21o_mirror.npz")
    if not os.path.exists(k22):
        k22 = os.path.join(RUNS, "seetalk", "seetalk_k21d.npz")
    if not os.path.exists(k22):
        k22 = os.path.join(RUNS, "seetalk", "seetalk_k22_big_fixedgrad.npz")
    if not os.path.exists(k22):
        k22 = os.path.join(RUNS, "seetalk", "seetalk_k22_big.npz")
    # see default = ktest_real_s0 (K=23 extended concepts, 84.36 test vs the K=12 ladder's 82.0;
    # the +2.5 K-gain was proven k1 and the checkpoint loads through the frozen registry with
    # explain identity 0.00e+00 — same transparency contract as every ship artifact).
    see_k23 = os.path.join(RUNS, "image", "ktest_real_s0_ep12.npz")
    cv = json.load(open(os.path.join(PROJ, "data", "chat", "prepared", CHAT_VOCAB_FILE)))
    return Variants(
        see=see_k23 if os.path.exists(see_k23) else os.path.join(
            RUNS, "image", "lad_image_cls_ep24.npz"),
        talk=CHAT_FINAL if os.path.exists(CHAT_FINAL) else None,
        see_talk=k22 if os.path.exists(k22) else os.path.join(RUNS, "seetalk",
                                                               "seetalk_small2.npz"),
        device=device, chat_vocab=cv,
    )


if __name__ == "__main__":
    import vl_data
    print("=" * 96)
    print("VARIANTS SMOKE — one interface: see-only / talk-only / see+talk")
    print("=" * 96)
    d = vl_data.prepare_vqa("fashion", seed=0)
    names = d["meta"]["names"]
    X, y = d["Xte"], d["yte"]
    v = build_default()
    print(f"  loaded capabilities: {v.capabilities()}")

    r = v.see(X[0])
    print(f"  see(img0): {r['answer']}  (truth {names[int(y[0])]})  conf {r['confidence']:.2f}")

    # vocab + eval checkpoint = whatever build_default ACTUALLY loaded (the K22 shipping ckpt when
    # present) — reading a hardcoded small2 path would score a different model than the one demoed.
    npz = v._paths.get("see_talk", os.path.join(RUNS, "seetalk", "seetalk_small2.npz"))
    vocab = json.loads(str(np.load(npz, allow_pickle=True)["meta"][0]))["vocab"]
    vi = {w: i for i, w in enumerate(vocab)}
    prefix = [vi["<pad>"], vi["<pad>"], vi["<bos>"], vi["this"], vi["is"], vi["a"]]
    r2 = v.see_and_talk(X[0], prefix)
    print(f"  see_and_talk(img0,'this is a ___'): {r2['word']}  conf {r2['confidence']:.2f} "
          f"(truth {names[int(y[0])]})")
    ex = v.explain(X[0], prefix)
    print(f"  explain: top image-driven contributions -> "
          f"{[(c['concept'], c['contribution']) for c in ex['image_driven_contributions'][:3]]}")

    if "talk" in v.capabilities():
        r3 = v.talk([0] * 16)
        print(f"  talk(zeros window): next id {r3['next_id']} word {r3['word']!r} "
              f"conf {r3['confidence']:.2f}  (vocab file: data/chat/prepared/vocab_v3500.json)")
        s1 = v.talk_sentence("Human: hello , how are you ?\n\nAssistant:", n_new=20, seed=0)
        s2 = v.talk_sentence("Human: hello , how are you ?\n\nAssistant:", n_new=20, seed=0)
        same = s1["reply"] == s2["reply"]
        print(f"  talk_sentence(seed 0, deterministic={same}): {s1['reply'][:80]}")
        if not same:
            raise AssertionError("same seed produced different replies — decoder is not seeded")

    # refusal test: a see-only build asked to talk must REFUSE, not guess
    v2 = Variants(see=os.path.join(RUNS, "image", "lad_image_cls_ep24.npz"))
    try:
        v2.talk([0] * 16)
        print("  REFUSAL TEST: FAIL — see-only model accepted a talk request")
    except ValueError as e:
        print(f"  REFUSAL TEST: OK — {str(e)[:80]}")

    # see+talk batch correctness on TEST images at the noun position (first and only use of this
    # test split for THIS model — logged)
    import train_seetalk
    net, meta = train_seetalk.SeeTalkNet.load(npz)
    ok = 0
    for i in range(min(500, len(X))):
        lg = net.forward_arrays(X[i:i + 1], np.asarray([prefix], dtype=np.int64))["logits"][0]
        word = meta["vocab"][int(lg.argmax())]
        ok += 1 if word in names[int(y[i])].split("_") else 0
    print(f"  see_and_talk noun position, 500 TEST images: {ok}/500 = {ok/5:.1f}% chose the "
          f"class word")
    tu = os.path.join(PROJ, "TEST-USE.md")
    with open(tu, "a") as f:
        f.write(f"\n- 2026-10-08 variants.py smoke on {os.path.basename(npz)}: see_and_talk "
                "evaluated on first 500 Fashion-MNIST TEST images at the noun position (one test "
                "pass per checkpoint; prior pass was seetalk_small2).\n")
    print("=" * 96)
