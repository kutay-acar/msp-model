#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path
from collections import defaultdict
import torch

from flair.data import Sentence, Token, Corpus
from joint_msp_model.dataset import MSPDatasetEnhanced, get_uriel_vector
from joint_msp_model.joint_model import Joint_Model_


def read_conllu_with_cols(path: Path):
    """Read CoNLL-U as meta, rows with (idx, form, upos)."""
    data = []
    with open(path, encoding="utf-8") as fh:
        sent = []
        meta = []
        for line in fh:
            line = line.rstrip("\n")
            if line == "":
                if sent:
                    data.append((meta, sent))
                sent, meta = [], []
            elif line.startswith("#"):
                meta.append(line)
            else:
                parts = line.split("\t")
                if len(parts) >= 6:
                    idx, form, upos = parts[0], parts[1], parts[3]
                    # skip multiword and abstract lines
                    if "-" in idx or "." in idx:
                        continue
                    sent.append((idx, form, upos))
        if sent:
            data.append((meta, sent))
    return data


def build_flair_sentence(tokens, lang_code: str):
    """
    Build a minimal Flair Sentence from (idx, form, upos) tuples.
    We only set UPOS and lang here; typology (URIEL) is attached later
    once the model is loaded and we know model.uriel_dim.
    """
    s_tokens = []
    for idx, form, upos in tokens:
        t = Token(form)
        t._conllu_idx = idx
        t.add_label("upos", upos if upos else "_")
        s_tokens.append(t)
    sent = Sentence(s_tokens, use_tokenizer=False)
    setattr(sent, "lang", lang_code)
    return sent


def nearest_verb_head(source_pos: int, upos_seq: list) -> int:
    """Return nearest index (0-based) of a VERB; default to self if none found."""
    best = None
    best_dist = 1e9
    for i, tag in enumerate(upos_seq):
        if tag == "VERB":
            d = abs(i - source_pos)
            if d < best_dist:
                best_dist = d
                best = i
    return source_pos if best is None else best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_file", required=True, help="Input CoNLL-U (gold tokens)")
    ap.add_argument("--train_conllu", required=True, help="Training CoNLL-U (for dicts/logging)")
    ap.add_argument("--joint_model", required=True, help="Checkpoint")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5, help="Morph threshold")
    ap.add_argument("--lang", required=True, help="ISO lang code for file (e.g., tr, cs)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Read gold tokens/upos only; mask other cols
    data = read_conllu_with_cols(Path(args.raw_file))
    print(f"Loaded {len(data)} sentences")

    # Build Flair sentences with lang attribute (typology will be added after model load)
    sentences = [build_flair_sentence(sent, args.lang) for (meta, sent) in data]

    # Dicts from train file (primarily for logging / sanity)
    train_ds = MSPDatasetEnhanced(
        Path(args.train_conllu),
        in_memory=True,
        lang_code=args.lang,
    )
    corpus = Corpus(train=train_ds.sentences, dev=[], test=[])
    deprel_dict = corpus.make_label_dictionary(label_type="deprel", add_unk=False)
    morph_dict = corpus.make_label_dictionary(label_type="ms_feat_val", add_unk=True)
    print(f"Deprel dict: {len(deprel_dict)}  Morph dict: {len(morph_dict)}")

    # Load joint model (this restores dictionaries, lang2idx, uriel_dim, abs inventories, etc.)
    model: Joint_Model_ = Joint_Model_.load(args.joint_model).to(device).eval()
    model.morph_threshold = args.threshold

    # Attach typology vectors (UDapter-style) if the model expects them
    # The model stores `uriel_dim`; if > 0, we mimic training-time behavior.
    if getattr(model, "uriel_dim", 0) > 0:
        u_dim = model.uriel_dim
        lang_code = args.lang
        print(f"Attaching URIEL typology for lang={lang_code} (dim={u_dim})")
        # One shared vector per language/file is fine (same as training).
        uriel_vec = get_uriel_vector(lang_code, dim=u_dim)
        for s in sentences:
            setattr(s, "uriel", uriel_vec)
    else:
        # No typology (baseline / ablation)
        print("Model has uriel_dim=0 → no typology used at inference.")
        for s in sentences:
            setattr(s, "uriel", None)

    # Predict in batches
    with torch.no_grad():
        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i:i + args.batch_size]
            model.predict(
                batch,
                mini_batch_size=len(batch),
                predict_parser=True,
                predict_morph=True,
                predict_wordtype=True,
                predict_abs=True,
                embedding_storage_mode="none",
            )

    # Post-process: write CoNLL-U + insert abstract nodes when predicted
    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fout:
        for (meta, base_rows), sent in zip(data, sentences):
            # keep original meta but ensure # text line exists
            text = " ".join([form for _, form, _ in base_rows])
            for m in meta:
                if not m.startswith("# text"):
                    fout.write(m + "\n")
            fout.write(f"# text = {text}\n")

            # Collect UPOS for nearest-VERB heuristic
            upos_seq = []
            for t in sent:
                labs = t.get_labels("upos")
                upos_seq.append(labs[-1].value if labs else "_")

            # Map linear position -> original ID
            position2id = [idx for idx, form, upos in base_rows]

            def conllu_line(cols):
                return "\t".join(cols) + "\n"

            # Iterate over tokens in order, adding base + possible abstract node
            for pos, ((idx, form, upos), tok) in enumerate(zip(base_rows, sent)):
                # Decide content/function via predicted word type
                is_content = tok.get_label("predicted_word_type").value == "content"

                if is_content:
                    head_lbls = tok.get_labels("predicted_head")
                    head_v = head_lbls[-1].value if head_lbls else "_"
                    deprel_lbls = tok.get_labels("predicted_deprel")
                    deprel_v = deprel_lbls[-1].value if deprel_lbls else "_"

                    morph_vals = (
                        [l.value for l in tok.get_labels("predicted_ms_feat_val")]
                        if head_v != "_"
                        else "_"
                    )
                    clean_vals = [v for v in morph_vals if v not in ("", "_", "|")]
                    if clean_vals:
                        feat_map = defaultdict(list)
                        for fv in clean_vals:
                            if "=" in fv:
                                k, v = fv.split("=", 1)
                                feat_map[k].append(v)
                        parts = []
                        for k in sorted(feat_map.keys()):
                            vals = sorted(set(feat_map[k]))
                            parts.append(f"{k}={';'.join(vals)}")
                        feats_out = "|".join(parts)
                    else:
                        feats_out = "|"

                    # map head position->original ID
                    if head_v in ("_", "-1"):
                        head_id = "_"
                    elif head_v == "0":
                        head_id = "0"
                    else:
                        try:
                            head_pos = int(head_v) - 1
                            head_id = (
                                position2id[head_pos]
                                if 0 <= head_pos < len(position2id)
                                else "_"
                            )
                        except Exception:
                            head_id = "_"
                else:
                    head_id = "_"
                    deprel_v = "_"
                    feats_out = "_"

                # Prepare to maybe add an abstract node according to predictions
                want_abs = False
                abs_pos = "_"  # '_', ABOVE, BELOW
                abs_dep = "_"
                abs_feats = []

                pres_lab = tok.get_labels("predicted_abs_pres")
                if pres_lab and pres_lab[-1].value == "1":
                    want_abs = True
                    ppos = tok.get_labels("predicted_abs_pos")
                    abs_pos = ppos[-1].value if ppos else "_"
                    pdep = tok.get_labels("predicted_abs_deprel")
                    abs_dep = pdep[-1].value if pdep else "_"
                    for fl in tok.get_labels("predicted_abs_feat_val"):
                        abs_feats.append(fl.value)

                # Abstract insertion rule:
                # If abs_dep != '_' AND at least 1 abs_feat, we insert; else skip.
                if want_abs and abs_dep != "_" and len(abs_feats) > 0 and abs_pos in ("ABOVE", "BELOW"):
                    abs_id = f"{idx}.1"

                    # Determine abstract head:
                    #  - for Turkish: head is self (X)
                    #  - else: nearest VERB by gold UPOS
                    if (args.lang or "").lower().startswith("tr"):
                        abs_head = idx
                    else:
                        nearest = nearest_verb_head(pos, upos_seq)
                        abs_head = position2id[nearest]

                    abs_feats_str = "|".join(sorted(set(abs_feats))) if abs_feats else "_"

                    # ABOVE: write abstract line before base token
                    if abs_pos == "ABOVE":
                        cols_abs = [
                            abs_id, "_", "_", "_", "_",
                            abs_feats_str, abs_head, abs_dep, "_", "_",
                        ]
                        fout.write(conllu_line(cols_abs))

                # Write base row
                cols_base = [
                    idx,
                    form,
                    "_",
                    upos if upos else "_",
                    "_",
                    feats_out if is_content else "_",
                    head_id,
                    deprel_v,
                    "_",
                    "_",
                ]
                fout.write(conllu_line(cols_base))

                # If BELOW, write abstract after base line
                if want_abs and abs_dep != "_" and len(abs_feats) > 0 and abs_pos == "BELOW":
                    abs_id = f"{idx}.1"

                    if (args.lang or "").lower().startswith("tr"):
                        abs_head = idx
                    else:
                        nearest = nearest_verb_head(pos, upos_seq)
                        abs_head = position2id[nearest]

                    abs_feats_str = "|".join(sorted(set(abs_feats))) if abs_feats else "_"
                    cols_abs = [
                        abs_id, "_", "_", "_", "_",
                        abs_feats_str, abs_head, abs_dep, "_", "_",
                    ]
                    fout.write(conllu_line(cols_abs))

            fout.write("\n")

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
