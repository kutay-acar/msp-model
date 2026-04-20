#!/usr/bin/env python3
import math
import argparse
from pathlib import Path
import torch
from flair.data import Corpus
from flair.trainers import ModelTrainer

from joint_msp_model.dataset import MSPDatasetEnhanced
from joint_msp_model.joint_model import Joint_Model_


def parse_manifest(p: Path):
    items = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path, lang = line.split()
            items.append((Path(path), lang))
    return items


def build_abs_inventories(sentences):
    dep = set(["_"])
    feats = set()
    for s in sentences:
        for t in s:
            if t.has_label("abs_pres") and t.get_label("abs_pres").value in ("1", "true", "True"):
                if t.has_label("abs_deprel"):
                    dep.add(t.get_label("abs_deprel").value)
                for fl in t.get_labels("abs_feat_val"):
                    feats.add(fl.value)
    dep_items = {l: i for i, l in enumerate(sorted(dep))}
    feat_items = {l: i for i, l in enumerate(sorted(feats))}
    return dep_items, feat_items


def per_lang_abs_priors_and_thresholds(dev_sents):
    pri = {}
    thr = {}
    buckets = {}
    for s in dev_sents:
        lang = getattr(s, "lang", None) or "xx"
        buckets.setdefault(lang, []).append(s)
    for lang, sents in buckets.items():
        above = below = 0
        for s in sents:
            for t in s:
                if t.has_label("abs_pres") and t.get_label("abs_pres").value in ("1", "true", "True"):
                    p = t.get_label("abs_pos").value if t.has_label("abs_pos") else "_"
                    if p == "ABOVE":
                        above += 1
                    elif p == "BELOW":
                        below += 1
        tot = above + below
        pri[lang] = {
            "ABOVE": (above / tot if tot else 0.5),
            "BELOW": (below / tot if tot else 0.5),
        }
        thr[lang] = {"pres": 0.5, "feat": 0.5}
    return pri, thr


def get_args():
    p = argparse.ArgumentParser()
    # Single-language classic mode:
    p.add_argument('--train', type=str, help="train.conllu")
    p.add_argument('--dev', type=str, help="dev.conllu")
    p.add_argument('--lang', type=str, help="ISO code for single-language mode")
    # Multilingual manifest mode:
    p.add_argument('--train_manifest', type=str, help="TSV with: <path> <lang>")
    p.add_argument('--dev_manifest', type=str, help="TSV with: <path> <lang>")

    p.add_argument('--output', type=str, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--arc_mlp', type=int, default=256)
    p.add_argument('--rel_mlp', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.33)
    p.add_argument('--seed', type=int, default=42)

    # base weights
    p.add_argument('--parser_weight', type=float, default=2.0)
    p.add_argument('--morph_weight', type=float, default=2.0)
    p.add_argument('--wordtype_weight', type=float, default=1.5)

    # abstract single-stage ramp config (flip)
    p.add_argument('--abs_weight_min', type=float, default=0.01)
    p.add_argument('--abs_weight_max', type=float, default=0.5)
    p.add_argument('--abs_ramp_power', type=float, default=2.0)

    # --- UDapter-style typology / adapters control ---
    p.add_argument('--uriel_dim', type=int, default=64,
                  help="Dimensionality of typology vector attached as Sentence.uriel.")
    p.add_argument('--no_typology', action='store_true',
                  help="Disable typology: do not attach URIEL vectors and set uriel_dim=0 in the model.")
    p.add_argument('--no_adapters', action='store_true',
                  help="Disable contextual adapters (use identity adapters instead).")

    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Decide typology usage and dimension
    use_typology = not args.no_typology
    uriel_dim = 0 if args.no_typology else args.uriel_dim

    # --- Load sentences ---
    train_sents = []
    dev_sents = []
    langs = set()

    if args.train_manifest and args.dev_manifest:
        # Multilingual mode: manifest lines are: <path> <lang>
        tr_items = parse_manifest(Path(args.train_manifest))
        dv_items = parse_manifest(Path(args.dev_manifest))

        for path, lang in tr_items:
            ds = MSPDatasetEnhanced(
                path,
                in_memory=True,
                lang_code=lang,
                uriel_dim=uriel_dim,
                use_typology=use_typology,
            )
            train_sents += ds.sentences
            langs.add(lang)

        for path, lang in dv_items:
            ds = MSPDatasetEnhanced(
                path,
                in_memory=True,
                lang_code=lang,
                uriel_dim=uriel_dim,
                use_typology=use_typology,
            )
            dev_sents += ds.sentences
            langs.add(lang)

        print(f"[multi] Train: {len(train_sents)}  Dev: {len(dev_sents)}  Langs: {sorted(langs)}")

    else:
        # single-language mode
        if not args.train or not args.dev:
            raise SystemExit(
                "Provide either --train/--dev/--lang for single-lang "
                "or --train_manifest/--dev_manifest for multilingual."
            )

        lang = args.lang or MSPDatasetEnhanced._infer_lang_from_path(Path(args.train).name)

        tr = MSPDatasetEnhanced(
            Path(args.train),
            in_memory=True,
            lang_code=lang,
            uriel_dim=uriel_dim,
            use_typology=use_typology,
        )
        dv = MSPDatasetEnhanced(
            Path(args.dev),
            in_memory=True,
            lang_code=lang,
            uriel_dim=uriel_dim,
            use_typology=use_typology,
        )

        train_sents = tr.sentences
        dev_sents = dv.sentences
        langs.add(lang)

        print(f"[single] Lang={lang} Train={len(train_sents)} Dev={len(dev_sents)}")

    corpus = Corpus(train=train_sents, dev=dev_sents, test=[])
    corpus.corpus_tokenizer = None

    # base dicts
    deprel_dict = corpus.make_label_dictionary(label_type="deprel", add_unk=False)
    morph_dict = corpus.make_label_dictionary(label_type="ms_feat_val", add_unk=True)

    # abstract inventories from train
    abs_dep_items, abs_feat_items = build_abs_inventories(corpus.train)

    # language inventory
    lang2idx = {l: i for i, l in enumerate(sorted(langs))}

    # init model
    model = Joint_Model_(
        deprel_dictionary=deprel_dict,
        morph_dictionary=morph_dict,
        num_abs_deprel=max(1, len(abs_dep_items)),
        num_abs_feats=max(1, len(abs_feat_items)),
        num_langs=len(lang2idx),
        uriel_dim=uriel_dim,
        embedding_name="xlm-roberta-large",
        arc_mlp_size=args.arc_mlp,
        rel_mlp_size=args.rel_mlp,
        dropout=args.dropout,
        use_char_embeddings=True,
        use_layer_norm=True,
        morph_threshold=0.5,
        parser_weight=args.parser_weight,
        morph_weight=args.morph_weight,
        wordtype_weight=args.wordtype_weight,
        abs_weight_min=args.abs_weight_min,
        abs_weight_max=args.abs_weight_max,
        abs_ramp_power=args.abs_ramp_power,
        use_contextual_adapters=not args.no_adapters,
    )

    model.set_language_inventory(lang2idx, uriel_dim)
    model.abs_deprel_items = abs_dep_items
    model.abs_feat_items = abs_feat_items
    model.abs_deprel_items_inv = {i: l for l, i in abs_dep_items.items()}
    model.abs_feat_items_inv = {i: l for l, i in abs_feat_items.items()}

    # per-language priors + thresholds from dev (for abstract arcs)
    pri, thr = per_lang_abs_priors_and_thresholds(corpus.dev)
    model.abs_lang_priors = pri
    model.abs_lang_thresholds = thr

    # rough steps estimate for ramp (for abstract loss schedule)
    num_batches_per_epoch = max(1, math.ceil(len(corpus.train) / args.batch_size))
    model.expected_total_steps = max(1, num_batches_per_epoch * args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer(model, corpus)
    trainer.train(
        base_path=str(outdir),
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        mini_batch_chunk_size=2,
        max_epochs=args.epochs,
        optimizer=torch.optim.AdamW,
        patience=1,
        anneal_factor=0.5,
        min_learning_rate=1e-7,
        train_with_dev=False,
        monitor_train_sample=0.1,
        monitor_test=False,
        save_final_model=True,
        save_optimizer_state=True,
        main_evaluation_metric=("las", "las"),
        embeddings_storage_mode='none',
        create_file_logs=True,
        create_loss_file=True,
        shuffle=True,
    )
    print(f"Done. Model at: {outdir}")


if __name__ == "__main__":
    main()

