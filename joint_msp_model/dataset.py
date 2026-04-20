import re
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from flair.data import Sentence, Token, FlairDataset

# ------------- URIEL getter -------------
# Tries lang2vec; falls back to deterministic vector.
def get_uriel_vector(lang_code: str, dim: int = 64) -> List[float]:
    """
    Return a fixed-size typological vector for `lang_code`.

    - First tries lang2vec with several URIEL-based spaces:
        ["syntax_knn", "phonology_knn", "inventory_knn",
         "genetic_knn", "geography_knn"]
      and concatenates them, then truncates / repeats to `dim`.
    - If anything fails (no lang2vec, unknown language, etc.),
      falls back to a deterministic hash-based pseudo-random vector.
    """
    try:
        from lang2vec.lang2vec import Lang2Vec
        l2v = Lang2Vec([
            "syntax_knn",
            "phonology_knn",
            "inventory_knn",
            "genetic_knn",
            "geography_knn",
        ])
        vec: List[float] = []
        for space in ["syntax_knn", "phonology_knn", "inventory_knn",
                      "genetic_knn", "geography_knn"]:
            try:
                v = l2v.get_vector(lang_code, space=space)[0]
                vec.extend(v)
            except Exception:
                continue

        if not vec:
            raise RuntimeError("empty vec from lang2vec")

        # Compact / expand to desired dimension
        if len(vec) < dim:
            k = (dim + len(vec) - 1) // len(vec)
            vec = (vec * k)[:dim]
        else:
            vec = vec[:dim]

        return [float(x) for x in vec]

    except Exception:
        # fallback: stable hash-based pseudo-random vector
        import hashlib
        h = hashlib.sha256(lang_code.encode()).digest()
        nums: List[float] = []
        for i in range(dim):
            j = i % len(h)
            v = (h[j] / 255.0) * 2 - 1  # [-1, 1]
            nums.append(float(v))
        return nums


class MSPDatasetEnhanced(FlairDataset):
    """
    CoNLL-U loader with:
      • always 'deprel' label (even '_')
      • head '_' -> -1 (or convert_head_underscore_to)
      • 'ms_feat_val' multi-labels + 'ms_feats_presence'
      • 'word_type' from FEATS ('_' => function, else content)
      • NEW: attach language code + uriel vector in Sentence metadata
      • NEW: derive ABS_* supervision on base tokens from abstract nodes X.Y
          - detect abstract lines where:
                '.' in ID AND HEAD, FEATS, DEPREL are non-empty (not '_')
                and ID is not like '0.Y'
          - if line X.Y is physically ABOVE X -> set base(X).ABS_POS=ABOVE
            else if BELOW -> ABS_POS=BELOW
          - copy X.Y's DEPREL -> base(X).ABS_DEPREL
          - copy each FEAT=VAL -> base(X).ABS_FEAT_VAL (multi)

      • Typology control:
          - uriel_dim: dimension of URIEL / lang2vec vector attached as `sent.uriel`
          - use_typology: if False, no `sent.uriel` is set (for ablation)
    """
    _DEPREL_SUBTYPE_REGEX = re.compile(r":.*")

    def __init__(
        self,
        path_to_conllu_file: Union[str, Path],
        in_memory: bool = True,
        split_multiwords: bool = True,
        keep_abstract_nodes: bool = True,
        convert_head_underscore_to: int = -1,
        lang_code: Optional[str] = None,
        uriel_dim: int = 64,
        use_typology: bool = True,
    ) -> None:
        self.path_to_conllu_file = Path(path_to_conllu_file)
        if not self.path_to_conllu_file.exists():
            raise FileNotFoundError(self.path_to_conllu_file)

        self.in_memory = in_memory
        self.split_multiwords = split_multiwords
        self.keep_abstract_nodes = keep_abstract_nodes
        self.convert_head_underscore_to = convert_head_underscore_to

        # language code + typology config
        self.lang_code = (
            lang_code or self._infer_lang_from_path(self.path_to_conllu_file.name)
        )
        self.uriel_dim = uriel_dim
        self.use_typology = use_typology

        if self.in_memory:
            self.sentences: List[Sentence] = []
            with open(self.path_to_conllu_file, encoding="utf-8") as f:
                while True:
                    s = self._read_next_sentence(f)
                    if s is None:
                        break
                    self.sentences.append(s)
            self.total_sentence_count = len(self.sentences)
        else:
            self.indices = [0]
            with open(self.path_to_conllu_file, encoding="utf-8") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip() == "":
                        self.indices.append(f.tell())
            if self.indices and self.indices[-1] == self.path_to_conllu_file.stat().st_size:
                self.indices.pop()
            self.total_sentence_count = len(self.indices)

    @staticmethod
    def _infer_lang_from_path(name: str) -> str:
        # msp.turkish.train.conllu -> 'tr' (best-effort)
        low = name.lower()
        # quick mapping
        table = {
            "turkish": "tr",
            "czech": "cs",
            "polish": "pl",
            "portuguese": "pt",
            "english": "en",
            "swedish": "sv",
            "serbian": "sr",
            "italian": "it",
            "hebrew": "he",
        }
        for k, v in table.items():
            if k in low:
                return v
        return "xx"

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self) -> int:
        return self.total_sentence_count

    def __getitem__(self, index: int) -> Sentence:
        if self.in_memory:
            return self.sentences[index]

        with open(self.path_to_conllu_file, encoding="utf-8") as f:
            f.seek(self.indices[index])
            return self._read_next_sentence(f) or Sentence("")

    def _read_next_sentence(self, fh) -> Optional[Sentence]:
        rows: List[List[str]] = []
        meta: List[str] = []
        while True:
            line = fh.readline()
            if not line:
                break
            line = line.rstrip("\n")
            if line == "":
                if rows:
                    return self._build_sentence(rows, meta)
                meta = []
                continue
            if line.startswith("#"):
                meta.append(line)
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            rows.append(cols)

        if rows:
            return self._build_sentence(rows, meta)
        return None

    def _build_sentence(self, rows: List[List[str]], meta: List[str]) -> Sentence:
        tokens: List[Token] = []

        # collect base and abstract rows with original order
        base_rows: List[Tuple[int, List[str]]] = []  # (order, cols)
        abs_rows: List[Tuple[int, List[str]]] = []   # (order, cols)

        for order, cols in enumerate(rows):
            tok_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = cols
            if "-" in tok_id:
                # ignore MWToken expansions (we output split tokens directly)
                continue
            if "." in tok_id:
                # abstract candidate
                if tok_id.startswith("0."):  # ignore 0.Y
                    continue
                is_valid = (
                    head not in ("", "_")
                    and feats not in ("", "_")
                    and deprel not in ("", "_")
                )
                if is_valid:
                    abs_rows.append((order, cols))
                # do not add abstract rows as tokens (training on base only)
                continue

            # base token
            base_rows.append((order, cols))

        # create tokens for base rows
        for _, cols in base_rows:
            tok_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = cols
            text = form if form != "_" else "_"
            head_id = (
                self.convert_head_underscore_to
                if head in ("", "_")
                else int(head)
            )
            t = Token(text, head_id=head_id)
            t._conllu_idx = tok_id  # keep original idx for mapping

            # lemma / upos / xpos
            if lemma != "_":
                t.add_label("lemma", lemma)
            if upos != "_":
                t.add_label("upos", upos)
            if xpos != "_":
                t.add_label("xpos", xpos)

            # deprel (canonicalize)
            clean = self._DEPREL_SUBTYPE_REGEX.sub("", deprel or "_")
            t.add_label("deprel", clean if clean else "_")

            # FEATS to ms_feat_val + presence
            keys: List[str] = []
            if feats and feats != "_":
                t.add_label("ms_feats", feats)
                for part in feats.split("|"):
                    if "=" not in part:
                        continue
                    k, vals = part.split("=", 1)
                    k = k.strip()
                    keys.append(k)
                    for v in vals.split(";"):
                        vv = v.strip()
                        if not vv:
                            continue
                        t.add_label(k, vv)
                        t.add_label("ms_feat_val", f"{k}={vv}")

            presence = "|".join(sorted(keys)) if keys else "_"
            t.add_label("ms_feats_presence", presence)

            # word_type: simple heuristic (no feats => function)
            t.add_label(
                "word_type",
                "function" if (feats == "_" or feats == "") else "content",
            )

            tokens.append(t)

        sent = Sentence(tokens)

        # ---- LANGUAGE + TYPOLOGY ATTACHMENT ----
        lang = self.lang_code
        setattr(sent, "lang", lang)

        # UDapter-style “typology feature vector”: we just keep it as a dense float list.
        # The model's `uriel_dim` should be set to the same value as `self.uriel_dim`.
        if self.use_typology and self.uriel_dim > 0:
            vec = get_uriel_vector(lang, dim=self.uriel_dim)
            setattr(sent, "uriel", vec)
        else:
            # For ablations: either leave unset, or explicitly set None.
            setattr(sent, "uriel", None)

        # ---- derive abstract labels and attach to base tokens ----
        # need quick map from ID -> (order, token_position)
        id2order: Dict[str, int] = {}
        for order, cols in base_rows:
            id2order[cols[0]] = order

        # map id to index in 'tokens' (surface position list)
        id2tokpos: Dict[str, int] = {}
        for i, (_, cols) in enumerate(base_rows):
            id2tokpos[cols[0]] = i

        # For each abstract row X.Y, find base X and compare orders
        for abs_order, cols in abs_rows:
            tok_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = cols
            base_id = tok_id.split(".")[0]
            if base_id not in id2tokpos:
                continue
            base_order = id2order.get(base_id, None)
            if base_order is None:
                continue

            pos = "ABOVE" if abs_order < base_order else "BELOW"
            tok_pos = id2tokpos[base_id]
            t = sent[tok_pos]

            # mark presence
            t.add_label("abs_pres", "1")
            # pos
            t.add_label("abs_pos", pos)

            # copy deprel to abs_deprel
            clean_dep = self._DEPREL_SUBTYPE_REGEX.sub("", deprel or "_")
            t.add_label("abs_deprel", clean_dep if clean_dep else "_")

            # feats into abs_feat_val
            if feats and feats != "_":
                for part in feats.split("|"):
                    if "=" not in part:
                        continue
                    k, vals = part.split("=", 1)
                    for v in vals.split(";"):
                        vv = v.strip()
                        if not vv:
                            continue
                        t.add_label("abs_feat_val", f"{k}={vv}")

        return sent
