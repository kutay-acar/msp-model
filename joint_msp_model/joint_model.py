import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path
from flair.data import Dictionary, Sentence
from flair.nn import Classifier
from flair.embeddings import StackedEmbeddings, TransformerWordEmbeddings, CharacterEmbeddings
from flair.training_utils import Result
from torch_struct import DependencyCRF


# ---------------------------
# UDapter-style Typology & Adapters
# ---------------------------

class TypologyEncoder(nn.Module):
    """
    UDapter-style language representation:
      lang_id embedding + URIEL vector -> typology-aware language embedding h_lang.

    In UDapter, URIEL/language typology comes from lang2vec and uses
    103 syntactic, 28 phonological and 158 phonetic-inventory features (total 289)
    as input slices. We assume that data loading code already selects the same
    slices and passes them in Sentence.uriel as a float vector of length uriel_dim.
    """

    def __init__(
        self,
        num_langs: int,
        uriel_dim: int,
        lang_emb_dim: int = 64,
        typo_hidden_dim: int = 64,
        out_dim: int = 128,
    ):
        super().__init__()
        self.num_langs = max(1, num_langs)
        self.uriel_dim = max(0, uriel_dim)

        self.lang_embed = nn.Embedding(self.num_langs, lang_emb_dim)

        if self.uriel_dim > 0:
            self.uriel_proj = nn.Linear(self.uriel_dim, typo_hidden_dim)
        else:
            self.uriel_proj = None

        in_dim = lang_emb_dim + (typo_hidden_dim if self.uriel_proj is not None else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, lang_ids: torch.LongTensor, uriel_vecs: Optional[torch.Tensor]) -> torch.Tensor:
        """
        lang_ids: [B]
        uriel_vecs: [B, uriel_dim] or None
        returns: h_lang [B, out_dim]
        """
        B = lang_ids.size(0)
        device = lang_ids.device

        lang_emb = self.lang_embed(lang_ids)  # [B, lang_emb_dim]

        if self.uriel_proj is not None and self.uriel_dim > 0:
            if uriel_vecs is None:
                uriel_vecs = torch.zeros(B, self.uriel_dim, device=device)
            h_typo = torch.tanh(self.uriel_proj(uriel_vecs))   # [B, typo_hidden_dim]
            h = torch.cat([lang_emb, h_typo], dim=-1)
        else:
            h = lang_emb

        return self.mlp(h)  # [B, out_dim]


class ContextualAdapter(nn.Module):
    """
    UDapter-style contextual language adapter:

      - base_down/base_up are shared across languages.
      - For each mini-batch, small language-specific deltas are generated
        from h_lang and added to the base matrices (contextual parameter generation).

    x: [B, L, D]
    h_lang: [B, H]
    """

    def __init__(self, dim: int, bottleneck: int, lang_repr_dim: int):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck

        # base adapter params (shared)
        self.base_down = nn.Linear(dim, bottleneck)
        self.base_up = nn.Linear(bottleneck, dim)

        # hypernets for language-specific deltas
        self.delta_down = nn.Linear(lang_repr_dim, bottleneck * dim)
        self.delta_up = nn.Linear(lang_repr_dim, dim * bottleneck)

        # keep deltas small at init
        self.delta_scale = 0.1

    def forward(self, x: torch.Tensor, h_lang: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        h_lang: [B, H]
        returns: [B, L, D]
        """
        B, L, D = x.size()
        assert D == self.dim

        Wd = self.base_down.weight   # [bottleneck, D]
        bd = self.base_down.bias     # [bottleneck]
        Wu = self.base_up.weight     # [D, bottleneck]
        bu = self.base_up.bias       # [D]

        dWd = self.delta_down(h_lang) * self.delta_scale  # [B, bottleneck*D]
        dWu = self.delta_up(h_lang) * self.delta_scale    # [B, D*bottleneck]
        dWd = dWd.view(B, self.bottleneck, D)
        dWu = dWu.view(B, D, self.bottleneck)

        out_batches = []
        for b in range(B):
            Wd_b = Wd + dWd[b]
            Wu_b = Wu + dWu[b]

            xb = x[b]                          # [L, D]
            down = F.linear(xb, Wd_b, bd)      # [L, bottleneck]
            down = F.relu(down)
            up = F.linear(down, Wu_b, bu)      # [L, D]
            out_batches.append(xb + up)

        return torch.stack(out_batches, dim=0)


class IdentityAdapter(nn.Module):
    """
    Simple identity adapter used when use_contextual_adapters=False.
    Signature matches ContextualAdapter (x, h_lang) but ignores h_lang.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, h_lang: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class Joint_Model_(Classifier[Sentence]):
    """
    Unified model:
      • Dependency parsing (content-only, CRF)
      • Morphological multi-label tagging (content-only)
      • Word-type classification (content/function)
      • Abstract heads (presence/pos/deprel/feats) ungated by word-type
      • Multilingual typology conditioning (UDapter-style typology encoder + contextual adapters)
    """

    def __init__(
        self,
        deprel_dictionary,
        morph_dictionary,
        # abstract label space sizes (built in trainer)
        num_abs_deprel: int,
        num_abs_feats: int,
        # multilingual
        num_langs: int = 1,
        uriel_dim: int = 0,
        embedding_name: str = "xlm-roberta-large",
        # Parser params
        arc_mlp_size: int = 256,
        rel_mlp_size: int = 128,
        # Shared params
        dropout: float = 0.33,
        use_char_embeddings: bool = True,
        use_layer_norm: bool = True,
        morph_threshold: float = 0.5,
        # Base loss weights (parser, morph, wordtype)
        parser_weight: float = 1.0,
        morph_weight: float = 1.0,
        wordtype_weight: float = 1.0,
        # Abstract heads ramp (single-stage flip)
        abs_weight_min: float = 0.05,
        abs_weight_max: float = 1.0,
        abs_ramp_power: float = 1.5,
        # NEW: explicit ablation flag
        use_contextual_adapters: bool = True,
    ):
        super().__init__()

        # Dicts
        self.deprel_dictionary = deprel_dictionary
        self.morph_dictionary = morph_dictionary
        self.relations = (deprel_dictionary.get_items()
                          if hasattr(deprel_dictionary, "get_items")
                          else deprel_dictionary)
        self.relation_map = {r: i for i, r in enumerate(self.relations)}

        self.arc_mlp_size = arc_mlp_size
        self.rel_mlp_size = rel_mlp_size
        self.morph_threshold = morph_threshold
        self.use_layer_norm = use_layer_norm

        self.parser_weight = parser_weight
        self.morph_weight = morph_weight
        self.wordtype_weight = wordtype_weight

        # abstract-head dynamic weight schedule settings
        self.abs_weight_min = abs_weight_min
        self.abs_weight_max = abs_weight_max
        self.abs_ramp_power = abs_ramp_power
        self.train_step = 0
        self.expected_total_steps = 1000  # set by trainer

        # Ablation flag
        self.use_contextual_adapters = use_contextual_adapters

        # Embeddings
        emb_list = [TransformerWordEmbeddings(model=embedding_name, fine_tune=True)]
        if use_char_embeddings:
            emb_list.append(CharacterEmbeddings(char_embedding_dim=100, hidden_size_char=300))
        self.embeddings = StackedEmbeddings(emb_list)
        self.embedding_dim = self.embeddings.embedding_length

        # Shared trunk
        self.dropout = nn.Dropout(dropout)
        if use_layer_norm:
            self.embedding_norm = nn.LayerNorm(self.embedding_dim)

        self.shared_intermediate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----- UDapter-style typology + contextual adapters -----
        self.num_langs = num_langs
        self.uriel_dim = uriel_dim

        # language representation h_lang (shared across tasks)
        self.lang_repr_dim = 128
        self.typology_encoder = TypologyEncoder(
            num_langs=self.num_langs,
            uriel_dim=self.uriel_dim,
            lang_emb_dim=64,
            typo_hidden_dim=64,
            out_dim=self.lang_repr_dim,
        )

        # NEW: per-language abstract-loss gate (logits -> sigmoid)
        self.abs_lang_gate = nn.Embedding(self.num_langs, 1)
        nn.init.zeros_(self.abs_lang_gate.weight)  # all gates start at sigmoid(0)=0.5

        if self.use_contextual_adapters:
            # contextual adapters:
            #   - one for parsing (content-only dependency)
            #   - one for sequence labeling-style tasks (morph, word-type, ABS)
            self.parser_adapter = ContextualAdapter(
                dim=self.embedding_dim,
                bottleneck=128,
                lang_repr_dim=self.lang_repr_dim,
            )
            self.tagger_adapter = ContextualAdapter(
                dim=self.embedding_dim,
                bottleneck=128,
                lang_repr_dim=self.lang_repr_dim,
            )
        else:
            # no typology-based adaptation: identity
            self.parser_adapter = IdentityAdapter(dim=self.embedding_dim)
            self.tagger_adapter = IdentityAdapter(dim=self.embedding_dim)

        # typology prediction head (UDapter-style multi-task)
        # we predict URIEL features from h_lang; masking of missing features is applied in the loss
        self.typology_weight = 0.0  # tune this
        if self.uriel_dim > 0:
            self.typology_linear = nn.Linear(self.lang_repr_dim, self.uriel_dim)
        else:
            self.typology_linear = None

        # Parser MLPs
        self.arc_dep_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, arc_mlp_size),
            nn.LayerNorm(arc_mlp_size) if use_layer_norm else nn.Identity(),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.arc_head_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, arc_mlp_size),
            nn.LayerNorm(arc_mlp_size) if use_layer_norm else nn.Identity(),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.rel_dep_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, rel_mlp_size),
            nn.LayerNorm(rel_mlp_size) if use_layer_norm else nn.Identity(),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.rel_head_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, rel_mlp_size),
            nn.LayerNorm(rel_mlp_size) if use_layer_norm else nn.Identity(),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.arc_W = nn.Parameter(torch.zeros(arc_mlp_size + 1, arc_mlp_size + 1))
        self.rel_W = nn.Parameter(torch.zeros(len(self.relations), rel_mlp_size + 1, rel_mlp_size + 1))

        # Morph head
        num_morph_labels = len(self.morph_dictionary.get_items())
        self.morph_linear = nn.Linear(self.embedding_dim, num_morph_labels)

        # Word-type head (BiLSTM + linear)
        self.wordtype_lstm = nn.LSTM(self.embedding_dim, hidden_size=256, num_layers=1,
                                     bidirectional=True, batch_first=True, dropout=0.0)
        self.wordtype_locked_dropout = nn.Dropout(0.5)
        self.wordtype_word_dropout = 0.05
        self.wordtype_linear = nn.Linear(512, 2)
        self.wordtype_class_weights = None

        # Abstract heads
        self.abs_pres_linear = nn.Linear(self.embedding_dim, 1)  # presence sigmoid
        self.abs_pos_linear = nn.Linear(self.embedding_dim, 3)   # '_', ABOVE, BELOW (single-label)
        self.abs_deprel_linear = nn.Linear(self.embedding_dim, num_abs_deprel)  # single-label
        self.abs_feats_linear = nn.Linear(self.embedding_dim, num_abs_feats)    # multi-label

        # Inventories (filled by trainer)
        self.lang2idx: Dict[str, int] = {}
        self.abs_lang_priors = {}      # per-lang priors for ABOVE/BELOW at inference
        self.abs_lang_thresholds = {}  # per-lang thresholds dict

        self._init_weights()

    def set_language_inventory(self, lang2idx: Dict[str, int], uriel_dim: int):
        self.lang2idx = lang2idx
        self.uriel_dim = uriel_dim

    @property
    def label_type(self) -> str:
        return "multitask"

    # ---------------------- utils ----------------------
    def _init_weights(self):
        nn.init.xavier_uniform_(self.arc_W, gain=1.0)
        nn.init.xavier_uniform_(self.rel_W, gain=1.0)

        def init_linear(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for module in [self.arc_dep_mlp, self.arc_head_mlp, self.rel_dep_mlp, self.rel_head_mlp]:
            for layer in module:
                init_linear(layer)
        for layer in self.shared_intermediate:
            init_linear(layer)
        init_linear(self.morph_linear)
        init_linear(self.wordtype_linear)
        init_linear(self.abs_pres_linear)
        init_linear(self.abs_pos_linear)
        init_linear(self.abs_deprel_linear)
        init_linear(self.abs_feats_linear)

        for name, p in self.wordtype_lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # dynamic abs weight in single-stage (flip/ramp)
    def _abs_weight(self):
        if self.expected_total_steps <= 0:
            return self.abs_weight_max
        frac = min(1.0, max(0.0, self.train_step / self.expected_total_steps))
        ramp = (frac ** self.abs_ramp_power)
        return self.abs_weight_min + (self.abs_weight_max - self.abs_weight_min) * ramp

    # NEW: per-language gate for abstract-loss, based on lang_ids
    def _abs_lang_gate_weight(self, lang_ids: torch.LongTensor) -> torch.Tensor:
        """
        lang_ids: [B]
        Returns a scalar gate in [0,1] for this batch.
        If your sampler is per-language, all lang_ids in the batch are identical,
        so this is effectively a per-language gate.
        """
        gate_logits = self.abs_lang_gate(lang_ids).view(-1)      # [B]
        gate_vals = torch.sigmoid(gate_logits)                   # [B]
        return gate_vals.mean()                                  # scalar

    # ---- embedding (no typology here; typology goes via adapters) ----
    def _embed(self, sentences: List[Sentence]):
        self.embeddings.embed(sentences)
        device = next(self.parameters()).device
        lengths = [len(s) for s in sentences]
        max_len = max(lengths) if lengths else 1
        B = len(sentences)
        D = self.embedding_dim

        emb = torch.zeros(B, max_len, D, device=device)
        for i, sent in enumerate(sentences):
            for j, token in enumerate(sent):
                emb[i, j] = token.embedding

        if self.use_layer_norm:
            mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
            for i, L in enumerate(lengths):
                mask[i, :L] = True
            emb = self.embedding_norm(emb)
            emb = emb * mask.unsqueeze(-1)

        emb = self.shared_intermediate(emb)
        return emb, lengths

    # ---- typology: language representation + URIEL gold + mask ----
    def _get_language_repr(self, sentences: List[Sentence], device):
        """
        Build UDapter-style language representation h_lang and URIEL targets
        plus an optional mask for missing features.

        We expect:
          - s.lang: UD language code (mapped via self.lang2idx)
          - s.uriel: list/array of floats of length uriel_dim (selected URIEL slices)
          - optionally s.uriel_mask: same length, 1.0 for observed features, 0.0 for missing
        """
        lang_ids_list = []
        uriel_list = []
        mask_list = []

        for s in sentences:
            lang = getattr(s, "lang", None) or "xx"
            lang_idx = self.lang2idx.get(lang, 0)
            lang_ids_list.append(lang_idx)

            # URIEL feature vector
            if self.uriel_dim > 0:
                uv = getattr(s, "uriel", None)
                if uv is None:
                    uriel_list.append(torch.zeros(self.uriel_dim, device=device))
                else:
                    uv_t = torch.tensor(uv, device=device, dtype=torch.float)
                    if uv_t.numel() != self.uriel_dim:
                        # fall back to zeros if dimension mismatch
                        uv_t = torch.zeros(self.uriel_dim, device=device)
                    uriel_list.append(uv_t)

                # mask for missing features (1 = observed, 0 = missing)
                um = getattr(s, "uriel_mask", None)
                if um is None:
                    mask_list.append(torch.ones(self.uriel_dim, device=device))
                else:
                    um_t = torch.tensor(um, device=device, dtype=torch.float)
                    if um_t.numel() != self.uriel_dim:
                        um_t = torch.ones(self.uriel_dim, device=device)
                    mask_list.append(um_t)
            else:
                uriel_list.append(torch.zeros(1, device=device))
                mask_list.append(torch.ones(1, device=device))

        lang_ids = torch.tensor(lang_ids_list, device=device, dtype=torch.long)  # [B]
        uriel_gold = None
        uriel_mask = None
        if self.uriel_dim > 0:
            uriel_gold = torch.stack(uriel_list)  # [B, uriel_dim]
            uriel_mask = torch.stack(mask_list)   # [B, uriel_dim]

        h_lang = self.typology_encoder(lang_ids, uriel_gold if self.uriel_dim > 0 else None)
        # now also return lang_ids for gating
        return h_lang, uriel_gold, uriel_mask, lang_ids

    # ---- parser scoring ----
    def _get_arc_scores(self, emb, mask):
        dep = self.arc_dep_mlp(emb)
        head = self.arc_head_mlp(emb)
        if self.training:
            noise = 0.01
            dep = dep + torch.randn_like(dep) * noise
            head = head + torch.randn_like(head) * noise
        dep = torch.cat([dep, dep.new_ones(*dep.shape[:-1], 1)], dim=-1)
        head = torch.cat([head, head.new_ones(*head.shape[:-1], 1)], dim=-1)
        scale = (self.arc_mlp_size + 1) ** -0.5
        scores = torch.einsum("bmd,dh,bnh->bmn", dep, self.arc_W, head) * scale
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        scores = scores.masked_fill(~mask.unsqueeze(2), -1e4)
        return scores

    def _get_rel_scores(self, emb, mask):
        dep = self.rel_dep_mlp(emb)
        head = self.rel_head_mlp(emb)
        if self.training:
            noise = 0.01
            dep = dep + torch.randn_like(dep) * noise
            head = head + torch.randn_like(head) * noise
        dep = torch.cat([dep, dep.new_ones(*dep.shape[:-1], 1)], dim=-1)
        head = torch.cat([head, head.new_ones(*head.shape[:-1], 1)], dim=-1)
        scale = (self.rel_mlp_size + 1) ** -0.5
        scores = torch.einsum("bmd,rdh,bnh->bmnr", dep, self.rel_W, head) * scale
        scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(-1), -1e4)
        scores = scores.masked_fill(~mask.unsqueeze(2).unsqueeze(-1), -1e4)
        return scores

    # ---- content-only selection for parser/morph ----
    @staticmethod
    def _content_mask_from_sentences(sentences: List[Sentence]):
        idxs = []
        for s in sentences:
            curr = []
            for i, tok in enumerate(s):
                # Inference: prefer predicted_word_type if present
                if tok.has_label("predicted_word_type"):
                    wt = tok.get_label("predicted_word_type").value
                # Training / gold datasets: fall back to gold word_type
                elif tok.has_label("word_type"):
                    wt = tok.get_label("word_type").value
                else:
                    # If nothing is set, assume content (safe default)
                    wt = "content"

                if wt == "content":
                    curr.append(i)
            idxs.append(curr)
        return idxs

    def _forward_parser_content_only(self, emb, lengths, sentences, h_lang):
        B, N, D = emb.shape
        device = emb.device

        # UDapter-style: language-specific parser adapter (or identity)
        emb = self.parser_adapter(emb, h_lang)

        content_idxs = self._content_mask_from_sentences(sentences)
        content_lengths = [len(x) for x in content_idxs]
        max_len = max(content_lengths) if content_lengths else 1
        content_emb = torch.zeros(B, max_len, D, device=device)
        content_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for b, idx_list in enumerate(content_idxs):
            for j, src_i in enumerate(idx_list):
                content_emb[b, j] = emb[b, src_i]
                content_mask[b, j] = True
        arc_scores = self._get_arc_scores(content_emb, content_mask)
        rel_scores = self._get_rel_scores(content_emb, content_mask)
        return arc_scores, rel_scores, content_idxs, content_lengths, content_mask

    # ---- losses ----
    def _compute_parser_loss(self, arc_scores, rel_scores, content_idxs, content_lengths, sentences):
        if not any(content_lengths):
            return torch.tensor(0.0, device=arc_scores.device), 0
        B = len(sentences)
        device = arc_scores.device
        gold_heads = []
        gold_rels = []
        for b, sent in enumerate(sentences):
            if not content_idxs[b]:
                gold_heads.append(torch.zeros(1, dtype=torch.long, device=device))
                gold_rels.append(torch.zeros(1, dtype=torch.long, device=device))
                continue
            idx2pos = {idx: j for j, idx in enumerate(content_idxs[b])}
            heads = []
            rels = []
            for dep_pos, token_idx in enumerate(content_idxs[b]):
                tok = sent[token_idx]
                hid = tok.head_id
                if hid == 0:
                    heads.append(dep_pos)
                    rels.append(self.relation_map.get("root", 0))
                elif hid > 0:
                    head_tok_idx = hid - 1
                    heads.append(idx2pos.get(head_tok_idx, dep_pos))
                    rels.append(self.relation_map.get(tok.get_label("deprel").value, 0))
                else:
                    heads.append(dep_pos)
                    rels.append(0)
            gold_heads.append(torch.tensor(heads, dtype=torch.long, device=device))
            gold_rels.append(torch.tensor(rels, dtype=torch.long, device=device))

        max_len = max(len(h) for h in gold_heads)
        padded_heads = torch.full((B, max_len), -1, dtype=torch.long, device=device)
        padded_rels = torch.full((B, max_len), 0, dtype=torch.long, device=device)
        for b, (h, r) in enumerate(zip(gold_heads, gold_rels)):
            padded_heads[b, :len(h)] = h
            padded_rels[b, :len(r)] = r

        arc_scores_t = arc_scores.transpose(1, 2).contiguous()
        dist = DependencyCRF(arc_scores_t, content_lengths, multiroot=False)
        gold_arcs = torch.zeros_like(arc_scores)
        for b in range(B):
            for d in range(content_lengths[b]):
                head = padded_heads[b, d]
                if head >= 0:
                    gold_arcs[b, d, head] = 1
        arc_loss = -dist.log_prob(gold_arcs.transpose(1, 2)).sum()

        rel_loss = 0.0
        n = 0
        for b in range(B):
            for d in range(content_lengths[b]):
                head = padded_heads[b, d]
                if head >= 0:
                    rel_loss += F.cross_entropy(rel_scores[b, d, head], padded_rels[b, d], reduction="sum")
                    n += 1
        n_arcs = sum(content_lengths)
        total = (arc_loss + rel_loss) / (n_arcs if n_arcs > 0 else 1)
        return total, n_arcs

    def _compute_morph_loss(self, emb, lengths, sentences, h_lang):
        B, L, D = emb.size()
        device = emb.device

        # UDapter-style: tagging adapter (or identity)
        emb = self.tagger_adapter(emb, h_lang)

        x = self.dropout(emb).view(-1, D)
        logits = self.morph_linear(x)
        num_labels = logits.size(-1)
        gold = torch.zeros((B * L, num_labels), device=device)
        valid = []
        for b, sent in enumerate(sentences):
            for i, tok in enumerate(sent):
                if i >= lengths[b]:
                    break
                if tok.get_label("word_type").value == "function":
                    continue
                idx = b * L + i
                valid.append(idx)
                for lbl in tok.get_labels("ms_feat_val"):
                    li = self.morph_dictionary.get_idx_for_item(lbl.value)
                    if li >= 0:
                        gold[idx, li] = 1.0
        if valid:
            lv = logits[valid]
            gv = gold[valid]
            loss = F.binary_cross_entropy_with_logits(lv, gv, reduction="sum") / len(valid)
            return loss, len(valid)
        return torch.tensor(0.0, device=device), 0

    def _compute_wordtype_loss(self, emb, lengths, sentences, h_lang):
        B, L, D = emb.size()
        device = emb.device

        # UDapter-style: tagging adapter (or identity)
        emb = self.tagger_adapter(emb, h_lang)

        if self.wordtype_class_weights is None and self.training:
            cnt = {"content": 0, "function": 0}
            for s in sentences:
                for t in s:
                    v = t.get_label("word_type").value
                    if v in cnt:
                        cnt[v] += 1
            tot = cnt["content"] + cnt["function"]
            if tot > 0 and cnt["content"] > 0 and cnt["function"] > 0:
                w = torch.tensor([tot / (2 * cnt["content"]), tot / (2 * cnt["function"])], device=device)
                self.wordtype_class_weights = w

        if self.training and self.wordtype_word_dropout > 0:
            mask = (torch.rand(B, L, device=device) > self.wordtype_word_dropout).unsqueeze(-1).expand(B, L, D)
            emb = emb * mask.float()

        out, _ = self.wordtype_lstm(emb)
        out = self.wordtype_locked_dropout(out)
        logits = self.wordtype_linear(out.reshape(-1, 512))
        gold = torch.full((B * L,), -100, dtype=torch.long, device=device)
        valid = 0
        for b, s in enumerate(sentences):
            for i, t in enumerate(s):
                if i >= lengths[b]:
                    break
                gold[b * L + i] = 1 if t.get_label("word_type").value == "function" else 0
                valid += 1
        if self.wordtype_class_weights is not None:
            loss = F.cross_entropy(logits, gold, weight=self.wordtype_class_weights, reduction="sum")
        else:
            loss = F.cross_entropy(logits, gold, reduction="sum")
        return (loss / valid if valid else torch.tensor(0.0, device=device)), valid

    # Abstract heads loss (typology via tagging adapter or identity)
    def _compute_abs_loss(self, emb, lengths, sentences, h_lang):
        B, L, D = emb.size()
        device = emb.device

        emb = self.tagger_adapter(emb, h_lang)

        x = self.dropout(emb).view(-1, D)

        # logits
        pres_logits = self.abs_pres_linear(x).squeeze(-1)       # [B*L]
        pos_logits  = self.abs_pos_linear(x)                    # [B*L, 3]
        dep_logits  = self.abs_deprel_linear(x)                 # [B*L, A_dep]
        feat_logits = self.abs_feats_linear(x)                  # [B*L, A_feat]

        # golds
        pres_gold = torch.zeros(B * L, device=device)
        pos_gold  = torch.full((B * L,), 0, device=device, dtype=torch.long)  # 0:'_'
        dep_gold  = torch.full((B * L,), -100, device=device, dtype=torch.long)
        # multi-label feats -> a bag
        feat_gold = torch.zeros_like(feat_logits)

        valid = 0
        for b, s in enumerate(sentences):
            for i, t in enumerate(s):
                if i >= lengths[b]:
                    break
                idx = b * L + i
                # presence
                if t.has_label("abs_pres") and t.get_label("abs_pres").value in ("1", "true", "True"):
                    pres_gold[idx] = 1.0
                    # pos
                    if t.has_label("abs_pos"):
                        v = t.get_label("abs_pos").value
                        pos_gold[idx] = 1 if v == "ABOVE" else (2 if v == "BELOW" else 0)
                    # deprel single
                    if t.has_label("abs_deprel"):
                        # trainer sets inventory and maps string->index via model.abs_deprel_items
                        label = t.get_label("abs_deprel").value
                        li = getattr(self, "abs_deprel_items", {}).get(label, 0)
                        dep_gold[idx] = li
                    # feats multi
                    for fl in t.get_labels("abs_feat_val"):
                        li = getattr(self, "abs_feat_items", {}).get(fl.value, None)
                        if li is not None:
                            feat_gold[idx, li] = 1.0
                else:
                    # no-abstract => keep '_' everywhere
                    dep_gold[idx] = -100
                valid += 1

        # losses
        pres_loss = F.binary_cross_entropy_with_logits(pres_logits, pres_gold, reduction="sum")
        pos_loss  = F.cross_entropy(pos_logits, pos_gold, reduction="sum")
        dep_loss  = F.cross_entropy(dep_logits, dep_gold, ignore_index=-100, reduction="sum")
        feat_loss = F.binary_cross_entropy_with_logits(feat_logits, feat_gold, reduction="sum")

        denom = max(1, valid)
        loss = (pres_loss + pos_loss + dep_loss + feat_loss) / denom
        return loss, valid

    # ---- typology prediction (URIEL reconstruction) ----
    def _compute_typology_loss(self, h_lang, uriel_gold, uriel_mask):
        """
        UDapter-style typology prediction: predict URIEL features from h_lang.

        We apply a mask so that missing features (mask=0) do not contribute to the loss,
        matching the idea of only supervising on observed typology values.
        """
        device = h_lang.device
        if self.typology_linear is None or uriel_gold is None or uriel_mask is None:
            return torch.tensor(0.0, device=device), 0

        logits = self.typology_linear(h_lang)  # [B, uriel_dim]
        # BCE per-feature
        loss_vec = F.binary_cross_entropy_with_logits(logits, uriel_gold, reduction="none")  # [B, U]
        loss_vec = loss_vec * uriel_mask  # zero out missing positions
        denom = uriel_mask.sum()
        if denom.item() == 0:
            return torch.tensor(0.0, device=device), 0
        loss = loss_vec.sum() / denom
        return loss, int(denom.item())

    # ----------------- forward_loss -----------------
    def forward_loss(self, sentences: List[Sentence]):
        emb, lengths = self._embed(sentences)
        device = emb.device

        # UDapter-style: language representation from ID + typology
        h_lang, uriel_gold, uriel_mask, lang_ids = self._get_language_repr(sentences, device)

        # base tasks with contextual adapters (or identity if disabled)
        arc_scores, rel_scores, cidxs, clen, _ = self._forward_parser_content_only(
            emb, lengths, sentences, h_lang
        )
        parser_loss, n_arcs = self._compute_parser_loss(arc_scores, rel_scores, cidxs, clen, sentences)

        morph_loss, n_morph = self._compute_morph_loss(emb, lengths, sentences, h_lang)
        wt_loss, n_wt = self._compute_wordtype_loss(emb, lengths, sentences, h_lang)
        abs_loss, n_abs = self._compute_abs_loss(emb, lengths, sentences, h_lang)

        # typology prediction multi-task loss
        typo_loss, n_typo = self._compute_typology_loss(h_lang, uriel_gold, uriel_mask)

        abs_w_global = self._abs_weight()

        total = torch.tensor(0.0, device=device)
        if n_arcs:
            total = total + self.parser_weight * parser_loss
        if n_morph:
            total = total + self.morph_weight * morph_loss
        if n_wt:
            total = total + self.wordtype_weight * wt_loss
        if n_abs:
            # language-specific abstract gate
            lang_gate = self._abs_lang_gate_weight(lang_ids)  # scalar in [0,1]
            total = total + (abs_w_global * lang_gate) * abs_loss
        if n_typo:
            total = total + self.typology_weight * typo_loss

        # step advance (trainer increments externally too; we guard here)
        self.train_step += 1

        total_tokens = sum(lengths)
        return total, total_tokens

    # ----------------- prediction helpers -----------------
    def _predict_wordtype(self, emb, lengths, sentences, h_lang):
        B, L, D = emb.size()

        emb = self.tagger_adapter(emb, h_lang)

        out, _ = self.wordtype_lstm(emb)
        logits = self.wordtype_linear(out.reshape(-1, 512)).view(B, L, 2)
        probs = torch.softmax(logits, dim=-1)
        for b, s in enumerate(sentences):
            for i, t in enumerate(s):
                if i >= lengths[b]:
                    break
                idx = probs[b, i].argmax().item()
                lab = "function" if idx == 1 else "content"
                t.remove_labels("predicted_word_type")
                t.add_label("predicted_word_type", lab, probs[b, i, idx].item())

    def _predict_parser(self, emb, lengths, sentences, h_lang):
        arc_scores, rel_scores, cidxs, clen, _ = self._forward_parser_content_only(
            emb, lengths, sentences, h_lang
        )
        if not any(clen):
            for s in sentences:
                for t in s:
                    t.set_label("predicted_head", "_")
                    t.set_label("predicted_deprel", "_")
            return
        arc_scores_t = arc_scores.transpose(1, 2).contiguous()
        dist = DependencyCRF(arc_scores_t, clen, multiroot=False)
        pred = dist.argmax
        for b, s in enumerate(sentences):
            for t in s:
                t.set_label("predicted_head", "_")
                t.set_label("predicted_deprel", "_")
            if not cidxs[b]:
                continue
            for dep_pos, tok_idx in enumerate(cidxs[b]):
                head_pos = None
                for h in range(clen[b]):
                    if pred[b, h, dep_pos] > 0.5:
                        head_pos = h
                        break
                if head_pos is None or head_pos == dep_pos:
                    s[tok_idx].set_label("predicted_head", "0")
                    s[tok_idx].set_label("predicted_deprel", "root")
                else:
                    head_tok_idx = cidxs[b][head_pos]
                    s[tok_idx].set_label("predicted_head", str(head_tok_idx + 1))
                    rel_idx = rel_scores[b, dep_pos, head_pos].argmax().item()
                    rel = self.relations[rel_idx]
                    if rel == "root":
                        # fallback to next
                        probs = torch.softmax(rel_scores[b, dep_pos, head_pos], dim=-1)
                        for k in torch.argsort(probs, descending=True)[1:]:
                            rr = self.relations[k]
                            if rr != "root":
                                rel = rr
                                break
                        else:
                            rel = "dep"
                    s[tok_idx].set_label("predicted_deprel", rel)

    def _predict_morph(self, emb, lengths, sentences, h_lang):
        B, L, D = emb.size()

        emb = self.tagger_adapter(emb, h_lang)

        x = self.dropout(emb).view(-1, D)
        logits = self.morph_linear(x).view(B, L, -1)
        probs = torch.sigmoid(logits)
        items = self.morph_dictionary.get_items()

        for b, s in enumerate(sentences):
            for i, t in enumerate(s):
                if i >= lengths[b]:
                    break

                # Decide content/function
                if t.has_label("predicted_word_type"):
                    wt = t.get_label("predicted_word_type").value
                else:
                    wt = "content"  # safest default

                t.remove_labels("predicted_ms_feat_val")

                if wt == "function":
                    # functional → no ms features
                    t.add_label("predicted_ms_feat_val", "_")
                    continue

                # content → predict ms feats
                chosen = [
                    items[j]
                    for j in range(len(items))
                    if probs[b, i, j] > self.morph_threshold
                ]
                if chosen:
                    for v in chosen:
                        t.add_label("predicted_ms_feat_val", v)
                else:
                    t.add_label("predicted_ms_feat_val", "|")

    # Abstract heads prediction (presence, pos, deprel, feats)
    def _predict_abs(self, emb, lengths, sentences, h_lang, pres_thr: float = 0.5):
        B, L, D = emb.size()

        emb = self.tagger_adapter(emb, h_lang)

        x = self.dropout(emb).view(-1, D)
        pres = torch.sigmoid(self.abs_pres_linear(x)).view(B, L)
        pos  = self.abs_pos_linear(x).view(B, L, 3)
        dep  = self.abs_deprel_linear(x).view(B, L, -1)
        feat = torch.sigmoid(self.abs_feats_linear(x)).view(B, L, -1)

        dep_inv = getattr(self, "abs_deprel_items_inv", {})
        feat_inv = getattr(self, "abs_feat_items_inv", {})

        for b, s in enumerate(sentences):
            lang = getattr(s, "lang", None) or "xx"
            thrs = self.abs_lang_thresholds.get(lang, {"pres": pres_thr, "feat": 0.5})
            for i, t in enumerate(s):
                if i >= lengths[b]:
                    break
                p = pres[b, i].item()
                t.remove_labels("predicted_abs_pres")
                t.add_label("predicted_abs_pres", "1" if p >= thrs["pres"] else "0", p)
                if p < thrs["pres"]:
                    continue
                # pos
                pos_idx = pos[b, i].argmax().item()
                t.remove_labels("predicted_abs_pos")
                t.add_label("predicted_abs_pos", ["_", "ABOVE", "BELOW"][pos_idx])

                # deprel single
                d_idx = dep[b, i].argmax().item()
                dep_str = dep_inv.get(d_idx, "_")
                t.remove_labels("predicted_abs_deprel")
                t.add_label("predicted_abs_deprel", dep_str)

                # feats multi
                t.remove_labels("predicted_abs_feat_val")
                for j in range(feat.shape[-1]):
                    if feat[b, i, j].item() > thrs.get("feat", 0.5):
                        fv = feat_inv.get(j, None)
                        if fv:
                            t.add_label("predicted_abs_feat_val", fv)

    # ----------------- public predict -----------------
    def predict(
        self,
        sentences: Union[Sentence, List[Sentence]],
        mini_batch_size: int = 32,
        return_loss: bool = False,
        embedding_storage_mode: str = "none",
        predict_parser: bool = True,
        predict_morph: bool = True,
        predict_wordtype: bool = True,
        predict_abs: bool = True,
        **kwargs,
    ):
        if return_loss:
            return self.forward_loss(sentences)

        single = isinstance(sentences, Sentence)
        if single:
            sentences = [sentences]
        self.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), mini_batch_size):
                batch = sentences[i : i + mini_batch_size]
                emb, lengths = self._embed(batch)
                device = emb.device
                h_lang, _, _, _ = self._get_language_repr(batch, device)

                if predict_wordtype:
                    self._predict_wordtype(emb, lengths, batch, h_lang)
                if predict_parser:
                    self._predict_parser(emb, lengths, batch, h_lang)
                if predict_morph:
                    self._predict_morph(emb, lengths, batch, h_lang)
                if predict_abs:
                    self._predict_abs(emb, lengths, batch, h_lang)
        if embedding_storage_mode == "none":
            for s in sentences:
                s.clear_embeddings()
        return sentences[0] if single else sentences

    def evaluate(
            self,
            data_points,
            gold_label_type: str = "deprel",
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            main_evaluation_metric: Union[Tuple[str, str], List[Tuple[str, str]]] = ("las", "las"),
            exclude_labels: List[str] = [],
            gold_label_dictionary: Optional["Dictionary"] = None,
            return_loss: bool = True,
            **kwargs,
    ) -> Result:
        """
        Evaluate LAS · UAS · word-type accuracy (+ optional abstract stats) in a
        Flair-compatible way, matching the reference implementation style.
        """
        import torch
        from flair.training_utils import Result

        # ---- normalize input to list[Sentence] ----
        if hasattr(data_points, "sentences"):
            sentences = data_points.sentences
        elif hasattr(data_points, "__len__") and hasattr(data_points, "__getitem__"):
            sentences = [data_points[i] for i in range(len(data_points))]
        else:
            sentences = data_points

        # simple manual batching
        data_loader = [
            sentences[i: i + mini_batch_size]
            for i in range(0, len(sentences), mini_batch_size)
        ]

        self.eval()

        eval_sentences: List[Sentence] = []
        with torch.no_grad():
            for batch in data_loader:
                if not isinstance(batch, list):
                    batch = [batch]

                # We predict parser + word-type here; morph/abs optional
                self.predict(
                    batch,
                    mini_batch_size=len(batch),
                    embedding_storage_mode=embedding_storage_mode,
                    predict_parser=True,
                    predict_morph=False,
                    predict_wordtype=True,
                    predict_abs=False,  # abstracts not needed for LAS/UAS
                )
                eval_sentences.extend(batch)

        # ───────────── PARSING METRICS (LAS/UAS on content words) ─────────────
        total_content = 0
        correct_uas = 0
        correct_las = 0

        for sent in eval_sentences:
            for tok in sent:
                # only content tokens are evaluated
                if tok.get_label("word_type").value != "content":
                    continue

                total_content += 1
                gold_head = tok.head_id
                gold_rel = tok.get_label("deprel").value

                ph = tok.get_label("predicted_head").value
                # robust int conversion
                if isinstance(ph, int):
                    pred_head = ph
                else:
                    ph_str = str(ph)
                    pred_head = int(ph_str) if ph_str.isdigit() else -1

                pred_rel = tok.get_label("predicted_deprel").value

                if gold_head == pred_head:
                    correct_uas += 1
                    if gold_rel == pred_rel:
                        correct_las += 1

        uas = correct_uas / total_content if total_content > 0 else 0.0
        las = correct_las / total_content if total_content > 0 else 0.0

        # ───────────── WORD-TYPE ACCURACY ─────────────
        wt_correct = 0
        wt_total = 0
        for sent in eval_sentences:
            for tok in sent:
                gold_lbl = tok.get_label("word_type").value
                pred_lbl = tok.get_label("predicted_word_type").value
                wt_total += 1
                if gold_lbl == pred_lbl:
                    wt_correct += 1
        wt_acc = wt_correct / wt_total if wt_total > 0 else 0.0

        # ───────────── OPTIONAL: ABSTRACT-HEAD DIAGNOSTICS (LIGHTWEIGHT) ─────────────
        abs_pres_gold = abs_pres_pred = abs_pres_tp = 0

        for sent in eval_sentences:
            for tok in sent:
                if tok.has_label("abs_pres"):
                    g = tok.get_label("abs_pres").value in ("1", "true", "True")
                    p_lab = tok.get_label("predicted_abs_pres").value if tok.has_label("predicted_abs_pres") else "0"
                    p = p_lab in ("1", "true", "True")
                    if g:
                        abs_pres_gold += 1
                    if p:
                        abs_pres_pred += 1
                    if g and p:
                        abs_pres_tp += 1

        # simple F1 for presence
        if abs_pres_gold + abs_pres_pred > 0:
            prec = abs_pres_tp / abs_pres_pred if abs_pres_pred > 0 else 0.0
            rec = abs_pres_tp / abs_pres_gold if abs_pres_gold > 0 else 0.0
            if prec + rec > 0:
                abs_pres_f1 = 2 * prec * rec / (prec + rec)
            else:
                abs_pres_f1 = 0.0
        else:
            abs_pres_f1 = 0.0

        # ───────────── GLOBAL LOSS (for AnnealOnPlateau aux='loss') ─────────────
        total_loss = 0.0
        total_batches = 0
        if return_loss:
            with torch.no_grad():
                for batch in data_loader:
                    if not isinstance(batch, list):
                        batch = [batch]
                    loss, cnt = self.forward_loss(batch)
                    total_loss += loss.item()
                    total_batches += 1
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

        # ───────────── BUILD Result OBJECT (NO log_line/log_header KWARGS!) ─────────────
        detailed = (
            f"LAS: {las:.4f}\n"
            f"UAS: {uas:.4f}\n"
            f"Word-type ACC: {wt_acc:.4f}\n"
            f"Abstract presence F1: {abs_pres_f1:.4f}\n"
            f"Content words evaluated: {total_content}\n"
        )

        scores = {
            "las": las,
            "uas": uas,
            "wordtype_acc": wt_acc,
            "loss": avg_loss,  # <- for aux_metric='loss'
            "abs_pres_f1": abs_pres_f1,
        }

        result = Result(
            main_score=las,
            detailed_results=detailed,
            scores=scores,
        )

        return result

    # ----------------- save/load -----------------
    def _get_state_dict(self):
        return {
            "state_dict": self.state_dict(),
            "deprel_dictionary": list(self.relations),
            "morph_dictionary": (self.morph_dictionary.get_items()
                                 if hasattr(self.morph_dictionary, "get_items")
                                 else list(self.morph_dictionary)),
            "num_langs": self.num_langs,
            "uriel_dim": self.uriel_dim,
            "abs_weight_min": self.abs_weight_min,
            "abs_weight_max": self.abs_weight_max,
            "abs_ramp_power": self.abs_ramp_power,
            "lang2idx": self.lang2idx,
            "abs_deprel_items": getattr(self, "abs_deprel_items", {}),
            "abs_feat_items": getattr(self, "abs_feat_items", {}),
            "abs_lang_priors": self.abs_lang_priors,
            "abs_lang_thresholds": self.abs_lang_thresholds,
            "use_contextual_adapters": self.use_contextual_adapters,
        }

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        dep = Dictionary(add_unk=False)
        for it in state.get("deprel_dictionary", []):
            dep.add_item(it)
        morph = Dictionary(add_unk=True)
        for it in state.get("morph_dictionary", []):
            morph.add_item(it)

        model = cls(
            deprel_dictionary=dep,
            morph_dictionary=morph,
            num_abs_deprel=max(1, len(state.get("abs_deprel_items", {}))),
            num_abs_feats=max(1, len(state.get("abs_feat_items", {}))),
            num_langs=state.get("num_langs", 1),
            uriel_dim=state.get("uriel_dim", 0),
            abs_weight_min=state.get("abs_weight_min", 0.05),
            abs_weight_max=state.get("abs_weight_max", 1.0),
            abs_ramp_power=state.get("abs_ramp_power", 1.5),
            use_contextual_adapters=state.get("use_contextual_adapters", True),
        )
        missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
        if missing or unexpected:
            print("[Joint_Model_.load] missing keys:", missing)
            print("[Joint_Model_.load] unexpected keys:", unexpected)
        model.lang2idx = state.get("lang2idx", {})
        model.abs_deprel_items = state.get("abs_deprel_items", {})
        model.abs_feat_items = state.get("abs_feat_items", {})
        model.abs_deprel_items_inv = {i: l for l, i in model.abs_deprel_items.items()}
        model.abs_feat_items_inv = {i: l for l, i in model.abs_feat_items.items()}
        model.abs_lang_priors = state.get("abs_lang_priors", {})
        model.abs_lang_thresholds = state.get("abs_lang_thresholds", {})
        return model
