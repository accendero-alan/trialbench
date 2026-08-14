"""Tier B: deep tabular methods.

Each runs on the dense **tabular** view. Heavy deps (torch, tabpfn,
pytorch-tabnet) are imported lazily inside ``fit`` per CLAUDE.md's golden
rule #5, so a missing install is recorded as "skipped", never a crash.
Networks are kept small and epochs capped so CPU training stays fast.
"""
from __future__ import annotations

import numpy as np

from .base import BaseMethod
from .registry import register


def _stratified_subsample_idx(y, max_n, rng):
    """Indices for a class-stratified subsample of size <= max_n."""
    y = np.asarray(y)
    n = len(y)
    if n <= max_n:
        return np.arange(n)
    classes, counts = np.unique(y, return_counts=True)
    frac = max_n / n
    idx = []
    for c, cnt in zip(classes, counts):
        c_idx = np.flatnonzero(y == c)
        take = min(len(c_idx), max(1, int(round(cnt * frac))))
        idx.append(rng.choice(c_idx, size=take, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx[:max_n]


@register("tabpfn")
class TabPFN(BaseMethod):
    """Tabular foundation model (near-instant "training" via in-context
    learning). Hard limits from the pretrained model itself: <=10_000 train
    rows, <=500 features, <=10 classes -- rows are stratified-subsampled and
    features reduced via SelectKBest (fit on train only) when exceeded; a
    task with more classes than the cap is not run at all.

    Requires a one-time, free license acceptance: visit
    https://ux.priorlabs.ai, accept the license, copy your API key, then
    ``export TABPFN_TOKEN="<key>"`` before running (see deploy/README.md).
    Without it, this method is recorded as "skipped" (a setup gap, not a
    bug) -- every other method still runs normally.
    """
    feature_view = "tabular"

    MAX_ROWS = 10_000
    MAX_FEATURES = 500
    MAX_CLASSES = 10

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if self.num_classes > self.MAX_CLASSES:
            raise NotImplementedError(
                f"tabpfn supports at most {self.MAX_CLASSES} classes; "
                f"this task has {self.num_classes}"
            )
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.errors import TabPFNLicenseError
        except ImportError as e:
            raise ImportError(f"tabpfn not installed ({e}); pip install tabpfn") from e

        # Check for a token *before* calling fit(): without one, TabPFN falls
        # back to an interactive browser-login flow, which isn't just
        # unwanted in an automated run -- on Windows (and in headless SSH
        # sessions generally) it can raise a raw, unhelpful OSError instead
        # of a clean TabPFNLicenseError. Failing fast here keeps this a
        # predictable "skipped" every time, on every platform.
        try:
            from tabpfn.browser_auth import get_cached_token
            has_token = get_cached_token() is not None
        except Exception:
            import os as _os
            has_token = bool(_os.environ.get("TABPFN_TOKEN"))
        if not has_token:
            raise ImportError(
                "tabpfn needs a one-time license acceptance -- see "
                "https://ux.priorlabs.ai, accept the license, then "
                'export TABPFN_TOKEN="<api key>" (see deploy/README.md)'
            )

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)

        self._selector = None
        if X_train.shape[1] > self.MAX_FEATURES:
            from sklearn.feature_selection import SelectKBest, f_classif
            self._selector = SelectKBest(f_classif, k=self.MAX_FEATURES).fit(X_train, y_train)
            X_train = self._selector.transform(X_train)

        rng = np.random.default_rng(self.seed)
        idx = _stratified_subsample_idx(y_train, self.MAX_ROWS, rng)
        X_train, y_train = X_train[idx], y_train[idx]

        self.model_ = TabPFNClassifier(device="cpu", random_state=self.seed, n_jobs=1)
        try:
            self.model_.fit(X_train, y_train)
        except TabPFNLicenseError as e:
            raise ImportError(
                "tabpfn model weights need a one-time license acceptance -- "
                f"see https://ux.priorlabs.ai and set TABPFN_TOKEN ({e})"
            ) from e
        self._classes = self.model_.classes_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self._selector is not None:
            X = self._selector.transform(X)
        proba = self.model_.predict_proba(X)
        if self.task_type == "binary":
            return self._binary_scores(proba)
        full = np.zeros((len(X), self.num_classes), dtype=float)
        for j, c in enumerate(self._classes):
            full[:, int(c)] = proba[:, j]
        return full


@register("tabnet")
class TabNet(BaseMethod):
    """pytorch-tabnet TabNetClassifier. Small network, capped epochs with
    early stopping on a held-out validation set, sized for CPU.

    T3 in newsletter-part2-test-plan.md: as originally shipped, this scored
    within noise of the majority prior (mean 0.5115 vs 0.4319) on several
    cells. `experiments/tabnet_fix_compare.py` found three fixes that moved
    real-data PR-AUC by +0.054 across 3 seeds (see rerun.md) and this class
    now ships them:
      1. class weighting (`weights=1`, pytorch-tabnet's inverse-freq option)
      2. StandardScaler fit on train only, applied to valid/test -- TabNet is
         one of the scale-sensitive methods that, unlike logreg/knn/svm here,
         never got a scaler
      3. loss-based early stopping (`eval_metric=["logloss"]`) instead of the
         accuracy-ish pytorch-tabnet default, which stops too early/late on
         these imbalanced cells
    """
    feature_view = "tabular"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        from pytorch_tabnet.tab_model import TabNetClassifier
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler().fit(np.asarray(X_train, dtype=float))
        X_train = self.scaler_.transform(np.asarray(X_train, dtype=float)).astype(np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        n = len(X_train)
        batch_size = max(8, min(256, n))
        virtual_batch_size = max(4, min(64, batch_size))

        eval_set, eval_name = None, None
        if X_valid is not None and len(X_valid) > 0:
            Xva = self.scaler_.transform(np.asarray(X_valid, dtype=float)).astype(np.float32)
            eval_set = [(Xva, np.asarray(y_valid, dtype=np.int64))]
            eval_name = ["valid"]

        self.model_ = TabNetClassifier(
            n_d=16, n_a=16, n_steps=3, gamma=1.3,
            seed=self.seed, verbose=0, device_name="cpu",
        )
        self.model_.fit(
            X_train, y_train, eval_set=eval_set, eval_name=eval_name,
            eval_metric=["logloss"], weights=1,
            max_epochs=100, patience=15,
            batch_size=batch_size, virtual_batch_size=virtual_batch_size,
            drop_last=False,
        )
        self._classes = self.model_.classes_
        return self

    def predict_proba(self, X):
        X = self.scaler_.transform(np.asarray(X, dtype=float)).astype(np.float32)
        proba = self.model_.predict_proba(X)
        if self.task_type == "binary":
            return self._binary_scores(proba)
        full = np.zeros((len(X), self.num_classes), dtype=float)
        for j, c in enumerate(self._classes):
            full[:, int(c)] = proba[:, j]
        return full


@register("ft_transformer")
class FTTransformer(BaseMethod):
    """A compact FT-Transformer, hand-rolled in plain PyTorch rather than via
    the `rtdl` package (old, unmaintained) or `pytorch_tabular` (pulls in a
    pandas<3.0 pin that conflicts with this repo's pandas). Each of the
    already-encoded tabular columns is linearly tokenized to a d_token
    embedding, a learned CLS token is prepended, a small Transformer encoder
    mixes tokens via self-attention, and a linear head on the CLS output
    produces class logits. Kept small (2 layers, 4 heads) with capped epochs
    and early stopping to stay CPU-fast on these dataset sizes.
    """
    feature_view = "tabular"

    D_TOKEN = 32
    N_LAYERS = 2
    N_HEADS = 4
    MAX_EPOCHS = 100
    PATIENCE = 10

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        n_features = X_train.shape[1]
        n_classes = max(2, self.num_classes)
        d_token, n_layers, n_heads = self.D_TOKEN, self.N_LAYERS, self.N_HEADS

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
                self.bias = nn.Parameter(torch.zeros(n_features, d_token))
                self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                    dropout=0.1, batch_first=True, activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
                self.norm = nn.LayerNorm(d_token)
                self.head = nn.Linear(d_token, n_classes)

            def forward(self, x):
                tok = x.unsqueeze(-1) * self.weight + self.bias   # (B, F, D)
                cls = self.cls.expand(x.shape[0], -1, -1)          # (B, 1, D)
                seq = torch.cat([cls, tok], dim=1)                 # (B, F+1, D)
                out = self.encoder(seq)
                return self.head(self.norm(out[:, 0]))

        self.model_ = _Net()
        opt = torch.optim.AdamW(self.model_.parameters(), lr=1e-3, weight_decay=1e-4)

        y_train = np.asarray(y_train, dtype=np.int64)
        classes, counts = np.unique(y_train, return_counts=True)
        w = np.ones(n_classes, dtype=np.float32)
        for c, cnt in zip(classes, counts):
            w[int(c)] = len(y_train) / (len(classes) * cnt)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w))

        Xt = torch.tensor(np.asarray(X_train, dtype=np.float32))
        yt = torch.tensor(y_train)
        batch_size = max(8, min(128, len(Xt)))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        have_valid = X_valid is not None and len(X_valid) > 0
        if have_valid:
            Xv = torch.tensor(np.asarray(X_valid, dtype=np.float32))
            yv = torch.tensor(np.asarray(y_valid, dtype=np.int64))

        best_state, best_loss, bad_epochs = None, float("inf"), 0
        for _epoch in range(self.MAX_EPOCHS):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()

            if have_valid:
                self.model_.eval()
                with torch.no_grad():
                    vloss = loss_fn(self.model_(Xv), yv).item()
                if vloss < best_loss - 1e-4:
                    best_loss, bad_epochs = vloss, 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.PATIENCE:
                        break
        if have_valid and best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(np.asarray(X, dtype=np.float32)))
            proba = torch.softmax(logits, dim=1).numpy()
        if self.task_type == "binary":
            return self._binary_scores(proba)
        return proba
