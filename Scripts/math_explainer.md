# Math & Code Explainer: `binomial_rep_analysis.py`

This document walks through every mathematical concept in the pipeline, connecting
equations to the code that implements them.

---

## 1. What We're Measuring

For each binomial expression (e.g. *bread and butter*) we have two orderings:

- **AB** = "bread and butter" (alphabetically canonical)
- **BA** = "butter and bread" (reversed)

We embed each ordering in ~500 distinct sentences using an OPT model, extract
the hidden-state vectors at each layer for the phrase tokens, and pool them
into one vector per sentence.  For each ordering we then have a matrix of
shape **(n × D)** where n ≈ 500 (sentences) and D is the hidden dimension
(768 / 1024 / 2048 for 125 M / 350 M / 1.3 B).

Call these matrices:
```
A  (n × D) : pooled phrase reps for ordering AB
B  (n × D) : pooled phrase reps for ordering BA
```

We compute two scores per (binomial, layer, checkpoint):

| Score | Intuition |
|---|---|
| `self_sim_AB` | How **consistent** is the AB representation across sentence contexts? |
| `procrustes_dist` | How well can you **rotate** the AB cloud onto the BA cloud? |

---

## 2. Representation Extraction

**File:** `extract_representations` (line ~354)

### 2.1 Phrase-span masking

For a sentence like *"I always buy bread and butter at the store"*, we want the
average of the hidden-state vectors at the tokens covering "bread and butter",
not the whole sentence.

The tokenizer provides an **offset mapping**: a list of (char_start, char_end)
pairs, one per token.  `find_phrase_span_in_tokens` does an O(n) scan to find
the first and last token whose character offsets fall inside the phrase span.

```python
span_mask[b_idx, span[0]:span[1] + 1] = True   # inclusive token span
```

### 2.2 Vectorised masked mean

For each layer's hidden-state tensor `layer_h` of shape (B, T, D):

```python
span_f  = span_mask.to(dtype).unsqueeze(-1)           # (B, T, 1)
counts  = span_f.squeeze(-1).sum(dim=1, keepdim=True) # (B, 1) = #tokens in phrase
pooled  = (layer_h * span_f).sum(dim=1) / counts      # (B, D)
```

Mathematically, for sentence *i* and layer *l*:

$$v_i^{(l)} = \frac{1}{|S_i|} \sum_{t \in S_i} h_t^{(l)}$$

where $S_i$ is the set of token indices in the phrase span and $h_t^{(l)}$ is
the hidden state at token $t$.

**Why vectorised?**  The old code looped over sentences.  The vectorised form
runs the masked mean for all B sentences in one kernel, and moves all valid
rows to CPU in a single `.cpu()` call per layer instead of per sentence.

---

## 3. Self-Similarity

**File:** `_batch_self_similarity` (line ~455)

### 3.1 What it measures

Self-similarity asks: across n different sentences that contain the same phrase,
does the model assign *consistent* representations?

- **High (less negative) self-similarity** → the representation barely varies
  with sentence context.
- **Low (more negative) self-similarity** → context changes the representation a
  lot.

### 3.2 The naive approach — O(n²D)

Build the n×n **Gram matrix** K = X Xᵀ, where X is the (n, D) representation
matrix.  **Double-centre** it:

$$K_c = K - \text{row\_mean} - \text{col\_mean} + \text{grand\_mean}$$

More concretely, defining the centring matrix H = I − (1/n) **1 1**ᵀ:

$$K_c = H K H$$

Self-similarity is the **mean of the off-diagonal elements** of K_c.

### 3.3 The shortcut — O(nD), no Gram matrix

Two facts about K_c let us bypass the n×n matrix entirely.

**Fact 1 — sum of ALL elements of K_c = 0.**

Double-centering subtracts every row mean and every column mean, so the total
always cancels:

$$\sum_{i,j} (K_c)_{ij} = 0$$

**Fact 2 — the diagonal of K_c equals squared distance to the mean.**

Let $\mu = \frac{1}{n}\sum_i x_i$.  Then:

$$K_c[i,i] = K[i,i] - 2\cdot(\text{row mean of row }i) + \text{grand mean}$$

Substituting $K[i,i]=\|x_i\|^2$, row mean $= x_i^\top\mu$, grand mean $= \|\mu\|^2$:

$$K_c[i,i] = \|x_i\|^2 - 2x_i^\top\mu + \|\mu\|^2 = \|x_i - \mu\|^2$$

**Combining Facts 1 and 2:**

$$\text{sum(off-diagonal)} = \underbrace{\text{sum(all)}}_{\displaystyle 0} - \text{sum(diagonal)} = -\sum_i \|x_i - \mu\|^2 = -\|X_c\|_F^2$$

There are $n(n-1)$ off-diagonal entries, so:

$$\boxed{\text{mean(off-diagonal }K_c) = \frac{-\|X_c\|_F^2}{n(n-1)}}$$

where $X_c = X - \mu$ is the mean-centred matrix.  No Gram matrix needed.

### 3.4 Code

```python
def _batch_self_similarity(X: torch.Tensor) -> torch.Tensor:
    # X : (N, n, D)  — N binomials, n sentences, D-dim representations
    _, n, _ = X.shape
    mu    = X.mean(dim=1, keepdim=True)          # (N, 1, D)  mean over sentences
    X_c   = X - mu                               # (N, n, D)  centred
    return -X_c.pow(2).sum(dim=(1, 2)) / (n * (n - 1))  # (N,)
```

`X_c.pow(2).sum(dim=(1, 2))` computes $\|X_c\|_F^2$ for each of the N binomials
simultaneously.

---

## 4. Orthogonal Procrustes Distance

**File:** `compute_scores_batched`, Procrustes section (line ~625)

### 4.1 What it measures

Orthogonal Procrustes asks: **can you rotate the AB cloud to match the BA cloud?**

Formally, find the orthogonal matrix R* (rotation / reflection) that minimises:

$$R^* = \arg\min_{R^\top R = I} \|AR - B\|_F$$

A small residual means the two orderings live in geometrically aligned
subspaces — the model has learned a representation that is "rotation-equivalent"
across orderings.

### 4.2 The optimal rotation

Take the **SVD of AᵀB**:

$$A^\top B = U S V^\top \quad (U, V \text{ orthogonal},\ S = \text{diag}(\sigma_i))$$

The optimal rotation is $R^* = UV^\top$.

**Why?** We want to maximise $\text{tr}(B^\top A R)$ (which minimises the
residual).  Setting $M = B^\top A = V S U^\top$ and $Q = U^\top R V$ (orthogonal):

$$\text{tr}(MR) = \text{tr}(VSU^\top R) = \text{tr}(S \cdot U^\top R V) = \text{tr}(SQ)$$

This is maximised when Q = I, i.e. $R = UV^\top$, giving maximum value $\sum_i \sigma_i(A^\top B)$.

### 4.3 Residual without forming R*

Expand $\|AR^* - B\|_F^2$:

$$\|AR^* - B\|_F^2 = \text{tr}(R^{*\top} A^\top A R^*) - 2\,\text{tr}(R^{*\top} A^\top B) + \text{tr}(B^\top B)$$

Since $R^*$ is orthogonal, $\text{tr}(R^{*\top} A^\top A R^*) = \text{tr}(A^\top A) = \|A\|_F^2$.

At $R^* = UV^\top$ and $A^\top B = USV^\top$:

$$\text{tr}(R^{*\top} A^\top B) = \text{tr}(VU^\top \cdot USV^\top) = \text{tr}(VSV^\top) = \text{tr}(S) = \sum_i \sigma_i(A^\top B)$$

Therefore:

$$\boxed{\|AR^* - B\|_F^2 = \|A\|_F^2 + \|B\|_F^2 - 2\sum_i \sigma_i(A^\top B)}$$

**We only need the singular values of AᵀB** — never construct R* or compute AR*−B.

### 4.4 Thin-factorisation shortcut (avoids D×D matrices)

**The problem:** A and B are (n, D) with n=500, D=2048 for the 1.3 B model.
Computing AᵀB directly gives a (D, D) = (2048, 2048) matrix per binomial.
With N=594 binomials batched, that's a (594, 2048, 2048) tensor ≈ 10 GB.
SVD of each 2048×2048 matrix costs O(D³) ≈ 8.6 × 10⁹ flops.

**The trick:** Use A's **thin SVD**:

$$A = U_A \,\text{diag}(S_A)\, V_A^\top$$

where $U_A$ is $n \times n$, $S_A \in \mathbb{R}^n$, $V_A$ is $D \times n$, and
$V_A^\top V_A = I_n$ (V_A has **orthonormal columns**).

Then:

$$A^\top B = V_A \,\text{diag}(S_A)\, U_A^\top B = V_A C, \quad C = \text{diag}(S_A)(U_A^\top B)$$

C has shape $n \times D$ — the same as A and B.

**Key lemma: left-multiplying by $V_A$ does not change singular values.**

Proof via the AB ∼ BA non-zero eigenvalue property:

$$\sigma_i^2(V_A C) = \text{eig}_i\!\left((V_A C)(V_A C)^\top\right) = \text{eig}_i\!\left(V_A C C^\top V_A^\top\right)$$

The non-zero eigenvalues of $V_A(CC^\top V_A^\top)$ equal those of
$(CC^\top V_A^\top)V_A = CC^\top (V_A^\top V_A) = CC^\top$
(using $V_A^\top V_A = I_n$).

$$\Rightarrow\quad \sigma_i^2(V_A C) = \sigma_i^2(C) \quad\Rightarrow\quad \sigma_i(A^\top B) = \sigma_i(C)$$

So we can replace SVD of the (D, D) matrix AᵀB with SVD of the (n, D) matrix C.

**Cost comparison for 1.3 B (D=2048, n=500):**

| Step | Old | New |
|---|---|---|
| Form AᵀB | O(nD²) per binomial | avoided |
| SVD | O(D³) = O(D³) | O(n²D) via svdvals(C) |
| Savings | — | ~D/n ≈ **17×** |

### 4.5 Obtaining U_A and S_A without a full SVD

We need U_A (left singular vectors) and S_A (singular values) of A.
Rather than running a full SVD, we compute the **eigendecomposition of AAᵀ**:

$$AA^\top = U_A \,\text{diag}(S_A^2)\, U_A^\top$$

Since AAᵀ is symmetric positive semi-definite, `torch.linalg.eigh` gives exact
eigenvalues in ascending order.  Then $S_A = \sqrt{\text{eigenvalues}}$.

Bonus: $\|A\|_F^2 = \text{tr}(AA^\top) = \sum_i \lambda_i(AA^\top)$, so we can
read off $\|A\|_F^2$ directly from the eigenvalues L_A.

### 4.6 Code

```python
# Step 1 — eigh gives U_A (left sing. vecs) and L_A = S_A² (eigenvalues of AAᵀ)
AAT       = torch.bmm(A, A.transpose(1, 2))    # (N, n, n)
L_A, U_A  = torch.linalg.eigh(AAT)             # ascending eigenvalues, eigenvectors

# Reuse eigenvalues for ||A||_F² and S_A
norm_A_sq = L_A.clamp(min=0).sum(dim=1)        # ||A||_F² = Σ eigenvalues
norm_B_sq = B.pow(2).sum(dim=(1, 2))           # ||B||_F²
S_A       = L_A.clamp(min=0).sqrt()            # (N, n) singular values of A

# Step 2 — form C = diag(S_A) U_Aᵀ B  (N, n, D)
C = S_A.unsqueeze(-1) * torch.bmm(U_A.transpose(1, 2), B)

# Step 3 — sum of singular values of AᵀB = sum of singular values of C
S        = torch.linalg.svdvals(C)             # (N, n)
resid_sq = (norm_A_sq + norm_B_sq - 2.0 * S.sum(dim=1)).clamp(min=0.0)

# Step 4 — normalise residual by ||B||_F
proc = resid_sq.sqrt() / norm_B_sq.sqrt().clamp(min=1e-10)
```

The `.clamp(min=0)` calls guard against tiny negative eigenvalues that can
arise from floating-point arithmetic on near-zero eigenvalues of a PSD matrix.
The final `.clamp(min=0.0)` on `resid_sq` similarly guards against numerical
cancellation producing a tiny negative value before the square root.

### 4.7 Normalisation choice

The reported `procrustes_dist` is:

$$\text{procrustes\_dist} = \frac{\|AR^* - B\|_F}{\|B\|_F}$$

This normalises by the scale of B so scores are comparable across layers and
model sizes where hidden norms may differ.

---

## 5. Batching Strategy

**File:** `compute_scores_batched` (line ~501)

### 5.1 Why batching matters

The naive approach loops over each of N=594 binomials and issues one small
GPU call per binomial.  Small GPU kernels have Python overhead and kernel-launch
latency that typically exceeds the actual compute time.  With 594 binomials ×
25 layers = ~15,000 calls, the GPU spends most of its time idle.

### 5.2 What we do instead

For each layer, all N binomials are stacked into 3-D tensors:

```python
A = torch.from_numpy(np.stack([a[:n] for a in A_arrs])).to(device).float()  # (N, n, D)
B = torch.from_numpy(np.stack([b[:n] for b in B_arrs])).to(device).float()  # (N, n, D)
```

Every operation (`bmm`, `eigh`, `svdvals`) is batched over the N dimension,
issuing ~6 GPU operations per layer instead of ~594 × 4 = 2,376.

### 5.3 n truncation

All binomials are truncated to the same n = minimum available count across
the batch.  This is required for rectangular stacking.  In practice n ≈ 500
for all binomials; the soft warning at MIN_SENTENCES_SOFT_WARN = 500 ensures
outliers are flagged before they shrink the whole batch.

---

## 6. Resume Logic

**File:** `load_completed` (line ~692)

The output CSV is append-only.  On resume, we count how many layer rows exist
for each (model, checkpoint, phrase_AB) key.  The expected count is inferred
as the **most common count** across all keys (i.e. the mode of the
layer-count distribution).  Any key with fewer rows than that mode is treated
as incomplete and re-run.

This handles partial writes from a crash mid-layer without needing to know the
model's layer count ahead of time.

---

## 7. Score Interpretation Summary

| Score | Formula | High value means | Low value means |
|---|---|---|---|
| `self_sim_AB` | $-\|X_c^{AB}\|_F^2 / (n(n-1))$ | Consistent AB reps | Context-dependent AB reps |
| `self_sim_BA` | $-\|X_c^{BA}\|_F^2 / (n(n-1))$ | Consistent BA reps | Context-dependent BA reps |
| `self_sim_ratio` | `self_sim_AB / self_sim_BA` | AB more consistent than BA | BA more consistent than AB |
| `procrustes_dist` | $\|AR^*-B\|_F / \|B\|_F$ | Poor geometric alignment | Good geometric alignment |

Note: self-similarity scores are ≤ 0 (they are negative squared norms divided
by a positive constant).  "Higher" means closer to 0, i.e. less spread.
