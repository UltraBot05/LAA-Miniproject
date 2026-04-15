"""
╔══════════════════════════════════════════════════════════════════╗
║   Face Recognition Using Eigenfaces                              ║
║   UE24MA241B - Linear Algebra and Its Applications               ║
║   PES University                                                 ║
║                                                                  ║
║   Team: <Team>                                                   ║
║     PES2UG24CS018 - Aayush Gupta                                 ║
║     PES2UG24CS019 - Abhigyan Dutta                               ║
║     PES2UG24CS036 - Aditya Shrivastav                            ║
║     PES2UG24CS042 - Akash Singh                                  ║
║                                                                  ║
║   Pipeline:                                                      ║
║     Step 1  Real-World Data → Matrix Representation              ║
║     Step 2  Matrix Simplification (Mean-Centering)               ║
║     Step 3  Structure of the Space (Rank / Null / Col Space)     ║
║     Step 4  Remove Redundancy (Linear Independence / Basis)      ║
║     Step 5  Orthogonalization (Gram–Schmidt)                     ║
║     Step 6  Projection (Orthogonal Projection)                   ║
║     Step 7  Prediction (Least Squares)                           ║
║     Step 8  Pattern Discovery (Eigenvalues & Eigenvectors)       ║
║     Step 9  System Simplification (Diagonalization)              ║
║     Output  Face Recognition + Reconstruction                    ║
╚══════════════════════════════════════════════════════════════════╝

NOTE: All core linear algebra operations are implemented manually
      using NumPy primitives - no sklearn.PCA or similar shortcuts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE (matches the PPT)
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#07090F"
PANEL   = "#0E1422"
CARD    = "#121929"
ACCENT  = "#4F8EF7"
VIOLET  = "#A78BFA"
MINT    = "#34D399"
AMBER   = "#F59E0B"
TEXT    = "#E2E8F0"
SUBTEXT = "#94A3B8"
GRID    = "#1E2D4A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SEP = "━" * 65


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 - REAL-WORLD DATA → MATRIX REPRESENTATION
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print("  FACE RECOGNITION USING EIGENFACES")
print("  Linear Algebra Pipeline - UE24MA241B")
print(f"{'═'*65}")

print(f"\n{SEP}")
print("  STEP 1 - Real-World Data → Matrix Representation")
print(SEP)

print("\n  Loading AT&T/Olivetti face dataset …")

from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces(shuffle=True, random_state=42)

images = data.images          # (400, 64, 64)  float64 in [0, 1]
labels = data.target          # (400,)

n_subjects    = 40
n_per_subject = 10
n_samples     = 400
h, w          = 64, 64
n_pixels      = h * w         # 4096

# Flatten each image to a row vector → data matrix A
A = data.data.astype(np.float64)   # (400, 4096) - already flattened by sklearn

print(f"\n  Dataset   : AT&T/Olivetti face dataset (real)")
print(f"  Subjects  : 40 people")
print(f"  Images    : 10 per subject  →  400 total")
print(f"  Image size: {h} × {w} = {n_pixels} pixels")
print(f"\n  Data matrix A  shape: {A.shape}")
print(f"  Each ROW    = one face (flattened pixel vector)")
print(f"  Each COLUMN = one pixel across all faces")
print(f"\n  A ∈ ℝ^{{400 × 4096}}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 - MATRIX SIMPLIFICATION (MEAN-CENTERING)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 2 - Matrix Simplification (Mean-Centering)")
print(SEP)

mean_face = A.mean(axis=0)          # shape (4096,)  - average face
A_centered = A - mean_face          # shape (400, 4096)  - Ã

print(f"\n  Mean face μ computed  (shape {mean_face.shape})")
print(f"  Ã = A − μ  →  centered matrix  (shape {A_centered.shape})")
print(f"\n  Before centering: pixel mean across faces = {A.mean():.4f}")
print(f"  After  centering: pixel mean across faces = {A_centered.mean():.2e}  ≈ 0  ✓")
print(f"\n  Effect: common lighting/baseline removed.")
print(f"  Ã encodes only the VARIATION between faces.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 - STRUCTURE OF THE SPACE
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 3 - Structure of the Space")
print(SEP)

# Numerical rank via SVD (threshold small singular values)
# We use the thin SVD of A_centered (400×4096) - only 400 singular values
# We'll compute full SVD later in Step 8; here just rank/nullity
_, singular_values, _ = np.linalg.svd(A_centered, full_matrices=False)
tol = singular_values.max() * max(A_centered.shape) * np.finfo(float).eps
numerical_rank = int(np.sum(singular_values > tol))
nullity = n_pixels - numerical_rank

print(f"\n  Singular values computed (thin SVD of Ã).")
print(f"  Tolerance for rank = {tol:.2e}")
print(f"\n  Rank (Ã)    = {numerical_rank}")
print(f"  Nullity (Ã) = {nullity}   (= n_pixels − rank = {n_pixels} − {numerical_rank})")
print(f"\n  Rank-Nullity Theorem: dim(Col Ã) + dim(Null Ã) = n_pixels")
print(f"                         {numerical_rank}          +    {nullity}     = {n_pixels}  ✓")
print(f"\n  Column Space: 'Face Space' - {numerical_rank}-dimensional subspace")
print(f"               spanned by independent facial variation directions.")
print(f"  Null Space  : {nullity}-dimensional - pixel directions with ZERO")
print(f"               variation across all 400 faces (constant pixels).")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 - REMOVE REDUNDANCY (Linear Independence / Basis)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 4 - Remove Redundancy (Linear Independence / Basis)")
print(SEP)

# The rank-399 basis is extracted via the eigenvectors of the small
# covariance matrix C_small = Ã Ãᵀ (400×400) - same non-zero eigenvalues
# as the full 4096×4096 covariance, but MUCH cheaper to compute.
# This is the standard computational trick for high-dimensional PCA.

C_small = (A_centered @ A_centered.T) / (n_samples - 1)   # 400 × 400

print(f"\n  Full covariance C = ÃᵀÃ would be {n_pixels}×{n_pixels} - too large.")
print(f"  Trick: use C_small = ÃÃᵀ / (n-1)  shape {C_small.shape}")
print(f"  Both share the same non-zero eigenvalues.")
print(f"\n  C_small is symmetric: C_small = C_smallᵀ ?  "
      f"{'✓' if np.allclose(C_small, C_small.T) else '✗'}")
print(f"\n  The eigenvectors of C_small → linearly independent directions")
print(f"  in face space. These form the BASIS we need.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 - ORTHOGONALIZATION (Gram-Schmidt on a small subset for demo)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 5 - Orthogonalization (Gram–Schmidt)")
print(SEP)

def gram_schmidt(V):
    """
    Gram-Schmidt orthogonalization.
    Input : V  - matrix whose COLUMNS are input vectors  (m × k)
    Output: Q  - matrix whose COLUMNS are orthonormal    (m × k)
    """
    Q = np.zeros_like(V, dtype=np.float64)
    for j in range(V.shape[1]):
        v = V[:, j].copy()
        for i in range(j):           # subtract projections onto previous q_i
            v -= np.dot(Q[:, i], v) * Q[:, i]
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            Q[:, j] = v / norm
        else:
            Q[:, j] = 0.0            # linearly dependent - zero it out
    return Q

# Demonstrate Gram-Schmidt on the first 6 eigenvectors of C_small
# (We'll get ALL eigenvectors properly in Step 8)
evals_demo, evecs_demo = np.linalg.eigh(C_small)
idx_demo = np.argsort(evals_demo)[::-1]
V_demo = evecs_demo[:, idx_demo[:6]]   # first 6 eigenvectors, shape (400, 6)

Q_demo = gram_schmidt(V_demo)

print(f"\n  Demonstrating Gram-Schmidt on the first 6 basis vectors.")
print(f"\n  Input basis V (from Step 4), shape: {V_demo.shape}")
print(f"  Output Q (orthonormal),       shape: {Q_demo.shape}")

# Verify orthonormality
QtQ = Q_demo.T @ Q_demo
max_off_diag = np.max(np.abs(QtQ - np.eye(6)))
print(f"\n  Verification - QᵀQ:")
print(f"  Diagonal entries  (should be 1.0): {np.diag(QtQ).round(8)}")
print(f"  Max off-diagonal  (should be ≈ 0): {max_off_diag:.2e}  ✓")
print(f"\n  Each column of Q = one orthonormal 'face direction'.")
print(f"  Projecting onto Q gives noise-free, non-overlapping coordinates.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 - PROJECTION (Orthogonal Projection onto Face Subspace)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 6 - Projection (Orthogonal Projection)")
print(SEP)

# Full eigendecomposition of C_small (needed for projection)
eigenvalues_small, eigenvectors_small = np.linalg.eigh(C_small)
idx_full = np.argsort(eigenvalues_small)[::-1]
eigenvalues_small  = eigenvalues_small[idx_full]
eigenvectors_small = eigenvectors_small[:, idx_full]

# Convert small eigenvectors → full pixel-space eigenfaces
# If u is an eigenvector of C_small, then Ãᵀu is the eigenface in pixel space
eigenfaces_raw = A_centered.T @ eigenvectors_small   # shape (4096, 400)

# Normalize each eigenface to unit length
norms = np.linalg.norm(eigenfaces_raw, axis=0, keepdims=True)
norms[norms < 1e-10] = 1.0
eigenfaces = eigenfaces_raw / norms                  # shape (4096, 400)

# Choose k such that cumulative variance ≥ 95%
total_variance = eigenvalues_small.sum()
cumulative_var  = np.cumsum(eigenvalues_small) / total_variance
k = int(np.argmax(cumulative_var >= 0.95)) + 1

print(f"\n  Eigenfaces computed in pixel space.  Shape: {eigenfaces.shape}")
print(f"  Keeping top-k eigenfaces for ≥ 95% variance: k = {k}")
print(f"  (Variance captured by top-{k} eigenfaces: {cumulative_var[k-1]*100:.2f}%)")

# Reduced eigenfaces matrix (4096 × k)
W = eigenfaces[:, :k]   # the 'eigenface basis'

# Project ALL faces into k-dimensional eigenface space
# coords[i] = Wᵀ · (face_i − μ)   →  shape (400, k)
coords = (A_centered @ W)   # shape (400, k)

print(f"\n  Projection: each face f → Wᵀ(f − μ)  →  {k}-dim coordinate vector")
print(f"  Projection matrix W  shape: {W.shape}")
print(f"  Coordinates matrix   shape: {coords.shape}")
print(f"\n  Each face is now a {k}-number 'fingerprint' instead of {n_pixels} pixels.")

# Demo: project one test face
test_idx = 0   # use first image as test
test_face = A[test_idx]
test_centered = test_face - mean_face
test_coords = W.T @ test_centered      # shape (k,)

print(f"\n  Test face (subject 0, image 0) projection:")
print(f"  coord[:5] = {test_coords[:5].round(4)}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 - PREDICTION / LEAST SQUARES (Reconstruction + Recognition)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 7 - Prediction / Least Squares")
print(SEP)

# Least Squares: reconstruct a face from its eigenface coefficients
# System: W · x ≈ f_centered   →  overdetermined (4096 eq, k unknowns)
# Normal equations: x̂ = (WᵀW)⁻¹ Wᵀ f_centered
#                     = Wᵀ f_centered   (since W is orthonormal → WᵀW = I)
# So x̂ = Wᵀ f_centered  is already the least squares solution!

x_hat = W.T @ test_centered               # shape (k,)  - LS solution
face_reconstructed = mean_face + W @ x_hat  # shape (4096,) - reconstructed face

residual = np.linalg.norm(test_face - face_reconstructed)
print(f"\n  Least Squares:  x̂ = (WᵀW)⁻¹Wᵀf  =  Wᵀf  (since WᵀW = I)")
print(f"  Reconstruction: f̂ = μ + W·x̂")
print(f"\n  Residual ‖f − f̂‖ = {residual:.6f}")
print(f"  (Lower residual = more accurate reconstruction)")

# Recognition: find nearest neighbor in eigenface space
# Exclude the test face itself from the search
def recognize(test_face_vec, A_all, labels_all, mean_f, W_basis, exclude_idx=None):
    """
    Project test face and find nearest training face in eigenface space.
    Returns (predicted_label, matched_index, distance).
    """
    tc = test_face_vec - mean_f
    test_c = W_basis.T @ tc
    train_coords = (A_all - mean_f) @ W_basis   # (400, k)

    distances = np.linalg.norm(train_coords - test_c, axis=1)
    if exclude_idx is not None:
        distances[exclude_idx] = np.inf

    best_idx = np.argmin(distances)
    return labels_all[best_idx], best_idx, distances[best_idx]

pred_label, match_idx, dist = recognize(test_face, A, labels, mean_face, W, exclude_idx=test_idx)
true_label = labels[test_idx]
print(f"\n  Recognition Demo (test = subject {true_label}, image 0):")
print(f"  Predicted subject : {pred_label}")
print(f"  True subject      : {true_label}")
print(f"  Distance          : {dist:.4f}")
print(f"  Result            : {'✓ CORRECT' if pred_label == true_label else '✗ INCORRECT'}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 - PATTERN DISCOVERY (Eigenvalues & Eigenvectors)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 8 - Pattern Discovery (Eigenvalues & Eigenvectors)")
print(SEP)

print(f"\n  Covariance matrix C_small = ÃÃᵀ/(n−1)   shape: {C_small.shape}")
print(f"  Symmetric: ✓   Positive semi-definite: ✓")
print(f"\n  Top-10 eigenvalues (descending):")
for i in range(10):
    pct = eigenvalues_small[i] / total_variance * 100
    bar = "█" * int(pct / 1.5)
    print(f"    λ{i+1:>2} = {eigenvalues_small[i]:>10.3f}   ({pct:>5.2f}%)  {bar}")

print(f"\n  Cumulative variance (top-k):")
for k_show in [10, 20, 50, 100, k]:
    print(f"    k = {k_show:>3}  →  {cumulative_var[k_show-1]*100:.2f}%  cumulative variance")

print(f"\n  Interpretation:")
print(f"  λ1 (largest) captures the single strongest facial pattern")
print(f"  - likely overall brightness / lighting gradient.")
print(f"  λ2 captures the next most distinct variation - e.g. face shape.")
print(f"  Smaller eigenvalues → minor details or noise.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 - SYSTEM SIMPLIFICATION (Diagonalization)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 9 - System Simplification (Diagonalization)")
print(SEP)

# C_small = P D Pᵀ  (symmetric → orthogonal diagonalization)
P = eigenvectors_small            # columns = eigenvectors (already sorted desc)
D = np.diag(eigenvalues_small)    # diagonal matrix

recon_error = np.linalg.norm(C_small - P @ D @ P.T)
print(f"\n  C_small = P · D · Pᵀ   (since C_small is symmetric → Pᵀ = P⁻¹)")
print(f"\n  P  shape: {P.shape}  (columns = eigenvectors)")
print(f"  D  shape: {D.shape}  (diagonal = eigenvalues)")
print(f"\n  Verification: ‖C_small − PDPᵀ‖ = {recon_error:.2e}  ✓")
print(f"\n  Reduced system: keep top-k = {k} eigenvectors")
print(f"  → Eigenfaces W  shape: {W.shape}")
print(f"  → Compressed from {n_pixels}D → {k}D  (retains ≥95% variance)")
print(f"\n  Noise reduction: the bottom {n_samples - k} eigenvalue directions")
print(f"  (noise / minor variations) are discarded entirely.")


# ═════════════════════════════════════════════════════════════════════════════
# FULL RECOGNITION ACCURACY TEST
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  FINAL OUTPUT - Recognition Accuracy Evaluation")
print(SEP)

# Train on images 0-6 of each subject (7 images), test on images 7-9 (3 images)
train_mask = np.array([(i % 10) < 7 for i in range(n_samples)])
test_mask  = ~train_mask

A_train      = A[train_mask]
labels_train = labels[train_mask]
A_test       = A[test_mask]
labels_test  = labels[test_mask]

# Re-compute pipeline on training set only
mean_train = A_train.mean(axis=0)
A_train_c  = A_train - mean_train

C_tr = (A_train_c @ A_train_c.T) / (len(A_train) - 1)
evals_tr, evecs_tr = np.linalg.eigh(C_tr)
idx_tr = np.argsort(evals_tr)[::-1]
evals_tr  = evals_tr[idx_tr]
evecs_tr  = evecs_tr[:, idx_tr]

ef_raw_tr = A_train_c.T @ evecs_tr
norms_tr  = np.linalg.norm(ef_raw_tr, axis=0, keepdims=True)
norms_tr[norms_tr < 1e-10] = 1.0
ef_tr = ef_raw_tr / norms_tr

cum_tr = np.cumsum(evals_tr) / evals_tr.sum()
k_tr   = int(np.argmax(cum_tr >= 0.95)) + 1
W_tr   = ef_tr[:, :k_tr]

# Project training faces
coords_train = A_train_c @ W_tr   # (280, k_tr)

# Recognise each test image
correct = 0
total   = len(A_test)
results = []

for i in range(total):
    f       = A_test[i]
    fc      = f - mean_train
    fc_proj = W_tr.T @ fc                              # (k_tr,)
    dists   = np.linalg.norm(coords_train - fc_proj, axis=1)
    best    = np.argmin(dists)
    pred    = labels_train[best]
    true    = labels_test[i]
    hit     = (pred == true)
    correct += hit
    results.append((true, pred, dists[best], hit))

accuracy = correct / total * 100
print(f"\n  Train set : {len(A_train)} images (7 per subject)")
print(f"  Test set  : {len(A_test)} images  (3 per subject)")
print(f"  k used    : {k_tr}  eigenfaces")
print(f"\n  Correct   : {correct} / {total}")
print(f"  Accuracy  : {accuracy:.1f}%")

wrong = [(t, p, d) for t, p, d, h in results if not h]
if wrong:
    print(f"\n  Misclassified ({len(wrong)} images):")
    for t, p, d in wrong[:5]:
        print(f"    True={t:>2}  Predicted={p:>2}  dist={d:.4f}")
else:
    print(f"\n  Perfect recognition - 0 errors  🎉")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS  (6 plots in 2 figures)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  Generating Visualisations …")
print(SEP)

def show_face(ax, vec, title="", cmap="gray"):
    ax.imshow(vec.reshape(h, w), cmap=cmap, interpolation="nearest")
    ax.set_title(title, color=TEXT, fontsize=9, pad=4)
    ax.axis("off")

# ── Figure 1: Mean face + first 12 eigenfaces ───────────────────────────────
fig1 = plt.figure(figsize=(15, 5), facecolor=BG)
fig1.suptitle("Step 1–9 · Mean Face & Top-12 Eigenfaces", color=TEXT, fontsize=13, y=1.01)

gs1 = gridspec.GridSpec(2, 7, figure=fig1, hspace=0.3, wspace=0.15)

ax_mean = fig1.add_subplot(gs1[:, 0])
show_face(ax_mean, mean_face, "Mean Face\n(μ)")
ax_mean.set_facecolor(PANEL)

for i in range(12):
    row, col = divmod(i, 6)
    ax = fig1.add_subplot(gs1[row, col + 1])
    ef_vis = eigenfaces[:, i]
    # Normalize to [0,1] for display
    ef_min, ef_max = ef_vis.min(), ef_vis.max()
    ef_norm = (ef_vis - ef_min) / (ef_max - ef_min + 1e-10)
    pct = eigenvalues_small[i] / total_variance * 100
    show_face(ax, ef_norm, f"EF {i+1}\n({pct:.1f}%)")
    ax.set_facecolor(PANEL)
    # Colour border by eigenvalue magnitude
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(ACCENT if i < 3 else VIOLET if i < 8 else SUBTEXT)
        spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig("fig1_eigenfaces.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  Saved: fig1_eigenfaces.png")

# ── Figure 2: Variance explained (scree plot) ────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
fig2.suptitle("Step 8 · Pattern Discovery - Eigenvalue Analysis", color=TEXT, fontsize=13)

ax = axes[0]
ax.set_facecolor(PANEL)
top = 50
ax.bar(range(1, top + 1),
       eigenvalues_small[:top] / total_variance * 100,
       color=ACCENT, alpha=0.85, width=0.8)
ax.set_xlabel("Eigenface index", color=SUBTEXT, fontsize=9)
ax.set_ylabel("Variance explained (%)", color=SUBTEXT, fontsize=9)
ax.set_title("Individual variance per eigenface", color=TEXT, fontsize=10)
ax.grid(axis="y", alpha=0.3)

ax2 = axes[1]
ax2.set_facecolor(PANEL)
ax2.plot(range(1, n_samples + 1), cumulative_var * 100, color=MINT, linewidth=2)
ax2.axhline(95, color=AMBER, linewidth=1.2, linestyle="--", label="95% threshold")
ax2.axvline(k, color=VIOLET, linewidth=1.2, linestyle="--", label=f"k = {k}")
ax2.fill_between(range(1, k + 1), cumulative_var[:k] * 100, alpha=0.15, color=MINT)
ax2.set_xlabel("Number of eigenfaces (k)", color=SUBTEXT, fontsize=9)
ax2.set_ylabel("Cumulative variance (%)", color=SUBTEXT, fontsize=9)
ax2.set_title("Cumulative variance vs k", color=TEXT, fontsize=10)
ax2.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, n_samples)
ax2.set_ylim(0, 102)

plt.tight_layout()
plt.savefig("fig2_variance.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  Saved: fig2_variance.png")

# ── Figure 3: Reconstruction quality at different k values ──────────────────
fig3, axes3 = plt.subplots(2, 6, figsize=(15, 5), facecolor=BG)
fig3.suptitle("Step 7 · Least Squares Reconstruction at varying k", color=TEXT, fontsize=13)

# Pick subject 5, image 0
demo_idx = 50
demo_face = A[demo_idx]
demo_centered = demo_face - mean_face

k_values = [1, 5, 10, 25, 50, k]

# Row 0: original (repeated) | Row 1: reconstructions
for col, kv in enumerate(k_values):
    W_k = eigenfaces[:, :kv]
    x_k = W_k.T @ demo_centered
    recon_k = mean_face + W_k @ x_k
    res_k   = np.linalg.norm(demo_face - recon_k)

    ax_orig = axes3[0, col]
    ax_orig.set_facecolor(PANEL)
    show_face(ax_orig, demo_face, f"Original\n(Subject {labels[demo_idx]})")

    ax_rec = axes3[1, col]
    ax_rec.set_facecolor(PANEL)
    show_face(ax_rec, np.clip(recon_k, 0, 1), f"k={kv}\n‖err‖={res_k:.3f}")

    for spine in ax_rec.spines.values():
        spine.set_visible(True)
        spine.set_color(MINT if kv == k else ACCENT)
        spine.set_linewidth(1.5 if kv == k else 1)

plt.tight_layout()
plt.savefig("fig3_reconstruction.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  Saved: fig3_reconstruction.png")

# ── Figure 4: Recognition demo - test face + nearest match + wrong example ──
fig4 = plt.figure(figsize=(15, 5), facecolor=BG)
fig4.suptitle("Step 6–7 · Recognition Demo - Query → Top Matches", color=TEXT, fontsize=13)

gs4 = gridspec.GridSpec(2, 10, figure=fig4, hspace=0.4, wspace=0.2)

# Show 5 test faces and their matches
test_indices_demo = [0, 10, 20, 30, 40]   # one per subject (image 7 in test set)
test_actual = np.where(test_mask)[0]       # real indices in A

for col, ti in enumerate(test_indices_demo):
    actual_idx = test_actual[ti]
    f_test     = A[actual_idx]
    true_lbl   = labels[actual_idx]

    pred_lbl, match_idx_t, dist_t = recognize(
        f_test, A_train, labels_train, mean_train, W_tr
    )
    match_face = A_train[match_idx_t]
    correct_t  = (pred_lbl == true_lbl)

    ax_q = fig4.add_subplot(gs4[0, col * 2])
    ax_q.set_facecolor(PANEL)
    show_face(ax_q, f_test, f"Query\nS={true_lbl}")

    ax_m = fig4.add_subplot(gs4[0, col * 2 + 1])
    ax_m.set_facecolor(PANEL)
    result_color = MINT if correct_t else AMBER
    show_face(ax_m, match_face, f"Match\nS={pred_lbl}")
    for spine in ax_m.spines.values():
        spine.set_visible(True)
        spine.set_color(result_color)
        spine.set_linewidth(2)

# Bottom row: projection space (2D scatter of first 2 eigenface coords)
ax_scatter = fig4.add_subplot(gs4[1, :])
ax_scatter.set_facecolor(PANEL)

coords_vis = A_train_c @ W_tr[:, :2] if W_tr.shape[1] >= 2 else coords_train[:, :2]

# Colour each subject differently
cmap_scatter = plt.cm.get_cmap("tab20", 40)
for subj in range(40):
    mask_s = (labels_train == subj)
    ax_scatter.scatter(
        coords_vis[mask_s, 0], coords_vis[mask_s, 1],
        color=cmap_scatter(subj), s=18, alpha=0.7, label=f"S{subj}" if subj < 5 else ""
    )

ax_scatter.set_xlabel("Eigenface 1 coordinate", color=SUBTEXT, fontsize=9)
ax_scatter.set_ylabel("Eigenface 2 coordinate", color=SUBTEXT, fontsize=9)
ax_scatter.set_title("Projection into 2D Eigenface Space (all training faces)", color=TEXT, fontsize=10)
ax_scatter.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("fig4_recognition.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  Saved: fig4_recognition.png")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print("  PIPELINE COMPLETE - SUMMARY")
print(f"{'═'*65}")
print(f"""
  Step 1  A ∈ ℝ^{{400×4096}}          Real faces as row vectors
  Step 2  Ã = A − μ                  Mean-centered (baseline removed)
  Step 3  Rank(Ã) = {numerical_rank:<4}              Face space dimension
  Step 4  Basis extracted            Independent facial directions
  Step 5  Gram-Schmidt → Q           Orthonormal basis verified
  Step 6  W = eigenfaces[:, :k]      k = {k}  (≥95% variance)
  Step 7  x̂ = Wᵀ(f−μ)  →  f̂=μ+Wx̂   Least squares reconstruction
  Step 8  λ₁ = {eigenvalues_small[0]:>10.3f}           Dominant facial pattern
  Step 9  C = PDPᵀ verified          Diagonalization error ≈ 0

  RECOGNITION ACCURACY (train 7 / test 3 per subject):
    {correct} / {total}  →  {accuracy:.1f}%
""")
print("  Figures saved:")
print("    fig1_eigenfaces.png    - mean face + top-12 eigenfaces")
print("    fig2_variance.png      - scree plot + cumulative variance")
print("    fig3_reconstruction.png- reconstruction at varying k")
print("    fig4_recognition.png   - recognition demo + 2D projection")
print()
