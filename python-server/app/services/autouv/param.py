"""Per-chart flattening.

Three methods, best-of kept per chart:

* **LSCM** (Levy et al. 2002): minimises *angle* distortion, needs two pinned
  vertices. Conformal, so it preserves angles at the cost of *area* — a large
  chart spanning curvature comes out with uneven texel density.

* **ARAP** (Liu et al. 2008, "A Local/Global Approach to Mesh Parameterization"):
  minimises deviation from a *rigid* (isometric) map, so it balances angle *and*
  area distortion. It needs an initial flattening to start from; we seed it with
  the LSCM result (or planar) and run a few local/global iterations. This is what
  flattens the area distortion that LSCM leaves on the larger islands the weld +
  merge passes now produce.

* **planar**: orthographic projection onto the chart's best-fit plane. Always
  succeeds; the only sane answer for closed charts with no boundary (LSCM
  degenerate) and a very good one for near-flat charts.

For each chart we compute the viable candidates and keep whichever has the
lowest combined (angle + area) distortion with no flips.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, splu, MatrixRankWarning

from .metrics import triangle_frames, signed_uv_area, jacobians, singular_values, island_overlap
from . import topology as _topology


# --------------------------------------------------------------------- helpers
def _chart_local(vertices, faces, face_ids):
    """Extract a chart as a compact (V',3) / (F',3) local mesh."""
    f = faces[face_ids]
    uniq, inv = np.unique(f.reshape(-1), return_inverse=True)
    local_faces = inv.reshape(-1, 3)
    local_verts = vertices[uniq]
    return local_verts, local_faces, uniq


def _boundary_vertices(local_faces, n_verts):
    """Indices of vertices lying on the chart boundary (open edges)."""
    e = np.concatenate([local_faces[:, [0, 1]],
                        local_faces[:, [1, 2]],
                        local_faces[:, [2, 0]]], axis=0)
    es = np.sort(e, axis=1)
    # an edge is on the boundary if it occurs exactly once
    uniq, counts = np.unique(es, axis=0, return_counts=True)
    bedges = uniq[counts == 1]
    if len(bedges) == 0:
        return np.array([], dtype=np.int64)
    return np.unique(bedges.reshape(-1))


# ------------------------------------------------------------------------ LSCM
def lscm(local_verts, local_faces):
    """Solve LSCM. Returns (uv, ok)."""
    n = len(local_verts)
    bnd = _boundary_vertices(local_faces, n)
    if len(bnd) < 2:
        return None, False  # closed chart -> degenerate for LSCM

    # pin the two most distant boundary vertices
    bpos = local_verts[bnd]
    c = bpos.mean(axis=0)
    far = bnd[np.argmax(np.linalg.norm(bpos - c, axis=1))]
    far_pos = local_verts[far]
    p1 = bnd[np.argmax(np.linalg.norm(local_verts[bnd] - far_pos, axis=1))]
    p0 = far
    if p0 == p1:
        return None, False

    # One complex conformality equation per triangle (real + imaginary row),
    # assembled for every triangle at once.
    nt = len(local_faces)
    P, _ = triangle_frames(local_verts, local_faces)
    valid = P[:, 1, 0] >= 1e-12                 # skip zero-length first edges
    x0, y0 = P[:, 0, 0], P[:, 0, 1]
    x1, y1 = P[:, 1, 0], P[:, 1, 1]
    x2, y2 = P[:, 2, 0], P[:, 2, 1]
    # W = opposite edge of each vertex, as a complex number a + i b
    Wre = np.stack([x2 - x1, x0 - x2, x1 - x0], axis=1)
    Wim = np.stack([y2 - y1, y0 - y2, y1 - y0], axis=1)
    area2 = np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    w = 1.0 / np.sqrt(area2 + 1e-12)
    a = (w[:, None] * Wre)[valid]               # (T, 3)
    b = (w[:, None] * Wim)[valid]
    vtx = local_faces[valid]
    t = np.nonzero(valid)[0][:, None].repeat(3, axis=1)
    r_real = 2 * t
    r_imag = 2 * t + 1
    # Real:  a*u - b*v      Imag:  b*u + a*v
    rows = np.concatenate([r_real, r_real, r_imag, r_imag], axis=None)
    cols = np.concatenate([vtx, n + vtx, vtx, n + vtx], axis=None)
    vals = np.concatenate([a, -b, b, a], axis=None)

    M = coo_matrix((vals, (rows, cols)), shape=(2 * nt, 2 * n)).tocsc()

    pinned = np.array([p0, n + p0, p1, n + p1])
    pin_val = np.array([0.0, 0.0, 1.0, 0.0])  # p0->(0,0), p1->(1,0)
    free = np.setdiff1d(np.arange(2 * n), pinned)

    Mf = M[:, free]
    Mp = M[:, pinned]
    rhs = -Mp.dot(pin_val)
    A = (Mf.T @ Mf).tocsc()
    bvec = Mf.T @ rhs
    try:
        # A singular system (degenerate chart) is caught by the finiteness test
        # below and falls back to another flattening; the warning is only noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MatrixRankWarning)
            xf = spsolve(A, bvec)
    except Exception:
        return None, False
    if not np.all(np.isfinite(xf)):
        return None, False

    x = np.zeros(2 * n)
    x[free] = xf
    x[pinned] = pin_val
    uv = np.stack([x[:n], x[n:]], axis=1)
    return uv, True


# ------------------------------------------------------------------- planar
def planar(local_verts, local_faces):
    """Project onto the area-weighted mean-normal plane.

    Projecting along the surface normal guarantees no triangle flips as long as
    every face normal stays within 90 degrees of the mean -- which is exactly
    the invariant the bounded-cone segmentation maintains. In-plane axes are
    aligned to the chart's principal direction for a tidy result.
    """
    v = local_verts[local_faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    a = np.linalg.norm(fn, axis=1, keepdims=True)
    n = (fn).sum(axis=0)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / nn
    # build an in-plane basis
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= max(np.linalg.norm(t1), 1e-12)
    t2 = np.cross(n, t1)
    c = local_verts.mean(axis=0)
    X = local_verts - c
    uv = np.stack([X @ t1, X @ t2], axis=1)
    return uv


# --------------------------------------------------------------------- ARAP
def _flatten_triangles(local_verts, local_faces):
    """Isometric 2D coords + corner cotangents for every triangle (vectorised).

    Returns (P, cot, area) where P is (F,3,2) the isometrically flattened corners,
    cot is (F,3) the cotangent of the angle at corner 0/1/2, and area is (F,).
    """
    v = local_verts[local_faces]                      # (F,3,3)
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    len1 = np.linalg.norm(e1, axis=1)
    len1 = np.where(len1 < 1e-12, 1e-12, len1)
    x_axis = e1 / len1[:, None]
    proj = np.einsum("ij,ij->i", e2, x_axis)
    perp = e2 - proj[:, None] * x_axis
    h = np.linalg.norm(perp, axis=1)
    P = np.zeros((len(local_faces), 3, 2))
    P[:, 1, 0] = len1
    P[:, 2, 0] = proj
    P[:, 2, 1] = h
    area = 0.5 * len1 * h                              # = 0.5 * base * height

    # cotangent of the angle at each corner, from the flattened coords
    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
    twoA = np.where(2.0 * area < 1e-12, 1e-12, 2.0 * area)
    cot = np.empty((len(local_faces), 3))
    cot[:, 0] = np.einsum("ij,ij->i", p1 - p0, p2 - p0) / twoA   # angle at 0
    cot[:, 1] = np.einsum("ij,ij->i", p0 - p1, p2 - p1) / twoA   # angle at 1
    cot[:, 2] = np.einsum("ij,ij->i", p0 - p2, p1 - p2) / twoA   # angle at 2
    return P, cot, area


def arap(local_verts, local_faces, uv_init, iters=4):
    """Refine an initial UV toward an as-rigid-as-possible map.

    Local/global iteration: (local) fit the best rotation per triangle between
    the flattened reference and the current UV; (global) re-solve vertex
    positions for those rotations through a cotangent-Laplacian system whose
    factorisation is reused across iterations. Returns (uv, ok).
    """
    n = len(local_verts)
    if n < 3 or len(local_faces) < 1:
        return None, False
    P, cot, area = _flatten_triangles(local_verts, local_faces)

    # the three within-triangle edges (a,b) and the cotangent weighting each:
    # edge (0,1) is opposite corner 2, etc.
    E = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    ai = np.concatenate([local_faces[:, a] for a, _, _ in E])
    bi = np.concatenate([local_faces[:, b] for _, b, _ in E])
    w = np.concatenate([cot[:, c] for _, _, c in E])     # (3F,)
    # reference edge vectors P[a]-P[b] for each edge (3F,2)
    dP = np.concatenate([P[:, a] - P[:, b] for a, b, _ in E], axis=0)

    # cotangent Laplacian A (constant across iterations) -------------------
    rows = np.concatenate([ai, bi, ai, bi])
    cols = np.concatenate([ai, bi, bi, ai])
    data = np.concatenate([w, w, -w, -w])
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tolil()
    # pin vertex 0 to remove the translational null space
    A[0, :] = 0.0
    A[0, 0] = 1.0
    try:
        lu = splu(csc_matrix(A))
    except Exception:
        return None, False

    nf = len(local_faces)
    uv = uv_init.astype(np.float64).copy()
    fa = local_faces                                     # (F,3)

    for _ in range(max(1, iters)):
        # ---- local step: best rotation per triangle (batched 2x2 SVD) -----
        # S_t = sum_edges w * du * dx^T  (du = current uv diff, dx = ref diff)
        du = np.concatenate([uv[fa[:, a]] - uv[fa[:, b]] for a, b, _ in E], axis=0)
        we = w[:, None]
        # accumulate per-triangle 2x2 covariance
        S = np.zeros((nf, 2, 2))
        contrib = (we * du)[:, :, None] * dP[:, None, :]  # (3F,2,2)
        for k in range(3):
            S += contrib[k * nf:(k + 1) * nf]
        U, _, Vt = np.linalg.svd(S)
        R = np.matmul(U, Vt)                              # (F,2,2)
        det = np.linalg.det(R)
        flip = det < 0
        if np.any(flip):                                  # enforce proper rotation
            U[flip, :, -1] *= -1.0
            R = np.matmul(U, Vt)

        # ---- global step: solve A uv = rhs --------------------------------
        # rhs[a] += w * R_t dx ; rhs[b] -= w * R_t dx
        Rt_per_edge = np.concatenate([R, R, R], axis=0)   # (3F,2,2)
        rot_dP = np.einsum("eij,ej->ei", Rt_per_edge, dP) * w[:, None]
        rhs = np.zeros((n, 2))
        np.add.at(rhs, ai, rot_dP)
        np.add.at(rhs, bi, -rot_dP)
        rhs[0] = uv_init[0]                               # match the pin
        try:
            uv = np.stack([lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])], axis=1)
        except Exception:
            return None, False
        if not np.all(np.isfinite(uv)):
            return None, False
    return uv, True


# ------------------------------------------------------------- distortion
def distortion(local_verts, local_faces, uv):
    """Mean per-triangle (angle, area) distortion. Lower is better.

    Angle distortion uses a *quasi-conformal* measure 0.5*(s0/s1 + s1/s0) from
    the singular values of the per-triangle Jacobian, clamped so a few folded
    triangles cannot send the average to infinity. A perfect conformal map
    gives 1.0.
    """
    eps = 1e-12
    cap = 50.0
    P, a3 = triangle_frames(local_verts, local_faces)
    a2 = np.abs(signed_uv_area(uv, local_faces))
    keep = a3 >= eps
    if not keep.any():
        return cap, cap
    J, ok = jacobians(P[keep], uv, local_faces[keep])
    a3k, a2k = a3[keep][ok], a2[keep][ok]
    if len(a3k) == 0:
        return cap, cap
    sv = np.clip(singular_values(J[ok]), eps, None)
    qc = np.minimum(0.5 * (sv[:, 0] / sv[:, 1] + sv[:, 1] / sv[:, 0]), cap)
    angle_d = float((a3k * qc).sum() / a3k.sum())
    if a2k.sum() > 0:
        ratio = a2k / a3k
        ratio /= (ratio * a3k).sum() / a3k.sum()
        area_d = float(np.sqrt(np.average((ratio - 1.0) ** 2, weights=a3k)))
    else:
        area_d = cap
    return angle_d, area_d


def count_flips(local_faces, uv):
    """Number of triangles whose orientation flipped in UV space."""
    s = signed_uv_area(uv, local_faces)
    return int(min(np.sum(s > 0), np.sum(s < 0)))


def _flatten(lv, lf, method="auto", arap_iters=4):
    """Best-of flattening of one manifold chart.

    Returns ``(name, uv, angle_d, area_d, flips, overlap)``. ``overlap`` is the
    island's self-overlap share (see :func:`metrics.island_overlap`): a layout
    can be flip-free and still fold its boundary back over its own interior,
    which a flip count never sees, so candidates are ranked on it too.
    """
    candidates = []
    lscm_uv = None
    if method in ("auto", "lscm", "arap"):
        uv, ok = lscm(lv, lf)
        if ok:
            # fix mirrored orientation if the whole chart came out flipped
            if count_flips(lf, uv) > len(lf) // 2:
                uv[:, 1] *= -1.0
            lscm_uv = uv
            if method in ("auto", "lscm"):
                ad, ar = distortion(lv, lf, uv)
                candidates.append(("lscm", uv, ad, ar, count_flips(lf, uv), island_overlap(uv, lf)))
    if method in ("auto", "planar", "arap") or not candidates:
        uv = planar(lv, lf)
        ad, ar = distortion(lv, lf, uv)
        candidates.append(("planar", uv, ad, ar, count_flips(lf, uv), island_overlap(uv, lf)))

    # ARAP: refine the best available initialisation toward an isometric map.
    # Conformal LSCM preserves angles but lets area (texel density) drift on the
    # larger islands; ARAP pulls that back. Seed from LSCM when we have it, else
    # from the planar projection.
    if method in ("auto", "arap") and arap_iters > 0:
        seed = lscm_uv if lscm_uv is not None else candidates[-1][1]
        uv, ok = arap(lv, lf, seed, iters=arap_iters)
        if ok:
            if count_flips(lf, uv) > len(lf) // 2:
                uv[:, 1] *= -1.0
            ad, ar = distortion(lv, lf, uv)
            candidates.append(("arap", uv, ad, ar, count_flips(lf, uv), island_overlap(uv, lf)))

    return min(candidates, key=_score)


# A flattening whose island overlaps itself by more than this share is not
# accepted as it is: the merge validation refuses it, and the fast path falls
# back to the full best-of search.
OVERLAP_TOL = 0.02
# Weight of the self-overlap share against distortion when ranking candidates:
# 1% of the island painted twice costs as much as 0.05 of area distortion. A
# hard "no overlap first" rule picked badly stretched planar projections over
# ARAP layouts with a small folded tip -- worse texel density everywhere to
# save a few texels in one place.
OVERLAP_WEIGHT = 10.0


def _score(c):
    """Fewest flips, then distortion with self-overlap as a weighted penalty."""
    _, _, ad, ar, fl, ov = c
    return (fl, ad + 2.0 * ar + OVERLAP_WEIGHT * ov)


def parameterize_chart(vertices, faces, face_ids, method="auto", arap_iters=4,
                       vertex_weight=None, make_disk=True, fast=False):
    """Flatten one chart. Returns (local_uv, uniq_vertex_ids, local_faces, info).

    The chart is first made manifold (:func:`topology.split_corners`) and, with
    ``make_disk``, cut to a disk (:func:`topology.make_disk`). ``uniq`` maps
    every local vertex to its vertex in ``vertices``; after a cut several local
    vertices share one source vertex, which is exactly what a seam is.
    ``local_faces`` stays row-aligned with ``face_ids``.

    A chart with holes but no handles can be flattened as it is -- LSCM copes
    with an annulus -- so it is flattened both ways and only cut when the cut
    layout is clearly better (fewer flips, or 10% lower distortion). Closed and
    handled charts have no valid planar embedding and are always cut.

    ``vertex_weight`` (per vertex of ``vertices``) makes cuts avoid expensive
    (visible) places. ``fast`` tries LSCM alone and only falls back to the full
    best-of search when LSCM flips or folds over itself; it is what the
    chart-merge and border-move validations use.
    """
    lv0, lf0, uniq0 = _chart_local(vertices, faces, face_ids)
    info = {"method": None, "angle_d": None, "area_d": None, "flips": 0,
            "cut": "disk", "cut_edges": 0, "components": 1}

    fn = _topology.face_normals(lv0, lf0)
    lf1, orig1, _, n_comp = _topology.split_corners(lf0, len(lv0), fn)
    info["components"] = n_comp
    if n_comp > 1:
        # Only reachable through the merge validation (the unwrapper splits
        # disconnected charts up front): flatten the raw chart as before and
        # let the caller's flip test decide.
        variants = [("raw", lv0, lf0, uniq0)]
    else:
        lv1, uniq1 = lv0[orig1], uniq0[orig1]
        variants = []
        topo = _topology.topology(lf1, len(lv1))
        if topo.is_disk or (topo.n_loops > 0 and topo.genus <= 0) or not make_disk:
            variants.append(("disk" if topo.is_disk else "none", lv1, lf1, uniq1))
        if make_disk and not topo.is_disk:
            w = vertex_weight[uniq1] if vertex_weight is not None else None
            lf2, orig2, kind, n_cut = _topology.make_disk(lv1, lf1, w)
            if kind not in ("failed", "skip", "disk"):
                variants.append((kind, lv1[orig2], lf2, uniq1[orig2], n_cut))
        if not variants:
            variants.append(("failed", lv1, lf1, uniq1))

    def run(lv, lf):
        if fast and method in ("auto", "arap"):
            best = _flatten(lv, lf, "lscm", 0)
            if best[4] == 0 and best[5] <= OVERLAP_TOL:
                return best
        return _flatten(lv, lf, method, arap_iters)

    chosen = None
    for v in variants:
        kind, lv, lf, uniq = v[:4]
        res = run(lv, lf)
        if chosen is None:
            chosen = (v, res)
            continue
        (_, pres) = chosen
        pf, ps = _score(pres)
        cf, cs = _score(res)
        # prefer the uncut layout unless the cut one is clearly better
        if cf < pf or (cf == pf and cs < 0.9 * ps) or chosen[0][0] == "failed":
            chosen = (v, res)

    v, (name, uv, ad, ar, fl, ov) = chosen
    kind, lv, lf, uniq = v[:4]
    info.update(method=name, angle_d=float(ad), area_d=float(ar), flips=int(fl), overlap=float(ov),
                cut=kind, cut_edges=int(v[4]) if len(v) > 4 else 0)
    return uv, uniq, lf, info
