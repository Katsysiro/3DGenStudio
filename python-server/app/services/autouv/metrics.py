"""Layout quality measurements, all vectorised.

What the unwrapper reports about its own output, and what later passes use to
decide whether a change helped:

* **stretch** -- symmetric Dirichlet energy per triangle, 0.25 * (|J|^2 + |J^-1|^2)
  after one global rescale so total UV area equals total surface area. It is 1.0
  for an isometry and grows with both shear and scale error, so unlike an
  angle-only measure it also sees texel-density drift.
* **density share** -- the share of surface area whose texel density is off by
  more than 25% (scale outside [0.8, 1.25]). It is the number an artist sees as
  "checker squares of different sizes".
* **overlaps** -- texels that more than one triangle covers. Flipped
  triangles only catch local folds; an island boundary that loops back over its
  own interior, or two islands packed on top of each other, is invisible to a
  flip count, and this is what catches it.
* **seams** -- total seam length relative to the model size, and how much of it
  lies in hidden places when a hidden-surface field is available.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12
ENERGY_CAP = 50.0


# ------------------------------------------------------------------ frames
def triangle_frames(vertices, faces):
    """Isometric 2D coordinates of every triangle in its own plane.

    Returns ``(P, area)``: ``P`` is (F, 3, 2) with corner 0 at the origin and
    corner 1 on +x, ``area`` is (F,) the 3D area. Degenerate triangles get a
    zero area and should be masked by the caller.
    """
    v = vertices[faces]
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    len1 = np.linalg.norm(e1, axis=1)
    safe = np.where(len1 < _EPS, 1.0, len1)
    x_axis = e1 / safe[:, None]
    proj = np.einsum("ij,ij->i", e2, x_axis)
    perp = e2 - proj[:, None] * x_axis
    h = np.linalg.norm(perp, axis=1)
    P = np.zeros((len(faces), 3, 2))
    P[:, 1, 0] = len1
    P[:, 2, 0] = proj
    P[:, 2, 1] = h
    area = 0.5 * len1 * h
    area[len1 < _EPS] = 0.0
    return P, area


def signed_uv_area(uv, faces):
    """(F,) signed UV area; the sign carries the triangle's UV orientation."""
    q = uv[faces]
    return 0.5 * ((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                  - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1]))


def jacobians(P, uv, faces):
    """(F, 2, 2) Jacobian of the 3D->UV map per triangle, and a validity mask."""
    q = uv[faces]
    Pm = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)   # columns
    Qm = np.stack([q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]], axis=2)
    det = Pm[:, 0, 0] * Pm[:, 1, 1] - Pm[:, 0, 1] * Pm[:, 1, 0]
    ok = np.abs(det) > _EPS
    inv = np.zeros_like(Pm)
    d = np.where(ok, det, 1.0)
    inv[:, 0, 0] = Pm[:, 1, 1] / d
    inv[:, 1, 1] = Pm[:, 0, 0] / d
    inv[:, 0, 1] = -Pm[:, 0, 1] / d
    inv[:, 1, 0] = -Pm[:, 1, 0] / d
    return np.matmul(Qm, inv), ok


def singular_values(J):
    """(F, 2) singular values (descending) of a stack of 2x2 matrices, closed form."""
    a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    e = a * a + b * b + c * c + d * d
    det = a * d - b * c
    disc = np.sqrt(np.maximum(e * e - 4.0 * det * det, 0.0))
    s1 = np.sqrt(np.maximum(0.5 * (e + disc), 0.0))
    s2 = np.sqrt(np.maximum(0.5 * (e - disc), 0.0))
    return np.stack([s1, s2], axis=1)


# ---------------------------------------------------------------- stretch
def face_energy(vertices, faces, uv, rescale=True):
    """Per-face symmetric Dirichlet energy (1.0 = isometric) and the 3D areas.

    With ``rescale`` the whole layout is first scaled so its UV area equals the
    surface area, which makes the number independent of where the packer put
    the atlas. Degenerate faces get ``ENERGY_CAP``.
    """
    P, a3 = triangle_frames(vertices, faces)
    a2 = np.abs(signed_uv_area(uv, faces))
    s = 1.0
    if rescale and a2.sum() > _EPS:
        s = np.sqrt(a3.sum() / a2.sum())
    J, ok = jacobians(P, uv * s, faces)
    fro = np.einsum("fij,fij->f", J, J)
    det = np.abs(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
    e = np.full(len(faces), ENERGY_CAP)
    good = ok & (det > _EPS) & (a3 > _EPS)
    e[good] = 0.25 * (fro[good] + fro[good] / (det[good] ** 2))
    return np.minimum(e, ENERGY_CAP), a3


def density_share(vertices, faces, uv, lo=0.8, hi=1.25):
    """Share of surface area whose texel density is outside ``[lo, hi]``."""
    _, a3 = triangle_frames(vertices, faces)
    a2 = np.abs(signed_uv_area(uv, faces))
    tot3, tot2 = a3.sum(), a2.sum()
    if tot3 <= _EPS or tot2 <= _EPS:
        return 1.0
    g = np.sqrt(tot3 / tot2)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(a2 / np.where(a3 > _EPS, a3, np.inf)) * g
    off = (a3 > _EPS) & ((scale < lo) | (scale > hi))
    return float(a3[off].sum() / tot3)


# --------------------------------------------------------------- overlaps
def uv_overlap(uv, faces, resolution=1024, chunk=2_000_000, return_hits=False):
    """Texels of a ``resolution``^2 texture that more than one triangle covers.

    This is overlap as texturing sees it: a texel painted by two triangles
    shows the same colour in two places of the model. Each triangle is
    rasterised by pixel centre with a *strict* inside test, so neighbours that
    share an edge never both claim a texel on it; what is left is real overlap
    -- folds, an island boundary crossing its own interior, stacked islands.
    Sliver triangles that merely touch cannot produce false positives the way
    a geometric intersection test does.

    Returns ``(face_mask, share)``: faces covering at least one doubly-covered
    texel, and doubly-covered texels / covered texels. With ``return_hits`` a
    third value holds ``(tri, pix)`` for every doubly-covered texel hit.
    """
    F = len(faces)
    mask = np.zeros(F, dtype=bool)
    if F == 0:
        return mask, 0.0
    res = int(resolution)
    T = np.clip(uv[faces], 0.0, 1.0) * res                     # (F, 3, 2)
    lo = np.ceil(T.min(axis=1) - 0.5).astype(np.int64)
    hi = np.floor(T.max(axis=1) - 0.5).astype(np.int64)
    lo = np.clip(lo, 0, res - 1)
    hi = np.clip(hi, 0, res - 1)
    nx = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0)
    ny = np.maximum(hi[:, 1] - lo[:, 1] + 1, 0)
    cnt = nx * ny
    area = signed_uv_area(uv, faces)
    sgn = np.sign(area)
    cnt[sgn == 0] = 0

    cover = np.zeros(res * res, dtype=np.int32)
    hits_tri, hits_pix = [], []
    tri_all = np.nonzero(cnt)[0]
    # walk the triangles in groups whose candidate texels fit in one chunk
    csum = np.cumsum(cnt[tri_all])
    g0 = 0
    while g0 < len(tri_all):
        base = csum[g0 - 1] if g0 else 0
        g1 = int(np.searchsorted(csum, base + chunk, side="right"))
        g1 = max(g1, g0 + 1)
        tri = tri_all[g0:g1]
        c = cnt[tri]
        t = np.repeat(tri, c)
        off = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        px = lo[t, 0] + off % nx[t]
        py = lo[t, 1] + off // nx[t]
        cx = px + 0.5
        cy = py + 0.5
        inside = np.ones(len(t), dtype=bool)
        for k in range(3):
            a0 = T[t, k]
            a1 = T[t, (k + 1) % 3]
            e = (a1[:, 0] - a0[:, 0]) * (cy - a0[:, 1]) - (a1[:, 1] - a0[:, 1]) * (cx - a0[:, 0])
            inside &= e * sgn[t] > 0
        pix = py[inside] * res + px[inside]
        cover += np.bincount(pix, minlength=res * res).astype(np.int32)
        hits_tri.append(t[inside])
        hits_pix.append(pix)
        g0 = g1
    covered = int((cover > 0).sum())
    double = cover >= 2
    hits = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if hits_tri and double.any():
        ht = np.concatenate(hits_tri)
        hp = np.concatenate(hits_pix)
        on = double[hp]
        mask[np.unique(ht[on])] = True
        hits = (ht[on], hp[on])
    share = float(double.sum() / covered) if covered else 0.0
    if return_hits:
        return mask, share, hits
    return mask, share


def overlap_pairs(uv, faces, resolution=256, max_pairs=20000):
    """(P, 2) pairs of faces of one island that cover a common texel."""
    mn = uv.min(axis=0)
    ext = float((uv.max(axis=0) - mn).max())
    if ext <= _EPS or len(faces) < 2:
        return np.zeros((0, 2), np.int64)
    norm = (uv - mn) / ext * (1.0 - 1e-6)
    _, _, (tri, pix) = uv_overlap(norm, faces, resolution=resolution, return_hits=True)
    if len(tri) == 0:
        return np.zeros((0, 2), np.int64)
    o = np.argsort(pix, kind="stable")
    tri, pix = tri[o], pix[o]
    same = pix[1:] == pix[:-1]
    a, b = tri[:-1][same], tri[1:][same]
    pairs = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1), axis=0)
    return pairs[:max_pairs]


def _orient(a, b, c):
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def boundary_crosses(uv, faces, max_pairs=3_000_000):
    """True when two boundary edges of an island properly cross each other.

    For a flip-free map of a disk this is exactly the case where the island
    overlaps itself (a local homeomorphism of a disk is injective when its
    boundary curve is simple), and boundary edges are a small fraction of
    the faces, so this is the cheap first test.
    """
    a = faces.reshape(-1)
    b = faces[:, [1, 2, 0]].reshape(-1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * np.int64(len(uv) + 1) + hi
    _, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    bl, bh = lo[idx[cnt == 1]], hi[idx[cnt == 1]]
    B = len(bl)
    if B < 4:
        return False
    P, Q = uv[bl], uv[bh]
    if B * (B - 1) // 2 > max_pairs:
        return True          # too big to check exhaustively; let the raster decide
    i, j = np.triu_indices(B, 1)
    share = (bl[i] == bl[j]) | (bl[i] == bh[j]) | (bh[i] == bl[j]) | (bh[i] == bh[j])
    i, j = i[~share], j[~share]
    mn_i, mx_i = np.minimum(P[i], Q[i]), np.maximum(P[i], Q[i])
    mn_j, mx_j = np.minimum(P[j], Q[j]), np.maximum(P[j], Q[j])
    box = np.all((mn_i <= mx_j) & (mn_j <= mx_i), axis=1)
    i, j = i[box], j[box]
    if len(i) == 0:
        return False
    o1 = _orient(P[i], Q[i], P[j])
    o2 = _orient(P[i], Q[i], Q[j])
    o3 = _orient(P[j], Q[j], P[i])
    o4 = _orient(P[j], Q[j], Q[i])
    return bool(np.any((o1 * o2 < 0) & (o3 * o4 < 0)))


def island_overlap(uv, faces, resolution=64):
    """Self-overlap share of one island, rasterised in its own bounding box.

    Cheap enough to run on every flattening candidate. A flip-free island
    whose boundary does not cross itself cannot overlap and costs only the
    boundary test; otherwise the island is scaled so its longer side spans
    ``resolution`` texels and rasterised.
    """
    if len(faces) < 2:
        return 0.0
    area = signed_uv_area(uv, faces)
    if (np.all(area > 0) or np.all(area < 0)) and not boundary_crosses(uv, faces):
        return 0.0
    mn = uv.min(axis=0)
    ext = float((uv.max(axis=0) - mn).max())
    if ext <= _EPS:
        return 0.0
    norm = (uv - mn) / ext * (1.0 - 1e-6)
    return uv_overlap(norm, faces, resolution=resolution)[1]


# ------------------------------------------------------------------ seams
def seam_edges(src_vertex, faces, uv):
    """Seam edges of an unwrapped mesh, as pairs of *source* vertex ids.

    ``src_vertex`` maps each output (seam-split) vertex back to the vertex it
    was copied from, so two faces that were neighbours on the source surface
    can be recognised as such even though the split gave them different vertex
    ids. An edge is a seam when its two faces disagree on the UVs at its ends.
    Returns ``(pairs, n_interior)`` with ``pairs`` (S, 2) source vertex ids.
    """
    ha = faces.reshape(-1)
    hb = faces[:, [1, 2, 0]].reshape(-1)
    sa, sb = src_vertex[ha], src_vertex[hb]
    lo, hi = np.minimum(sa, sb), np.maximum(sa, sb)
    n = int(src_vertex.max()) + 1 if len(src_vertex) else 1
    key = lo * n + hi
    # UVs at the (lo, hi) ends, whichever way round the half-edge runs
    fwd = sa <= sb
    u_lo = np.where(fwd[:, None], uv[ha], uv[hb])
    u_hi = np.where(fwd[:, None], uv[hb], uv[ha])
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    # interior edges = keys that occur exactly twice
    starts = np.r_[0, np.nonzero(~same)[0] + 1]
    counts = np.diff(np.r_[starts, len(ks)])
    two = starts[counts == 2]
    h1, h2 = order[two], order[two + 1]
    diff = (np.abs(u_lo[h1] - u_lo[h2]).max(axis=1) > 1e-9) | \
           (np.abs(u_hi[h1] - u_hi[h2]).max(axis=1) > 1e-9)
    seam = h1[diff]
    return np.stack([lo[seam], hi[seam]], axis=1), int(len(two))


def layout_quality(vertices, faces, uv, src_vertex, src_positions, hidden=None,
                   resolution=1024):
    """The quality block of the unwrap stats. See the module docstring."""
    e, a3 = face_energy(vertices, faces, uv)
    tot = float(a3.sum()) or 1.0
    order = np.argsort(e)
    cum = np.cumsum(a3[order]) / tot
    p95 = float(e[order][min(np.searchsorted(cum, 0.95), len(e) - 1)]) if len(e) else 1.0
    over, over_share = uv_overlap(uv, faces, resolution=resolution)
    pairs, _ = seam_edges(src_vertex, faces, uv)
    seg = src_positions[pairs[:, 1]] - src_positions[pairs[:, 0]]
    seam_len = np.linalg.norm(seg, axis=1)
    out = {
        "stretch_energy": float((e * a3).sum() / tot),
        "stretch_energy_p95": p95,
        "density_off_share": density_share(vertices, faces, uv),
        "overlap_faces": int(over.sum()),
        # share of the used texture area that more than one triangle covers
        "overlap_share": over_share,
        "seam_length_rel": float(seam_len.sum() / np.sqrt(tot)),
    }
    if hidden is not None and len(pairs):
        h = 0.5 * (hidden[pairs[:, 0]] + hidden[pairs[:, 1]])
        w = seam_len.sum() or 1.0
        out["seam_hidden_mean"] = float((h * seam_len).sum() / w)
        # share of seam length running over clearly visible surface
        out["seam_visible_share"] = float(seam_len[h < 0.25].sum() / w)
    return out
