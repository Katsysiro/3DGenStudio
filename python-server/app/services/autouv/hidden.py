"""Where a seam is least visible: a per-vertex "hidden" score in [0, 1].

A seam is where texture detail breaks, where bakes leave a line and where mips
bleed, so an artist puts it where nobody looks: in creases, armpits, under
overhangs, inside folds. Two cues find those places:

* **concavity** -- how far a vertex's neighbours rise above its tangent plane.
  Positive inside a crease, zero on flat ground, negative on a ridge.
* **occlusion** -- a hemisphere of short rays around the normal. A vertex whose
  rays mostly hit nearby geometry sits in a pocket the camera rarely sees.

The score is ``max(concave, occluded)``, lightly smoothed, then made *relative*:
divided by its 85th percentile, so the most hidden 15% of the surface always
reads as fully hidden. Without that, a smooth model with only gentle creases
would produce a flat field and no preference at all.

Rays go through trimesh's intersector, which uses embreex when installed (it is
a mesh-service dependency). Only a sample of vertices is ray-cast; the rest are
filled by diffusing the sampled values over the vertex graph.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags

try:  # same probe as services/segment.py
    import embreex  # noqa: F401
    HAS_EMBREE = True
except Exception:  # pragma: no cover
    HAS_EMBREE = False


def _fib_hemisphere(n):
    """``n`` roughly uniform directions on the +z hemisphere, cosine-biased."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n                      # uniform in z on the hemisphere
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _vertex_normals(vertices, faces):
    v = vertices[faces]
    cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(n, faces[:, k], cross)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def _edges(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _averaging_operator(n, edges):
    """Row-normalised vertex adjacency (neighbour mean)."""
    a, b = edges[:, 0], edges[:, 1]
    A = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return diags(1.0 / deg) @ A


def hidden_field(vertices, faces, rays=16, distance=0.08, max_samples=20000, seed=0,
                 progress=None):
    """Per-vertex hidden score in [0, 1] (1 = best place for a seam).

    ``distance`` is the occlusion reach as a fraction of the bounding-box
    diagonal. ``max_samples`` caps how many vertices are ray-cast (the rest
    are interpolated); without embreex it is reduced further, since pure-numpy
    ray casting is orders of magnitude slower.
    """
    import trimesh

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(vertices)
    if n == 0 or len(faces) == 0:
        return np.zeros(n)
    normals = _vertex_normals(vertices, faces)
    edges = _edges(faces)
    avg = _averaging_operator(n, edges)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0

    # ---- concavity: neighbours rising above the tangent plane --------------
    a, b = edges[:, 0], edges[:, 1]
    d = vertices[b] - vertices[a]
    ln = np.linalg.norm(d, axis=1)
    ln[ln == 0] = 1.0
    d /= ln[:, None]
    ca = np.einsum("ij,ij->i", normals[a], d)       # b above a's plane -> > 0
    cb = -np.einsum("ij,ij->i", normals[b], d)      # a above b's plane -> > 0
    acc = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(acc, a, ca)
    np.add.at(acc, b, cb)
    np.add.at(cnt, a, 1)
    np.add.at(cnt, b, 1)
    concave = np.clip(3.0 * acc / np.maximum(cnt, 1), 0.0, 1.0)

    # ---- occlusion: short hemisphere rays -----------------------------------
    occlusion = np.zeros(n)
    if rays > 0 and distance > 0:
        cap = max_samples if HAS_EMBREE else min(max_samples, 1500)
        rng = np.random.default_rng(seed)
        sample = np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))
        nrm = normals[sample]
        # tangent frame around each normal
        ref = np.where(np.abs(nrm[:, [0]]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        t1 = np.cross(nrm, ref)
        t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
        t2 = np.cross(nrm, t1)
        hemi = _fib_hemisphere(rays)                          # (R, 3)
        dirs = (hemi[None, :, 0:1] * t1[:, None] + hemi[None, :, 1:2] * t2[:, None]
                + hemi[None, :, 2:3] * nrm[:, None])          # (S, R, 3)
        origins = vertices[sample] + nrm * (1e-4 * diag)
        o = np.repeat(origins, rays, axis=0)
        dr = dirs.reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        reach = distance * diag
        hits = np.zeros(len(o), dtype=bool)
        step = 400_000
        for s in range(0, len(o), step):
            loc, idx, _ = mesh.ray.intersects_location(o[s:s + step], dr[s:s + step],
                                                       multiple_hits=False)
            if len(idx):
                dist = np.linalg.norm(loc - o[s:s + step][idx], axis=1)
                hits[s + idx[dist <= reach]] = True
            if progress is not None:
                progress(min(1.0, (s + step) / len(o)))
        occ = hits.reshape(-1, rays).mean(axis=1)
        known = np.zeros(n, dtype=bool)
        known[sample] = True
        field = np.zeros(n)
        field[sample] = occ
        if not known.all():
            # fill unsampled vertices by diffusion with sampled values pinned
            field[~known] = occ.mean()
            for _ in range(30):
                field = np.where(known, field, avg @ field)
        occlusion = np.clip((field - 0.15) / 0.55, 0.0, 1.0)

    hidden = np.maximum(concave, occlusion)
    for _ in range(2):
        hidden = 0.5 * hidden + 0.5 * (avg @ hidden)
    p85 = float(np.percentile(hidden, 85))
    if p85 > 1e-6:
        hidden = hidden / p85
    return np.clip(hidden, 0.0, 1.0)


def vertex_weight(hidden, strength):
    """Per-vertex seam cost multiplier (1 where hidden, steep where visible)."""
    vis = 1.0 - np.asarray(hidden, dtype=np.float64)
    return 1.0 + strength * (8.0 * vis ** 2 + 60.0 * vis ** 6)
