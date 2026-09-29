"""Move chart borders onto the cheapest nearby seam: hidden, sharp and short.

Segmentation decides *which* charts exist; where exactly two neighbouring
charts meet is mostly an accident of the greedy merge order. The border it
leaves is one triangle wide, jagged, and indifferent to whether it runs
across a cheek or down a crease. Weighting the merge order does not fix that:
a chart stops growing where its normal cone fills up, whatever order the
merges came in (measured: a hidden-surface term in the merge cost moved the
share of seam length in hidden places by less than 1%).

So the border is optimised directly. For every pair of adjacent charts
``A | B``:

1. take a *band* of faces within ``rings`` face-rings of their shared border;
2. tie the rest of A to a source and the rest of B to a sink;
3. give every dual edge (a mesh edge between two band faces) a capacity equal
   to the cost of putting the seam there -- length x visibility weight,
   discounted on sharp edges;
4. the minimum s-t cut is the cheapest closed border through the band.

A cut prefers hidden and creased edges and, all else equal, the *shortest*
route, so it also straightens the staircase the greedy merge left behind. The
move is kept only when ``validate`` says both charts still flatten at least as
well as before, so the border cannot drift into a layout that stretches or
folds.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


def seam_costs(mesh, vertex_weight=None, sharp_discount=2.0):
    """Cost of a seam along each ``mesh.adjacency`` edge."""
    ev = mesh.adjacency_edges
    length = np.linalg.norm(mesh.vertices[ev[:, 1]] - mesh.vertices[ev[:, 0]], axis=1)
    cost = length.copy()
    if vertex_weight is not None:
        cost *= 0.5 * (vertex_weight[ev[:, 0]] + vertex_weight[ev[:, 1]])
    if sharp_discount > 0:
        cost /= 1.0 + sharp_discount * np.clip(mesh.adjacency_angle / (np.pi / 3), 0.0, 1.0)
    return np.maximum(cost, 1e-12)


def _min_cut(band, core_a, core_b, adj, cost, scale):
    """Source-side band faces of the min cut between ``core_a`` and ``core_b``."""
    n = len(band)
    S, T = n, n + 1
    fa, fb = adj[:, 0], adj[:, 1]
    local = np.full(len(core_a), -1, dtype=np.int64)
    local[band] = np.arange(n)
    in_band = local >= 0
    cap = np.clip(np.round(cost / scale), 1, 10**6).astype(np.int64)

    us, vs, cs = [], [], []
    both = in_band[fa] & in_band[fb]
    us.append(local[fa[both]])
    vs.append(local[fb[both]])
    cs.append(cap[both])
    for core, term in ((core_a, S), (core_b, T)):
        t1 = in_band[fa] & core[fb]
        t2 = in_band[fb] & core[fa]
        f = np.r_[fa[t1], fb[t2]]
        us.append(local[f])
        vs.append(np.full(len(f), term, dtype=np.int64))
        cs.append(np.r_[cap[t1], cap[t2]])
    u = np.concatenate(us)
    v = np.concatenate(vs)
    c = np.concatenate(cs)
    if len(u) == 0:
        return None
    rows = np.r_[u, v]
    cols = np.r_[v, u]
    C = coo_matrix((np.r_[c, c].astype(np.int32), (rows, cols)), shape=(n + 2, n + 2)).tocsr()
    C.sum_duplicates()
    res = maximum_flow(C, S, T)
    R = (C - res.flow).tocsr()
    R.data[R.data < 0] = 0
    R.eliminate_zeros()
    reach = breadth_first_order(R, S, directed=True, return_predecessors=False)
    side = np.zeros(n + 2, dtype=bool)
    side[reach] = True
    return side[:n]


def refine_borders(mesh, labels, cost, rings=4, validate=None, passes=1, progress=None):
    """Re-route every chart border along the cheapest route near it.

    ``cost`` is one seam cost per ``mesh.adjacency`` row (see
    :func:`seam_costs`). ``validate(old_a, old_b, new_a, new_b)`` receives
    face-id arrays and returns whether to keep a move. Returns
    ``(labels, stats)``.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    adj = mesh.adjacency
    F = mesh.n_faces
    stats = {"pairs": 0, "moved": 0, "rejected": 0, "faces_moved": 0}
    if len(adj) == 0:
        return labels, stats
    G = coo_matrix((np.ones(2 * len(adj)), (np.r_[adj[:, 0], adj[:, 1]],
                                           np.r_[adj[:, 1], adj[:, 0]])), shape=(F, F)).tocsr()
    G.data[:] = 1.0
    scale = float(np.median(cost)) / 100.0 or 1e-12

    for p in range(passes):
        la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
        diff = la != lb
        if not diff.any():
            break
        lo, hi = np.minimum(la[diff], lb[diff]), np.maximum(la[diff], lb[diff])
        key = lo * (int(labels.max()) + 1) + hi
        uk, inv = np.unique(key, return_inverse=True)
        blen = np.bincount(inv, weights=cost[diff])
        order = np.argsort(-blen)
        pairs = [(int(uk[i] // (labels.max() + 1)), int(uk[i] % (labels.max() + 1))) for i in order]
        stats["pairs"] += len(pairs)
        for k, (A, B) in enumerate(pairs):
            if progress is not None:
                progress((p + (k + 1) / len(pairs)) / passes)
            inA = labels == A
            inB = labels == B
            la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
            border = ((la == A) & (lb == B)) | ((la == B) & (lb == A))
            if not border.any():
                continue
            band = np.zeros(F, dtype=bool)
            band[adj[border, 0]] = True
            band[adj[border, 1]] = True
            ab = inA | inB
            for _ in range(rings):
                band |= (G @ band.astype(np.float64) > 0) & ab
            core_a = inA & ~band
            core_b = inB & ~band
            if not core_a.any() or not core_b.any():
                continue
            band_ids = np.nonzero(band)[0]
            side = _min_cut(band_ids, core_a, core_b, adj, cost, scale)
            if side is None:
                continue
            new_a_band = band_ids[side]
            new_b_band = band_ids[~side]
            moved = int((labels[new_a_band] != A).sum() + (labels[new_b_band] != B).sum())
            if moved == 0:
                continue
            old_a, old_b = np.nonzero(inA)[0], np.nonzero(inB)[0]
            new_a = np.sort(np.r_[np.nonzero(core_a)[0], new_a_band])
            new_b = np.sort(np.r_[np.nonzero(core_b)[0], new_b_band])
            if validate is not None and not validate(old_a, old_b, new_a, new_b):
                stats["rejected"] += 1
                continue
            labels[new_a] = A
            labels[new_b] = B
            stats["moved"] += 1
            stats["faces_moved"] += moved
    return labels, stats
