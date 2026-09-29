"""Chart topology: make every chart one manifold disk before it is flattened.

LSCM, ARAP and every other flattener assume a *disk*: one connected piece with
exactly one boundary loop and no handles. Segmentation does not promise that.
A chart is just a set of faces, so it can be

* **pinched** -- two fans that only touch at a vertex (a bowtie), or faces
  joined across an edge shared by three or more faces (doubled sheets, fins);
* **in several pieces** once those pinches are taken apart;
* **an annulus** -- a flat plate with a hole in it;
* **closed or handled** -- a tube, a torus, a whole sphere once the cone cap is
  raised high enough.

Before this module the flattener only asked "does the chart have at least two
boundary vertices", so a tube went to LSCM and folded, and a closed chart fell
through to a planar projection that overlapped itself.

Here vertex identity is first re-derived from the chart's own faces
(:func:`split_corners`): corners are the same vertex only when a chain of
glued edges joins their faces, and an edge shared by more than two faces glues
only its most coplanar, opposite-winding pair. That makes the chart manifold by
construction. Then :func:`disk_cuts` computes the cheapest set of edges to cut
so the chart becomes a disk:

* a closed sphere-like chart gets one slit between its two farthest points;
* a disk with holes gets each hole joined to the rest by the cheapest path
  (grown Prim-style from the longest boundary, like a Steiner tree);
* anything with handles gets the classic cut graph -- a maximum spanning tree
  of the dual graph keeps the expensive edges glued, and leaf-pruning the rest
  leaves just the loops that must be cut.

"Cheap" is edge length weighted by an optional per-vertex visibility weight and
discounted on sharp edges, so cuts land on creases and in hidden places.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree


# ------------------------------------------------------------------ helpers
def _half_edges(lf):
    a = lf.reshape(-1)
    b = lf[:, [1, 2, 0]].reshape(-1)
    return a, b


def _edge_keys(a, b, n):
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return lo * np.int64(n) + hi, lo, hi


def face_normals(lv, lf):
    v = lv[lf]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return fn / ln


def _groups(key):
    """Sort ``key``; return (order, group starts, group counts)."""
    order = np.argsort(key, kind="stable")
    ks = key[order]
    if len(ks) == 0:
        return order, np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1]
    counts = np.diff(np.r_[starts, len(ks)])
    return order, starts, counts


# ------------------------------------------------------------ corner split
def split_corners(lf, n_verts, normals=None, cut_keys=None, return_unpaired=False):
    """Re-derive vertex identity from which edges actually glue faces together.

    ``lf`` is (F, 3) local faces over ``n_verts`` vertices. ``cut_keys`` is an
    optional array of edge keys (``lo * n_verts + hi`` over the *input* vertex
    ids) that must stay open. Returns ``(new_lf, orig, face_comp, n_comp)``:
    ``new_lf`` indexes new vertices, ``orig[new]`` is the input vertex a new
    vertex was copied from, and ``face_comp`` labels the edge-connected pieces.
    With ``return_unpaired`` a fifth value lists the faces left unglued on an
    edge shared by three or more faces (fins, doubled sheets).
    """
    lf = np.asarray(lf, dtype=np.int64)
    F = len(lf)
    a, b = _half_edges(lf)
    key, _, _ = _edge_keys(a, b, n_verts)
    order, starts, counts = _groups(key)

    two = starts[counts == 2]
    h1 = order[two]
    h2 = order[two + 1]

    # Edges shared by 3+ faces: glue the most coplanar opposite-winding pairs,
    # each half-edge at most once. Rare, so a plain loop is fine.
    many = np.nonzero(counts > 2)[0]
    unpaired = []
    if len(many):
        if normals is None:
            normals = np.zeros((F, 3))
        x1, x2 = [], []
        for g in many:
            hs = order[starts[g]:starts[g] + counts[g]]
            cand = []
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    hi_, hj = hs[i], hs[j]
                    if a[hi_] == b[hj]:                       # opposite winding
                        d = float(normals[hi_ // 3] @ normals[hj // 3])
                        cand.append((-d, hi_, hj))
            cand.sort()
            used = set()
            for _, hi_, hj in cand:
                if hi_ in used or hj in used:
                    continue
                used.add(hi_)
                used.add(hj)
                x1.append(hi_)
                x2.append(hj)
            unpaired.extend(int(h) // 3 for h in hs if h not in used)
        if x1:
            h1 = np.concatenate([h1, np.asarray(x1, np.int64)])
            h2 = np.concatenate([h2, np.asarray(x2, np.int64)])

    if cut_keys is not None and len(cut_keys) and len(h1):
        keep = ~np.isin(key[h1], np.asarray(cut_keys, dtype=np.int64))
        h1, h2 = h1[keep], h2[keep]

    # corner ids: start corner of half-edge h is h, end corner is the next slot
    def end(h):
        return (h // 3) * 3 + (h % 3 + 1) % 3

    same_dir = a[h1] == a[h2]
    s1, e1, s2, e2 = h1, end(h1), h2, end(h2)
    ra = np.concatenate([s1, e1])
    rb = np.concatenate([np.where(same_dir, s2, e2), np.where(same_dir, e2, s2)])
    n_c = 3 * F
    g = coo_matrix((np.ones(len(ra)), (ra, rb)), shape=(n_c, n_c))
    _, lab = connected_components(g, directed=False)
    _, first, new_ids = np.unique(lab, return_index=True, return_inverse=True)
    new_ids = np.asarray(new_ids).ravel()
    new_lf = new_ids.reshape(F, 3)
    orig = lf.reshape(-1)[first]

    fg = coo_matrix((np.ones(len(h1)), (h1 // 3, h2 // 3)), shape=(F, F))
    n_comp, face_comp = connected_components(fg, directed=False)
    if return_unpaired:
        return new_lf, orig, face_comp, int(n_comp), np.unique(np.asarray(unpaired, np.int64))
    return new_lf, orig, face_comp, int(n_comp)


# ---------------------------------------------------------------- topology
@dataclass
class Topology:
    n_verts: int
    n_edges: int
    n_faces: int
    euler: int
    loops: list            # list of boundary-loop vertex arrays
    edge_lo: np.ndarray    # (E,) unique edge endpoints
    edge_hi: np.ndarray
    edge_count: np.ndarray  # (E,) faces per edge
    edge_faces: np.ndarray  # (E, 2) the two faces of an interior edge, -1 otherwise

    @property
    def n_loops(self) -> int:
        return len(self.loops)

    @property
    def genus(self) -> int:
        return (2 - self.n_loops - self.euler) // 2

    @property
    def is_disk(self) -> bool:
        return self.euler == 1 and self.n_loops == 1


def topology(lf, n_verts):
    """Euler characteristic, boundary loops and edge table of a manifold chart."""
    lf = np.asarray(lf, dtype=np.int64)
    F = len(lf)
    a, b = _half_edges(lf)
    key, lo, hi = _edge_keys(a, b, n_verts)
    order, starts, counts = _groups(key)
    e_lo = lo[order][starts]
    e_hi = hi[order][starts]
    faces2 = np.full((len(starts), 2), -1, dtype=np.int64)
    two = counts == 2
    faces2[two, 0] = order[starts[two]] // 3
    faces2[two, 1] = order[starts[two] + 1] // 3

    bnd = counts == 1
    loops = []
    if bnd.any():
        bl, bh = e_lo[bnd], e_hi[bnd]
        g = coo_matrix((np.ones(len(bl)), (bl, bh)), shape=(n_verts, n_verts))
        _, lab = connected_components(g, directed=False)
        verts = np.unique(np.concatenate([bl, bh]))
        vl = lab[verts]
        o = np.argsort(vl, kind="stable")
        verts, vl = verts[o], vl[o]
        cut = np.nonzero(vl[1:] != vl[:-1])[0] + 1
        loops = np.split(verts, cut)
    E = len(starts)
    return Topology(n_verts, E, F, n_verts - E + F, loops, e_lo, e_hi, counts, faces2)


# --------------------------------------------------------------- disk cuts
def edge_costs(lv, lf, topo, vertex_weight=None, sharp_discount=2.0):
    """Cost of cutting each edge: length x visibility weight / sharpness bonus."""
    length = np.linalg.norm(lv[topo.edge_hi] - lv[topo.edge_lo], axis=1)
    w = np.ones(len(length))
    if vertex_weight is not None:
        w = 0.5 * (vertex_weight[topo.edge_lo] + vertex_weight[topo.edge_hi])
    cost = length * w
    inner = topo.edge_faces[:, 0] >= 0
    if inner.any() and sharp_discount > 0:
        fn = face_normals(lv, lf)
        f1, f2 = topo.edge_faces[inner, 0], topo.edge_faces[inner, 1]
        dih = np.arccos(np.clip(np.einsum("ij,ij->i", fn[f1], fn[f2]), -1.0, 1.0))
        cost[inner] /= 1.0 + sharp_discount * np.clip(dih / (np.pi / 3), 0.0, 1.0)
    return np.maximum(cost, 1e-12)


def _vertex_graph(topo, cost):
    n = topo.n_verts
    g = coo_matrix((cost, (topo.edge_lo, topo.edge_hi)), shape=(n, n)).tocsr()
    return (g + g.T).tocsr()


def _path(pred, end):
    out = [int(end)]
    while pred[out[-1]] >= 0:
        out.append(int(pred[out[-1]]))
    return out


def _path_keys(path, n):
    p = np.asarray(path, dtype=np.int64)
    if len(p) < 2:
        return np.zeros(0, np.int64)
    k, _, _ = _edge_keys(p[:-1], p[1:], n)
    return k


def _slit(topo, G):
    """One cut between the two points farthest apart (double sweep)."""
    d0 = dijkstra(G, indices=0)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))
    da, pred = dijkstra(G, indices=a, return_predecessors=True)
    da[~np.isfinite(da)] = -1
    b = int(np.argmax(da))
    return _path_keys(_path(pred, b), topo.n_verts)


def _join_loops(topo, G):
    """Join every boundary loop to the longest one through the cheapest paths."""
    n = topo.n_verts
    loops = sorted(topo.loops, key=len, reverse=True)
    loop_of = np.full(n, -1, dtype=np.int64)
    for i, lp in enumerate(loops):
        loop_of[lp] = i
    joined = np.zeros(len(loops), dtype=bool)
    joined[0] = True
    tree = list(loops[0])
    keys = []
    while not joined.all():
        dist, pred, _ = dijkstra(G, indices=np.asarray(tree), return_predecessors=True,
                                 min_only=True)
        open_v = np.nonzero((loop_of >= 0) & ~joined[np.maximum(loop_of, 0)])[0]
        open_v = open_v[np.isfinite(dist[open_v])]
        if len(open_v) == 0:
            break
        tgt = open_v[np.argmin(dist[open_v])]
        path = _path(pred, tgt)
        keys.append(_path_keys(path, n))
        li = int(loop_of[tgt])
        joined[li] = True
        tree.extend(path)
        tree.extend(loops[li].tolist())
    return np.concatenate(keys) if keys else np.zeros(0, np.int64)


def _cut_graph(topo, cost):
    """Dual maximum spanning tree + leaf pruning: the loops a handle needs cut."""
    n = topo.n_verts
    inner = np.nonzero(topo.edge_faces[:, 0] >= 0)[0]
    f1 = topo.edge_faces[inner, 0]
    f2 = topo.edge_faces[inner, 1]
    lo_f, hi_f = np.minimum(f1, f2), np.maximum(f1, f2)
    # the dual may carry two edges between the same face pair; keep the first
    pk = lo_f * np.int64(topo.n_faces) + hi_f
    _, first = np.unique(pk, return_index=True)
    c = cost[inner[first]]
    wgt = (c.max() - c) + 1e-9 * (c.max() + 1.0)      # max tree -> min of complement
    G = coo_matrix((wgt, (lo_f[first], hi_f[first])),
                   shape=(topo.n_faces, topo.n_faces)).tocsr()
    T = minimum_spanning_tree(G).tocoo()
    tree_pk = set((np.minimum(T.row, T.col) * np.int64(topo.n_faces)
                   + np.maximum(T.row, T.col)).tolist())
    in_tree = np.zeros(len(inner), dtype=bool)
    in_tree[first] = np.fromiter((int(k) in tree_pk for k in pk[first]), bool, len(first))
    cand = inner[~in_tree]

    # prune dangling cut edges; boundary edges are fixed anchors
    deg = np.zeros(n, dtype=np.int64)
    bnd = topo.edge_count == 1
    np.add.at(deg, topo.edge_lo[bnd], 1)
    np.add.at(deg, topo.edge_hi[bnd], 1)
    np.add.at(deg, topo.edge_lo[cand], 1)
    np.add.at(deg, topo.edge_hi[cand], 1)
    inc: dict[int, set] = {}
    for e in cand.tolist():
        inc.setdefault(int(topo.edge_lo[e]), set()).add(e)
        inc.setdefault(int(topo.edge_hi[e]), set()).add(e)
    alive = set(cand.tolist())
    stack = [v for v in inc if deg[v] == 1]
    while stack:
        v = stack.pop()
        if deg[v] != 1:
            continue
        es = [e for e in inc.get(v, ()) if e in alive]
        if not es:
            continue
        e = es[0]
        alive.discard(e)
        for u in (int(topo.edge_lo[e]), int(topo.edge_hi[e])):
            deg[u] -= 1
            inc[u].discard(e)
            if deg[u] == 1:
                stack.append(u)
    kept = np.fromiter(alive, np.int64, len(alive))
    k, _, _ = _edge_keys(topo.edge_lo[kept], topo.edge_hi[kept], n)
    return k


def disk_cuts(lv, lf, topo, vertex_weight=None, holes=True):
    """Edge keys to cut so the chart becomes a disk (empty if it already is).

    Returns ``(keys, kind)`` where ``kind`` names what was done: "disk" (no
    cut), "slit", "holes", "handles", or "skip" (holes present but ``holes``
    off).
    """
    if topo.is_disk:
        return np.zeros(0, np.int64), "disk"
    cost = edge_costs(lv, lf, topo, vertex_weight)
    G = _vertex_graph(topo, cost)
    if topo.genus <= 0 and (2 - topo.n_loops - topo.euler) % 2 == 0:
        if topo.n_loops == 0:
            return _slit(topo, G), "slit"
        if not holes:
            return np.zeros(0, np.int64), "skip"
        return _join_loops(topo, G), "holes"
    keys = _cut_graph(topo, cost)
    if topo.n_loops == 0 and len(keys) == 0:
        return _slit(topo, G), "slit"
    return keys, "handles"


def make_disk(lv, lf, vertex_weight=None, holes=True, max_rounds=3):
    """Cut a manifold, connected chart until it is a disk.

    Returns ``(lf, orig, kind, n_cut_edges)`` -- new faces over new vertices,
    the map new vertex -> input vertex, what kind of cut was made, and how
    many edges were opened. Gives up (returns the input unchanged, kind
    "failed") if it cannot reach a disk in ``max_rounds``.
    """
    orig_total = np.arange(len(lv))
    cur_lf = lf
    cur_lv = lv
    cur_w = vertex_weight
    kinds = []
    n_cut = 0
    for _ in range(max_rounds):
        topo = topology(cur_lf, len(cur_lv))
        if topo.is_disk:
            break
        keys, kind = disk_cuts(cur_lv, cur_lf, topo, cur_w, holes=holes)
        if kind == "skip":
            return lf, np.arange(len(lv)), "skip", 0
        if len(keys) == 0:
            return lf, np.arange(len(lv)), "failed", 0
        kinds.append(kind)
        n_cut += len(np.unique(keys))
        new_lf, orig, _, _ = split_corners(cur_lf, len(cur_lv), face_normals(cur_lv, cur_lf),
                                           cut_keys=keys)
        cur_lf = new_lf
        cur_lv = cur_lv[orig]
        cur_w = cur_w[orig] if cur_w is not None else None
        orig_total = orig_total[orig]
    else:
        if not topology(cur_lf, len(cur_lv)).is_disk:
            return lf, np.arange(len(lv)), "failed", 0
    if not kinds:
        return lf, np.arange(len(lv)), "disk", 0
    return cur_lf, orig_total, "+".join(dict.fromkeys(kinds)), n_cut
