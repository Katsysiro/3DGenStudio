"""End-to-end unwrap orchestrator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh
from . import segment as _seg
from . import param as _param
from . import postprocess as _post
from . import pack as _pack
from . import weld as _weld
from . import hidden as _hidden
from . import metrics as _metrics
from . import topology as _topology
from . import border as _border


def _split_disconnected(mesh, labels):
    """Give every edge-connected, manifold piece of a chart its own label.

    Segmentation joins faces across any shared edge, including edges shared by
    three or more faces and vertices where two fans merely touch. Once those
    pinches are taken apart (:func:`topology.split_corners`) a chart can fall
    into pieces, and a flattener given several pieces at once has no way to
    place them relative to each other.

    Two degenerate shapes that no cut can open are also given one island per
    face: the faces left over on an edge shared by three or more faces (a fin,
    one layer of a doubled sheet), and tiny closed charts -- typically two
    copies of one triangle with opposite winding, a closed "pillow" whose only
    possible slit is a single edge, which opens nothing.
    """
    out = labels.copy()
    nxt = int(labels.max()) + 1
    order = np.argsort(labels, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(labels[order]))[0] + 1, len(labels)]
    for i in range(len(bounds) - 1):
        fids = order[bounds[i]:bounds[i + 1]]
        f = mesh.faces[fids]
        uniq, inv = np.unique(f.reshape(-1), return_inverse=True)
        lf = inv.reshape(-1, 3)
        lv = mesh.vertices[uniq]
        fn = _topology.face_normals(lv, lf)
        new_lf, orig, comp, n_comp, loose = _topology.split_corners(
            lf, len(lv), fn, return_unpaired=True)
        comp = comp.copy()
        single = np.zeros(len(fids), dtype=bool)
        if len(loose) and (_topology.topology(new_lf, len(orig)).edge_count > 2).any():
            single[loose] = True
            sub = np.nonzero(~single)[0]
            if len(sub):
                # re-derive the pieces of what is left once the loose faces go
                _, _, c2, _ = _topology.split_corners(lf[sub], len(lv), fn[sub])
                comp[sub] = c2
        # tiny closed pieces: no cut opens them, so one island per face
        for k in np.unique(comp[~single]):
            piece = np.nonzero((comp == k) & ~single)[0]
            if len(piece) > 8:
                continue
            p_lf, p_orig, _, _ = _topology.split_corners(lf[piece], len(lv), fn[piece])
            if _topology.topology(p_lf, len(p_orig)).n_loops == 0:
                single[piece] = True
        if single.any():
            comp[single] = int(comp.max()) + 1 + np.arange(int(single.sum()))
        _, comp = np.unique(comp, return_inverse=True)
        comp = np.asarray(comp).ravel()
        for k in range(1, int(comp.max()) + 1):
            out[fids[comp == k]] = nxt
            nxt += 1
    _, out = np.unique(out, return_inverse=True)
    return np.asarray(out).ravel().astype(np.int64)


def _fold_halves(mesh, fids, uv=None, lf=None, seeds="overlap"):
    """Split a folding chart in two along its sharpest crease.

    A chart that folds over itself in every flattening almost always spans a
    ridge or a pocket whose two sides face different ways. Two regions are
    grown over the chart's face graph from its two most opposed faces, and an
    edge costs more to grow across the sharper it is, so the regions meet on a
    crease. (Clustering the face normals instead shatters a noisy AI chart into
    a dozen speckled fragments.)

    With the chart's layout (``uv`` over local faces ``lf``, row-aligned with
    ``fids``) the seeds are the two faces that cover a common texel and lie
    farthest apart on the model -- one on each of the layers that overlap.
    With ``seeds="normals"`` (or no layout) the two most opposed faces are
    used instead; neither choice wins everywhere. Returns a list of face-id
    arrays, one per edge-connected piece, or ``[]`` when the split is
    degenerate.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    n = mesh.face_normals[fids]
    a = mesh.face_areas[fids]
    local = np.full(mesh.n_faces, -1, dtype=np.int64)
    local[fids] = np.arange(len(fids))
    adj = mesh.adjacency
    inside = (local[adj[:, 0]] >= 0) & (local[adj[:, 1]] >= 0)
    fa, fb = local[adj[inside, 0]], local[adj[inside, 1]]
    if len(fa) == 0:
        return []
    cen = mesh.vertices[mesh.faces[fids]].mean(axis=1)
    crease = np.clip(mesh.adjacency_angle[inside] / (np.pi / 2), 0.0, 1.0)
    w = np.linalg.norm(cen[fa] - cen[fb], axis=1) * (1.0 + 25.0 * crease ** 2) + 1e-12
    G = coo_matrix((w, (fa, fb)), shape=(len(fids), len(fids))).tocsr()
    s0 = s1 = -1
    if uv is not None and lf is not None and seeds == "overlap":
        pairs = _metrics.overlap_pairs(uv, lf)
        if len(pairs):
            d = np.linalg.norm(cen[pairs[:, 0]] - cen[pairs[:, 1]], axis=1)
            s0, s1 = (int(x) for x in pairs[int(np.argmax(d))])
    if s0 < 0:
        mean = (n * a[:, None]).sum(axis=0)
        mean /= max(np.linalg.norm(mean), 1e-12)
        s0 = int(np.argmin(n @ mean))
        s1 = int(np.argmin(n @ n[s0]))
    if s0 == s1:
        return []
    _, _, src = dijkstra(G, directed=False, indices=[s0, s1], min_only=True,
                         return_predecessors=True)
    lab = (src == s1).astype(np.int64)
    lab[src < 0] = 0
    if lab.min() == lab.max():
        return []
    pieces = []
    for k in (0, 1):
        sub = fids[lab == k]
        f = mesh.faces[sub]
        uniq, inv = np.unique(f.reshape(-1), return_inverse=True)
        lf = inv.reshape(-1, 3)
        lv = mesh.vertices[uniq]
        _, _, comp, n_comp = _topology.split_corners(lf, len(lv), _topology.face_normals(lv, lf))
        pieces.extend(sub[comp == c] for c in range(n_comp))
    return pieces


def _duplicate_faces(faces):
    """Split faces into keepers and exact duplicates (same three vertices).

    Returns ``(keep_ids, dup_ids, dup_keeper)``: the face ids to unwrap, the
    duplicate face ids, and for each duplicate the id of the face it copies.
    Duplicates are common in AI meshes (a triangle doubled with the opposite
    winding) and poison every chart they land in: they overlap by construction
    and glue edges into three- and four-face fans. They are left out of the
    unwrap and stacked back onto their twin's UVs at the end -- they occupy the
    same place in 3D, so sharing the twin's texels is exactly right.
    """
    key = np.sort(faces, axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    keeper = first[np.asarray(inv).ravel()]
    ids = np.arange(len(faces))
    dup = ids != keeper
    return ids[~dup], ids[dup], keeper[dup]


def _corner_normals(mesh, pre_weld_faces, weld_info, source_normals,
                    preserve_normals, normal_smooth_deg):
    """Resolve one normal per face corner, plus a label saying where it came from.

    Returns ``(corner_group, group_normal, source)``: ``corner_group`` is (F, 3)
    group ids per face corner and ``group_normal`` is (G, 3) unit normals, so a
    corner's normal is ``group_normal[corner_group[f, k]]``. Corners sharing a
    group id are interchangeable, which is what the assembly step keys on.

    **Preferring the input's own normals is the whole point.** Auto UV changes the
    UV layout, not the shape, so it must not change shading either -- and no angle
    heuristic can reconstruct what an artist (or a decimator) authored. Decimated
    organic meshes are the case that proves it: a simplified head measures a
    90th-percentile dihedral around 55 degrees over a surface that is genuinely
    smooth, so any fixed smoothing angle in the usual range shatters it into
    facets while the untouched original renders smooth.

    Carrying them through is exact rather than approximate: the weld maps say
    which input vertex sits behind each welded face corner, so a corner keeps its
    input normal unchanged. Corners are then grouped by normal *value*, which is a
    property of the input geometry alone -- charts never enter into it, so the
    result is automatically continuous across a UV seam, while an authored hard
    edge stays hard because its two sides carry different values.

    Falls back to `Mesh.corner_groups` (dihedral-angle smoothing groups) when the
    input carried no normals, or when the caller asks for a recompute.
    """
    if preserve_normals and source_normals is not None:
        cn = _input_corner_normals(mesh, pre_weld_faces, weld_info, source_normals)
        if cn is not None:
            # Group by normal value. Quantising to a fine grid folds together
            # copies that are equal bar float noise; a pair that straddles a
            # bucket edge merely costs one extra output vertex carrying its own
            # near-identical normal, so the failure mode is harmless.
            flat = cn.reshape(-1, 3)
            q = np.round(flat * 10000.0).astype(np.int64)
            _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
            inv = np.asarray(inv).ravel()
            return inv.reshape(-1, 3), flat[first], "input"

    return (*mesh.corner_groups(normal_smooth_deg), "recomputed")


def _input_corner_normals(mesh, pre_weld_faces, weld_info, source_normals):
    """(F, 3, 3) unit normals per welded face corner, taken from the input mesh.

    None only when the array cannot be lined up with the geometry at all (wrong
    shape, or an index past the end), or when so much of it is degenerate that it
    is not worth trusting -- the caller then falls back to recomputing.

    A handful of zero-length input normals is normal and NOT a reason to discard
    the rest: any vertex whose incident face normals happen to cancel comes out
    zero, and real meshes have a few. Those corners alone fall back to their own
    face normal. (Bailing globally on the first bad corner is a trap worth
    naming: on the reported head mesh exactly 4 corners out of 16836 were
    degenerate, which was enough to throw away every good normal in the file.)
    """
    src = np.asarray(source_normals, dtype=np.float64)
    faces = np.asarray(pre_weld_faces)
    if src.ndim != 2 or src.shape[1] != 3:
        return None

    # Which input vertex is behind each welded face corner. Welding relabels and
    # drops faces but never reorders their corners, so face_index is enough.
    corner_vert = faces[weld_info["face_index"]] if weld_info is not None else faces
    if corner_vert.shape != mesh.faces.shape:
        return None
    if corner_vert.size == 0 or int(corner_vert.max()) >= len(src):
        return None

    cn = src[corner_vert]
    ln = np.linalg.norm(cn, axis=2, keepdims=True)
    bad = ln[..., 0] <= 1e-12
    if bad.mean() > 0.5:  # the channel is junk, not merely imperfect
        return None
    if bad.any():
        # Substitute the corner's own face normal, the best local estimate.
        fn = mesh.face_normals[np.nonzero(bad)[0]]
        cn = cn.copy()
        cn[bad] = fn
        ln = np.linalg.norm(cn, axis=2, keepdims=True)
        ln[ln <= 1e-12] = 1.0
    return cn / ln


@dataclass
class UnwrapResult:
    vertices: np.ndarray          # (Vn,3) output positions (seam-duplicated)
    faces: np.ndarray             # (F,3) indices into vertices
    uv: np.ndarray                # (Vn,2) in [0,1]
    face_chart: np.ndarray        # (F,) chart id per face
    normals: np.ndarray = None    # (Vn,3) normals sampled before the seam split
    stats: dict = field(default_factory=dict)


def unwrap(
    mesh: Mesh,
    max_cone_deg: float = 50.0,
    sharp_weight: float = 0.35,
    min_faces: int = 20,
    min_area_frac: float = 0.004,
    fold_cap_deg: float = 88.0,
    refine: bool = True,
    refine_target_faces: int = 80,
    refine_ad_thresh: float = 1.32,
    resolution: int = 1024,
    padding_texels: int = 4,
    method: str = "auto",
    arap_iters: int = 4,
    weld: bool = True,
    weld_tol_frac: float = 0.1,
    normal_smooth_deg: float = 180.0,
    source_normals=None,
    preserve_normals: bool = True,
    hide_seams: bool = True,
    hide_strength: float = 1.0,
    refine_borders: bool = True,
    border_rings: int = 4,
    ensure_disks: bool = True,
    progress=None,
    verbose: bool = True,
) -> UnwrapResult:
    t0 = time.time()

    def report(stage, frac):
        if progress is not None:
            progress(stage, float(frac))

    # ---- topology repair: weld coincident-but-unshared vertices -------------
    # The hard floor on chart count is the number of connected components, so a
    # mesh shattered into hundreds of shells (AI/scan output) is forced to
    # hundreds of charts regardless of how good the parameterisation is. Welding
    # by true distance stitches those shells back into a few components; on
    # already-clean meshes it is a no-op. See autouv.weld.
    comps_before = int(len(np.unique(mesh.components)))
    weld_info = None
    pre_weld_faces = mesh.faces          # needed to trace corners back to input verts
    welded = False
    report("weld", 0.0)
    if weld:
        nv, nf, weld_info = _weld.proximity_weld(
            mesh.vertices, mesh.faces, tol_frac=weld_tol_frac
        )
        if weld_info["verts_after"] != weld_info["verts_before"]:
            mesh = Mesh(nv, nf)
            welded = True
    comps_after = int(len(np.unique(mesh.components)))
    if verbose and weld:
        print(f"[weld] {comps_before} -> {comps_after} components "
              f"({weld_info['verts_before']}->{weld_info['verts_after']} verts, "
              f"tol={weld_info['tol']:.5f})")
    report("weld", 1.0)

    full_mesh = mesh
    keep_ids, dup_ids, dup_keeper = _duplicate_faces(mesh.faces)
    if len(dup_ids):
        mesh = Mesh(full_mesh.vertices, full_mesh.faces[keep_ids])

    # ---- where seams are least visible -------------------------------------
    # One score per vertex (see autouv.hidden). It prices every seam: the chart
    # borders (border.refine_borders) and any cut made inside a chart
    # (topology.make_disk) both route toward hidden places.
    t_h0 = time.time()
    hidden = None
    vertex_weight = None
    if hide_seams and hide_strength > 0:
        report("hidden", 0.0)
        hidden = _hidden.hidden_field(mesh.vertices, mesh.faces,
                                      progress=lambda p: report("hidden", p))
        vertex_weight = _hidden.vertex_weight(hidden, hide_strength)
        report("hidden", 1.0)
    t_hidden = time.time() - t_h0
    if verbose and hidden is not None:
        print(f"[hidden] {t_hidden:.2f}s (mean {hidden.mean():.2f})")

    labels = _seg.segment(
        mesh,
        max_cone_deg=max_cone_deg,
        sharp_weight=sharp_weight,
        min_faces=min_faces,
        min_area_frac=min_area_frac,
        fold_cap_deg=fold_cap_deg,
    )
    if verbose:
        print(f"[segment] {int(labels.max()) + 1} charts in "
              f"{time.time() - t0:.2f}s")
    report("segment", 1.0)
    if refine:
        labels = _seg.refine_merge(
            mesh, labels,
            target_faces=refine_target_faces,
            ad_thresh=refine_ad_thresh,
            progress=lambda p: report("refine", p),
            method=method,
            arap_iters=arap_iters,
            vertex_weight=vertex_weight,
            make_disk=ensure_disks,
        )

    # ---- move each chart border onto the cheapest seam near it --------------
    t_b0 = time.time()
    border_stats = None
    if refine_borders and border_rings > 0 and int(labels.max()) > 0:
        score_cache: dict = {}
        areas = mesh.face_areas

        def chart_score(fids):
            key = fids.tobytes()
            hit = score_cache.get(key)
            if hit is None:
                _, _, _, info = _param.parameterize_chart(
                    mesh.vertices, mesh.faces, fids, method=method, arap_iters=arap_iters,
                    vertex_weight=vertex_weight, make_disk=ensure_disks, fast=True)
                hit = (info["flips"], info["overlap"],
                       info["angle_d"] + 2.0 * info["area_d"], float(areas[fids].sum()))
                score_cache[key] = hit
            return hit

        def keep_move(old_a, old_b, new_a, new_b):
            old = [chart_score(old_a), chart_score(old_b)]
            new = [chart_score(new_a), chart_score(new_b)]
            if sum(n[0] for n in new) > sum(o[0] for o in old):
                return False
            worst_old = max(o[1] for o in old)
            if any(n[1] > max(_param.OVERLAP_TOL, worst_old) for n in new):
                return False
            d_old = sum(o[2] * o[3] for o in old) / max(sum(o[3] for o in old), 1e-12)
            d_new = sum(n[2] * n[3] for n in new) / max(sum(n[3] for n in new), 1e-12)
            return d_new <= d_old * 1.01 + 1e-3

        labels, border_stats = _border.refine_borders(
            mesh, labels, _border.seam_costs(mesh, vertex_weight),
            rings=int(border_rings), validate=keep_move,
            progress=lambda p: report("borders", p))
        if verbose:
            print(f"[borders] {border_stats}")
    t_border = time.time()

    labels = _split_disconnected(mesh, labels)
    n_charts = int(labels.max()) + 1
    t_seg = time.time()
    if verbose and refine:
        print(f"[refine]  -> {n_charts} charts (total seg {t_seg - t0:.2f}s)")

    faces = mesh.faces
    verts = mesh.vertices
    # Per-face-corner normals, resolved on the welded topology -- i.e. *before* the
    # seam split below -- so both copies of a seam vertex get the same value and
    # the two sides of every chart boundary shade identically.
    # Resolved on the full welded mesh (duplicates included) so the weld maps
    # still line up; charts index it through keep_ids.
    corner_group, group_normal, normal_source = _corner_normals(
        full_mesh, pre_weld_faces, weld_info if welded else None,
        source_normals, preserve_normals, normal_smooth_deg,
    )

    islands = []          # per-chart uv (local)
    island_uniq = []      # per-chart global vertex ids
    island_faces = []     # per-chart local faces
    island_fids = []      # per-chart global face ids (row-aligned to island_faces)
    island_area3d = []    # per-chart surface area
    angle_ds, area_ds, flips, methods = [], [], [], []
    cut_kinds, cut_edges, not_disk = [], 0, 0

    def flatten(fids):
        res = _param.parameterize_chart(
            verts, faces, fids, method=method, arap_iters=arap_iters,
            vertex_weight=vertex_weight, make_disk=ensure_disks,
        )
        # Candidates are ranked on a coarse 64-texel raster, which misses thin
        # folded slivers; measure the chosen layout properly before deciding
        # whether it needs splitting.
        uv, _, lf, info = res
        if info["overlap"] > 0 or info["flips"] > 0 or _metrics.boundary_crosses(uv, lf):
            info["overlap"] = _metrics.island_overlap(uv, lf, resolution=256)
        return res

    def flatten_unfolded(fids, depth=0):
        """Flatten a chart; split it by normals while it folds over itself.

        A split is kept only when it lowers the folded area -- a chart whose
        halves fold just as badly stays whole, since more seams would buy
        nothing.
        """
        res = flatten(fids)
        info = res[3]
        if info["overlap"] <= _param.OVERLAP_TOL or depth >= 3 or len(fids) < 16:
            return [(fids, res)]
        area = mesh.face_areas
        before = info["overlap"] * float(area[fids].sum())
        best = None
        for seeds in ("overlap", "normals"):
            pieces = _fold_halves(mesh, fids, res[0], res[2], seeds=seeds)
            if len(pieces) < 2 or len(pieces) > 6:
                continue
            parts = [(p, flatten(p)) for p in pieces]
            if sum(r[3]["flips"] for _, r in parts) > info["flips"]:
                continue
            after = sum(r[3]["overlap"] * float(area[p].sum()) for p, r in parts)
            if best is None or after < best[0]:
                best = (after, parts)
        if best is None or best[0] >= 0.8 * before:
            return [(fids, res)]
        parts = best[1]
        out = []
        for p, r in parts:
            out.extend(flatten_unfolded(p, depth + 1) if r[3]["overlap"] > _param.OVERLAP_TOL
                       else [(p, r)])
        return out

    work = []
    for c in range(n_charts):
        work.extend(flatten_unfolded(np.nonzero(labels == c)[0]))
        report("parameterize", (c + 1) / n_charts)
    n_folds_split = len(work) - n_charts
    n_charts = len(work)

    for c, (fids, (uv, uniq, lf, info)) in enumerate(work):
        if info["cut"] in ("failed", "raw"):
            not_disk += 1
        elif info["cut"] not in ("disk", "none"):
            cut_kinds.append(info["cut"])
            cut_edges += info["cut_edges"]
        uv = _post.align_island(uv)
        islands.append(uv)
        island_uniq.append(uniq)
        island_faces.append(lf)
        island_fids.append(fids)
        island_area3d.append(float(mesh.face_areas[fids].sum()))
        angle_ds.append(info["angle_d"])
        area_ds.append(info["area_d"])
        flips.append(info["flips"])
        methods.append(info["method"])

    t_param = time.time()
    if verbose:
        print(f"[param] {n_charts} charts in {t_param - t_seg:.2f}s")

    islands = _post.normalize_texel_density(islands, island_faces, island_area3d)
    packed, fill = _pack.pack(
        islands, resolution=resolution, padding_texels=padding_texels
    )
    t_pack = time.time()
    if verbose:
        print(f"[pack] fill={fill:.2%} in {t_pack - t_param:.2f}s")

    # ---- assemble output mesh (duplicate vertices per chart for seam UVs) ----
    out_v = []
    out_src = []          # welded vertex each output vertex was copied from
    out_n = []
    out_uv = []
    out_f = []
    out_fc = []
    row_of_face = np.full(mesh.n_faces, -1, dtype=np.int64)   # face -> out_f row
    voff = 0
    foff = 0
    for c in range(n_charts):
        uniq = island_uniq[c]
        lf = island_faces[c]
        uv = packed[c]

        # An output vertex is one (chart vertex, smoothing group) pair, not just
        # one chart vertex. Splitting only per chart would be enough for a smooth
        # mesh, but a sharp edge running *inside* a chart puts corners from
        # several smoothing groups on the same chart vertex, and collapsing those
        # to one normal re-rounds the edge (a 2-chart cube comes out as a blob).
        # The extra copies are co-located and share the chart vertex's UV, so the
        # UV layout, the face count and the packing are all untouched -- only the
        # vertex count grows, and only where the mesh really is creased.
        #
        # _chart_local builds lf row-aligned to fids, so lf[i, k] and
        # corner_group[fids[i], k] describe the same corner.
        cg = corner_group[keep_ids[island_fids[c]]]            # (Fc, 3) group ids
        corner_key = np.stack([lf.reshape(-1), cg.reshape(-1)], axis=1)
        key_uniq, key_inv = np.unique(corner_key, axis=0, return_inverse=True)
        # numpy >= 2.0 shapes return_inverse after the input when axis is given;
        # flatten so the reshape below is version-independent.
        key_inv = np.asarray(key_inv).ravel()
        local_of_key = key_uniq[:, 0]                           # -> chart vertex
        group_of_key = key_uniq[:, 1]                           # -> smoothing group

        out_v.append(verts[uniq[local_of_key]])
        out_src.append(uniq[local_of_key])
        out_n.append(group_normal[group_of_key])
        out_uv.append(uv[local_of_key])
        out_f.append(key_inv.reshape(-1, 3) + voff)
        out_fc.append(np.full(len(lf), c))
        row_of_face[island_fids[c]] = foff + np.arange(len(lf))
        foff += len(lf)
        voff += len(key_uniq)

    out_v = np.concatenate(out_v, axis=0)
    out_src = np.concatenate(out_src, axis=0)
    out_n = np.concatenate(out_n, axis=0)
    out_uv = np.concatenate(out_uv, axis=0)
    out_f = np.concatenate(out_f, axis=0)
    out_fc = np.concatenate(out_fc, axis=0)

    report("metrics", 0.0)
    quality = _metrics.layout_quality(out_v, out_f, out_uv, out_src, verts, hidden=hidden,
                                      resolution=min(int(resolution), 2048))
    t_metrics = time.time()
    if verbose:
        print(f"[quality] {quality}")

    # ---- stack duplicate triangles onto their twin's UVs ---------------------
    # After the metrics on purpose: a stacked pair covers the same texels by
    # design and must not read as overlap.
    if len(dup_ids):
        pos = np.full(full_mesh.n_faces, -1, dtype=np.int64)
        pos[keep_ids] = np.arange(len(keep_ids))
        rows = row_of_face[pos[dup_keeper]]
        kf = out_f[rows]                                   # twin's output corners
        dfaces = full_mesh.faces[dup_ids]
        match = dfaces[:, :, None] == out_src[kf][:, None, :]
        src_out = np.take_along_axis(kf, np.argmax(match, axis=2), axis=1)
        n_new = src_out.size
        out_v = np.concatenate([out_v, out_v[src_out].reshape(-1, 3)])
        out_uv = np.concatenate([out_uv, out_uv[src_out].reshape(-1, 2)])
        # each copy keeps its own normal: a back-to-back twin faces the other way
        out_n = np.concatenate([out_n, group_normal[corner_group[dup_ids]].reshape(-1, 3)])
        out_src = np.concatenate([out_src, dfaces.reshape(-1)])
        out_f = np.concatenate([out_f, voff + np.arange(n_new).reshape(-1, 3)])
        out_fc = np.concatenate([out_fc, out_fc[rows]])

    total_flips = int(np.sum(flips))
    stats = {
        "n_faces": int(full_mesh.n_faces),
        "duplicate_faces_stacked": int(len(dup_ids)),
        "n_charts": n_charts,
        "components_before_weld": comps_before,
        "components_after_weld": comps_after,
        "fill_ratio": float(fill),
        "flipped_triangles": total_flips,
        "mean_angle_distortion": float(np.average(
            angle_ds, weights=island_area3d)),
        "mean_area_distortion": float(np.average(
            area_ds, weights=island_area3d)),
        "method_counts": {m: int(methods.count(m)) for m in set(methods)},
        "normal_source": normal_source,
        "borders_moved": border_stats["moved"] if border_stats else 0,
        # extra charts made by splitting charts that folded over themselves
        "fold_splits": int(n_folds_split),
        "charts_cut": len(cut_kinds),
        "cut_edges": int(cut_edges),
        "cut_kinds": {k: cut_kinds.count(k) for k in set(cut_kinds)},
        # charts that stayed non-manifold (sheets glued along an edge) and were
        # flattened as they are
        "charts_not_disk": not_disk,
        "hide_seams": hidden is not None,
        **quality,
        "time_seconds": round(time.time() - t0, 3),
        "time_breakdown": {
            "hidden": round(t_hidden, 3),
            "segment": round(t_b0 - t0 - t_hidden, 3),
            "borders": round(t_border - t_b0, 3),
            "parameterize": round(t_param - t_seg, 3),
            "pack": round(t_pack - t_param, 3),
            "metrics": round(t_metrics - t_pack, 3),
        },
    }
    return UnwrapResult(out_v, out_f, out_uv, out_fc, out_n, stats)
