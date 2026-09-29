"""High-to-low poly texture bake worker (headless Blender).

Runs `bpy` in ISOLATION: invoked as a subprocess by app/services/bake.py
(`python bake_worker.py --low low.glb --high high.glb --outdir dir --options o.json`).
Never import this module from the service — bpy is not thread-safe, holds ~1GB
RSS once imported, and a crash inside Blender must not take the API down.

This is the step that makes Auto Retopo and Optimize non-destructive. On their
own they hand back clean topology with the detail *deleted*; baking captures that
detail as a normal map (plus AO and a base-colour transfer) so the low-poly mesh
still reads as the high-poly one.

Blender's "selected to active" bake casts rays from the low-poly surface out to
the high-poly one, which is why both meshes are loaded into a single scene and
the low-poly is made active. The low-poly must carry UVs — there is nowhere to
write otherwise.

Protocol: progress/result JSON lines on stdout prefixed with GENSTUDIO_EVT
(bpy prints its own "Info:" noise, the parent ignores non-matching lines).
Exit codes: 0 ok, 2 bake error, 3 validation failed, 4 bpy missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SENTINEL = "GENSTUDIO_EVT "  # keep in sync with app/services/bake.py

# name -> (bake pass, colour space, principled input to rewire or None).
#
# Blender has no METALLIC bake pass (its passes are AO, COMBINED, DIFFUSE, EMIT,
# ENVIRONMENT, GLOSSY, NORMAL, POSITION, ROUGHNESS, SHADOW, TRANSMISSION, UV), so
# metallic is captured by temporarily routing the high-poly's Metallic input into
# an Emission shader and baking EMIT. Roughness has a native pass and needs no
# such trick.
#
# Base colour goes through that same rewire rather than through the DIFFUSE pass,
# which looks like the obvious choice and is a trap: DIFFUSE returns the *diffuse
# lobe* albedo, which for a Principled BSDF is base_color * (1 - metallic). A 60%
# metallic source therefore bakes at 40% brightness and a fully metallic one bakes
# pure black — and glTF's default metallicFactor is 1.0, so this also hits meshes
# that simply never wrote the field. EMIT off Base Color is exact at any metallic
# (measured: a 200/140/60 source round-trips byte-for-byte, where DIFFUSE gave
# 144/100/41 at metallic 0.5 and 0/0/0 at metallic 1.0).
#
# Everything except base colour is DATA, not colour, so it is written Non-Color:
# an sRGB-tagged roughness map would come back gamma-encoded and read wrong.
#
# Order matters and is the iteration order below: a rewire replaces the material's
# Surface link, so every pass that reads the real shader (ROUGHNESS) has to run
# before the first rewired one.
BAKE_PASSES = {
    "normal": ("NORMAL", "Non-Color", None),
    "ao": ("AO", "Non-Color", None),
    "roughness": ("ROUGHNESS", "Non-Color", None),
    "base_color": ("EMIT", "sRGB", "Base Color"),
    "metallic": ("EMIT", "Non-Color", "Metallic"),
}

BAKE_ORDER = ["normal", "ao", "roughness", "base_color", "metallic"]

# Passes that bake natively but still map to a Principled input, so the "this came
# from a constant, the map is flat" check applies to them as well.
PROBE_INPUTS = {"roughness": "Roughness"}

# glTF packs occlusion/roughness/metallic into one texture's R/G/B. Producing that
# packed form means three.js can hand it to all three material slots as a single
# object, which is both what the format wants and what lets its exporter skip
# recompositing the channels.
ORM_CHANNELS = ["ao", "roughness", "metallic"]
# Neutral values for channels that were not baked: no occlusion, fully rough,
# non-metal. Only the baked channels are ever read back on the client, so these
# are padding rather than claims about the material.
ORM_NEUTRAL = {"ao": 255, "roughness": 255, "metallic": 0}

# ── Alignment ───────────────────────────────────────────────────────────────
# A "selected to active" bake is PURELY SPATIAL: rays leave the low-poly surface
# and whatever they hit on the high-poly is what gets written. So the two meshes
# have to occupy the same world space, and nothing upstream guarantees that — the
# editor's automatic snapshots share whichever space the mesh was in at the time,
# but a source picked from the asset library arrives in raw file space. Move the
# pivot in between (Game-Ready's "set pivot on the ground" is one click) and the
# source is silently offset by the model's half-height from then on, with no
# symptom until the bake comes back black wherever the two no longer overlap.
#
# Two meshes at the same SCALE whose bounding boxes are merely offset are the same
# object with different pivots, and re-centring the source is unambiguously right.
#
# A UNIFORM scale difference is the same story one step out, and it is not a guess
# when it is measured rather than assumed: if the target's box is the source's box
# times the same ratio on all three axes, the two are the same shape in different
# units. That case is common and arrives entirely from outside this editor — a mesh
# simplified in a ComfyUI graph rather than by Auto Retopo/Optimize keeps the
# original's space, while the texturing pass it is meant to be baked against
# normalises its output to a unit bounding box at the origin (Trellis2 does exactly
# this). The two then differ by a clean uniform factor, which is recoverable; only
# a NON-uniform mismatch means different objects, and that is still reported and
# left alone. Uniform is also the only scale a bake tolerates: it leaves normals
# and ray directions untouched, where a per-axis one would skew both.
ALIGN_SCALE_TOLERANCE = 0.05  # per-axis extent agreement required to re-centre
# How far the three per-axis ratios may disagree (relative to their mean) and still
# count as one uniform scale. Sized well above the noise simplification introduces
# — measured at 0.06% on a 60k → 4.7k decimation — and far below any coincidence
# between two genuinely different objects. Ratios are only taken on axes with real
# extent; a flat axis divides by nothing useful.
ALIGN_UNIFORM_TOLERANCE = 0.02
ALIGN_MIN_AXIS_FRAC = 0.01  # axes thinner than this fraction of the diagonal give no ratio
ALIGN_MIN_SCALE_DELTA = 0.005  # ratios this close to 1 are the same scale, not a rescale
ALIGN_MIN_OFFSET_FRAC = 0.001  # offsets below this fraction of the diagonal are noise
# Simplification does not only shave a hair off every extremity — it can drop a
# whole spike (a chimney, a finial, a horn tip), and the axis it stood on then
# comes up several percent SHORT on the low-poly while the other two still agree
# to a fraction of a percent. Assets 5902/7512 (a Trellis2-normalised house and
# its ComfyUI simplification) were exactly that: X and Z said 1.982x, Y said
# 1.812x because the low-poly had lost 4cm of roof, and the strict test above
# refused the rescale and baked a half-size source in place (23% coverage). An
# axis may therefore trail the agreeing pair by up to this share of its extent —
# only ever SHORT, since simplification removes and never adds.
ALIGN_TRIM_MAX = 0.35
# An axis whose rescaled extents still differ by more than this fraction of the
# diagonal is lined up by each end as well as by its centre, and the placement
# that lands the target's surface closest to the source wins. Centre-to-centre is
# only right when both ends were shaved alike; a lost spike moves one end only.
ALIGN_TRIM_ANCHOR_FRAC = 0.005
# Points sampled (area-weighted) on the target surface to measure how much of it
# the source reaches. Enough to resolve a percent; few enough that ranking a
# dozen candidate placements stays well under a second.
ALIGN_REACH_SAMPLES = 1024
# A sample counts as reached when the source surface lies within this multiple of
# the cage extrusion. The bake's rays start one cage out and travel inward, so
# detail up to about a cage either side of the target is what they can find.
ALIGN_REACH_FACTOR = 2.0


def emit(stage: str, frac: float, message: str = "") -> None:
    print(f"{SENTINEL}{json.dumps({'type': 'progress', 'stage': stage, 'frac': round(frac, 4), 'message': message})}", flush=True)


def fail(code: int, error: str) -> None:
    print(f"{SENTINEL}{json.dumps({'type': 'result', 'ok': False, 'error': error})}", flush=True)
    sys.exit(code)


def hidden_from_render(obj) -> bool:
    """Is this object disabled for rendering, directly or through a collection?

    `bpy.ops.object.bake` errors on the first selected object it finds disabled,
    so this is the gate every object has to pass before it can be a bake source
    or target. Collections carry the flag as well as objects, and a collection
    inherits it from its parents — which is exactly how the glTF importer hides
    its bone widgets, in a `glTF_not_exported` collection.
    """
    import bpy

    if obj.hide_render:
        return True
    parent_of = {}
    for collection in bpy.data.collections:
        for child in collection.children:
            parent_of[child.name] = collection
    for collection in obj.users_collection:
        node = collection
        seen = set()
        while node is not None and node.name not in seen:
            if node.hide_render:
                return True
            seen.add(node.name)
            node = parent_of.get(node.name)
    return False


def principled_input_is_linked(objects, input_name: str) -> bool:
    """Is a Principled BSDF input driven by a node graph rather than a constant?

    Used by the passes that need no rewiring (roughness has a native bake) so they
    can report a flat result for the same reason the rewired ones do. Without this
    a constant-roughness source would silently produce a flat map with no warning,
    while constant metallic warned — an inconsistency that only shows up as
    surprise later.
    """
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes:
                continue
            bsdf = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            socket = bsdf.inputs.get(input_name) if bsdf else None
            if socket is not None and socket.is_linked:
                return True
    return False


REWIRE_NODE_NAME = "GenStudioBakeEmit"


def unlit_color_socket(tree):
    """The colour socket of a KHR_materials_unlit material, or None.

    Blender's glTF importer builds an unlit material with no Principled BSDF: the
    base colour (texture x factor) feeds an Emission node, which reaches the output
    through a Light Path "Is Camera Ray" mix against a Transparent BSDF. Both of the
    usual routes bake that BLACK at 100% coverage — there is no BSDF to rewire, the
    DIFFUSE fallback finds no diffuse lobe, and even a plain EMIT bake loses the
    emission to the light-path mix, because a bake ray is not a camera ray. So the
    Emission node's Color input is the base colour, and it is re-routed from there.

    Every mesh the editor has viewed unlit is exported unlit too (GLTFLoader loads
    it as MeshBasicMaterial and the exporter writes the extension back), so this
    also covers the automatic Before-Optimize/Retopo snapshots of such a mesh.
    """
    for node in tree.nodes:
        if node.type == "EMISSION" and node.name != REWIRE_NODE_NAME:
            return node.inputs.get("Color")
    return None


def rewire_to_emit(objects, input_name: str) -> tuple[bool, bool]:
    """Route a Principled BSDF input into an Emission shader so EMIT can bake it.

    Returns (rewired, driven_by_graph).

    `driven_by_graph` is True when a *texture* (or any node graph) actually drives
    the input on at least one material. When it is only a constant, the bake still
    succeeds but produces a flat map — which is strictly worse than the scalar it
    came from, so the caller reports that rather than pretending the map is useful.

    `rewired` is False when no material had a Principled BSDF (or, for base
    colour, an unlit Emission — see unlit_color_socket) to read, which the
    caller has to know about: an EMIT bake against a shader graph we never touched
    comes back black rather than wrong-but-plausible.

    Blender's glTF importer wires metallic/roughness through a Separate Color node
    fed by the packed ORM texture, so the upstream socket here is normally that
    node's B (or G) output. Linking a single float output into Emission's Color
    broadcasts it across RGB, which is exactly what a data bake wants.
    """
    rewired = False
    driven_by_graph = False
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes:
                continue
            tree = material.node_tree
            bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
            if not output:
                continue
            if bsdf:
                socket = bsdf.inputs.get(input_name)
            elif input_name == "Base Color":
                socket = unlit_color_socket(tree)
            else:
                socket = None
            if socket is None:
                continue

            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = REWIRE_NODE_NAME
            if socket.is_linked:
                tree.links.new(socket.links[0].from_socket, emission.inputs["Color"])
                driven_by_graph = True
            else:
                # Scalar inputs (Metallic) broadcast across RGB; Base Color is
                # already a 4-float and is copied straight through.
                raw = socket.default_value
                try:
                    value = float(raw)
                    rgba = (value, value, value, 1.0)
                except TypeError:
                    rgba = (raw[0], raw[1], raw[2], 1.0)
                emission.inputs["Color"].default_value = rgba
            tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
            rewired = True
    return rewired, driven_by_graph


def world_bounds(objects) -> tuple[list, list]:
    """Axis-aligned world-space bounds over a set of objects, as ([min], [max]).

    Read off each object's `bound_box` through its world matrix rather than from
    the vertices: the corners are eight points instead of hundreds of thousands,
    and an axis-aligned box of the transformed corners is exactly what the overlap
    measure below needs.
    """
    from mathutils import Vector

    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
    return low, high


def box_overlap(target: tuple, source: tuple) -> float:
    """How much of `target`'s box the `source` box covers, as the WORST axis.

    The worst axis rather than the volume ratio, because the volume ratio hides
    exactly the failure this is here to catch: a source offset along one axis only
    still has two axes overlapping perfectly, so its volume ratio stays
    respectable while half the mesh has nothing to sample. Axes whose target
    extent is degenerate (a flat plane) are skipped rather than counted as a
    total miss.
    """
    t_min, t_max = target
    s_min, s_max = source
    scale = max(t_max[axis] - t_min[axis] for axis in range(3))
    worst = 1.0
    for axis in range(3):
        extent = t_max[axis] - t_min[axis]
        if extent <= scale * 1e-4:
            continue
        span = min(t_max[axis], s_max[axis]) - max(t_min[axis], s_min[axis])
        worst = min(worst, max(span, 0.0) / extent)
    return worst


def uniform_scale_ratio(t_extent, s_extent, diagonal) -> float | None:
    """The single factor that takes the source's box to the target's, or None.

    None whenever the three axes disagree about what that factor is — which is the
    whole point: one ratio repeated on every axis is a measurement of a units
    change, three different ratios are two different objects, and only the first
    can be undone. Axes too thin to divide by (a flat panel's depth) contribute no
    ratio rather than a huge one; fewer than two usable axes is not enough evidence
    to rescale on, since a single axis agrees with itself by construction.
    """
    ratios = []
    floor = ALIGN_MIN_AXIS_FRAC * max(diagonal, 1e-9)
    for axis in range(3):
        if t_extent[axis] <= floor or s_extent[axis] <= floor:
            continue
        ratios.append(t_extent[axis] / s_extent[axis])
    if len(ratios) < 2:
        return None
    mean = sum(ratios) / len(ratios)
    if mean <= 0:
        return None
    if any(abs(r - mean) > ALIGN_UNIFORM_TOLERANCE * mean for r in ratios):
        return None
    return mean


def trimmed_scale_ratio(t_extent, s_extent, diagonal) -> tuple[float, list] | None:
    """The uniform factor when two axes agree and the third is merely TRIMMED.

    The fallback for uniform_scale_ratio's refusal, for a low-poly that lost a
    spike (see ALIGN_TRIM_MAX). Takes the agreeing pair's factor and returns it
    with the axes that fell short of it. Needs all three axes: with a flat axis
    out of the running there is no majority to say which of the other two is the
    odd one out. The odd axis must be the SHORTER one on the target — a target
    that sticks out further than the rescaled source has something the source
    lacks, which is a different object, not a simplification.
    """
    floor = ALIGN_MIN_AXIS_FRAC * max(diagonal, 1e-9)
    if any(t_extent[axis] <= floor or s_extent[axis] <= floor for axis in range(3)):
        return None
    ratios = [t_extent[axis] / s_extent[axis] for axis in range(3)]
    top = max(ratios)
    agreeing = [axis for axis in range(3) if top - ratios[axis] <= ALIGN_UNIFORM_TOLERANCE * top]
    if len(agreeing) != 2:
        return None
    ratio = sum(ratios[axis] for axis in agreeing) / len(agreeing)
    trimmed = [axis for axis in range(3) if axis not in agreeing]
    if any(ratios[axis] < (1.0 - ALIGN_TRIM_MAX) * ratio for axis in trimmed):
        return None
    return ratio, trimmed


def surface_samples(obj, count: int) -> list:
    """Area-weighted world-space points on `obj`'s surface, deterministic.

    Area-weighted because the question they answer is how much of the UV layout a
    bake can fill, and texels follow area, not vertex density — a low-poly's
    vertices crowd into its detailed corners and would over-report those.
    """
    import random
    from mathutils import Vector

    mesh = obj.data
    mesh.calc_loop_triangles()
    matrix = obj.matrix_world
    verts = [matrix @ v.co for v in mesh.vertices]
    tris = [tuple(verts[i] for i in tri.vertices) for tri in mesh.loop_triangles]
    areas = [((b - a).cross(c - a)).length for a, b, c in tris]
    if not tris or sum(areas) <= 0:
        return []
    rng = random.Random(0)
    points = []
    for index in rng.choices(range(len(tris)), weights=areas, k=count):
        a, b, c = tris[index]
        u, v = rng.random(), rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        points.append(Vector(a + (b - a) * u + (c - a) * v))
    return points


class SourceSurface:
    """Nearest-point queries against the high-poly, in world space.

    One BVH per object in its own local space (FromObject builds it there), with
    the query taken in through the inverse world matrix and the hit brought back
    out, so a parented or scaled import is measured where it really renders.
    """

    def __init__(self, high_objects):
        import bpy
        from mathutils.bvhtree import BVHTree

        depsgraph = bpy.context.evaluated_depsgraph_get()
        self.trees = []
        for obj in high_objects:
            evaluated = obj.evaluated_get(depsgraph)
            tree = BVHTree.FromObject(evaluated, depsgraph)
            self.trees.append((tree, obj.matrix_world.copy(), obj.matrix_world.inverted()))

    def distance(self, point) -> float:
        best = float("inf")
        for tree, matrix, inverse in self.trees:
            hit = tree.find_nearest(inverse @ point)
            if hit[0] is not None:
                best = min(best, (matrix @ hit[0] - point).length)
        return best


def measure_reach(samples, surface, scale: float, translation, reach: float) -> dict:
    """How much of the target the source reaches with the source moved by (scale, translation).

    The source is never moved to ask: each target sample goes through the inverse
    of the placement into the source's current space instead, and the distance
    found there comes back out multiplied by the scale. Ranking a dozen candidate
    placements this way costs queries, not a dozen BVH rebuilds.
    """
    if not samples:
        return {"reach": 1.0, "median": 0.0}
    distances = sorted(
        surface.distance((point - translation) / scale) * scale for point in samples
    )
    return {
        "reach": sum(1 for d in distances if d <= reach) / len(distances),
        "median": distances[len(distances) // 2],
    }


AXIS_NAMES = ("width", "depth", "height")  # Blender X / Y / Z; a glTF's up axis imports as Z


def align_source(low, high_objects, samples, surface, reach: float) -> dict:
    """Put the high-poly into the low-poly's space when the two are the same object.

    Returns a report dict (always), having transformed `high_objects` in place when
    it decided to. The bounding boxes PROPOSE placements — a scale (1, a uniform
    factor, or the agreeing pair's factor when one axis was trimmed) and, per
    axis, centre-to-centre or either end — and the target's own surface picks
    between them: each candidate is scored by how much of the target lands within
    reach of the source (see measure_reach), and "leave it where it is" is always
    one of the candidates, so a source is only moved when moving it measurably
    helps. Boxes alone cannot make that call: a lost spike shifts one end of one
    axis, and a half-size source nested inside the target covers half of every
    axis while its surface reaches almost none of the target's.

    `report["overlap"]` is that reach after alignment — the share of the target's
    surface the bake's rays can find the source from — and is what the pre-flight
    gate in main() tests. The worst-axis box overlap is kept alongside as
    `box_overlap` for the error message and for comparison with the client, which
    only has the boxes.
    """
    import itertools

    import bpy
    from mathutils import Matrix, Vector

    target = world_bounds([low])
    source = world_bounds(high_objects)
    t_min, t_max = target
    s_min, s_max = source

    t_extent = [t_max[i] - t_min[i] for i in range(3)]
    s_extent = [s_max[i] - s_min[i] for i in range(3)]
    diagonal = sum(e * e for e in t_extent) ** 0.5

    offset = [((t_min[i] + t_max[i]) - (s_min[i] + s_max[i])) / 2 for i in range(3)]
    distance = sum(o * o for o in offset) ** 0.5

    zero = Vector((0.0, 0.0, 0.0))
    untouched = measure_reach(samples, surface, 1.0, zero, reach)
    report = {
        "target_bounds": [t_min, t_max],
        "source_bounds": [s_min, s_max],
        "offset": offset,
        "reach_distance": round(reach, 6),
        "box_overlap_before": round(box_overlap(target, source), 4),
        "overlap_before": round(untouched["reach"], 4),
    }

    def keep(mode: str) -> dict:
        report["mode"] = mode
        report["overlap"] = report["overlap_before"]
        report["box_overlap"] = report["box_overlap_before"]
        report["median_distance"] = round(untouched["median"], 6)
        return report

    # A scale check has to tolerate the extremities simplification removes, which
    # is what ALIGN_SCALE_TOLERANCE is sized for. Compared against the diagonal
    # rather than each axis's own extent so a thin axis (a 2cm-deep relief on a 2m
    # panel) is not judged by a hair's breadth.
    scale_matches = all(
        abs(t_extent[i] - s_extent[i]) <= ALIGN_SCALE_TOLERANCE * max(diagonal, 1e-9)
        for i in range(3)
    )
    ratio, trimmed = 1.0, []
    if not scale_matches:
        # Different sizes, so the question is whether they differ by ONE factor —
        # on all three axes, or on two with the third trimmed short. If neither,
        # this is the honest refusal it has always been.
        ratio = uniform_scale_ratio(t_extent, s_extent, diagonal)
        if ratio is None:
            fallback = trimmed_scale_ratio(t_extent, s_extent, diagonal)
            if fallback is not None:
                ratio, trimmed = fallback
        if ratio is None:
            report["scale_axes"] = [
                round(t_extent[i] / s_extent[i], 4) if s_extent[i] > 1e-9 else None
                for i in range(3)
            ]
            return keep("skipped-scale")
        if abs(ratio - 1.0) <= ALIGN_MIN_SCALE_DELTA:
            # The same size after all, with one axis trimmed past the tolerance
            # above: a translation problem, and the anchors below solve it.
            ratio = 1.0

    # Per axis, where the rescaled source may sit: centred on the target always,
    # and flush with either end wherever the two extents still disagree enough
    # for the choice to matter.
    anchors = []
    for i in range(3):
        options = {"centre": (t_min[i] + t_max[i]) / 2 - ratio * (s_min[i] + s_max[i]) / 2}
        if abs(t_extent[i] - ratio * s_extent[i]) > ALIGN_TRIM_ANCHOR_FRAC * max(diagonal, 1e-9):
            options["min"] = t_min[i] - ratio * s_min[i]
            options["max"] = t_max[i] - ratio * s_max[i]
        anchors.append(list(options.items()))

    best = None
    for combo in itertools.product(*anchors):
        translation = Vector([value for _, value in combo])
        score = measure_reach(samples, surface, ratio, translation, reach)
        key = (round(score["reach"], 3), -score["median"])
        if best is None or key > best[0]:
            best = (key, combo, translation, score)
    _, combo, translation, score = best

    # Staying put wins ties: a placement that reaches no more of the target and
    # sits no closer is a move for nothing. So does a same-scale shift too small
    # to be anything but box noise.
    stay = (round(untouched["reach"], 3), -untouched["median"]) >= best[0]
    negligible = ratio == 1.0 and translation.length <= ALIGN_MIN_OFFSET_FRAC * max(diagonal, 1e-9)
    if stay or negligible:
        return keep("not-needed")

    transform = Matrix.Translation(translation) @ Matrix.Scale(ratio, 4)
    for obj in high_objects:
        # Through matrix_world so a mesh parented under an imported empty moves in
        # WORLD space, which is the space every measurement above was taken in.
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()

    report["mode"] = "applied" if ratio == 1.0 else "scaled"
    if ratio != 1.0:
        report["scale"] = round(ratio, 6)
    report["translation"] = [round(v, 6) for v in translation]
    report["distance"] = round(translation.length if ratio == 1.0 else distance, 6)
    ends = {AXIS_NAMES[i]: name for i, (name, _) in enumerate(combo) if name != "centre"}
    if ends:
        report["anchors"] = ends
    if trimmed:
        report["trimmed_axes"] = [AXIS_NAMES[i] for i in trimmed]
    report["overlap"] = round(score["reach"], 4)
    report["median_distance"] = round(score["median"], 6)
    report["box_overlap"] = round(box_overlap(target, world_bounds(high_objects)), 4)
    return report


def measure_coverage(low, outdir, written: dict, resolution: int) -> dict | None:
    """What fraction of the UV layout actually received ray hits?

    The one number that tells a good bake from a ruined one, and until now nothing
    computed it: a bake whose rays all miss still exits 0 with a full set of PNGs,
    every texel cleared to transparent black. Against a dark model that is
    invisible — which is precisely how a bake covering an eighth of the mesh got
    applied and saved.

    Measured as (baked texels ∩ UV layout) / (UV layout), where "baked" is the
    alpha channel the bake writes only for texels whose ray HIT and "UV layout" is
    the low-poly's render UV set rasterised at the bake resolution.

    Intersecting with the layout is not optional, and this is the trap: alpha is a
    hit mask ONLY INSIDE the layout. Outside it, margin dilation leaves alpha set
    across essentially the whole gutter (measured: 99.9% of it, 92% of that with no
    colour behind it), so alpha on its own reads as near-total coverage no matter
    how badly the bake went. Inside the layout it is clean — on the misaligned bake
    this was written for, 52.7% of layout texels came back alpha 0 against 43.3%
    carrying colour.

    The threshold is deliberately stricter than the one pack_orm uses. Here a
    partially-dilated edge texel should not count as a hit, because the number's
    whole job is to under-claim rather than over-claim success; pack_orm asks the
    opposite question ("is there definitely nothing here?") and so tests for
    exactly zero, which keeps it from overwriting real colour at island edges.

    Returns None when it cannot be measured — the maps are still perfectly good,
    so this must never be the reason a bake fails.
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        print(f"Coverage measurement unavailable ({exc}).", flush=True)
        return None

    source = next((written[name] for name in BAKE_ORDER if name in written), None)
    if not source:
        return None

    try:
        image = Image.open(outdir / source)
        if "A" not in image.getbands():
            return None
        baked = np.asarray(image.getchannel("A"), dtype=np.uint8) > 127

        height, width = baked.shape
        layout_mask = rasterize_uv_layout(low, width, height)
        if layout_mask is None:
            return None

        layout_texels = int(layout_mask.sum())
        if not layout_texels:
            return None
        covered = int((layout_mask & baked).sum())
        return {
            "coverage": round(covered / layout_texels, 4),
            "covered_texels": covered,
            "layout_texels": layout_texels,
            "layout_frac": round(layout_texels / float(width * height), 4),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"Coverage measurement failed ({exc}).", flush=True)
        return None


def rasterize_uv_layout(low, width: int, height: int, layer_name: str | None = None):
    """Boolean (height, width) mask of the texels the low-poly's UVs cover.

    None when it cannot be built. This is the authoritative "inside the layout"
    test — the texels the shader can actually sample — used both to score bake
    coverage and to decide which texels the gutter fill may overwrite.

    `low` may also be a LIST of objects, rasterised into one mask, and
    `layer_name` picks a UV set other than the active-render one — both for the
    flatten worker, which bakes every object of a scene into one shared atlas
    held in a second UV set.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        return None

    layout = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(layout)
    drawn = False
    for obj in (low if isinstance(low, (list, tuple)) else [low]):
        mesh = obj.data
        if layer_name is not None:
            uv_layer = mesh.uv_layers.get(layer_name)
        else:
            # Blender bakes into the ACTIVE RENDER uv set, which is not
            # necessarily the one selected in the UI — rasterising the other one
            # would describe a layout the bake never wrote to.
            uv_layer = next((layer for layer in mesh.uv_layers if layer.active_render), mesh.uv_layers.active)
        if uv_layer is None:
            continue
        # Removed in newer Blender, where the cache is maintained automatically.
        if hasattr(mesh, "calc_loop_triangles"):
            mesh.calc_loop_triangles()

        data = uv_layer.data
        # Blender's V runs bottom-up and its PNG writer flips on save, so the saved
        # image's top row is V=1 — hence (1 - v) here. Getting this backwards would
        # compare the layout against a mirror image of itself and report nonsense.
        for triangle in mesh.loop_triangles:
            draw.polygon([
                (data[loop].uv[0] * width, (1.0 - data[loop].uv[1]) * height)
                for loop in triangle.loops
            ], fill=1)
            drawn = True
    return np.asarray(layout, dtype=bool) if drawn else None


def _erode(mask, iterations: int = 1):
    """Shrink a boolean mask by `iterations` texels (4-neighbour).

    Unlike `scipy.ndimage.binary_erosion`'s default, the area outside the image
    counts as SET, so texels on the image border are not eroded. That is what the
    gutter fill wants: a border texel is clamp-sampled rather than blended with
    anything outside, so it is a perfectly good colour source and dropping it
    could empty the source mask for an island that only touches the edge.
    """
    out = mask
    for _ in range(max(iterations, 0)):
        m = out
        out = m.copy()
        out[1:, :] &= m[:-1, :]
        out[:-1, :] &= m[1:, :]
        out[:, 1:] &= m[:, :-1]
        out[:, :-1] &= m[:, 1:]
    return out


def fill_gutters(low, outdir, written: dict, layout=None):
    """Flood every baked map's empty gutter with the nearest in-layout colour.

    Returns {map_name: fraction_of_image_filled}, or None when it could not run
    — which is never fatal, the maps are still the maps.

    **Why the bake margin is not enough.** `bake.margin` dilates the islands by a
    fixed number of texels, so it can only protect the mip levels whose footprint
    is smaller than that. Every coarser level averages the gutter into the island
    edge, and a gutter of pure black paints a dark line along each UV seam that
    shows up *as the viewer zooms out* — because zooming out is exactly what
    selects the coarser mips. Measured on the reported head bake (2048px; the
    default margin of 8 yielded 2.8px median and 9px maximum of real dilation):
    0-1% of island-edge texels darkened at mips 0-2, then 13% at mip 3, 44% at
    mip 4, 60% at mip 5. Raising the margin only moves the level it starts at.

    Filling the whole gutter makes every mip level safe at any resolution, since
    there is no black left to average in. Texels inside the UV layout are never
    written, so the bake this runs after is still the bake that was measured.

    Must run AFTER pack_orm: that reads the bake's alpha and tests colour for
    exactly zero to decide a channel is empty, and both of those stop meaning
    what they mean once the gutter carries colour.

    `layout` is a precomputed rasterize_uv_layout mask, for callers whose layout
    is not the low-poly's active-render UV set (the flatten worker's atlas).
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy.ndimage import distance_transform_edt
    except Exception as exc:  # noqa: BLE001
        print(f"Gutter fill unavailable ({exc}); UV seams may darken at distance.", flush=True)
        return None

    first = next((written[name] for name in BAKE_ORDER if name in written), None)         or next(iter(written.values()), None)
    if not first:
        return None
    try:
        with Image.open(outdir / first) as probe:
            width, height = probe.size
        if layout is None:
            layout = rasterize_uv_layout(low, width, height)
        if layout is None or not layout.any():
            return None

        # Seed the flood from strictly *inside* the layout: the outermost texel
        # ring is only partially covered, so it is already blended toward the
        # gutter and would seed the fill with the very darkness being removed.
        source = _erode(layout.copy(), 1)
        # An island only one or two texels wide erodes away completely, and then
        # its gutter would be flooded with a *neighbouring* island's colour —
        # reintroducing the wrong-colour bleed this is here to remove. Keep those
        # islands whole instead: a slightly blended source beats a foreign one.
        try:
            from scipy.ndimage import label
            islands, _ = label(layout)
            survived = np.unique(islands[source])
            source = source | (layout & ~np.isin(islands, survived))
        except Exception:  # noqa: BLE001 — labelling is a refinement, not a need
            pass
        if not source.any():
            source = layout

        # Exact nearest-source lookup for every texel in one pass.
        _, (iy, ix) = distance_transform_edt(~source, return_indices=True)
        gutter = ~layout

        filled = {}
        for name, filename in list(written.items()):
            path = outdir / filename
            try:
                with Image.open(path) as image:
                    mode = image.mode
                    arr = np.asarray(image).copy()
            except Exception:  # noqa: BLE001 — skip anything unreadable
                continue
            if arr.ndim != 3 or arr.shape[:2] != (height, width):
                continue
            arr[gutter] = arr[iy[gutter], ix[gutter]]
            Image.fromarray(arr, mode=mode).save(path)
            filled[name] = round(float(gutter.mean()), 4)
        return filled
    except Exception as exc:  # noqa: BLE001 — never fail a good bake over this
        print(f"Gutter fill failed ({exc}).", flush=True)
        return None


def pack_orm(written: dict, outdir, resolution: int) -> tuple[str | None, list]:
    """Compose the baked AO/roughness/metallic into one R/G/B texture.

    Returns (filename, channels_actually_baked). Skipped unless at least two of
    the three exist — a single channel is better served by its own map.
    """
    present = [name for name in ORM_CHANNELS if name in written]
    if len(present) < 2:
        return None, present

    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # noqa: BLE001 — the individual maps are still returned
        print(f"ORM packing unavailable ({exc}); returning separate maps.", flush=True)
        return None, present

    planes = []
    for name in ORM_CHANNELS:
        if name in written:
            image = Image.open(outdir / written[name])
            if image.size != (resolution, resolution):
                image = image.resize((resolution, resolution), Image.LANCZOS)
            plane = np.asarray(image.convert("L"), dtype=np.uint8)
            # Texels whose ray missed carry the transparent-black clear value, and
            # 0 is a MEANINGFUL number in all three of these channels — fully
            # occluded, mirror-smooth, and (for metallic) the only one where 0 is
            # harmless. Substituting the neutral value keeps a partial bake from
            # ringing the mesh in black shadow and glossy patches; the alpha
            # channel is what says which texels those are. ORM itself stays RGB,
            # as glTF wants, so the mask cannot travel with it.
            if "A" in image.getbands():
                miss = np.asarray(image.getchannel("A"), dtype=np.uint8) == 0
                plane = np.where(miss, ORM_NEUTRAL[name], plane).astype(np.uint8)
            planes.append(plane)
        else:
            planes.append(np.full((resolution, resolution), ORM_NEUTRAL[name], dtype=np.uint8))

    Image.fromarray(np.dstack(planes), mode="RGB").save(outdir / "orm.png")
    return "orm.png", present


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", required=True)
    parser.add_argument("--high", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--options", required=True)
    args = parser.parse_args()

    options = json.loads(Path(args.options).read_text(encoding="utf-8"))
    resolution = int(options.get("resolution", 2048))
    maps = [m for m in options.get("maps", ["normal", "ao"]) if m in BAKE_PASSES]
    if not maps:
        fail(3, "No valid bake maps were requested.")

    try:
        import bpy
    except Exception as exc:  # noqa: BLE001
        fail(4, f"Blender (bpy) is not available on the mesh-tools service: {exc}")

    emit("scene", 0.05, "Preparing the scene…")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def import_glb(path: str) -> list:
        before = set(bpy.data.objects)
        # bone_heuristic: the default 'BLENDER' gives a rigged glTF's bones an
        # Icosphere display widget, and that widget is a real mesh object parked
        # in a hidden 'glTF_not_exported' collection. Both halves of the bake
        # then break: the widget joins the source selection, where Blender
        # refuses the whole bake outright ('Object "Icosphere" is not enabled for
        # rendering'), and it sorts before most mesh names, so it can be picked
        # as the target too. 'TEMPERANCE' only orients bones, no widgets — the
        # FBX and thumbnail workers import the same way, for the same reason.
        bpy.ops.import_scene.gltf(filepath=path, bone_heuristic="TEMPERANCE")
        new_meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        # Belt and braces on top of the heuristic: anything that reaches the bake
        # disabled for rendering fails it, so drop those rather than hand Blender
        # a selection it will reject.
        return [o for o in new_meshes if not hidden_from_render(o)]

    emit("import", 0.12, "Importing the low-poly mesh…")
    low_objects = import_glb(args.low)
    if not low_objects:
        fail(3, "The low-poly file contains no mesh.")
    low = low_objects[0]

    if not low.data.uv_layers:
        fail(3, "The low-poly mesh has no UVs. Run Auto UV before baking.")

    emit("import", 0.2, "Importing the high-poly mesh…")
    high_objects = import_glb(args.high)
    if not high_objects:
        fail(3, "The high-poly source file contains no mesh.")

    # Put the two meshes in the same space before anything expensive runs. See the
    # ALIGN_* constants: a bake is ray casting, so an offset source produces black
    # wherever the boxes stop overlapping, and this is the only place that holds
    # both meshes and can tell.
    #
    # Cage extrusion is a distance, so a fixed default is only ever right for one
    # mesh size: 5cm is generous on a 1m prop and invisible on a 20m building.
    # 0 means "scale it to this mesh" — 2% of the bounding-box diagonal, which
    # reaches far enough to catch protruding detail without punching through to
    # surfaces on the far side. Settled before alignment because it is also the
    # yardstick alignment measures reach with.
    cage = float(options.get("cage_extrusion", 0.0))
    if cage <= 0.0:
        diagonal = max(low.dimensions.x, 1e-6) ** 2 + low.dimensions.y ** 2 + low.dimensions.z ** 2
        cage = 0.02 * (diagonal ** 0.5)
        emit("scene", 0.21, f"Auto cage extrusion: {cage:.4f}m")

    emit("align", 0.22, "Checking source alignment…")
    samples = surface_samples(low, ALIGN_REACH_SAMPLES)
    surface = SourceSurface(high_objects)
    reach = ALIGN_REACH_FACTOR * cage
    if bool(options.get("align_source", True)):
        alignment = align_source(low, high_objects, samples, surface, reach)
    else:
        target, source = world_bounds([low]), world_bounds(high_objects)
        # The bounds go in even here: they are what the refusal below quotes, and
        # an error naming (0,0,0)..(0,0,0) tells the reader nothing.
        from mathutils import Vector
        measured = measure_reach(samples, surface, 1.0, Vector((0.0, 0.0, 0.0)), reach)
        alignment = {
            "mode": "disabled",
            "target_bounds": [target[0], target[1]],
            "source_bounds": [source[0], source[1]],
            "reach_distance": round(reach, 6),
            "overlap": round(measured["reach"], 4),
            "median_distance": round(measured["median"], 6),
            "box_overlap": round(box_overlap(target, source), 4),
        }
    if alignment["mode"] == "applied":
        shift = alignment["translation"]
        emit("align", 0.23,
             f"Source moved onto the target by ({shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f})m")
    elif alignment["mode"] == "scaled":
        emit("align", 0.23,
             f"Source rescaled onto the target by {alignment['scale']:.4f}x and lined up")

    # Pre-flight rather than post-mortem: a bake with nothing to hit costs the same
    # minutes of Cycles time as a good one and then hands back maps that look
    # plausible. Reach is the honest gate, not matching extents — a source with
    # extra geometry (a plinth the low-poly dropped) has mismatched extents and
    # bakes perfectly well, while a source that only reaches half the target cannot.
    # And reach, not box overlap: a half-size source nested inside the target
    # covers half of every axis (it passed this gate at 50.5% on assets 5902/7512)
    # while its surface was within reach of 13% of the target's.
    require_overlap = float(options.get("require_overlap", 0.5))
    if require_overlap > 0 and alignment.get("overlap", 1.0) < require_overlap:
        t_min, t_max = alignment.get("target_bounds", ([0, 0, 0], [0, 0, 0]))
        s_min, s_max = alignment.get("source_bounds", ([0, 0, 0], [0, 0, 0]))
        fmt = lambda v: "(" + ", ".join(f"{x:.3f}" for x in v) + ")"  # noqa: E731
        reason = {
            "skipped-scale": "their sizes differ by a different amount on each axis, so they are "
                             "not one object at two scales and no single factor can line them up",
            "disabled": "automatic alignment is switched off",
        }.get(alignment["mode"], "lining them up did not bring them together")
        fail(3,
             "The high-poly source does not overlap the mesh being baked to — "
             f"only {alignment['overlap'] * 100:.0f}% of the target's surface has the source within "
             f"{alignment.get('reach_distance', 0):.3f}m of it, and {reason}. "
             f"Target bounds {fmt(t_min)}..{fmt(t_max)}, source bounds {fmt(s_min)}..{fmt(s_max)}. "
             "A bake casts rays from the target onto the source, so a source that does not lie on "
             "the target's surface can only return blank or wrong texels. Pick the source this mesh "
             "was actually derived from, move it into the same space as the target, or raise the cage "
             "extrusion if its detail stands further off the surface than that.")

    # A bake target needs a material with an image node to write into.
    material = bpy.data.materials.new(name="BakeTarget")
    material.use_nodes = True
    low.data.materials.clear()
    low.data.materials.append(material)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(options.get("samples", 8))
    bake = scene.render.bake
    bake.use_selected_to_active = True

    bake.cage_extrusion = cage  # sized above, before alignment
    bake.max_ray_distance = float(options.get("max_ray_distance", 0.0))
    # Margin dilates the baked islands outward so mip-mapping and bilinear
    # filtering cannot sample the empty gutter and bleed seams into the surface.
    bake.margin = int(options.get("margin", 8))
    bake.use_clear = True

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = {}

    flat_channels = []
    ordered = [name for name in BAKE_ORDER if name in maps]

    for index, map_name in enumerate(ordered):
        pass_type, colorspace, rewire_input = BAKE_PASSES[map_name]
        frac = 0.25 + 0.7 * (index / len(ordered))
        emit("bake", frac, f"Baking {map_name.replace('_', ' ')}…")

        # Either way, a channel whose source is a constant bakes to a flat map,
        # which is strictly worse than the scalar it came from — report it. Base
        # colour is exempt: a constant there is still the colour the mesh should
        # have, and the paint canvas has no other route to receive it.
        if rewire_input:
            rewired, driven = rewire_to_emit(high_objects, rewire_input)
            if not rewired:
                if map_name != "base_color":
                    flat_channels.append(map_name)
                else:
                    # Nothing to rewire, so EMIT would bake black. DIFFUSE is the
                    # only pass that reads an arbitrary shader: it loses metallic
                    # surfaces, but a dark map beats an empty one.
                    pass_type = "DIFFUSE"
            elif not driven and map_name != "base_color":
                flat_channels.append(map_name)
        elif map_name in PROBE_INPUTS:
            if not principled_input_is_linked(high_objects, PROBE_INPUTS[map_name]):
                flat_channels.append(map_name)

        # alpha=True is what turns a miss into something detectable. Blender masks
        # a texel whose ray found no high-poly surface out of the write entirely,
        # so it keeps the clear value — and the clear value is transparent
        # (0,0,0,0) only for an image with an alpha channel; without one it is
        # OPAQUE black, indistinguishable from a surface that is genuinely black.
        # That is what let a bake covering an eighth of this mesh pass for a good
        # one. With alpha, the channel is a per-texel hit mask: measure_coverage
        # counts it, pack_orm substitutes neutral values through it, and the client
        # composites the base colour through it instead of painting misses black.
        image = bpy.data.images.new(f"bake_{map_name}", width=resolution, height=resolution,
                                    alpha=True, float_buffer=False)
        image.alpha_mode = "STRAIGHT"
        image.colorspace_settings.name = colorspace

        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        material.node_tree.nodes.active = node

        # Selection defines the bake: every high-poly object selected, the
        # low-poly selected *and* active as the destination.
        bpy.ops.object.select_all(action="DESELECT")
        for obj in high_objects:
            obj.select_set(True)
        low.select_set(True)
        bpy.context.view_layer.objects.active = low

        bake_kwargs = {"type": pass_type, "use_clear": True}
        if pass_type == "DIFFUSE":
            # Only the base-colour fallback above reaches this. Without the filter
            # the transfer would bake lighting into the albedo too.
            bake_kwargs["pass_filter"] = {"COLOR"}

        try:
            bpy.ops.object.bake(**bake_kwargs)
        except Exception as exc:  # noqa: BLE001
            fail(2, f"Baking {map_name} failed: {exc}")

        path = outdir / f"{map_name}.png"
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
        written[map_name] = path.name

        material.node_tree.nodes.remove(node)
        bpy.data.images.remove(image)

    # Before ORM packing, which reads the alpha this measures and then discards it.
    emit("measure", 0.94, "Measuring coverage…")
    coverage = measure_coverage(low, outdir, written, resolution)

    emit("pack", 0.96, "Packing ORM…")
    orm_name, orm_channels = pack_orm(written, outdir, resolution)
    if orm_name:
        written["orm"] = orm_name

    # After packing, for the reasons in fill_gutters' docstring.
    emit("dilate", 0.98, "Filling UV gutters…")
    gutter_filled = fill_gutters(low, outdir, written)

    emit("done", 1.0, "Bake complete.")
    stats = {
        "maps": written,
        "resolution": resolution,
        "low_faces": len(low.data.polygons),
        "high_faces": int(sum(len(o.data.polygons) for o in high_objects)),
        "samples": scene.cycles.samples,
        "cage_extrusion": round(cage, 6),
        # How much of the UV layout the rays actually reached, and what the source
        # had to be moved by to get there. Reported unconditionally: a partial bake
        # is not an error (protruding detail legitimately misses) but it must never
        # again be indistinguishable from a complete one.
        **(coverage or {}),
        "alignment": alignment,
        # Every mesh object past the first in the low-poly file is not baked to.
        # Said out loud rather than dropped silently; the editor only ever uploads
        # one merged mesh, so this is for imported targets.
        "low_objects_ignored": len(low_objects) - 1,
        # Which of the ORM channels carry real baked data, so the client only binds
        # the material slots that were actually measured.
        "orm_channels": orm_channels if orm_name else [],
        # Channels whose source was a constant, not a texture — the map is flat.
        "flat_channels": flat_channels,
        # Fraction of each map that was gutter-filled so mip-mapping cannot bleed
        # the empty atlas into the islands. None means the fill did not run.
        "gutter_filled": gutter_filled,
    }
    print(f"{SENTINEL}{json.dumps({'type': 'result', 'ok': True, 'stats': stats})}", flush=True)


if __name__ == "__main__":
    main()
