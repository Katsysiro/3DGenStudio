// Server-side smooth normals for GLB meshes that ship without them.
//
// Several generators (Hunyuan3D 2.x among them) write POSITION + TEXCOORD_0 and
// no NORMAL. The glTF spec says such a primitive MUST be drawn flat-shaded, and
// three's GLTFLoader does exactly that (material.flatShading). That is harmless
// on a 500k-triangle source, where the facets are too small to see — and very
// visible after simplification, where gltfpack hands back the same normal-less
// primitive at 5k triangles.
//
// The Mesh Editor never showed the problem because its loader computes normals
// before anything is sent to gltfpack (createMergedGeometryFromObject in
// src/utils/meshEditor.js), so the simplifier carries the DENSE mesh's smooth
// normals through to the result. The Batch Optimize stage, the LOD route and
// the MCP optimize tool all upload the stored file untouched, so they did not.
// This gives them the same input.
//
// Same design as meshPivot.js: the glTF JSON is edited and one bufferView per
// computed normal set is appended to the binary chunk. Materials, images, UVs,
// skins and animations are copied through byte-for-byte.
//
// The normals are area-weighted per VERTEX, like three's computeVertexNormals
// and so like the editor. Welding them by position looks like the better choice
// (a UV seam's copies each see only their own chart's faces, so per-vertex
// normals crease there) but measured worse: on a 500k Hunyuan mesh taken to
// ~12.6k triangles, welded normals came out at 14.7 deg mean / 42.8 deg p95
// from the source's smooth surface, per-vertex at 12.6 / 35.2 — even with the
// welded normals as the reference. At that density the crease is invisible, and
// gltfpack simplifies better against it.
import { Buffer } from 'node:buffer';
import { parseGlb, serializeGlb } from './meshPivot.js';

const TRIANGLES = 4;
const FLOAT = 5126;
const ARRAY_BUFFER = 34962;

const COMPONENT_READERS = {
  5120: (view, offset) => view.getInt8(offset),
  5121: (view, offset) => view.getUint8(offset),
  5122: (view, offset) => view.getInt16(offset, true),
  5123: (view, offset) => view.getUint16(offset, true),
  5125: (view, offset) => view.getUint32(offset, true),
  5126: (view, offset) => view.getFloat32(offset, true)
};
const COMPONENT_BYTES = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

// These compress bufferViews, so the raw bytes are not vertex data. gltfpack
// decodes them itself; this simply leaves such a file alone.
const COMPRESSION_EXTENSIONS = ['KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_meshopt_compression'];

// Returns null for anything it cannot read faithfully, which the caller treats
// as "leave this primitive as it is" — a missing normal is better than a wrong one.
function readAccessor(json, bin, index) {
  const accessor = (json.accessors || [])[index];
  if (!accessor || accessor.sparse || accessor.bufferView === undefined || !bin) return null;
  const components = TYPE_COMPONENTS[accessor.type];
  const read = COMPONENT_READERS[accessor.componentType];
  const size = COMPONENT_BYTES[accessor.componentType];
  const view = (json.bufferViews || [])[accessor.bufferView];
  // Only buffer 0 is the GLB's own binary chunk; anything else is external.
  if (!components || !read || !view || (view.buffer ?? 0) !== 0) return null;

  const base = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = view.byteStride || components * size;
  if (accessor.count && base + (accessor.count - 1) * stride + components * size > bin.byteLength) return null;

  // Quantised positions are left as raw integers on purpose: their scale lives
  // in the node transform, and a normal is a direction in the mesh's own space,
  // which is exactly the space these integers are in.
  const data = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const out = new Float64Array(accessor.count * components);
  for (let i = 0; i < accessor.count; i += 1) {
    for (let c = 0; c < components; c += 1) {
      out[i * components + c] = read(data, base + i * stride + c * size);
    }
  }
  return out;
}

// Area-weighted smooth normals, one per vertex (see the header for why not welded).
function computeVertexNormals(positions, indices) {
  const vertexCount = positions.length / 3;
  const sums = new Float64Array(vertexCount * 3);
  const triangleCount = indices ? indices.length / 3 : vertexCount / 3;
  for (let t = 0; t < triangleCount; t += 1) {
    const a = indices ? indices[t * 3] : t * 3;
    const b = indices ? indices[t * 3 + 1] : t * 3 + 1;
    const c = indices ? indices[t * 3 + 2] : t * 3 + 2;
    if (a >= vertexCount || b >= vertexCount || c >= vertexCount) continue;
    const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
    const e1x = positions[b * 3] - ax, e1y = positions[b * 3 + 1] - ay, e1z = positions[b * 3 + 2] - az;
    const e2x = positions[c * 3] - ax, e2y = positions[c * 3 + 1] - ay, e2z = positions[c * 3 + 2] - az;
    // The unnormalised cross product is twice the triangle's area, which is
    // what makes the average area-weighted.
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    for (const v of [a, b, c]) {
      sums[v * 3] += nx;
      sums[v * 3 + 1] += ny;
      sums[v * 3 + 2] += nz;
    }
  }

  const normals = new Float32Array(vertexCount * 3);
  for (let v = 0; v < vertexCount; v += 1) {
    const length = Math.hypot(sums[v * 3], sums[v * 3 + 1], sums[v * 3 + 2]);
    // A vertex only on degenerate triangles has no direction; +Y beats NaN.
    if (length > 0) {
      normals[v * 3] = sums[v * 3] / length;
      normals[v * 3 + 1] = sums[v * 3 + 1] / length;
      normals[v * 3 + 2] = sums[v * 3 + 2] / length;
    } else {
      normals[v * 3 + 1] = 1;
    }
  }
  return normals;
}

// Adds a NORMAL attribute to every triangle primitive that lacks one.
// Returns { buffer, added } — `buffer` is the input itself when nothing needed
// adding (or the file cannot be edited safely), so calling this twice is cheap.
export function addMissingNormals(buffer) {
  let parsed;
  try {
    parsed = parseGlb(buffer);
  } catch {
    return { buffer, added: 0 };
  }
  const { json, bin } = parsed;

  const used = [...(json.extensionsUsed || []), ...(json.extensionsRequired || [])];
  if (used.some(name => COMPRESSION_EXTENSIONS.includes(name))) return { buffer, added: 0 };

  const missing = (json.meshes || []).flatMap(mesh => (mesh.primitives || []))
    .filter(primitive => (primitive.mode ?? TRIANGLES) === TRIANGLES
      && primitive.attributes?.POSITION !== undefined
      && primitive.attributes.NORMAL === undefined);
  if (!missing.length) return { buffer, added: 0 };

  const appended = [];
  let binLength = bin ? bin.length : 0;
  // Primitives sharing the same POSITION + indices share the computed accessor.
  const computed = new Map();
  let added = 0;

  for (const primitive of missing) {
    const key = `${primitive.attributes.POSITION}:${primitive.indices ?? ''}`;
    if (!computed.has(key)) {
      const positions = readAccessor(json, bin, primitive.attributes.POSITION);
      const indices = primitive.indices !== undefined ? readAccessor(json, bin, primitive.indices) : null;
      if (!positions || (primitive.indices !== undefined && !indices)) {
        computed.set(key, null);
        continue;
      }
      const normals = computeVertexNormals(positions, indices);

      const pad = (4 - (binLength % 4)) % 4;
      if (pad) appended.push(Buffer.alloc(pad));
      const byteOffset = binLength + pad;
      const bytes = Buffer.from(normals.buffer, normals.byteOffset, normals.byteLength);
      appended.push(bytes);
      binLength = byteOffset + bytes.length;

      json.bufferViews = json.bufferViews || [];
      json.bufferViews.push({ buffer: 0, byteOffset, byteLength: bytes.length, target: ARRAY_BUFFER });
      json.accessors.push({
        bufferView: json.bufferViews.length - 1,
        componentType: FLOAT,
        count: normals.length / 3,
        type: 'VEC3'
      });
      computed.set(key, json.accessors.length - 1);
    }
    const accessorIndex = computed.get(key);
    if (accessorIndex == null) continue;
    primitive.attributes.NORMAL = accessorIndex;
    added += 1;
  }

  if (!added) return { buffer, added: 0 };

  const nextBin = Buffer.concat([bin || Buffer.alloc(0), ...appended]);
  json.buffers = json.buffers?.length ? json.buffers : [{}];
  json.buffers[0] = { ...json.buffers[0], byteLength: nextBin.length };
  delete json.buffers[0].uri;
  return { buffer: serializeGlb(json, nextBin), added };
}
