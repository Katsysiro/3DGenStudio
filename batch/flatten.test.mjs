// node batch/flatten.test.mjs
//
// The backend flatten (batch/flatten.js): the atlas it packs before the bake and
// the GLB it builds from the baked albedo — against hand-built glTF files, so
// every rule the browser version follows is checked without a browser.
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { prepareFlattenGlb } from './flatten.js';
import { parseGlb, serializeGlb } from '../meshPivot.js';

let passed = 0;
const queued = [];
function test(name, fn) { queued.push([name, fn]); }

// A PNG header saying `size` x `size`, padded — flatten.js only reads IHDR.
function fakePng(size, label = '') {
  const out = Buffer.alloc(40 + label.length);
  out.writeUInt32BE(0x89504e47, 0);
  out.writeUInt32BE(0x0d0a1a0a, 4);
  out.writeUInt32BE(13, 8);
  out.write('IHDR', 12, 'ascii');
  out.writeUInt32BE(size, 16);
  out.writeUInt32BE(size, 20);
  out.write(label, 40, 'ascii');
  return out;
}

// Build a GLB from typed arrays. `data` entries become one accessor each, in
// order; `images` become embedded images after them.
function buildGlb(json, data, images = []) {
  const parts = [];
  let length = 0;
  const place = (bytes) => {
    const pad = (4 - (length % 4)) % 4;
    if (pad) { parts.push(Buffer.alloc(pad)); length += pad; }
    parts.push(bytes);
    const view = { buffer: 0, byteOffset: length, byteLength: bytes.length };
    length += bytes.length;
    return view;
  };
  json.bufferViews = [];
  json.accessors = data.map(({ array, ...accessor }) => {
    json.bufferViews.push(place(Buffer.from(array.buffer, array.byteOffset, array.byteLength)));
    return { bufferView: json.bufferViews.length - 1, ...accessor };
  });
  json.images = images.map(bytes => {
    json.bufferViews.push(place(bytes));
    return { mimeType: 'image/png', bufferView: json.bufferViews.length - 1 };
  });
  if (!json.images.length) delete json.images;
  json.buffers = [{ byteLength: length }];
  return serializeGlb({ asset: { version: '2.0' }, ...json }, Buffer.concat(parts));
}

const f32 = values => Float32Array.from(values);
const quadPositions = (x) => f32([x, 0, 0, x + 1, 0, 0, x + 1, 1, 0, x, 1, 0]);
const FULL_UV = f32([0, 0, 1, 0, 1, 1, 0, 1]);
const QUAD_INDEX = Uint16Array.from([0, 1, 2, 0, 2, 3]);

// Two textured quads, each filling the whole 0..1 square of its own texture —
// the two-material layout that must repack. The first is skinned and animated.
function twoMaterialGlb({ doubleSidedSecond = false, blendSecond = false } = {}) {
  return buildGlb({
    meshes: [{
      primitives: [
        { attributes: { POSITION: 0, TEXCOORD_0: 1, JOINTS_0: 2, WEIGHTS_0: 3, TANGENT: 4 }, indices: 5, material: 0 },
        { attributes: { POSITION: 6, TEXCOORD_0: 7, COLOR_0: 8 }, indices: 9, material: 1 }
      ]
    }],
    materials: [
      { name: 'Body', pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicFactor: 1 }, normalTexture: { index: 1 } },
      {
        name: 'Glass',
        pbrMetallicRoughness: { baseColorTexture: { index: 1 } },
        ...(doubleSidedSecond ? { doubleSided: true } : {}),
        ...(blendSecond ? { alphaMode: 'BLEND' } : {}),
        extensions: { KHR_materials_emissive_strength: { emissiveStrength: 2 } }
      }
    ],
    textures: [{ source: 0 }, { source: 1 }],
    extensionsUsed: ['KHR_materials_emissive_strength'],
    nodes: [{ mesh: 0, skin: 0 }, { name: 'Hips' }],
    skins: [{ joints: [1] }],
    animations: [{ name: 'Wave', channels: [{ sampler: 0, target: { node: 1, path: 'translation' } }], samplers: [{ input: 10, output: 11 }] }],
    scenes: [{ nodes: [0, 1] }],
    scene: 0
  }, [
    { array: quadPositions(0), componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
    { array: FULL_UV, componentType: 5126, count: 4, type: 'VEC2' },
    { array: new Uint8Array(16), componentType: 5121, count: 4, type: 'VEC4' },
    { array: f32([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), componentType: 5126, count: 4, type: 'VEC4' },
    { array: f32(new Array(16).fill(1)), componentType: 5126, count: 4, type: 'VEC4' },
    { array: QUAD_INDEX, componentType: 5123, count: 6, type: 'SCALAR' },
    { array: quadPositions(2), componentType: 5126, count: 4, type: 'VEC3', min: [2, 0, 0], max: [3, 1, 0] },
    { array: FULL_UV, componentType: 5126, count: 4, type: 'VEC2' },
    { array: f32(new Array(16).fill(0.5)), componentType: 5126, count: 4, type: 'VEC4' },
    { array: QUAD_INDEX, componentType: 5123, count: 6, type: 'SCALAR' },
    { array: f32([0, 1]), componentType: 5126, count: 2, type: 'SCALAR', min: [0], max: [1] },
    { array: f32([0, 0, 0, 0, 1, 0]), componentType: 5126, count: 2, type: 'VEC3' }
  ], [fakePng(1024, 'body-texture'.repeat(200)), fakePng(512, 'glass-texture'.repeat(200))]);
}

function accessorValues(glb, index) {
  const { json, bin } = parseGlb(glb);
  const accessor = json.accessors[index];
  const view = json.bufferViews[accessor.bufferView];
  const components = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[accessor.type];
  const Ctor = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }[accessor.componentType];
  const stride = view.byteStride || components * Ctor.BYTES_PER_ELEMENT;
  const out = [];
  for (let i = 0; i < accessor.count; i += 1) {
    const start = bin.byteOffset + (view.byteOffset || 0) + (accessor.byteOffset || 0) + i * stride;
    const element = new Ctor(bin.buffer.slice(start, start + components * Ctor.BYTES_PER_ELEMENT));
    out.push(...element);
  }
  return out;
}

// Rasterised union area of a set of UV triangles, as a fraction of written area
// — 1.0 means nothing overlaps.
function overlapRatio(triangles) {
  const grid = 512;
  const hits = new Uint8Array(grid * grid);
  let written = 0;
  for (const [u0, v0, u1, v1, u2, v2] of triangles) {
    const det = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
    written += Math.abs(det) / 2;
    for (let py = 0; py < grid; py += 1) {
      for (let px = 0; px < grid; px += 1) {
        const x = (px + 0.5) / grid; const y = (py + 0.5) / grid;
        const w0 = ((x - u0) * (v2 - v0) - (u2 - u0) * (y - v0)) / det;
        const w1 = ((u1 - u0) * (y - v0) - (x - u0) * (v1 - v0)) / det;
        if (w0 >= 0 && w1 >= 0 && w0 + w1 <= 1) hits[py * grid + px] = 1;
      }
    }
  }
  const covered = hits.reduce((sum, hit) => sum + hit, 0) / (grid * grid);
  return written / covered;
}

function atlasTriangles(glb, channel) {
  const { json } = parseGlb(glb);
  const triangles = [];
  for (const primitive of json.meshes.flatMap(mesh => mesh.primitives)) {
    const uv = accessorValues(glb, primitive.attributes[`TEXCOORD_${channel}`]);
    const index = accessorValues(glb, primitive.indices);
    for (let t = 0; t < index.length; t += 3) {
      triangles.push([0, 1, 2].flatMap(k => [uv[index[t + k] * 2], uv[index[t + k] * 2 + 1]]));
    }
  }
  return triangles;
}

test('two materials that each fill their own square are repacked into one atlas', () => {
  const prepared = prepareFlattenGlb(twoMaterialGlb(), { resolution: 1024 });
  assert.equal(prepared.channel, 1, 'the atlas goes in the first UV set no texture reads');
  assert.equal(prepared.atlas.repacked, true);
  assert.ok(prepared.atlas.atlasWrites > 1.9, `a double-painted layout reads ~2x (got ${prepared.atlas.atlasWrites})`);
  assert.equal(prepared.atlas.islands, 2);

  const triangles = atlasTriangles(prepared.bakeTarget, 1);
  for (const triangle of triangles) {
    for (const value of triangle) assert.ok(value >= 0 && value <= 1, `atlas UV ${value} is inside the atlas`);
  }
  assert.ok(overlapRatio(triangles) < 1.02, 'the two quads no longer share texels');
});

test('the bake target keeps the source materials and UV0 but not the clips', () => {
  const { bakeTarget } = prepareFlattenGlb(twoMaterialGlb(), { resolution: 1024 });
  const { json } = parseGlb(bakeTarget);
  assert.equal(json.animations, undefined, 'clips only cost Blender time');
  assert.equal(json.materials.length, 2);
  assert.equal(json.images.length, 2);
  assert.deepEqual(accessorValues(bakeTarget, json.meshes[0].primitives[0].attributes.TEXCOORD_0), [...FULL_UV]);
  assert.equal(json.accessors.length, 12 - 2 + 2, 'animation accessors dropped, two atlas UV sets added');
});

test('the finished mesh has one texture and one material, and keeps its rig and clip', () => {
  const prepared = prepareFlattenGlb(twoMaterialGlb(), { resolution: 1024 });
  const albedo = fakePng(1024, 'albedo');
  const { buffer, materialCount } = prepared.finish(albedo, { unlit: true, name: 'knight' });
  const { json } = parseGlb(buffer);

  assert.equal(materialCount, 1);
  assert.equal(json.materials.length, 1);
  assert.equal(json.materials[0].name, 'knight_flat');
  assert.deepEqual(json.materials[0].pbrMetallicRoughness, { baseColorTexture: { index: 0 }, metallicFactor: 0, roughnessFactor: 1 });
  assert.deepEqual(json.materials[0].extensions, { KHR_materials_unlit: {} });
  assert.deepEqual(json.extensionsUsed, ['KHR_materials_unlit'], 'the old material extension went with its material');
  assert.equal(json.images.length, 1);
  assert.equal(json.textures.length, 1);

  const primitives = json.meshes[0].primitives;
  for (const primitive of primitives) {
    assert.equal(primitive.material, 0);
    assert.deepEqual(Object.keys(primitive.attributes).filter(name => /TEXCOORD|COLOR|TANGENT/.test(name)), ['TEXCOORD_0']);
  }
  // UV0 is now the atlas the bake wrote through.
  assert.deepEqual(atlasTriangles(buffer, 0), atlasTriangles(prepared.bakeTarget, 1));

  assert.equal(json.skins.length, 1);
  assert.deepEqual(accessorValues(buffer, primitives[0].attributes.WEIGHTS_0).slice(0, 4), [1, 0, 0, 0]);
  assert.equal(json.animations.length, 1);
  assert.deepEqual(accessorValues(buffer, json.animations[0].samplers[0].output), [0, 0, 0, 0, 1, 0]);

  // The source textures are gone from the binary, not just unreferenced.
  assert.equal(buffer.includes('body-texture'), false);
  assert.equal(buffer.includes('albedo'), true);
});

test('a simple-lit flatten saves a plain rough, non-metal material', () => {
  const prepared = prepareFlattenGlb(twoMaterialGlb(), { resolution: 1024 });
  const { json } = parseGlb(prepared.finish(fakePng(1024), { unlit: false, name: 'knight' }).buffer);
  assert.equal(json.materials[0].extensions, undefined);
  assert.equal(json.extensionsUsed, undefined);
});

test('render state is kept per part: double-sided and blended only where they were', () => {
  const prepared = prepareFlattenGlb(twoMaterialGlb({ doubleSidedSecond: true, blendSecond: true }), { resolution: 1024 });
  const opaqueAlbedo = prepared.finish(fakePng(1024), { hasAlpha: false, name: 'k' });
  let { json } = parseGlb(opaqueAlbedo.buffer);
  assert.equal(opaqueAlbedo.materialCount, 2, 'sidedness still differs without alpha');
  assert.equal(json.materials[json.meshes[0].primitives[0].material].doubleSided, undefined);
  assert.equal(json.materials[json.meshes[0].primitives[1].material].doubleSided, true);
  assert.equal(json.materials.some(material => material.alphaMode), false, 'no alpha in the bake, nothing blends');

  const withAlpha = prepareFlattenGlb(twoMaterialGlb({ blendSecond: true }), { resolution: 1024 })
    .finish(fakePng(1024), { hasAlpha: true, name: 'k' });
  ({ json } = parseGlb(withAlpha.buffer));
  assert.equal(json.materials[json.meshes[0].primitives[0].material].alphaMode, undefined, 'the opaque part stays opaque');
  assert.equal(json.materials[json.meshes[0].primitives[1].material].alphaMode, 'BLEND');
});

test('a single clean layout is kept as it is', () => {
  const glb = buildGlb({
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 2, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: {} }],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }]
  }, [
    { array: quadPositions(0), componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
    { array: f32([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]), componentType: 5126, count: 4, type: 'VEC2' },
    { array: QUAD_INDEX, componentType: 5123, count: 6, type: 'SCALAR' }
  ]);
  const prepared = prepareFlattenGlb(glb);
  assert.equal(prepared.atlas.repacked, false);
  const primitive = parseGlb(prepared.bakeTarget).json.meshes[0].primitives[0];
  assert.equal(primitive.attributes.TEXCOORD_1, primitive.attributes.TEXCOORD_0, 'the atlas IS UV0');
});

test('faces without UVs get an island each, on vertices of their own', () => {
  // The quad's second triangle shares vertices 0 and 2 with the first, so its
  // new UVs cannot be written in place.
  const glb = buildGlb({
    meshes: [{ primitives: [{ attributes: { POSITION: 0, JOINTS_0: 1 }, indices: 2 }] }],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }]
  }, [
    { array: quadPositions(0), componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
    { array: Uint8Array.from([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]), componentType: 5121, count: 4, type: 'VEC4' },
    { array: QUAD_INDEX, componentType: 5123, count: 6, type: 'SCALAR' }
  ]);
  const prepared = prepareFlattenGlb(glb, { resolution: 512 });
  assert.equal(prepared.atlas.unmapped, 2);
  assert.equal(prepared.atlas.islands, 2);

  const { json } = parseGlb(prepared.bakeTarget);
  const primitive = json.meshes[0].primitives[0];
  assert.equal(json.accessors[primitive.attributes.POSITION].count, 10, '4 originals + 3 per split face');
  assert.deepEqual(json.accessors[primitive.attributes.POSITION].max, [1, 1, 0], 'copies keep the bounds');
  // Every attribute was copied with its vertex, skin included.
  const joints = accessorValues(prepared.bakeTarget, primitive.attributes.JOINTS_0);
  const index = accessorValues(prepared.bakeTarget, primitive.indices);
  assert.deepEqual(index.map(v => joints[v * 4]), [1, 2, 3, 1, 3, 4]);
  assert.ok(overlapRatio(atlasTriangles(prepared.bakeTarget, 1)) < 1.02);
  // No UVs before: UV0 is zeroes, which is what its textures sampled then.
  assert.ok(accessorValues(prepared.bakeTarget, primitive.attributes.TEXCOORD_0).every(value => value === 0));
});

test('a mesh drawn by two nodes is split so each copy gets its own texels', () => {
  const glb = buildGlb({
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 2, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: {} }],
    nodes: [{ mesh: 0 }, { mesh: 0, translation: [5, 0, 0] }],
    scenes: [{ nodes: [0, 1] }]
  }, [
    { array: quadPositions(0), componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
    { array: FULL_UV, componentType: 5126, count: 4, type: 'VEC2' },
    { array: QUAD_INDEX, componentType: 5123, count: 6, type: 'SCALAR' }
  ]);
  const prepared = prepareFlattenGlb(glb, { resolution: 512 });
  const { json } = parseGlb(prepared.bakeTarget);
  assert.equal(json.meshes.length, 2);
  assert.deepEqual(json.nodes.map(node => node.mesh), [0, 1]);
  assert.equal(prepared.atlas.repacked, true);
  assert.ok(overlapRatio(atlasTriangles(prepared.bakeTarget, 1)) < 1.02);
});

test('a texture already reading UV1 pushes the atlas to UV2', () => {
  const glb = buildGlb({
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 2, material: 0 }] }],
    materials: [{ occlusionTexture: { index: 0, texCoord: 1 } }],
    textures: [{ source: 0 }],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }]
  }, [
    { array: quadPositions(0), componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
    { array: f32([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]), componentType: 5126, count: 4, type: 'VEC2' },
    { array: QUAD_INDEX, componentType: 5123, count: 6, type: 'SCALAR' }
  ], [fakePng(256)]);
  const prepared = prepareFlattenGlb(glb);
  assert.equal(prepared.channel, 2);
  const primitive = parseGlb(prepared.bakeTarget).json.meshes[0].primitives[0];
  assert.notEqual(primitive.attributes.TEXCOORD_1, undefined, 'the set below the atlas exists');
});

test('a compressed mesh is refused with a way out', () => {
  const { json, bin } = parseGlb(twoMaterialGlb());
  json.extensionsUsed = ['KHR_draco_mesh_compression'];
  assert.throws(() => prepareFlattenGlb(serializeGlb(json, bin)), /KHR_draco_mesh_compression.*Export dialog/);
});

for (const [name, fn] of queued) {
  try {
    await fn();
    passed += 1;
  } catch (err) {
    console.error(`FAIL ${name}\n`, err);
    process.exitCode = 1;
  }
}
console.log(`${passed}/${queued.length} batch flatten tests passed`);
