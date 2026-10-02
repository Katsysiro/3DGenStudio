import * as THREE from 'three'

/**
 * gpuTextureBake.js
 * ─────────────────────────────────────────────────────────────────────────────
 * GPU UV-space ("texture-space") projection bake.
 *
 * This is the relocation recommended in the projection-mode analysis: instead of
 * a single-threaded per-texel JS loop with a per-texel BVH raycast, the mesh is
 * rasterised directly into UV space on the GPU. Every texel of every triangle is
 * shaded in parallel, projective-textured from the source view, gated by a real
 * depth-map visibility test, and accumulated as Σ(color·w) / Σ(w).
 *
 * Pipeline (per the analysis):
 *   Pass 1  Depth pre-pass   — render the mesh from the view camera into a native
 *                              DepthTexture. Back faces are rendered so depth acne
 *                              is pushed behind the surface.
 *   Pass 2  UV-space bake    — vertex shader places each triangle by its UV, so
 *                              the rasteriser fills every texel. Fragment shader
 *                              projects to the view, rejects back-facing / off-screen
 *                              / occluded / behind-projector texels, weights the
 *                              sample by pow(n·v, ALPHA), and writes vec4(rgb·w, w).
 *   (accum) ping-pong add    — each view is rendered to a temp RGBA16F target and
 *                              summed into the accumulator with a full-screen add.
 *                              Avoids the EXT_float_blend gotcha (no hardware float
 *                              blending required; only render-to-float).
 *   Pass 3  Resolve          — rgb = accum.rgb / max(accum.a, eps) → RGBA8.
 *   (dilate) JFA gutter fill — jump-flooding nearest-valid-texel dilation, capped
 *                              at `dilatePixels`, to kill atlas/mip seam bleed.
 *
 * Why each of the four reported problems is addressed:
 *   3 (slow)   the per-texel JS loop and per-texel raycast both disappear; N
 *              rasterisation passes run at GPU fill-rate.
 *   4 (paints  the depth-map compare resolves which surface the view sees first
 *      behind) along the projector ray, in-shader, with no CPU readback.
 *   2 (leak)   there is no 2D UV-space feather and no screen-space seam smear;
 *              blending is purely per-texel accumulation of texels that each passed
 *              the visibility gate, so weight cannot reach an occluded surface or a
 *              neighbouring island.
 *   1 (seams)  a steep cosine (ALPHA≈6) makes the front-most view dominate sharply;
 *              weighted accumulation smooths the residual; JFA dilation removes the
 *              mip/island-border seam.
 *
 * The module is self-contained: it owns its renderer and render targets and never
 * mutates the caller's geometry. If the platform cannot render to a float target
 * it returns null so the caller can fall back to the existing CPU path.
 */

const ACCUM_EPS = 1e-5
// Minimum accumulated weight for a texel to SEED gutter dilation. Tied to the
// cosine exponent (DEFAULT_ALPHA): at alpha 6 this is ~facing > 0.52, so only
// reasonably head-on coverage extends into gutters — a view's grazing edge (matte)
// never seeds, which is what produced faint gray seams where two views' dilations
// met in a UV gutter.
const DILATE_SEED_MIN_WEIGHT = 0.02
const DEFAULT_ALPHA = 6.0
const DEFAULT_MIN_BIAS = 0.0008
const DEFAULT_MAX_BIAS = 0.0045
const SUV_EDGE_EPS = 0.0015
// Silhouette-edge fade (ported from image-to-3dlab's Pixel Match). A view's weight
// ramps from ~0 at the mesh's silhouette IN THAT VIEW to full over this many source
// -image pixels. The silhouette is where a generated view bleeds its backdrop/matte
// colour and where a view that drifted a few pixels from the geometry misses the
// surface, so those texels should be the first to give way to a better view.
const DEFAULT_EDGE_FADE_PX = 6
// Occlusion edges (an arm in front of the torso) fade too: neighbouring depth-map
// pixels whose LINEAR depth differs by more than this fraction of the mesh's
// bounding diameter are treated as a silhouette. 0.02 ≈ a surface tilted ~87° at
// 1024px, so ordinary slopes never trigger it. 0 = outer silhouette only.
const DEFAULT_EDGE_DEPTH_JUMP = 0.02

// ── Capability detection ────────────────────────────────────────────────────
let cachedSupport = null

export function isGpuBakeSupported() {
  if (cachedSupport !== null) {
    return cachedSupport
  }
  try {
    const canvas = typeof document !== 'undefined' ? document.createElement('canvas') : null
    const gl = canvas?.getContext?.('webgl2')
    if (!gl) {
      cachedSupport = false
      return false
    }
    // Float color attachments are required to accumulate in linear space.
    const hasColorFloat = Boolean(gl.getExtension('EXT_color_buffer_float'))
    cachedSupport = hasColorFloat
    // Tidy up the probe context.
    gl.getExtension('WEBGL_lose_context')?.loseContext?.()
    return cachedSupport
  } catch {
    cachedSupport = false
    return false
  }
}

// ── Shared GPU resources ──────────────────────────────────────────────────────
let sharedRenderer = null
let sharedQuadScene = null
let sharedQuadCamera = null
let sharedQuadMesh = null

// A persistent offscreen WebGLRenderer can silently lose its GPU context
// (driver reset, tab backgrounding, or the browser evicting the oldest WebGL
// context once too many exist — e.g. after a ComfyUI crash followed by retries).
// Three.js does NOT auto-recreate it, so a dead renderer bakes NOTHING onto the
// texture until a full page reload. Treat a lost context as "needs rebuild".
function isBakeContextLost(renderer) {
  if (!renderer) return false
  try {
    return renderer.getContext().isContextLost()
  } catch {
    return true
  }
}

function getBakeRenderer() {
  if (isBakeContextLost(sharedRenderer)) {
    sharedRenderer.dispose?.()
    sharedRenderer = null
  }
  if (!sharedRenderer) {
    sharedRenderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance'
    })
    sharedRenderer.setPixelRatio(1)
    sharedRenderer.autoClear = false
    sharedRenderer.outputColorSpace = THREE.LinearSRGBColorSpace
    sharedRenderer.toneMapping = THREE.NoToneMapping
  }
  return sharedRenderer
}

// A reusable clip-space quad for the full-screen resolve / add / JFA passes.
function getQuad() {
  if (!sharedQuadScene) {
    sharedQuadScene = new THREE.Scene()
    sharedQuadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const geom = new THREE.PlaneGeometry(2, 2)
    sharedQuadMesh = new THREE.Mesh(geom, null)
    sharedQuadMesh.frustumCulled = false
    sharedQuadScene.add(sharedQuadMesh)
  }
  return { scene: sharedQuadScene, camera: sharedQuadCamera, mesh: sharedQuadMesh }
}

function runFullScreenPass(renderer, target, material) {
  const { scene, camera, mesh } = getQuad()
  const prev = mesh.material
  mesh.material = material
  renderer.setRenderTarget(target)
  renderer.clear(true, true, false)
  renderer.render(scene, camera)
  renderer.setRenderTarget(null)
  mesh.material = prev
}

function makeFloatTarget(width, height, { type = THREE.HalfFloatType } = {}) {
  return new THREE.WebGLRenderTarget(width, height, {
    format: THREE.RGBAFormat,
    type,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false
  })
}

function makeByteTarget(width, height) {
  return new THREE.WebGLRenderTarget(width, height, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false
  })
}

// Depth target whose depth attachment is a sampleable DepthTexture.
// IMPORTANT: three.js only wires up the DepthTexture (setupRenderTarget calls
// setupDepthRenderbuffer → setupDepthTexture) when `depthBuffer` is true AND the
// depthTexture is present at setup time. Building a depthBuffer:false target and
// assigning `.depthTexture` afterwards leaves it UNATTACHED: the prepass writes no
// depth, the bake shader samples 0, and every texel fails `curDepth > 0 + bias`,
// so coverage comes back empty. Passing both here is the correct way.
function makeDepthTarget(width, height) {
  const depthTexture = new THREE.DepthTexture(width, height)
  depthTexture.type = THREE.UnsignedIntType
  depthTexture.minFilter = THREE.NearestFilter
  depthTexture.magFilter = THREE.NearestFilter
  return new THREE.WebGLRenderTarget(width, height, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    depthBuffer: true,   // ← required so three attaches the depth attachment
    stencilBuffer: false,
    depthTexture,        // ← present at setup so three wires it up
    generateMipmaps: false
  })
}

// ── GLSL ─────────────────────────────────────────────────────────────────────
// Pass 2 bake shader. RawShaderMaterial / GLSL3 so we control every attribute
// and uniform and inject nothing from three.js's MeshStandardMaterial chunks.
// NOTE: do NOT put '#version 300 es' here — three.js prepends it automatically
// because the material sets glslVersion: THREE.GLSL3. Declaring it twice makes the
// shader fail to compile ("Vertex shader is not compiled").
const BAKE_VERT = /* glsl */`precision highp float;

in vec3 position;     // object-space vertex position
in vec3 normal;       // object-space vertex normal
in vec2 aTexUV;       // PRE-TRANSFORMED texture-space UV in [0,1] (flipY/offset/etc already applied on CPU)

uniform mat4 uModel;        // mesh.matrixWorld
uniform mat3 uNormalMat;    // normal matrix (world)
uniform mat4 uProjView;     // projector projection * view

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec4 vProjCoord;

void main() {
  vec4 world = uModel * vec4(position, 1.0);
  vWorldPos = world.xyz;
  vWorldNormal = normalize(uNormalMat * normal);
  vProjCoord = uProjView * world;

  // Place the triangle in UV space: the rasteriser now fills every texel.
  // aTexUV already matches mapUvToCanvasPoint() (the CPU convention), so the
  // output texture is pixel-identical in layout to the CPU bake.
  gl_Position = vec4(aTexUV * 2.0 - 1.0, 0.0, 1.0);
}
`

const BAKE_FRAG = /* glsl */`precision highp float;
precision highp int;
precision highp sampler2D;

in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec4 vProjCoord;

uniform sampler2D uSourceView;   // the rendered/inpainted view image (flipY default = true)
uniform sampler2D uMaskView;     // crop / inpaint mask (alpha used); same orientation as uSourceView
uniform sampler2D uDepthMap;     // native depth texture from the projector depth pre-pass
uniform vec3  uProjectorPos;     // projector (camera) world position
uniform float uAlpha;            // cosine exponent (steep front weighting)
uniform float uViewWeight;       // per-view opacity multiplier
uniform float uMinBias;
uniform float uMaxBias;
uniform float uUseDepth;         // 1.0 if depth map valid, else 0.0
uniform float uUseMask;          // 1.0 if a separate mask texture is bound
uniform float uCullBackfaces;    // 1.0 reject back faces, 0.0 accept both sides
uniform float uMinMaskAlpha;
uniform float uMinFacing;        // reject texels seen more grazingly than this cos
uniform vec3  uGain;             // per-view per-channel Brown-Lowe gain (1,1,1 = identity)
uniform sampler2D uEdgeSeed;     // JFA nearest-silhouette-pixel field in projector space
uniform vec2  uEdgeSize;         // its size in pixels
uniform float uEdgePx;           // fade width in edge-field pixels (0 = no fade)
uniform float uOutputFade;       // 1.0 = write vec4(edgeFade, 1, 0, 1) instead of colour

out vec4 outColor;

void main() {
  // Behind the projector → reject (do NOT clamp w; clamping is the three-projected
  // -material bug the analysis calls out).
  if (vProjCoord.w <= 0.0) discard;

  vec2 sUV = (vProjCoord.xy / vProjCoord.w) * 0.5 + 0.5;
  if (sUV.x < SUV_EDGE_EPS || sUV.y < SUV_EDGE_EPS ||
      sUV.x > 1.0 - SUV_EDGE_EPS || sUV.y > 1.0 - SUV_EDGE_EPS) discard;

  // Facing test. The projector "view direction" at this texel.
  vec3 projDir = normalize(uProjectorPos - vWorldPos);
  vec3 n = normalize(vWorldNormal);
  float ndotv = dot(n, projDir);
  float facing = uCullBackfaces > 0.5 ? ndotv : abs(ndotv);
  // Reject grazing texels: a view should only contribute where it sees the surface
  // reasonably head-on. The near-vertical legs seen from a top view are extremely
  // grazing and there the projector samples the dark background at its silhouette
  // (matte) — projecting black lines onto legs a better-facing view already owns.
  // Rejecting them (not just down-weighting) keeps such texels OUT of this view's
  // coverage, so the composite leaves the better view's result untouched.
  if (facing <= max(uMinFacing, 0.0)) discard;

  // Occlusion: compare this texel's projector-space depth against the depth map.
  if (uUseDepth > 0.5) {
    float curDepth = (vProjCoord.z / vProjCoord.w) * 0.5 + 0.5;
    float stored = texture(uDepthMap, sUV).r;
    float bias = max(uMaxBias * (1.0 - facing), uMinBias); // slope-scaled
    if (curDepth > stored + bias) discard;                  // occluded
  }

  vec4 src = texture(uSourceView, sUV);
  float maskA = uUseMask > 0.5 ? texture(uMaskView, sUV).a : 1.0;
  float alpha = src.a * maskA;
  if (alpha <= uMinMaskAlpha) discard;

  // Silhouette-edge fade: distance (in projector pixels) from where this texel lands
  // to the nearest silhouette pixel of the mesh in this view, smoothstepped over
  // uEdgePx.
  float edgeFade = 1.0;
  if (uEdgePx > 0.0) {
    // The silhouette comes from a rasterised depth map, so it is a pixel staircase
    // and the distance to it wobbles by ~half a pixel along a smooth rim — comb
    // streaks wherever a later view takes the rim over. Measure from the sub-pixel
    // position (not the whole pixel) and average a 3x3 neighbourhood to iron the
    // staircase out. A pixel with no seed in range is interior (fade 1).
    vec2 p = sUV * uEdgeSize - 0.5;
    float sum = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        vec2 q = p + vec2(float(dx), float(dy));
        vec4 seed = texture(uEdgeSeed, (q + 0.5) / uEdgeSize);
        float t = seed.w > 0.5 ? clamp(distance(q, seed.xy) / uEdgePx, 0.0, 1.0) : 1.0;
        sum += t * t * (3.0 - 2.0 * t);
      }
    }
    edgeFade = sum / 9.0;
  }

  // Steep cosine weight is the seam-collapsing primary weight.
  float w = pow(facing, uAlpha) * uViewWeight * alpha;
  if (w <= 0.0) discard;
  // The fade must lower the weight WITHOUT uncovering anything: a grazing rim texel
  // already sits near ACCUM_EPS, and fading it below that (coverage is acc.a > eps)
  // would punch holes along every silhouette. Anything covered stays covered, at a
  // weight that loses to any real sample from another view.
  if (w > ACCUM_EPS) w = max(w * edgeFade, 2.0 * ACCUM_EPS);

  if (uOutputFade > 0.5) {
    outColor = vec4(edgeFade, 1.0, 0.0, 1.0);
    return;
  }
  vec3 rgb = clamp(src.rgb * uGain, 0.0, 8.0);
  outColor = vec4(rgb * w, w);   // Σ(c·w), Σ(w)
}
`.replace(/SUV_EDGE_EPS/g, SUV_EDGE_EPS.toFixed(5))
  .replace(/ACCUM_EPS/g, ACCUM_EPS.toExponential())

// Full-screen add: dst = a + b
const ADD_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D uA;
uniform sampler2D uB;
void main() {
  gl_FragColor = texture2D(uA, vUv) + texture2D(uB, vUv);
}
`

const QUAD_VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

// Resolve: normalise accumulator → straight RGBA8, alpha encodes coverage.
const RESOLVE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D uAccum;
uniform float uEps;
void main() {
  vec4 acc = texture2D(uAccum, vUv);
  if (acc.a <= uEps) { gl_FragColor = vec4(0.0); return; }
  gl_FragColor = vec4(acc.rgb / acc.a, 1.0);
}
`

// JFA seed init: only texels whose accumulated WEIGHT exceeds uSeedMin become
// seeds (carry their pixel coords); others carry (-1,-1). Seeding on weight (not
// mere presence) means the gutter fill extends only MEANINGFUL coverage. A view's
// grazing-garbage coverage (tiny weight) must not seed, or its gray/matte edge
// colour bleeds into UV gutters and averages with a good view's fill → gray seams.
const JFA_INIT_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D uAccum;
uniform vec2 uSize;
uniform float uSeedMin;
void main() {
  vec4 acc = texture2D(uAccum, vUv);
  vec2 px = floor(vUv * uSize);
  if (acc.a > uSeedMin) gl_FragColor = vec4(px, 0.0, 1.0);
  else gl_FragColor = vec4(-1.0, -1.0, 0.0, 0.0);
}
`

// JFA step: keep the nearest valid seed among the 3x3 neighbourhood at `uStep`.
const JFA_STEP_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D uSeed;
uniform vec2 uSize;
uniform float uStep;
void main() {
  vec2 px = floor(vUv * uSize);
  vec4 best = texture2D(uSeed, vUv);
  float bestD = best.w > 0.5 ? distance(px, best.xy) : 1e20;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2 nUv = (px + vec2(float(dx), float(dy)) * uStep + 0.5) / uSize;
      if (nUv.x < 0.0 || nUv.y < 0.0 || nUv.x > 1.0 || nUv.y > 1.0) continue;
      vec4 s = texture2D(uSeed, nUv);
      if (s.w < 0.5) continue;
      float d = distance(px, s.xy);
      if (d < bestD) { bestD = d; best = vec4(s.xy, 0.0, 1.0); }
    }
  }
  gl_FragColor = best;
}
`

// JFA apply: fill invalid texels from their nearest seed colour, capped at uMaxDist.
const JFA_APPLY_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D uColor;   // resolved RGBA8 (as float in [0,1])
uniform sampler2D uSeed;
uniform vec2 uSize;
uniform float uMaxDist;
void main() {
  vec4 c = texture2D(uColor, vUv);
  if (c.a > 0.5) { gl_FragColor = c; return; } // already valid
  vec4 seed = texture2D(uSeed, vUv);
  if (seed.w < 0.5) { gl_FragColor = vec4(0.0); return; }
  vec2 px = floor(vUv * uSize);
  if (distance(px, seed.xy) > uMaxDist) { gl_FragColor = vec4(0.0); return; }
  vec2 seedUv = (seed.xy + 0.5) / uSize;
  vec3 rgb = texture2D(uColor, seedUv).rgb;
  gl_FragColor = vec4(rgb, 1.0);
}
`

// Silhouette seeds for the edge fade, in PROJECTOR space (one pixel per depth-map
// pixel). A pixel seeds when it is background (nothing drawn: depth stays at the
// cleared 1.0) or when it sits on an occlusion edge. The JFA then gives every
// surface pixel its nearest seed, so the bake shader reads "how far inside this
// view's silhouette am I".
//
// Occlusion edge = a step to a 4-neighbour of more than uJump in LINEAR depth that is
// also several times the steps on either side of it. A size threshold alone fires
// all along a steep-but-continuous surface (a sphere's rim climbs fast near the
// tangent), and it fires irregularly with the pixel phase, which showed as banding
// across the whole rim. A real occlusion is an isolated step; a slope is not. Where
// either flanking pixel is background, the background seed already covers it.
const EDGE_SEED_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D uDepthMap;
uniform vec2 uSize;
uniform float uNear;
uniform float uFar;
uniform float uOrtho;
uniform float uJump;   // linear-depth step that counts as an edge (0 = off)
const float BG = 0.99999;
const float ISOLATION = 4.0;   // a step this many times its flanks is an occlusion
float linearDepth(float d) {
  if (uOrtho > 0.5) return uNear + d * (uFar - uNear);
  return (uNear * uFar) / max(uFar - d * (uFar - uNear), 1e-6);
}
// Raw depth at a pixel, or 1.0 (background) outside the map.
float depthAt(vec2 p) {
  if (p.x < 0.0 || p.y < 0.0 || p.x >= uSize.x || p.y >= uSize.y) return 1.0;
  return texture2D(uDepthMap, (p + 0.5) / uSize).r;
}
void main() {
  vec2 px = floor(vUv * uSize);
  float d = texture2D(uDepthMap, vUv).r;
  bool seed = d >= BG;
  if (!seed && uJump > 0.0) {
    float z = linearDepth(d);
    for (int k = 0; k < 4; k++) {
      vec2 off = k == 0 ? vec2(1.0, 0.0) : k == 1 ? vec2(-1.0, 0.0) : k == 2 ? vec2(0.0, 1.0) : vec2(0.0, -1.0);
      float dn = depthAt(px + off);
      float db = depthAt(px - off);
      float dn2 = depthAt(px + 2.0 * off);
      if (dn >= BG || db >= BG || dn2 >= BG) continue;
      float zn = linearDepth(dn);
      float step = abs(zn - z);
      float flank = max(abs(z - linearDepth(db)), abs(linearDepth(dn2) - zn));
      if (step > uJump && step > ISOLATION * flank) { seed = true; break; }
    }
  }
  if (seed) gl_FragColor = vec4(px, 0.0, 1.0);
  else gl_FragColor = vec4(-1.0, -1.0, 0.0, 0.0);
}
`

// ── UV transform (mirrors transformUvToTextureSpace in meshTexturing.js) ──────
function applyWrap(value, wrapMode) {
  if (wrapMode === THREE.RepeatWrapping) return value - Math.floor(value)
  if (wrapMode === THREE.MirroredRepeatWrapping) {
    if (Math.abs(Math.floor(value) % 2) === 1) return Math.ceil(value) - value
    return value - Math.floor(value)
  }
  return THREE.MathUtils.clamp(value, 0, 1)
}

// Build a lightweight bake geometry that SHARES the source position/normal/index
// buffers (no large-buffer duplication) and adds a precomputed texture-space UV
// attribute. Never mutates the source geometry.
function buildBakeGeometry(srcGeometry, textureConfig) {
  const uvAttr = srcGeometry.attributes.uv
  const posAttr = srcGeometry.attributes.position
  if (!uvAttr || !posAttr) return null

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', posAttr)

  let normalAttr = srcGeometry.attributes.normal
  if (!normalAttr) {
    // Compute normals on a throwaway clone so we don't touch the source.
    const tmp = srcGeometry.clone()
    tmp.computeVertexNormals()
    normalAttr = tmp.attributes.normal
  }
  geom.setAttribute('normal', normalAttr)
  if (srcGeometry.index) geom.setIndex(srcGeometry.index)

  const count = uvAttr.count
  const texUv = new Float32Array(count * 2)
  if (!textureConfig) {
    for (let i = 0; i < count; i += 1) {
      texUv[i * 2] = uvAttr.getX(i)
      texUv[i * 2 + 1] = uvAttr.getY(i)
    }
  } else {
    // Build the affine UV transform ONCE, not per vertex.
    const m = new THREE.Matrix3().setUvTransform(
      textureConfig.offset?.x || 0,
      textureConfig.offset?.y || 0,
      textureConfig.repeat?.x || 1,
      textureConfig.repeat?.y || 1,
      textureConfig.rotation || 0,
      textureConfig.center?.x || 0,
      textureConfig.center?.y || 0
    )
    const v = new THREE.Vector2()
    for (let i = 0; i < count; i += 1) {
      v.set(uvAttr.getX(i), uvAttr.getY(i)).applyMatrix3(m)
      v.x = applyWrap(v.x, textureConfig.wrapS)
      v.y = applyWrap(v.y, textureConfig.wrapT)
      if (textureConfig.flipY) v.y = 1 - v.y
      texUv[i * 2] = v.x
      texUv[i * 2 + 1] = v.y
    }
  }
  geom.setAttribute('aTexUV', new THREE.BufferAttribute(texUv, 2))
  return geom
}

// ── Shared materials ──────────────────────────────────────────────────────────
let depthMaterial = null
let bakeMaterial = null
// The full-screen add/resolve/stats/JFA passes are cached on *Ref objects
// (getFullScreenMaterial), not on module-level material vars.

function getDepthMaterial() {
  if (!depthMaterial) {
    depthMaterial = new THREE.MeshBasicMaterial({
      colorWrite: false,
      // DoubleSide + depthTest keeps the NEAREST fragment, so the depth map stores
      // the front-most surface the projector sees — exactly what "first face only"
      // needs, and (unlike FrontSide) robust to inverted/inconsistent winding in
      // imported meshes. BackSide stored the FAR depth, making the test pass almost
      // everywhere and letting the view leak through to the back. Self-acne on the
      // front surface is handled by the slope-scaled bias in the bake shader, and
      // the fitted near/far below gives the comparison enough precision to actually
      // separate the front surface from the back.
      side: THREE.DoubleSide
    })
  }
  return depthMaterial
}

// UV-occupancy material: rasterises every triangle into UV space (same placement
// as the bake) and writes solid white, with NO facing / depth / projection test.
// The result is the union of ALL triangles' texel footprints — i.e. which texels
// belong to SOME island. Used to stop the gutter dilation from bleeding a view's
// colour across a thin UV gutter onto a NEIGHBOURING island (the "front colour on
// the back of the mesh" leak when UVs are poorly packed / AI-generated).
let occupancyMaterial = null
function getOccupancyMaterial() {
  if (!occupancyMaterial) {
    occupancyMaterial = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: /* glsl */`
        in vec2 aTexUV;
        void main() { gl_Position = vec4(aTexUV * 2.0 - 1.0, 0.0, 1.0); }
      `,
      fragmentShader: /* glsl */`
        precision highp float;
        out vec4 outColor;
        void main() { outColor = vec4(1.0); }
      `,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide
    })
  }
  return occupancyMaterial
}

function getBakeMaterial() {
  if (!bakeMaterial) {
    bakeMaterial = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: BAKE_VERT,
      fragmentShader: BAKE_FRAG,
      transparent: false,
      // NoBlending: each texel is shaded by exactly one fragment (one triangle in
      // UV space), so the output vec4(rgb*w, w) must be written verbatim. The
      // default NormalBlending would multiply rgb by the fragment's own alpha and
      // collapse the result toward black — i.e. nothing shows on the mesh.
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide, // visibility handled in-shader; rasterise both sides
      uniforms: {
        uModel: { value: new THREE.Matrix4() },
        uNormalMat: { value: new THREE.Matrix3() },
        uProjView: { value: new THREE.Matrix4() },
        uProjectorPos: { value: new THREE.Vector3() },
        uSourceView: { value: null },
        uMaskView: { value: null },
        uDepthMap: { value: null },
        uAlpha: { value: DEFAULT_ALPHA },
        uViewWeight: { value: 1 },
        uMinBias: { value: DEFAULT_MIN_BIAS },
        uMaxBias: { value: DEFAULT_MAX_BIAS },
        uUseDepth: { value: 1 },
        uUseMask: { value: 0 },
        uCullBackfaces: { value: 1 },
        uMinMaskAlpha: { value: 0.02 },
        uMinFacing: { value: 0 },
        uGain: { value: new THREE.Vector3(1, 1, 1) },
        uEdgeSeed: { value: null },
        uEdgeSize: { value: new THREE.Vector2(1, 1) },
        uEdgePx: { value: 0 },
        uOutputFade: { value: 0 }
      }
    })
  }
  return bakeMaterial
}

function getFullScreenMaterial(ref, frag, uniforms) {
  if (!ref.mat) {
    ref.mat = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERT,
      fragmentShader: frag,
      uniforms,
      // The add / resolve / stats / JFA passes write their computed value directly;
      // NormalBlending would multiply the output by its own alpha and break the
      // ping-pong accumulation and the jump-flood coordinate propagation.
      blending: THREE.NoBlending,
      transparent: false,
      depthTest: false,
      depthWrite: false
    })
  }
  return ref.mat
}

const addRef = {}
const resolveRef = {}
const statsRef = {}
const jfaInitRef = {}
const jfaStepRef = {}
const jfaApplyRef = {}
const edgeSeedRef = {}

// ── Passes ────────────────────────────────────────────────────────────────────
function runDepthPrepass(renderer, depthRt, meshes, camera) {
  const scene = new THREE.Scene()
  const mat = getDepthMaterial()
  const proxies = []
  meshes.forEach(mesh => {
    if (!mesh?.geometry) return
    const proxy = new THREE.Mesh(mesh.geometry, mat)
    proxy.matrixAutoUpdate = false
    // The world transform must go in `matrix`, not only `matrixWorld`: render() runs
    // scene.updateMatrixWorld(), which recomputes every child's matrixWorld from its
    // local matrix (identity here) and drew every mesh UNTRANSFORMED into the depth
    // map. The colour bake was unaffected (it takes uModel from a closure), so any
    // mesh with a node transform got an occlusion test against the wrong depth.
    proxy.matrix.copy(mesh.matrixWorld)
    proxy.matrixWorld.copy(mesh.matrixWorld)
    proxy.frustumCulled = false
    scene.add(proxy)
    proxies.push(proxy)
  })
  renderer.setRenderTarget(depthRt)
  renderer.setClearColor(0x000000, 0)
  renderer.clear(true, true, false) // color + depth
  renderer.render(scene, camera)
  renderer.setRenderTarget(null)
  scene.clear()
}

// Rasterise EVERY triangle of every bake geometry into UV space and read back a
// per-texel occupancy mask (1 = this texel belongs to some island). Computed at
// the bake texture resolution so it aligns texel-for-texel with the coverage.
function computeUvOccupancy(renderer, bakeGeoms, width, height) {
  const target = makeByteTarget(width, height)
  const mat = getOccupancyMaterial()
  const scene = new THREE.Scene()
  bakeGeoms.forEach(({ geom, matrixWorld }) => {
    const proxy = new THREE.Mesh(geom, mat)
    proxy.matrixAutoUpdate = false
    proxy.matrixWorld.copy(matrixWorld)
    proxy.frustumCulled = false
    scene.add(proxy)
  })
  renderer.setRenderTarget(target)
  renderer.setClearColor(0x000000, 0)
  renderer.clear(true, true, false)
  renderer.render(scene, getQuad().camera) // camera unused; vertex shader ignores it
  renderer.setRenderTarget(null)
  scene.clear()
  const buf = new Uint8Array(width * height * 4)
  renderer.readRenderTargetPixels(target, 0, 0, width, height, buf)
  target.dispose()
  const occ = new Uint8Array(width * height)
  for (let i = 0; i < occ.length; i += 1) occ[i] = buf[i * 4 + 3] > 127 ? 1 : 0
  return occ
}

// Render every source mesh's UV-space contribution for ONE view into `temp`.
function bakeViewIntoTemp(renderer, temp, bakeGeoms, camera, depthTexture, viewTex, maskTex, opts) {
  const mat = getBakeMaterial()
  const u = mat.uniforms
  const projView = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
  u.uProjView.value.copy(projView)
  u.uProjectorPos.value.copy(camera.getWorldPosition(new THREE.Vector3()))
  u.uSourceView.value = viewTex
  u.uMaskView.value = maskTex || null
  u.uUseMask.value = maskTex ? 1 : 0
  u.uDepthMap.value = depthTexture || null
  u.uUseDepth.value = depthTexture ? 1 : 0
  u.uAlpha.value = opts.alpha
  u.uViewWeight.value = opts.viewWeight
  u.uMinBias.value = opts.minBias
  u.uMaxBias.value = opts.maxBias
  u.uCullBackfaces.value = opts.cullBackfaces ? 1 : 0
  u.uMinMaskAlpha.value = opts.minMaskAlpha
  u.uMinFacing.value = opts.minFacing ?? 0
  const g = opts.gain
  if (Array.isArray(g)) u.uGain.value.set(g[0] ?? 1, g[1] ?? 1, g[2] ?? 1)
  else if (g && typeof g === 'object' && 'x' in g) u.uGain.value.copy(g)
  else if (typeof g === 'number') u.uGain.value.set(g, g, g)
  else u.uGain.value.set(1, 1, 1)
  // The material is shared across views and calls, so the edge field must be set
  // (or cleared) every time, never left over from the previous view.
  const edge = opts.edgeField
  u.uEdgeSeed.value = edge?.texture || null
  u.uEdgePx.value = edge ? edge.fadePx : 0
  u.uEdgeSize.value.set(edge?.width || 1, edge?.height || 1)
  u.uOutputFade.value = opts.outputFade ? 1 : 0

  const scene = new THREE.Scene()
  const proxies = []
  bakeGeoms.forEach(({ geom, matrixWorld }) => {
    const proxy = new THREE.Mesh(geom, mat)
    proxy.matrixAutoUpdate = false
    proxy.matrixWorld.copy(matrixWorld)
    proxy.onBeforeRender = () => {
      u.uModel.value.copy(matrixWorld)
      u.uNormalMat.value.getNormalMatrix(matrixWorld)
    }
    proxy.frustumCulled = false
    scene.add(proxy)
    proxies.push(proxy)
  })

  renderer.setRenderTarget(temp)
  renderer.setClearColor(0x000000, 0)
  renderer.clear(true, true, false)
  renderer.render(scene, camera) // camera irrelevant; vertex shader ignores it
  renderer.setRenderTarget(null)
  scene.clear()
}

function addInto(renderer, dst, aTex, bTex) {
  const mat = getFullScreenMaterial(addRef, ADD_FRAG, {
    uA: { value: null }, uB: { value: null }
  })
  mat.uniforms.uA.value = aTex
  mat.uniforms.uB.value = bTex
  runFullScreenPass(renderer, dst, mat)
}

function resolveInto(renderer, dst, accumTex) {
  const mat = getFullScreenMaterial(resolveRef, RESOLVE_FRAG, {
    uAccum: { value: null }, uEps: { value: ACCUM_EPS }
  })
  mat.uniforms.uAccum.value = accumTex
  mat.uniforms.uEps.value = ACCUM_EPS
  runFullScreenPass(renderer, dst, mat)
}

function statsInto(renderer, dst, accumTex) {
  const mat = getFullScreenMaterial(statsRef, /* glsl */`
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uAccum;
    uniform float uEps;
    void main() {
      vec4 a = texture2D(uAccum, vUv);
      float cov = a.a > uEps ? 1.0 : 0.0;
      float conf = clamp(a.a, 0.0, 1.0);
      gl_FragColor = vec4(cov, conf, 0.0, 1.0);
    }
  `, { uAccum: { value: null }, uEps: { value: ACCUM_EPS } })
  mat.uniforms.uAccum.value = accumTex
  mat.uniforms.uEps.value = ACCUM_EPS
  runFullScreenPass(renderer, dst, mat)
}

// Jump-flood dilation. Returns a byte target with covered colour extended up to
// `maxDist` texels past the chart border (gutter / mip-bleed fix). Only coverage
// whose accumulated weight exceeds `seedMin` seeds the fill, so a view's grazing
// garbage does not bleed its matte edge into the gutters.
// Jump-flood steps from `startStep` down to 1. `seedA` holds the initial seeds;
// returns the pair with the result in `seedA`.
function runJfaSteps(renderer, seedA, seedB, size, startStep) {
  const stepMat = getFullScreenMaterial(jfaStepRef, JFA_STEP_FRAG, {
    uSeed: { value: null }, uSize: { value: new THREE.Vector2() }, uStep: { value: 1 }
  })
  let step = startStep
  while (step >= 1) {
    stepMat.uniforms.uSeed.value = seedA.texture
    stepMat.uniforms.uSize.value.copy(size)
    stepMat.uniforms.uStep.value = step
    runFullScreenPass(renderer, seedB, stepMat)
    const t = seedA; seedA = seedB; seedB = t
    step = Math.floor(step / 2)
  }
  return { seedA, seedB }
}

// Nearest-silhouette field for one view, from its depth pre-pass. Only distances up
// to `fadePx` matter (past that the fade is 1), so the flood starts at the power of
// two covering it instead of at the full image size — a handful of passes.
// Returns { target, texture, width, height, fadePx } or null when the fade is off;
// the caller disposes `target`.
function computeEdgeField(renderer, depthRt, camera, { fadePx, depthJump }) {
  if (!(fadePx > 0)) return null
  const width = depthRt.width
  const height = depthRt.height
  const size = new THREE.Vector2(width, height)
  let seedA = makeFloatTarget(width, height, { type: THREE.FloatType })
  let seedB = makeFloatTarget(width, height, { type: THREE.FloatType })
  const initMat = getFullScreenMaterial(edgeSeedRef, EDGE_SEED_FRAG, {
    uDepthMap: { value: null }, uSize: { value: new THREE.Vector2() },
    uNear: { value: 0.1 }, uFar: { value: 100 }, uOrtho: { value: 0 }, uJump: { value: 0 }
  })
  initMat.uniforms.uDepthMap.value = depthRt.depthTexture
  initMat.uniforms.uSize.value.copy(size)
  initMat.uniforms.uNear.value = camera.near ?? 0.1
  initMat.uniforms.uFar.value = camera.far ?? 100
  initMat.uniforms.uOrtho.value = camera.isOrthographicCamera ? 1 : 0
  initMat.uniforms.uJump.value = Math.max(0, depthJump || 0)
  runFullScreenPass(renderer, seedA, initMat)
  const startStep = Math.pow(2, Math.ceil(Math.log2(Math.max(1, fadePx))))
  ;({ seedA, seedB } = runJfaSteps(renderer, seedA, seedB, size, startStep))
  seedB.dispose()
  return { target: seedA, texture: seedA.texture, width, height, fadePx }
}

// Fade width in edge-field (depth-map) pixels for a fade given in source-image pixels.
function edgeFadeInDepthPx(edgeFadePx, depthWidth, imageWidth) {
  if (!(edgeFadePx > 0)) return 0
  return edgeFadePx * (imageWidth > 0 ? depthWidth / imageWidth : 1)
}

function jfaDilate(renderer, width, height, colorTex, accumTex, maxDist, seedMin = ACCUM_EPS) {
  const size = new THREE.Vector2(width, height)
  let seedA = makeFloatTarget(width, height, { type: THREE.FloatType })
  let seedB = makeFloatTarget(width, height, { type: THREE.FloatType })

  const initMat = getFullScreenMaterial(jfaInitRef, JFA_INIT_FRAG, {
    uAccum: { value: null }, uSize: { value: new THREE.Vector2() }, uSeedMin: { value: ACCUM_EPS }
  })
  initMat.uniforms.uAccum.value = accumTex
  initMat.uniforms.uSize.value.copy(size)
  initMat.uniforms.uSeedMin.value = seedMin
  runFullScreenPass(renderer, seedA, initMat)

  ;({ seedA, seedB } = runJfaSteps(renderer, seedA, seedB, size,
    Math.pow(2, Math.ceil(Math.log2(Math.max(width, height)))) / 2))

  const out = makeByteTarget(width, height)
  const applyMat = getFullScreenMaterial(jfaApplyRef, JFA_APPLY_FRAG, {
    uColor: { value: null }, uSeed: { value: null },
    uSize: { value: new THREE.Vector2() }, uMaxDist: { value: 0 }
  })
  applyMat.uniforms.uColor.value = colorTex
  applyMat.uniforms.uSeed.value = seedA.texture
  applyMat.uniforms.uSize.value.copy(size)
  applyMat.uniforms.uMaxDist.value = maxDist
  runFullScreenPass(renderer, out, applyMat)

  seedA.dispose()
  seedB.dispose()
  return out
}

// Read an RGBA8 render target into a fresh canvas. readRenderTargetPixels returns
// rows bottom-up (GL order); the CPU bake's canvas row 0 corresponds to texture-V
// 0, which is also GL row 0, so no vertical flip is needed to match the existing
// pipeline. `flipY` is exposed for the checker-test gotcha noted in the analysis.
function bufferToCanvas(buf, width, height, flipY = false) {
  const canvas = (typeof document !== 'undefined') ? document.createElement('canvas') : null
  if (!canvas) return null
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(width, height)
  if (flipY) {
    for (let y = 0; y < height; y += 1) {
      const src = (height - 1 - y) * width * 4
      const dst = y * width * 4
      img.data.set(buf.subarray(src, src + width * 4), dst)
    }
  } else {
    img.data.set(buf)
  }
  ctx.putImageData(img, 0, 0)
  return canvas
}

function byteTargetToCanvas(renderer, target, width, height, flipY = false) {
  const buf = new Uint8Array(width * height * 4)
  renderer.readRenderTargetPixels(target, 0, 0, width, height, buf)
  return bufferToCanvas(buf, width, height, flipY)
}

function readStats(renderer, target, width, height) {
  const buf = new Uint8Array(width * height * 4)
  renderer.readRenderTargetPixels(target, 0, 0, width, height, buf)
  const n = width * height
  const coverageMask = new Uint8Array(n)
  const confidenceMap = new Float32Array(n)
  let coveredTexels = 0
  for (let i = 0; i < n; i += 1) {
    const j = i * 4
    const covered = buf[j] > 127 ? 1 : 0
    coverageMask[i] = covered
    confidenceMap[i] = buf[j + 1] / 255
    if (covered) coveredTexels += 1
  }
  return { coverageMask, confidenceMap, coveredTexels }
}

function makeSourceTexture(image) {
  const tex = new THREE.Texture(image)
  // NoColorSpace = sample the raw bytes with NO sRGB→linear decode. The bake then
  // works entirely in gamma/byte space and writes the same bytes back, exactly like
  // the old CPU path (which did getImageData arithmetic in byte space). If this were
  // SRGBColorSpace, three uploads an sRGB internal format, the GPU decodes to linear,
  // and we'd store those linear values into an RGBA8 canvas that is later read as
  // sRGB — i.e. a missing re-encode that makes the whole mesh look too dark.
  tex.colorSpace = THREE.NoColorSpace
  tex.minFilter = THREE.LinearFilter
  tex.magFilter = THREE.LinearFilter
  tex.wrapS = THREE.ClampToEdgeWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  tex.flipY = true // canvas default; makes projective sUV land correctly
  tex.needsUpdate = true
  return tex
}

// ── Brown–Lowe per-view gain compensation (CPU, closed form) ─────────────────
// Solves a per-channel scalar gain per view that minimises pairwise colour
// disagreement in overlap regions, regularised toward 1.0. This is the right
// tool for independently-generated AI views with different global tint/exposure.
function solveLinearSystem(A, b, k) {
  // Gauss-Jordan with partial pivoting on a k×k system. Mutates copies.
  const M = A.map((row, i) => row.slice().concat(b[i]))
  for (let col = 0; col < k; col += 1) {
    let pivot = col
    for (let r = col + 1; r < k; r += 1) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r
    }
    if (Math.abs(M[pivot][col]) < 1e-9) continue
    const tmp = M[col]; M[col] = M[pivot]; M[pivot] = tmp
    const pv = M[col][col]
    for (let c = col; c <= k; c += 1) M[col][c] /= pv
    for (let r = 0; r < k; r += 1) {
      if (r === col) continue
      const f = M[r][col]
      if (f === 0) continue
      for (let c = col; c <= k; c += 1) M[r][c] -= f * M[col][c]
    }
  }
  return M.map(row => row[k])
}

export function solveViewGains(perViewColor, perViewCoverage, width, height, {
  // Tight clamp: gain may correct mild exposure / white-balance differences between
  // views, but must NEVER be wide enough to flip a hue. With minGain/maxGain at
  // 0.82/1.22 the largest per-channel ratio is ~1.49, so a blue surface can never be
  // dragged orange to "match" an opposite-facing view it does not truly share.
  lambda = 0.6, minGain = 0.82, maxGain = 1.22,
  // Per-view confidence maps (cosine weight, 0..1). When provided, an overlap texel
  // only informs the solve if BOTH views see it confidently (>= minConfidence). Two
  // opposite-facing views only "overlap" on grazing wrap-around surfaces, whose
  // colours legitimately differ — feeding those into the photometric solve is what
  // produced the runaway tint. Gating to head-on co-visible texels means a pair with
  // no reliable shared surface contributes nothing and both views keep their colour.
  perViewConfidence = null, minConfidence = 0.4
} = {}) {
  const k = perViewColor.length
  const gains = Array.from({ length: k }, () => [1, 1, 1])
  if (k <= 1) return gains
  const n = width * height

  for (let ch = 0; ch < 3; ch += 1) {
    // Pairwise overlap means for this channel, normalised to [0,1].
    const sum = Array.from({ length: k }, () => new Float64Array(k))
    const cnt = Array.from({ length: k }, () => new Float64Array(k))
    for (let p = 0; p < n; p += 1) {
      const j = p * 4
      for (let i = 0; i < k; i += 1) {
        if (!perViewCoverage[i][p]) continue
        if (perViewConfidence && (perViewConfidence[i]?.[p] ?? 0) < minConfidence) continue
        const ci = perViewColor[i][j + ch] / 255
        for (let q = i + 1; q < k; q += 1) {
          if (!perViewCoverage[q][p]) continue
          if (perViewConfidence && (perViewConfidence[q]?.[p] ?? 0) < minConfidence) continue
          const cq = perViewColor[q][j + ch] / 255
          sum[i][q] += ci; sum[q][i] += cq
          cnt[i][q] += 1; cnt[q][i] += 1
        }
      }
    }

    const A = Array.from({ length: k }, () => new Array(k).fill(0))
    const b = new Array(k).fill(0)
    for (let i = 0; i < k; i += 1) {
      A[i][i] += lambda // prior pulling g_i toward 1
      b[i] += lambda
      for (let q = 0; q < k; q += 1) {
        if (q === i || cnt[i][q] <= 0) continue
        const mi = sum[i][q] / cnt[i][q]
        const mq = sum[q][i] / cnt[q][i]
        const w = cnt[i][q]
        A[i][i] += w * mi * mi
        A[i][q] -= w * mi * mq
      }
    }
    const g = solveLinearSystem(A, b, k)
    for (let i = 0; i < k; i += 1) {
      gains[i][ch] = THREE.MathUtils.clamp(Number.isFinite(g[i]) ? g[i] : 1, minGain, maxGain)
    }
  }
  return gains
}

// ── Mesh collection ──────────────────────────────────────────────────────────
// Replicates getTextureKey() / getTextureKeyFromMaterial() from meshTexturing.js so the
// GPU bake selects exactly the meshes the CPU path would. NOTE: the app prefers the
// texture *source* uuid, not the map uuid — matching only map.uuid silently selects no
// meshes and forces the slow CPU fallback.
function materialTextureKey(material) {
  const tex = material?.map
  if (!tex) return ''
  return String(
    tex.source?.uuid
    || tex.uuid
    || tex.image?.currentSrc
    || tex.image?.src
    || tex.name
    || ''
  )
}

function collectMeshes(meshes, root, textureKey) {
  if (Array.isArray(meshes) && meshes.length) return meshes
  const allUv = []
  const matched = []
  const key = textureKey != null ? String(textureKey) : ''
  root?.traverse?.(child => {
    if (!child?.isMesh || !child.geometry?.attributes?.uv) return
    allUv.push(child)
    if (!key) return
    const mats = Array.isArray(child.material) ? child.material : [child.material]
    if (mats.some(m => materialTextureKey(m) === key)) matched.push(child)
  })
  if (!key) return allUv
  if (matched.length) return matched
  // Safety net: a key mismatch must not silently fall back to the 10–20 s CPU path.
  // For a single-texture mesh the full UV set is the correct set anyway.
  if (allUv.length && typeof console !== 'undefined') {
    console.warn('[gpuTextureBake] textureKey matched no meshes; baking all UV meshes instead.')
  }
  return allUv
}

function imageSize(image) {
  const w = image?.naturalWidth || image?.videoWidth || image?.width || 0
  const h = image?.naturalHeight || image?.videoHeight || image?.height || 0
  return { w, h }
}

// Bounding sphere of the meshes in world space (matrices must be up to date).
function computeSceneSphere(meshList) {
  const box = new THREE.Box3()
  for (const mesh of meshList) {
    if (mesh?.geometry) box.expandByObject(mesh)
  }
  if (box.isEmpty()) return null
  return box.getBoundingSphere(new THREE.Sphere())
}

// Clone the projector camera with near/far tightened to the mesh bounds.
// CRITICAL for occlusion precision: with the app's default near/far (~0.1/1000)
// the perspective depth buffer compresses the whole character into the last
// fraction of [0,1], so the front and back of the body differ by less than the
// depth-compare bias and the back leaks through. Fitting near/far to the bounding
// sphere spreads the depth across the full range so front vs back is unambiguous.
// near/far only affect clip-space z, NOT x/y/w, so the projected sUV (and thus the
// sampled view colour) is unchanged — only the depth comparison gets sharper.
function fittedProjector(camera, sphere) {
  if (!sphere || !camera?.clone) return camera
  const cam = camera.clone()
  cam.updateMatrixWorld(true)
  const camPos = cam.getWorldPosition(new THREE.Vector3())
  const dist = camPos.distanceTo(sphere.center)
  const r = sphere.radius * 1.2 + 1e-3
  if (cam.isPerspectiveCamera) {
    cam.near = Math.max(dist - r, dist * 0.02, 1e-3)
    cam.far = Math.max(cam.near + 1e-3, dist + r)
  }
  cam.updateProjectionMatrix()
  cam.updateMatrixWorld(true)
  return cam
}

// ── Public: single-view drop-in (Steps 1–3) ─────────────────────────────────
// Replaces accumulateProjectedPatch + finalizeProjectedPatch for one view.
// Hard-edged (no UV feather, no screen-space seam mask → the Step-0 leak fix is
// structural here). Output slots straight into the existing JSX layer composite:
// canvas alpha = 255 where covered / 0 elsewhere, confidenceMap = cosine weight.
export async function bakeViewToTextureGPU(params) {
  if (!isGpuBakeSupported()) return null
  const {
    meshes, root, textureKey, textureConfig,
    camera, viewImage, maskImage = null,
    textureWidth, textureHeight,
    alpha = DEFAULT_ALPHA, viewOpacity = 1,
    cullBackfaces = false, minMaskAlpha = 0.02,
    minFacing = 0,
    minBias = DEFAULT_MIN_BIAS, maxBias = DEFAULT_MAX_BIAS,
    depthResolution = null, flipOutputY = false,
    // Gutter padding to stop white UV-seam bleed under display-time filtering.
    // Small by design: a large radius would bleed one island's colour across a
    // thin gutter into its neighbour.
    dilatePixels = 4,
    // Silhouette-edge fade width in source-image pixels (0 = off), and the
    // occlusion-edge depth step as a fraction of the mesh diameter (0 = outer
    // silhouette only). See DEFAULT_EDGE_FADE_PX.
    edgeFadePx = DEFAULT_EDGE_FADE_PX,
    edgeDepthJump = DEFAULT_EDGE_DEPTH_JUMP
  } = params

  if (!camera || !viewImage || !textureWidth || !textureHeight) return null
  const meshList = collectMeshes(meshes, root, textureKey)
  if (!meshList.length) return null

  const startedAt = (typeof performance !== 'undefined' ? performance.now() : Date.now())
  const renderer = getBakeRenderer()
  camera.updateMatrixWorld?.(true)
  camera.updateProjectionMatrix?.()

  const bakeGeoms = []
  for (const mesh of meshList) {
    mesh.updateMatrixWorld?.(true)
    const geom = buildBakeGeometry(mesh.geometry, textureConfig)
    if (geom) bakeGeoms.push({ geom, matrixWorld: mesh.matrixWorld.clone() })
  }
  if (!bakeGeoms.length) return null

  const { w: imgW, h: imgH } = imageSize(viewImage)
  const dW = depthResolution?.w || imgW || 1024
  const dH = depthResolution?.h || imgH || 1024

  const depthRt = makeDepthTarget(dW, dH)

  const temp = makeFloatTarget(textureWidth, textureHeight)
  const resolved = makeByteTarget(textureWidth, textureHeight)
  const stats = makeByteTarget(textureWidth, textureHeight)

  const viewTex = makeSourceTexture(viewImage)
  const maskTex = maskImage ? makeSourceTexture(maskImage) : null

  // Fit the projector near/far so the depth compare can actually separate the
  // front surface from the back (otherwise the front view leaks onto the back).
  const sceneSphere = computeSceneSphere(meshList)
  const pcam = fittedProjector(camera, sceneSphere)

  let occlusionModeUsed = 'none'
  let dilatedRt = null
  let edgeField = null
  try {
    runDepthPrepass(renderer, depthRt, meshList, pcam)
    occlusionModeUsed = 'depth-prepass'
    edgeField = computeEdgeField(renderer, depthRt, pcam, {
      fadePx: edgeFadeInDepthPx(edgeFadePx, dW, imgW),
      depthJump: edgeDepthJump * 2 * (sceneSphere?.radius || 0)
    })
    const bakeOpts = {
      alpha, viewWeight: viewOpacity, minBias, maxBias, cullBackfaces, minMaskAlpha, minFacing, gain: 1, edgeField
    }

    // Single view → the temp target already holds Σ(rgb*w, w) for this view
    // (NoBlending, one fragment per texel). No cross-view accumulation is needed,
    // so resolve directly from it. (The previous addInto(accum, temp, accum) read
    // and wrote the same target — a framebuffer feedback loop that zeroed the
    // result on real GPUs, which is why nothing appeared on the mesh.)
    bakeViewIntoTemp(renderer, temp, bakeGeoms, pcam, depthRt.depthTexture, viewTex, maskTex, bakeOpts)

    resolveInto(renderer, resolved, temp.texture)
    statsInto(renderer, stats, temp.texture)
    const { coverageMask: coreCoverage, confidenceMap: coreConfidence, coveredTexels } =
      readStats(renderer, stats, textureWidth, textureHeight)

    // The fade on its own, per texel, for the layer composite: confidence mixes it
    // with the cosine, but only the fade says "this is the rim of the view" — the
    // band a later, better view is allowed to take over. Same shader and discards
    // as the colour pass, so it covers exactly the core texels. Texels it does not
    // reach (dilation pad, uncovered) read 1 = no rim, i.e. today's behaviour.
    let edgeFadeMap = null
    if (edgeField) {
      bakeViewIntoTemp(renderer, stats, bakeGeoms, pcam, depthRt.depthTexture, viewTex, maskTex, { ...bakeOpts, outputFade: true })
      const fadeBuf = new Uint8Array(textureWidth * textureHeight * 4)
      renderer.readRenderTargetPixels(stats, 0, 0, textureWidth, textureHeight, fadeBuf)
      edgeFadeMap = new Float32Array(textureWidth * textureHeight)
      for (let i = 0; i < edgeFadeMap.length; i += 1) {
        edgeFadeMap[i] = fadeBuf[i * 4 + 1] > 127 ? fadeBuf[i * 4] / 255 : 1
      }
    }

    // Gutter dilation: pad the covered colour a few texels past each UV-island
    // border so display-time bilinear/mip filtering does not pull the (often white)
    // base gutter across chart edges — the "white wireframe seams". The CPU path
    // avoided this with negative-baryEps triangle inflation + edge bleed; the GPU
    // rasteriser has no such inflation, so we dilate here.
    let colorTarget = resolved
    if (dilatePixels > 0) {
      dilatedRt = jfaDilate(renderer, textureWidth, textureHeight, resolved.texture, temp.texture, dilatePixels, DILATE_SEED_MIN_WEIGHT)
      colorTarget = dilatedRt
    }

    // UV-island occupancy: which texels belong to SOME triangle's footprint. The
    // gutter dilation grows colour in 2D texture space, blind to island borders, so
    // on a poorly-packed (e.g. AI-generated) UV layout a front-facing island's halo
    // bleeds across a thin gutter onto a NEIGHBOURING island that belongs to the back
    // of the mesh — front colour appears on untouched back faces. We allow the
    // dilation to fill genuinely EMPTY gutter texels (occ == 0, kills white mip
    // seams) but reject any dilated texel that lands on a DIFFERENT island
    // (occ == 1 && not core), so a view can never paint a surface it doesn't own.
    const uvOccupancyMask = computeUvOccupancy(renderer, bakeGeoms, textureWidth, textureHeight)

    // Coverage/confidence must include the padded gutter, or the layer composite
    // will not write those texels and the seam returns. Derive coverage from the
    // dilated alpha. CRITICAL: padded (dilated) texels get a TINY confidence, far
    // below any real cosine-weighted sample. The dilation exists only to fill UV
    // gutters that NO view covers — there it is the sole contributor, and the
    // composite normalises by ΣW so its colour still shows regardless of magnitude.
    // But where another view already has REAL coverage (e.g. the front view owns the
    // legs), this view's dilated silhouette — which carries matte/white edge colour —
    // must NOT compete, or it paints a white ring along this view's coverage border
    // over the other view's surface. A tiny confidence guarantees real coverage wins.
    const PAD_CONFIDENCE = 0.02
    let coverageMask = coreCoverage
    let confidenceMap = coreConfidence
    let canvas
    if (dilatedRt) {
      const padBuf = new Uint8Array(textureWidth * textureHeight * 4)
      renderer.readRenderTargetPixels(dilatedRt, 0, 0, textureWidth, textureHeight, padBuf)
      coverageMask = new Uint8Array(textureWidth * textureHeight)
      confidenceMap = new Float32Array(textureWidth * textureHeight)
      for (let i = 0; i < coverageMask.length; i += 1) {
        const isCore = coreCoverage[i] === 1
        const isDilated = padBuf[i * 4 + 3] > 127
        // Reject a dilated halo texel that bleeds onto a different island.
        const bleedsOntoOtherIsland = isDilated && !isCore && uvOccupancyMask[i] === 1
        if (isDilated && !bleedsOntoOtherIsland) {
          coverageMask[i] = 1
          confidenceMap[i] = isCore ? coreConfidence[i] : PAD_CONFIDENCE
        } else if (bleedsOntoOtherIsland) {
          // Strip the bled colour from the canvas too, so the single-view reproject
          // path (which draws gpu.canvas directly) does not paint it either.
          padBuf[i * 4] = 0
          padBuf[i * 4 + 1] = 0
          padBuf[i * 4 + 2] = 0
          padBuf[i * 4 + 3] = 0
        }
      }
      canvas = bufferToCanvas(padBuf, textureWidth, textureHeight, flipOutputY)
    } else {
      canvas = byteTargetToCanvas(renderer, colorTarget, textureWidth, textureHeight, flipOutputY)
    }

    return {
      canvas,
      coverageMask,
      confidenceMap,
      edgeFadeMap,
      uvOccupancyMask,
      coveredTexels,
      occlusionModeUsed,
      durationMs: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - startedAt
    }
  } catch (err) {
    console.warn('[gpuTextureBake] single-view bake failed, falling back to CPU:', err)
    return null
  } finally {
    bakeGeoms.forEach(({ geom }) => geom.dispose())
    depthRt.depthTexture?.dispose?.()
    depthRt.dispose()
    temp.dispose(); resolved.dispose(); stats.dispose()
    dilatedRt?.dispose?.()
    edgeField?.target.dispose()
    viewTex.dispose(); maskTex?.dispose?.()
  }
}

// ── Mask paint shader (3D-gated brush) ───────────────────────────────────────
// Rasterises the mesh in UV space (same aTexUV placement as the bake) and writes
// solid coverage where the texel's WORLD position lies within uRadius of the 3D
// brush segment [uHitA,uHitB]. Gating by WORLD distance — not UV distance — means a
// texel that is near in UV but far in 3D (mirrored/overlapping UV islands, or the
// far side of a thin part) is rejected. That is the structural fix for mask seam
// bleed: a stroke paints exactly the surface under the brush in 3D, regardless of
// how the UVs are laid out, and never leaks across a UV seam (nor gaps at one).
const MASK_PAINT_VERT = /* glsl */`precision highp float;
in vec3 position;
in vec2 aTexUV;
uniform mat4 uModel;
out vec3 vWorldPos;
void main() {
  vWorldPos = (uModel * vec4(position, 1.0)).xyz;
  gl_Position = vec4(aTexUV * 2.0 - 1.0, 0.0, 1.0);
}
`

const MASK_PAINT_FRAG = /* glsl */`precision highp float;
in vec3 vWorldPos;
uniform vec3 uHitA;
uniform vec3 uHitB;
uniform float uRadius;
out vec4 outColor;
void main() {
  // Distance from this texel's world position to the brush segment (capsule), so a
  // fast drag between frames paints a gap-free stroke from one rAF point dab.
  vec3 ab = uHitB - uHitA;
  float t = clamp(dot(vWorldPos - uHitA, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
  float d = distance(vWorldPos, uHitA + t * ab);
  if (d > uRadius) discard;
  outColor = vec4(1.0);
}
`

let maskPaintMaterial = null
function getMaskPaintMaterial() {
  if (!maskPaintMaterial) {
    maskPaintMaterial = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: MASK_PAINT_VERT,
      fragmentShader: MASK_PAINT_FRAG,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: {
        uModel: { value: new THREE.Matrix4() },
        uHitA: { value: new THREE.Vector3() },
        uHitB: { value: new THREE.Vector3() },
        uRadius: { value: 0.05 }
      }
    })
  }
  return maskPaintMaterial
}

// Public: paint one brush dab into a coverage canvas on the GPU. Returns a canvas
// (textureWidth×textureHeight) — opaque white where covered, fully transparent
// elsewhere, in the texture-canvas pixel layout (identical to the bake output) — or
// null if the GPU path is unavailable. The caller composites it onto the layer's
// persistent mask canvas (source-over to add, destination-out to erase).
export function paintProjectionMaskDabGPU(params) {
  if (!isGpuBakeSupported()) return null
  const {
    meshes, root, textureKey, textureConfig,
    hitA, hitB = hitA, radiusWorld,
    textureWidth, textureHeight
  } = params
  if (!hitA || !radiusWorld || !textureWidth || !textureHeight) return null
  const meshList = collectMeshes(meshes, root, textureKey)
  if (!meshList.length) return null

  const renderer = getBakeRenderer()
  const bakeGeoms = []
  for (const mesh of meshList) {
    mesh.updateMatrixWorld?.(true)
    const geom = buildBakeGeometry(mesh.geometry, textureConfig)
    if (geom) bakeGeoms.push({ geom, matrixWorld: mesh.matrixWorld.clone() })
  }
  if (!bakeGeoms.length) return null

  const target = makeByteTarget(textureWidth, textureHeight)
  const mat = getMaskPaintMaterial()
  const u = mat.uniforms
  u.uHitA.value.copy(hitA)
  u.uHitB.value.copy(hitB || hitA)
  u.uRadius.value = radiusWorld

  try {
    const scene = new THREE.Scene()
    bakeGeoms.forEach(({ geom, matrixWorld }) => {
      const proxy = new THREE.Mesh(geom, mat)
      proxy.matrixAutoUpdate = false
      proxy.matrixWorld.copy(matrixWorld)
      proxy.onBeforeRender = () => { u.uModel.value.copy(matrixWorld) }
      proxy.frustumCulled = false
      scene.add(proxy)
    })
    const camera = new THREE.Camera() // unused; the vertex shader places texels in UV space
    renderer.setRenderTarget(target)
    renderer.setClearColor(0x000000, 0)
    renderer.clear(true, true, false)
    renderer.render(scene, camera)
    renderer.setRenderTarget(null)
    scene.clear()
    return byteTargetToCanvas(renderer, target, textureWidth, textureHeight, false)
  } catch (err) {
    console.warn('[gpuTextureBake] mask dab failed:', err)
    return null
  } finally {
    bakeGeoms.forEach(({ geom }) => geom.dispose())
    target.dispose()
  }
}

// ── Public: unified multi-view bake (Steps 1–5) ──────────────────────────────
// Accumulates ALL views into one float target with a per-texel visibility gate
// and steep cosine weight, resolves, then JFA-dilates the gutter. This replaces
// the per-layer independent bake + hand-built recomposite + seam blur entirely,
// which is the structural fix for the seam (1) and leak (2) problems.
export async function bakeMultiViewTextureGPU(params) {
  if (!isGpuBakeSupported()) return null
  const {
    meshes, root, textureKey, textureConfig,
    textureWidth, textureHeight,
    views = [],
    baseCanvas = null,
    alpha = DEFAULT_ALPHA,
    dilatePixels = 32,
    gainCompensation = false,
    cullBackfaces = false, minMaskAlpha = 0.02,
    minFacing = 0,
    minBias = DEFAULT_MIN_BIAS, maxBias = DEFAULT_MAX_BIAS,
    depthResolution = null, flipOutputY = false,
    onProgress = null,
    edgeFadePx = DEFAULT_EDGE_FADE_PX,
    edgeDepthJump = DEFAULT_EDGE_DEPTH_JUMP
  } = params

  if (!views.length || !textureWidth || !textureHeight) return null
  const meshList = collectMeshes(meshes, root, textureKey)
  if (!meshList.length) return null

  const startedAt = (typeof performance !== 'undefined' ? performance.now() : Date.now())
  const renderer = getBakeRenderer()

  const bakeGeoms = []
  for (const mesh of meshList) {
    mesh.updateMatrixWorld?.(true)
    const geom = buildBakeGeometry(mesh.geometry, textureConfig)
    if (geom) bakeGeoms.push({ geom, matrixWorld: mesh.matrixWorld.clone() })
  }
  if (!bakeGeoms.length) return null

  // Fit projector near/far per view so the depth compare can separate front from
  // back (see fittedProjector). Computed once; the mesh does not move between views.
  const sceneSphere = computeSceneSphere(meshList)

  const accumA = makeFloatTarget(textureWidth, textureHeight)
  const accumB = makeFloatTarget(textureWidth, textureHeight)
  const temp = makeFloatTarget(textureWidth, textureHeight)
  const resolved = makeByteTarget(textureWidth, textureHeight)
  const stats = makeByteTarget(textureWidth, textureHeight)

  // depth target sized to the largest view image
  let dW = depthResolution?.w || 0
  let dH = depthResolution?.h || 0
  if (!dW || !dH) {
    for (const v of views) {
      const { w, h } = imageSize(v.image)
      dW = Math.max(dW, w); dH = Math.max(dH, h)
    }
    dW = dW || 1024; dH = dH || 1024
  }
  const depthRt = makeDepthTarget(dW, dH)

  const viewTextures = views.map(v => makeSourceTexture(v.image))
  const maskTextures = views.map(v => (v.mask ? makeSourceTexture(v.mask) : null))

  // Depth pre-pass + silhouette-edge field + UV bake of view i into `temp`.
  const bakeOneView = (i, cam, gain) => {
    runDepthPrepass(renderer, depthRt, meshList, cam)
    const edgeField = computeEdgeField(renderer, depthRt, cam, {
      fadePx: edgeFadeInDepthPx(edgeFadePx, dW, imageSize(views[i].image).w),
      depthJump: edgeDepthJump * 2 * (sceneSphere?.radius || 0)
    })
    try {
      bakeViewIntoTemp(renderer, temp, bakeGeoms, cam, depthRt.depthTexture, viewTextures[i], maskTextures[i], {
        alpha, viewWeight: views[i].opacity ?? 1, minBias, maxBias, cullBackfaces, minMaskAlpha, minFacing, gain, edgeField
      })
    } finally {
      edgeField?.target.dispose()
    }
  }

  // Optional gain pre-solve: bake each view to its own resolved canvas, read it
  // back, solve Brown–Lowe gains, then accumulate with gains applied.
  let gains = views.map(() => [1, 1, 1])
  try {
    if (gainCompensation && views.length > 1) {
      const perColor = []
      const perCov = []
      const perConf = []
      for (let i = 0; i < views.length; i += 1) {
        views[i].camera.updateMatrixWorld?.(true); views[i].camera.updateProjectionMatrix?.()
        const cam = fittedProjector(views[i].camera, sceneSphere)
        bakeOneView(i, cam, 1)
        resolveInto(renderer, resolved, temp.texture)
        statsInto(renderer, stats, temp.texture)
        const buf = new Uint8Array(textureWidth * textureHeight * 4)
        renderer.readRenderTargetPixels(resolved, 0, 0, textureWidth, textureHeight, buf)
        const { coverageMask, confidenceMap } = readStats(renderer, stats, textureWidth, textureHeight)
        perColor.push(buf); perCov.push(coverageMask); perConf.push(confidenceMap)
      }
      // Gate the photometric solve to head-on co-visible texels so grazing overlap
      // between opposite-facing views cannot drive a runaway whole-view tint.
      gains = solveViewGains(perColor, perCov, textureWidth, textureHeight, { perViewConfidence: perConf })
    }

    // Main accumulation pass.
    renderer.setRenderTarget(accumA)
    renderer.setClearColor(0x000000, 0)
    renderer.clear(true, true, false)
    renderer.setRenderTarget(null)
    let cur = accumA
    let other = accumB
    for (let i = 0; i < views.length; i += 1) {
      views[i].camera.updateMatrixWorld?.(true); views[i].camera.updateProjectionMatrix?.()
      const cam = fittedProjector(views[i].camera, sceneSphere)
      // per-channel Brown–Lowe gain ([r,g,b]); identity when compensation is off
      bakeOneView(i, cam, gains[i])
      addInto(renderer, other, cur.texture, temp.texture)
      const t = cur; cur = other; other = t
      onProgress?.((i + 1) / views.length)
    }

    resolveInto(renderer, resolved, cur.texture)
    statsInto(renderer, stats, cur.texture)
    const { coverageMask, confidenceMap, coveredTexels } = readStats(renderer, stats, textureWidth, textureHeight)

    const dilated = jfaDilate(renderer, textureWidth, textureHeight, resolved.texture, cur.texture,
      Math.max(0, dilatePixels))
    const projCanvas = byteTargetToCanvas(renderer, dilated, textureWidth, textureHeight, flipOutputY)
    dilated.dispose()

    // Composite over the base for uncovered texels.
    let outCanvas = projCanvas
    if (baseCanvas && projCanvas) {
      outCanvas = document.createElement('canvas')
      outCanvas.width = textureWidth
      outCanvas.height = textureHeight
      const octx = outCanvas.getContext('2d')
      octx.drawImage(baseCanvas, 0, 0, textureWidth, textureHeight)
      octx.drawImage(projCanvas, 0, 0)
    }

    return {
      canvas: outCanvas,
      coverageMask,
      confidenceMap,
      coveredTexels,
      perViewGain: gains,
      durationMs: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - startedAt
    }
  } catch (err) {
    console.warn('[gpuTextureBake] multi-view bake failed, falling back to CPU:', err)
    return null
  } finally {
    bakeGeoms.forEach(({ geom }) => geom.dispose())
    depthRt.depthTexture?.dispose?.()
    depthRt.dispose()
    accumA.dispose(); accumB.dispose(); temp.dispose(); resolved.dispose(); stats.dispose()
    viewTextures.forEach(t => t.dispose())
    maskTextures.forEach(t => t?.dispose?.())
  }
}

export function disposeGpuBakeResources() {
  sharedRenderer?.dispose?.()
  sharedRenderer = null
  sharedQuadScene = null
  sharedQuadCamera = null
  sharedQuadMesh = null
  depthMaterial = bakeMaterial = null
  addRef.mat = resolveRef.mat = statsRef.mat = jfaInitRef.mat = jfaStepRef.mat = jfaApplyRef.mat = edgeSeedRef.mat = null
}
