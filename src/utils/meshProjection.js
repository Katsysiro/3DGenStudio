// meshProjection.js
// Pure helpers extracted from src/pages/MeshEditorPage.jsx (behaviour-preserving move).
// No React, no component state.

import * as THREE from 'three'
import { mapUvToCanvasPoint } from './meshTexturing'

export function drawProjectionCheckerboard(context, width, height) {
  if (!context || !width || !height) {
    return
  }

  const cellSize = Math.max(16, Math.round(width / 64))
  for (let cy = 0; cy < height; cy += cellSize) {
    for (let cx = 0; cx < width; cx += cellSize) {
      context.fillStyle = (((cx / cellSize) + (cy / cellSize)) % 2 === 0) ? '#d4d4d4' : '#bcbcbc'
      context.fillRect(cx, cy, cellSize, cellSize)
    }
  }
}

export function computeProjectionDistanceInsideMask(mask, width, height) {
  const pixelCount = width * height
  const dist = new Int32Array(pixelCount)
  if (!mask || mask.length !== pixelCount || !width || !height) {
    return dist
  }

  const INF = 1 << 29
  const ORTHO = 10
  const DIAG = 14

  for (let i = 0; i < pixelCount; i += 1) {
    dist[i] = mask[i] > 0 ? INF : 0
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x
      let best = dist[i]
      if (x > 0) best = Math.min(best, dist[i - 1] + ORTHO)
      if (y > 0) best = Math.min(best, dist[i - width] + ORTHO)
      if (x > 0 && y > 0) best = Math.min(best, dist[i - width - 1] + DIAG)
      if (x + 1 < width && y > 0) best = Math.min(best, dist[i - width + 1] + DIAG)
      dist[i] = best
    }
  }

  for (let y = height - 1; y >= 0; y -= 1) {
    for (let x = width - 1; x >= 0; x -= 1) {
      const i = y * width + x
      let best = dist[i]
      if (x + 1 < width) best = Math.min(best, dist[i + 1] + ORTHO)
      if (y + 1 < height) best = Math.min(best, dist[i + width] + ORTHO)
      if (x > 0 && y + 1 < height) best = Math.min(best, dist[i + width - 1] + DIAG)
      if (x + 1 < width && y + 1 < height) best = Math.min(best, dist[i + width + 1] + DIAG)
      dist[i] = best
    }
  }

  return dist
}

// Distance (in texels, via BFS) from a layer's true VIEW border — the screen-space
// silhouette / self-occlusion edge where the projected view stops seeing the
// surface — measured inward through the layer's coverage. UNLIKE a plain coverage
// distance transform, this ignores UV-island edges: a coverage-border texel only
// seeds the BFS when an uncovered 4-neighbour is still INSIDE the UV layout
// (`uvOccupancyMask` === 1, i.e. the same/another chart). A neighbour that falls in
// a gutter (occupancy 0) is a UV-chart edge, NOT a view border, so it is never
// seeded — that is what prevents a "blue line at every UV seam" when feathering a
// projection into a kept base texture. Returns -1 for covered texels beyond
// `maxRadius` (the interior — full projection) and for uncovered texels.
export function computeProjectionViewBorderDistance(coverageMask, uvOccupancyMask, width, height, maxRadius) {
  const pixelCount = width * height
  const dist = new Int32Array(pixelCount).fill(-1)
  if (
    !coverageMask || coverageMask.length !== pixelCount
    || !uvOccupancyMask || uvOccupancyMask.length !== pixelCount
    || !width || !height
  ) {
    return dist
  }

  const radius = Math.max(1, Math.floor(maxRadius))
  let frontier = []
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x
      if (!coverageMask[i]) {
        continue
      }
      // View border = an uncovered 4-neighbour that is still part of the UV layout.
      const left  = x > 0           && !coverageMask[i - 1]     && uvOccupancyMask[i - 1]
      const right = x < width - 1   && !coverageMask[i + 1]     && uvOccupancyMask[i + 1]
      const up    = y > 0           && !coverageMask[i - width] && uvOccupancyMask[i - width]
      const down  = y < height - 1  && !coverageMask[i + width] && uvOccupancyMask[i + width]
      if (left || right || up || down) {
        dist[i] = 0
        frontier.push(i)
      }
    }
  }

  for (let d = 0; d < radius && frontier.length > 0; d += 1) {
    const next = []
    for (let k = 0; k < frontier.length; k += 1) {
      const i = frontier[k]
      const x = i % width
      const y = (i / width) | 0
      if (x > 0)            { const n = i - 1;     if (coverageMask[n] && dist[n] < 0) { dist[n] = d + 1; next.push(n) } }
      if (x < width - 1)    { const n = i + 1;     if (coverageMask[n] && dist[n] < 0) { dist[n] = d + 1; next.push(n) } }
      if (y > 0)            { const n = i - width; if (coverageMask[n] && dist[n] < 0) { dist[n] = d + 1; next.push(n) } }
      if (y < height - 1)   { const n = i + width; if (coverageMask[n] && dist[n] < 0) { dist[n] = d + 1; next.push(n) } }
    }
    frontier = next
  }

  return dist
}

// Rasterize the textured mesh's UV triangles once → per texel: interpolated 3D
// WORLD position, the local world-units-per-texel scale, and a "has position" flag.
// This is the surface information that lets a caller tell a genuine 3D silhouette
// border from a UV-atlas island border — two texels adjacent in the atlas but far
// apart on the mesh. UNLIKE the owned-only rasterization inside applySeamPostProcessing,
// this fills EVERY texel any triangle covers in UV, whether a projection covers it or
// not, because view-border detection needs positions on BOTH sides of the coverage
// edge (the uncovered neighbour must be locatable in 3D to test whether it is the same
// surface). Depends only on the mesh geometry/UVs and the texture size — none of which
// change while compositing — so callers should build it once per projection session and
// cache it. Returns null if the mesh has no UV'd geometry.
export function buildProjectionSurfacePositionMap(texturableMesh, width, height) {
  const pixelCount = width * height
  if (!texturableMesh?.root || !pixelCount) {
    return null
  }

  const meshes = []
  texturableMesh.root.traverse(obj => {
    if (obj.isMesh && obj.geometry && obj.geometry.attributes?.uv) meshes.push(obj)
  })
  if (meshes.length === 0) {
    return null
  }

  const posMap = new Float32Array(pixelCount * 3)
  const texelSize = new Float32Array(pixelCount)
  const hasPos = new Uint8Array(pixelCount)
  const textureConfig = texturableMesh.textureConfig

  const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3()
  const eAB = new THREE.Vector3(), eAC = new THREE.Vector3(), cross = new THREE.Vector3()
  const uvA = new THREE.Vector2(), uvB = new THREE.Vector2(), uvC = new THREE.Vector2()

  let sizeSum = 0, sizeN = 0
  for (let mi = 0; mi < meshes.length; mi++) {
    const mesh = meshes[mi]
    mesh.updateWorldMatrix(true, false)
    const matrixWorld = mesh.matrixWorld
    const geom = mesh.geometry
    const posAttr = geom.attributes.position
    const uvAttr = geom.attributes.uv
    const indexAttr = geom.index
    const triCount = indexAttr ? indexAttr.count / 3 : posAttr.count / 3

    for (let t = 0; t < triCount; t++) {
      const base = t * 3
      const i0 = indexAttr ? indexAttr.getX(base) : base
      const i1 = indexAttr ? indexAttr.getX(base + 1) : base + 1
      const i2 = indexAttr ? indexAttr.getX(base + 2) : base + 2

      vA.fromBufferAttribute(posAttr, i0).applyMatrix4(matrixWorld)
      vB.fromBufferAttribute(posAttr, i1).applyMatrix4(matrixWorld)
      vC.fromBufferAttribute(posAttr, i2).applyMatrix4(matrixWorld)

      uvA.set(uvAttr.getX(i0), uvAttr.getY(i0))
      uvB.set(uvAttr.getX(i1), uvAttr.getY(i1))
      uvC.set(uvAttr.getX(i2), uvAttr.getY(i2))
      const pA = mapUvToCanvasPoint(uvA, width, height, textureConfig)
      const pB = mapUvToCanvasPoint(uvB, width, height, textureConfig)
      const pC = mapUvToCanvasPoint(uvC, width, height, textureConfig)
      const x0 = pA.x, y0 = pA.y, x1 = pB.x, y1 = pB.y, x2 = pC.x, y2 = pC.y

      const denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
      if (Math.abs(denom) < 1e-10) continue

      // World-units per texel for this triangle = sqrt(3D area / UV-pixel area).
      eAB.subVectors(vB, vA); eAC.subVectors(vC, vA)
      const area3D = 0.5 * cross.crossVectors(eAB, eAC).length()
      const areaPx = 0.5 * Math.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
      const wpt = Math.sqrt(area3D / Math.max(areaPx, 1e-9))

      const invDenom = 1 / denom
      const minPx = Math.max(0, Math.floor(Math.min(x0, x1, x2)))
      const maxPx = Math.min(width - 1, Math.ceil(Math.max(x0, x1, x2)))
      const minPy = Math.max(0, Math.floor(Math.min(y0, y1, y2)))
      const maxPy = Math.min(height - 1, Math.ceil(Math.max(y0, y1, y2)))
      const baryEps = -1e-3

      for (let py = minPy; py <= maxPy; py++) {
        const rowBase = py * width
        for (let px = minPx; px <= maxPx; px++) {
          const idx = rowBase + px
          if (hasPos[idx]) continue
          const fx = px + 0.5, fy = py + 0.5
          const wa = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) * invDenom
          const wb = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) * invDenom
          const wc = 1 - wa - wb
          if (wa < baryEps || wb < baryEps || wc < baryEps) continue
          const m = idx * 3
          posMap[m]     = wa * vA.x + wb * vB.x + wc * vC.x
          posMap[m + 1] = wa * vA.y + wb * vB.y + wc * vC.y
          posMap[m + 2] = wa * vA.z + wb * vB.z + wc * vC.z
          texelSize[idx] = wpt
          hasPos[idx] = 1
          sizeSum += wpt; sizeN++
        }
      }
    }
  }
  if (sizeN === 0) {
    return null
  }
  return { posMap, texelSize, hasPos, avgWpt: sizeSum / sizeN, width, height }
}

// Silhouette-aware sibling of computeProjectionViewBorderDistance. Same output — the
// distance (in texels) inward through a layer's coverage from its VIEW border, -1 in
// the interior — but it decides what a "view border" is from the 3D SURFACE instead of
// UV occupancy. computeProjectionViewBorderDistance only asks "is the uncovered
// neighbour inside the UV layout?", which cannot distinguish a real silhouette from a
// packed UV-island boundary (two charts placed edge-to-edge with no gutter) and so
// feathered the kept base texture in along UV seams. This asks "is the uncovered
// neighbour on the SAME 3D surface?" using the rasterized world positions: a gutter
// neighbour (no position) or a different-island neighbour (far in 3D) never seeds, so
// the base feather lands ONLY on genuine silhouette / self-occlusion edges. `surfacePositions`
// is the object returned by buildProjectionSurfacePositionMap.
export function computeProjectionSurfaceBorderDistance(coverageMask, surfacePositions, width, height, maxRadius) {
  const pixelCount = width * height
  const dist = new Int32Array(pixelCount).fill(-1)
  if (
    !coverageMask || coverageMask.length !== pixelCount
    || !surfacePositions || !width || !height
  ) {
    return dist
  }
  const { posMap, texelSize, hasPos } = surfacePositions
  if (!posMap || !texelSize || !hasPos || hasPos.length !== pixelCount) {
    return dist
  }

  // Two image-adjacent texels are on the same surface when their 3D positions are
  // within a few texels' worth of world distance (matches applySeamPostProcessing).
  const ADJ = 4
  const onSurface = (a, b) => {
    if (!hasPos[a] || !hasPos[b]) return false
    const ma = a * 3, mb = b * 3
    const dx = posMap[ma] - posMap[mb]
    const dy = posMap[ma + 1] - posMap[mb + 1]
    const dz = posMap[ma + 2] - posMap[mb + 2]
    const lim = ADJ * Math.max(texelSize[a], texelSize[b])
    return dx * dx + dy * dy + dz * dz <= lim * lim
  }

  const radius = Math.max(1, Math.floor(maxRadius))
  let frontier = []
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x
      if (!coverageMask[i] || !hasPos[i]) {
        continue
      }
      // View border = an uncovered 4-neighbour that lies on the SAME 3D surface.
      const left  = x > 0          && !coverageMask[i - 1]     && onSurface(i, i - 1)
      const right = x < width - 1  && !coverageMask[i + 1]     && onSurface(i, i + 1)
      const up    = y > 0          && !coverageMask[i - width] && onSurface(i, i - width)
      const down  = y < height - 1 && !coverageMask[i + width] && onSurface(i, i + width)
      if (left || right || up || down) {
        dist[i] = 0
        frontier.push(i)
      }
    }
  }

  // Grow inward through the layer's own coverage, but only across on-surface edges so
  // the feather distance never jumps a UV seam into a co-packed island's interior.
  for (let d = 0; d < radius && frontier.length > 0; d += 1) {
    const next = []
    for (let k = 0; k < frontier.length; k += 1) {
      const i = frontier[k]
      const x = i % width
      const y = (i / width) | 0
      if (x > 0)          { const n = i - 1;     if (coverageMask[n] && dist[n] < 0 && onSurface(i, n)) { dist[n] = d + 1; next.push(n) } }
      if (x < width - 1)  { const n = i + 1;     if (coverageMask[n] && dist[n] < 0 && onSurface(i, n)) { dist[n] = d + 1; next.push(n) } }
      if (y > 0)          { const n = i - width; if (coverageMask[n] && dist[n] < 0 && onSurface(i, n)) { dist[n] = d + 1; next.push(n) } }
      if (y < height - 1) { const n = i + width; if (coverageMask[n] && dist[n] < 0 && onSurface(i, n)) { dist[n] = d + 1; next.push(n) } }
    }
    frontier = next
  }

  return dist
}

export function buildProjectionOverlapWeights(previousCoverage, previousSharedCoverage, layerCoverage, layerSharedMask, width, height, blendPixels = 0) {
  const pixelCount = width * height
  if (
    !previousCoverage
    || !layerCoverage
    || previousCoverage.length !== pixelCount
    || layerCoverage.length !== pixelCount
  ) {
    return null
  }

  const weights = new Float32Array(pixelCount)
  const radius = Math.max(0, Number(blendPixels) || 0)
  const maxOverlapInfluence = 0.68

  if (radius <= 0) {
    for (let i = 0; i < pixelCount; i += 1) {
      if (!layerCoverage[i]) {
        weights[i] = 0
      } else if (previousCoverage[i] <= 0) {
        weights[i] = 1
      } else if ((previousSharedCoverage?.[i] || 0) > 0 || (layerSharedMask?.[i] || 0) > 0) {
        weights[i] = maxOverlapInfluence
      } else {
        weights[i] = 0
      }
    }
    return weights
  }

  const distToUntouched = computeProjectionDistanceInsideMask(previousCoverage, width, height)

  const seamRadiusPx = Math.max(1, Math.min(6, Math.round(Math.max(1, radius) * 0.6)))
  const seamRadiusCost = Math.max(10, seamRadiusPx * 10)
  const seamDenom = seamRadiusCost + 1

  for (let i = 0; i < pixelCount; i += 1) {
    if (!layerCoverage[i]) {
      weights[i] = 0
      continue
    }

    if (previousCoverage[i] <= 0) {
      weights[i] = 1
      continue
    }

    if ((previousSharedCoverage?.[i] || 0) <= 0 && (layerSharedMask?.[i] || 0) <= 0) {
      weights[i] = 0
      continue
    }

    const dUntouched = distToUntouched[i]
    if (dUntouched > seamRadiusCost) {
      weights[i] = 0
      continue
    }

    const t = Math.max(0, Math.min(1, 1 - dUntouched / seamDenom))
    const eased = t * t * (3 - 2 * t)
    weights[i] = Math.min(maxOverlapInfluence, eased)
  }

  return weights
}

export function buildProjectionCoverageMaskFromBakedAlpha(alphaBytes, width, height, {
  minAlpha = 1,
  stitchEdges = true
} = {}) {
  const pixelCount = width * height

  const mask = new Uint8Array(pixelCount)
  if (!alphaBytes || alphaBytes.length !== pixelCount || !width || !height) {
    return mask
  }

  // Keep most anti-aliased edge texels so projected borders stay smooth.

  // Tiny pinholes are still patched in a second pass below.
  for (let i = 0; i < pixelCount; i += 1) {
    mask[i] = alphaBytes[i] > minAlpha ? 1 : 0
  }

  if (!stitchEdges) {
    return mask
  }

  const next = mask.slice()
  const has = (x, y) => {
    if (x < 0 || y < 0 || x >= width || y >= height) {
      return 0
    }
    return mask[y * width + x]
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x
      if (mask[i]) {
        continue
      }

      // Fill small 1px holes and stitch anti-aliased edge fragments so the
      // overlap mask does not create zipper seams.
      const ortho = has(x - 1, y) + has(x + 1, y) + has(x, y - 1) + has(x, y + 1)
      const diag = has(x - 1, y - 1) + has(x + 1, y - 1) + has(x - 1, y + 1) + has(x + 1, y + 1)
      if (ortho >= 3 || (ortho >= 2 && diag >= 2)) {
        next[i] = 1
      }
    }
  }

  return next
}

export function buildProjectionConfidenceMap(accumulatedWeight, coverageMask, alphaBytes) {
  const pixelCount = accumulatedWeight?.length || 0
  const confidence = new Float32Array(pixelCount)
  if (!accumulatedWeight || !pixelCount) {
    return confidence
  }

  for (let i = 0; i < pixelCount; i += 1) {
    if (coverageMask && !coverageMask[i]) {
      continue
    }

    const weight = Math.max(0, Number(accumulatedWeight[i]) || 0)
    if (weight <= 1e-6) {
      if (alphaBytes && alphaBytes[i] > 0) {
        // UV gap-filled fringes have 0 accumulated weight but are crucial to prevent black cracks.
        // We restore their confidence using the dilated alpha.
        confidence[i] = alphaBytes[i] / 255
      }
      continue
    }

    confidence[i] = Math.max(0, Math.min(1, 1 - Math.exp(-weight)))
  }

  return confidence
}

export function applyProjectionEdgeBleed(canvas, passes = 1) {
  if (!canvas?.width || !canvas?.height) {
    return
  }

  const width = canvas.width
  const height = canvas.height
  const context = canvas.getContext('2d', { willReadFrequently: true }) || canvas.getContext('2d')
  if (!context) {
    return
  }

  let imageData = context.getImageData(0, 0, width, height)
  let data = imageData.data
  const totalPasses = Math.max(1, Math.min(2, Math.floor(passes)))

  for (let pass = 0; pass < totalPasses; pass += 1) {
    const source = new Uint8ClampedArray(data)
    let changed = false

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x
        const offset = index * 4
        const a = source[offset + 3]
        if (a > 0) {
          continue
        }

        let sumR = 0
        let sumG = 0
        let sumB = 0
        let sumA = 0
        let count = 0

        for (let dy = -1; dy <= 1; dy += 1) {
          const ny = y + dy
          if (ny < 0 || ny >= height) {
            continue
          }

          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0) {
              continue
            }

            const nx = x + dx
            if (nx < 0 || nx >= width) {
              continue
            }

            const nOffset = (ny * width + nx) * 4
            const na = source[nOffset + 3]
            if (na <= 0) {
              continue
            }

            sumR += source[nOffset]
            sumG += source[nOffset + 1]
            sumB += source[nOffset + 2]
            sumA += na
            count += 1
          }
        }

        if (count > 0) {
          data[offset] = Math.round(sumR / count)
          data[offset + 1] = Math.round(sumG / count)
          data[offset + 2] = Math.round(sumB / count)
          data[offset + 3] = Math.max(1, Math.min(255, Math.round((sumA / count) * 0.38)))
          changed = true
        }
      }
    }

    if (!changed) {
      break
    }
  }

  context.putImageData(imageData, 0, 0)
}

export function clamp01(value) {
  return Math.max(0, Math.min(1, value))
}

export function blendRgbByMode(mode, dstR, dstG, dstB, srcR, srcG, srcB) {
  const m = String(mode || 'source-over').toLowerCase()
  if (m === 'multiply') {
    return [
      (dstR * srcR) / 255,
      (dstG * srcG) / 255,
      (dstB * srcB) / 255
    ]
  }

  if (m === 'screen') {
    return [
      255 - ((255 - dstR) * (255 - srcR)) / 255,
      255 - ((255 - dstG) * (255 - srcG)) / 255,
      255 - ((255 - dstB) * (255 - srcB)) / 255
    ]
  }

  if (m === 'overlay') {
    const overlayChannel = (d, s) => {
      if (d < 128) {
        return (2 * d * s) / 255
      }
      return 255 - (2 * (255 - d) * (255 - s)) / 255
    }
    return [overlayChannel(dstR, srcR), overlayChannel(dstG, srcG), overlayChannel(dstB, srcB)]
  }

  if (m === 'darken') {
    return [Math.min(dstR, srcR), Math.min(dstG, srcG), Math.min(dstB, srcB)]
  }

  if (m === 'lighten') {
    return [Math.max(dstR, srcR), Math.max(dstG, srcG), Math.max(dstB, srcB)]
  }

  // Normal / source-over fallback.
  return [srcR, srcG, srcB]
}

export function compositeProjectionLayerIntoImageData({
  outputData,
  layerData,
  layerCoverage,
  ownershipMask,
  sharedMask,
  layerConfidence,
  overlapWeights,
  composedCoverage,
  composedSharedCoverage,
  composedConfidence,
  seamAccumColor,
  seamAccumWeight,
  opacity,
  blendMode
}) {
  if (
    !outputData
    || !layerData
    || !layerCoverage
    || !overlapWeights
    || !composedCoverage
    || !composedSharedCoverage
    || !composedConfidence
    || !seamAccumColor
    || !seamAccumWeight
  ) {
    return 0
  }

  const pixelCount = layerCoverage.length
  let contributed = 0
  const op = clamp01(Number(opacity) || 0)

  if (op <= 0) {
    return 0
  }

  for (let i = 0; i < pixelCount; i += 1) {
    if (!layerCoverage[i]) {
      continue
    }

    const w = overlapWeights[i]
    if (w <= 1e-6) {
      continue
    }

    const alpha = clamp01(op * w)
    if (alpha <= 1e-6) {
      continue
    }

    const j = i * 4
    const dstR = outputData[j]
    const dstG = outputData[j + 1]
    const dstB = outputData[j + 2]
    const srcR = layerData[j]
    const srcG = layerData[j + 1]
    const srcB = layerData[j + 2]
    const srcAlpha = layerData[j + 3] / 255
    if (srcAlpha <= 1e-4) {
      continue
    }

    const [blendR, blendG, blendB] = blendRgbByMode(blendMode, dstR, dstG, dstB, srcR, srcG, srcB)
    const effectiveAlpha = clamp01(alpha * srcAlpha)
    if (effectiveAlpha <= 1e-6) {
      continue
    }
    const confidence = Math.max(0.05, Math.min(1, layerConfidence?.[i] || effectiveAlpha))

    const shouldAccumulateShared = Boolean(
      sharedMask?.[i]
      && (composedCoverage[i] > 0 || composedSharedCoverage[i] > 0)
    )

    if (shouldAccumulateShared) {
      if (seamAccumWeight[i] <= 1e-6) {
        const baseConfidence = Math.max(0.05, Math.min(1, composedConfidence[i] || 0.05))
        seamAccumColor[j] = dstR * baseConfidence
        seamAccumColor[j + 1] = dstG * baseConfidence
        seamAccumColor[j + 2] = dstB * baseConfidence
        seamAccumWeight[i] = baseConfidence
      }

      const seamWeight = effectiveAlpha * confidence
      seamAccumColor[j] += blendR * seamWeight
      seamAccumColor[j + 1] += blendG * seamWeight
      seamAccumColor[j + 2] += blendB * seamWeight
      seamAccumWeight[i] += seamWeight
      composedSharedCoverage[i] = 1
      composedConfidence[i] = Math.max(composedConfidence[i], confidence)
      contributed += 1
      continue
    }

    outputData[j] = Math.round(dstR * (1 - effectiveAlpha) + blendR * effectiveAlpha)
    outputData[j + 1] = Math.round(dstG * (1 - effectiveAlpha) + blendG * effectiveAlpha)
    outputData[j + 2] = Math.round(dstB * (1 - effectiveAlpha) + blendB * effectiveAlpha)
    outputData[j + 3] = 255
    composedConfidence[i] = Math.max(composedConfidence[i], confidence)

    if (sharedMask?.[i]) {
      composedSharedCoverage[i] = 1
    }

    if (ownershipMask?.[i]) {
      composedCoverage[i] = 1
      if (!sharedMask?.[i]) {
        composedSharedCoverage[i] = 0
      }
    }

    contributed += 1
  }

  return contributed
}

export function resolveProjectionSharedSeams(outputData, seamAccumColor, seamAccumWeight) {
  if (!outputData || !seamAccumColor || !seamAccumWeight) {
    return
  }

  for (let i = 0; i < seamAccumWeight.length; i += 1) {
    const weight = seamAccumWeight[i]
    if (weight <= 1e-6) {
      continue
    }

    const j = i * 4
    outputData[j] = Math.round(seamAccumColor[j] / weight)
    outputData[j + 1] = Math.round(seamAccumColor[j + 1] / weight)
    outputData[j + 2] = Math.round(seamAccumColor[j + 2] / weight)
    outputData[j + 3] = 255
  }
}

// Final-composite gutter padding. After the ownership composite resolves every
// covered texel, bleed those colours a few texels past each covered island edge
// into the still-uncovered base. The display texture uses LinearFilter with no
// mipmaps, so sampling reaches ~0.5 texel beyond a UV-island border at a seam;
// without padding it pulls the unpainted (white) base in, producing the thin
// white "wireframe" seams. The per-layer GPU dilation only seeds from head-on
// coverage (DILATE_SEED_MIN_WEIGHT, to avoid a multi-view gray seam), so it
// leaves grazing-but-covered islands' gutters white — this pass is ungated and
// fills them. It runs on the FINAL composite (colours already resolved by
// ownership), so it can never recreate that cross-view gray seam.
export const PROJECTION_GUTTER_PAD_PX = 4
export function dilateProjectionGutter(outputData, coverage, width, height, radius = PROJECTION_GUTTER_PAD_PX, uvOccupancyMask = null) {
  if (!outputData || !coverage || !width || !height) {
    return
  }
  const passes = Math.max(1, Math.floor(radius))
  const grown = Uint8Array.from(coverage)
  const useOccupancy = uvOccupancyMask && uvOccupancyMask.length === width * height
  for (let pass = 0; pass < passes; pass += 1) {
    const src = Uint8Array.from(grown)
    let changed = false
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const i = y * width + x
        if (src[i]) {
          continue
        }
        // An uncovered texel is either an island's OWN under-covered edge (occupancy
        // == 1 — what produced the thin white UV-seam lines: no view's bake reached it)
        // or an empty inter-island gutter (occupancy == 0). We now fill BOTH, but an
        // island texel may only borrow colour from neighbours that are ALSO on an island
        // (same-chart extension). Borrowing from a filled gutter (occupancy 0) is the
        // gutter→neighbour-chart step that leaks one view's colour onto a DIFFERENT
        // chart, so that single direction stays forbidden.
        const fillingIsland = useOccupancy && uvOccupancyMask[i]
        let sumR = 0
        let sumG = 0
        let sumB = 0
        let count = 0
        for (let dy = -1; dy <= 1; dy += 1) {
          const ny = y + dy
          if (ny < 0 || ny >= height) {
            continue
          }
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0) {
              continue
            }
            const nx = x + dx
            if (nx < 0 || nx >= width) {
              continue
            }
            const ni = ny * width + nx
            if (!src[ni]) {
              continue
            }
            // Same-chart extension only when filling an island texel (see above).
            if (fillingIsland && useOccupancy && !uvOccupancyMask[ni]) {
              continue
            }
            const no = ni * 4
            sumR += outputData[no]
            sumG += outputData[no + 1]
            sumB += outputData[no + 2]
            count += 1
          }
        }
        if (count > 0) {
          const o = i * 4
          outputData[o] = Math.round(sumR / count)
          outputData[o + 1] = Math.round(sumG / count)
          outputData[o + 2] = Math.round(sumB / count)
          outputData[o + 3] = 255
          grown[i] = 1
          changed = true
        }
      }
    }
    if (!changed) {
      break
    }
  }
}

// Ordered-ownership composite. Layers are applied IN ORDER (layer 0 = the first
// projection the user applied). Each layer:
//   • fully paints texels nothing has covered yet (gap fill — so no face the view
//     touches is left untextured and nothing ever "untextures");
//   • does NOT repaint the interior of a region an earlier layer already owns
//     (so later views never change already-textured faces);
//   • blends only within `blendPixels` of the already-owned boundary (the seam),
//     so the "Blend overlap" slider widens the cross-fade WITHOUT removing coverage;
//   • may take over the RIM of an earlier layer — texels that layer only saw near
//     its own silhouette (`edgeFadeMap` < 1, from the GPU bake) — in proportion to
//     how much better it sees them. Scaled by "Opacity seams" like the seam blend.
// This matches the intent: front view owns what it sees; the next view fills the
// rest and feathers across the join.
//
// `blendWithBase` is set when the session kept the mesh's ORIGINAL texture as the
// base (Keep texture). In that case `outputData` arrives pre-filled with real
// texture everywhere, so a layer must NOT hard-overwrite the base at its true VIEW
// border (its screen-space silhouette / self-occlusion edge) — it feathers across
// `blendPixels` there into the base. The feather is seeded from the view border
// only, NOT from UV-island edges, so it does not paint the base over every UV seam.
// With a fresh checkerboard base this stays off: the layer fully fills its footprint
// so the checker never bleeds in.
//
// `surfacePositions` (from buildProjectionSurfacePositionMap) is preferred for that
// view-border detection: it uses the mesh's 3D positions to seed the base feather ONLY
// at genuine silhouette edges (computeProjectionSurfaceBorderDistance), which the
// UV-occupancy fallback (computeProjectionViewBorderDistance) cannot do — the latter
// treats a packed UV-island boundary as a view border and so feathers the base across
// UV seams. When no mesh/position map is available it falls back to `uvOccupancyMask`.
export function resolveProjectionLayersIntoImageData(outputData, layerSnapshots, width, height, viewGains = null, uvOccupancyMask = null, blendWithBase = false, surfacePositions = null) {
  if (!outputData || !Array.isArray(layerSnapshots) || layerSnapshots.length === 0 || !width || !height) {
    return
  }

  const pixelCount = width * height
  const committed = new Uint8Array(pixelCount) // texels owned by already-processed layers
  // Best per-texel confidence among the layers committed so far. Drives the seam
  // cross-fade: a later view blends in proportional to how well IT sees the texel
  // RELATIVE to the best committed view there (not an absolute threshold).
  const committedConf = new Float32Array(pixelCount)
  // Strongest silhouette-edge fade among the committed layers per texel (see
  // `edgeFadeMap` on a snapshot; 1 = interior of some view, ~0 = only ever seen at a
  // view's silhouette). Where it is below 1 the owner is unsure of its own pixel —
  // the rim is where a generated view bleeds backdrop colour and where a view that
  // drifted from the geometry misses — so a later view may take the texel over in
  // proportion. Layers without a map (CPU bake) count as interior everywhere.
  const committedEdge = new Float32Array(pixelCount)
  // Base strength a later view blends into an owned seam texel at the very border;
  // ramps to 0 `blendPixels` inside. Scaled per layer by its "Opacity seams" knob.
  const SEAM_MAX = 0.7

  for (let layerIndex = 0; layerIndex < layerSnapshots.length; layerIndex += 1) {
    const layer = layerSnapshots[layerIndex]
    if (!layer?.coverageMask || !layer?.pixelData) {
      continue
    }
    const opacity = clamp01(layer.opacity ?? 1)
    if (opacity <= 0) {
      continue
    }

    // "Opacity seams" now controls how strongly this view blends across an already
    // -owned border: 1 = full seam cross-fade, 0 = hard ownership edge (no blend).
    const seamMax = SEAM_MAX * clamp01(layer.opacitySeams ?? 1)
    // Per-view colour gain (Brown–Lowe) to match this view's overall tone to the
    // others before blending — collapses photometric seams between views ComfyUI
    // coloured differently. Identity when compensation is off / single layer.
    const gain = viewGains?.[layerIndex] || null
    const gainR = gain ? gain[0] : 1
    const gainG = gain ? gain[1] : 1
    const gainB = gain ? gain[2] : 1

    const blendPx = Math.max(0, Number(layer.blendPixels) || 0)
    // Distance (in pixels) INWARD through the owned region from the genuine INTER-VIEW
    // SEAM — the boundary where the committed (earlier-view) region ends and THIS
    // layer's coverage continues. computeProjectionViewBorderDistance seeds exactly at
    // committed texels whose non-committed neighbour is covered by this layer, so the
    // feather follows view-ownership boundaries ONLY. A plain distance-from-committed
    // -border transform (the previous approach) also fired on UV-chart edges and
    // coverage holes, so the blend smeared around UV seams (patchy white blotches)
    // instead of the view seam. Returns -1 for texels farther than blendPx from a seam
    // and for non-committed texels → those stay locked.
    const ownedDist = (layerIndex > 0 && blendPx > 0 && seamMax > 0)
      ? computeProjectionViewBorderDistance(committed, layer.coverageMask, width, height, blendPx)
      : null

    // Distance from this layer's true VIEW border (silhouette/occlusion edge),
    // inward through its coverage — 0 at the border, growing inward, -1 in the
    // interior / on UV-chart edges. Used only when keeping the original texture, to
    // feather the projection into the base at the view border WITHOUT touching UV
    // seams. Prefers the mesh's 3D positions (surfacePositions) to locate genuine
    // silhouette edges; falls back to the UV occupancy mask when no mesh is available.
    const viewBorderDist = (blendWithBase && blendPx > 0)
      ? (surfacePositions
          // Preferred: seed the base feather only at true 3D silhouette edges.
          ? computeProjectionSurfaceBorderDistance(layer.coverageMask, surfacePositions, width, height, blendPx)
          // Fallback (no mesh available): UV-occupancy border — rejects gutters but
          // not packed UV-island boundaries, so it can feather the base across UV seams.
          : (uvOccupancyMask
              ? computeProjectionViewBorderDistance(layer.coverageMask, uvOccupancyMask, width, height, blendPx)
              : null))
      : null

    const edgeFade = layer.edgeFadeMap?.length === pixelCount ? layer.edgeFadeMap : null

    for (let i = 0; i < pixelCount; i += 1) {
      if (!layer.coverageMask[i]) {
        continue
      }
      const j = i * 4
      const srcAlpha = (layer.pixelData[j + 3] || 0) / 255
      if (srcAlpha <= 1e-4) {
        continue
      }

      let influence
      if (!committed[i]) {
        const dBorder = viewBorderDist ? viewBorderDist[i] : -1
        if (dBorder >= 0) {
          // Keep-texture mode, within blendPixels of the view's silhouette: the base
          // under this texel is REAL texture, so feather — full base at the border
          // (dBorder 0) ramping to full projection blendPixels inside. This is the
          // "projections fade with the current texture at their seams" behaviour, and
          // because dBorder is only set near the VIEW border (not UV-chart edges) it
          // does not bleed the base across UV seams.
          const t = clamp01(dBorder / blendPx)
          influence = (t * t * (3 - 2 * t)) * opacity
        } else {
          // Interior of the view (or fresh-checkerboard mode): paint fully — it is the
          // only data here, so no base/checker bleed and nothing is left untextured.
          influence = opacity
        }
        // Keep-texture mode: the base under this texel is real texture, so the view's
        // silhouette rim fades into it instead of painting the rim's backdrop-tinted
        // colour. With a fresh checkerboard base the rim still paints fully — it is
        // the only data there, and fading it would show the checker.
        if (blendWithBase && edgeFade) {
          influence = Math.min(influence, edgeFade[i] * opacity)
        }
      } else {
        // RELATIVE-confidence weight: how well THIS view sees the texel versus the
        // best view already committed here. This avoids every brittle absolute threshold:
        //  • both views see it comparably (a normal view-to-view seam) → ~0.5 → smooth
        //    50/50 cross-fade across the band (the arm seam now actually blends);
        //  • the owner only has the dilation PAD here (conf ≈ 0.02) while this view sees it
        //    for real → ratio ≈ 1 → this view takes over, so the pad never shows as a line
        //    offset from the seam;
        //  • this view only grazes its own silhouette here (washed/dark sample, low conf)
        //    while the owner saw it well → ratio ≈ 0 → owner kept, no washed/dark bleed.
        const thisConf = layer.confidenceMap?.[i] || 0
        const ownerConf = committedConf[i] || 0
        const relWeight = thisConf / (thisConf + ownerConf + 1e-4)

        let seamInfluence = 0
        if (ownedDist && ownedDist[i] >= 0) {
          // Within blendPixels of a genuine inter-view seam → cross-fade. Interior
          // texels (dist -1) stay locked, so the blend never touches UV-chart edges or
          // coverage holes — only the view-ownership boundary.
          const dEdge = ownedDist[i] // pixels from the seam (0 at the seam, growing inward)
          const t = clamp01(1 - dEdge / Math.max(1, blendPx)) // 1 at the seam → 0 blendPx inside
          seamInfluence = (t * t * (3 - 2 * t)) * relWeight * (seamMax / SEAM_MAX) * opacity
        }

        // Rim takeover: the owner only saw this texel near its own silhouette, so a
        // later view that sees it better replaces it in proportion. Deep inside every
        // owner (committedEdge 1) this is 0 — already-textured faces stay locked.
        // Both confidences already include each view's own edge fade.
        let rimInfluence = 0
        if (committedEdge[i] < 1) {
          rimInfluence = (1 - committedEdge[i]) * relWeight * (seamMax / SEAM_MAX) * opacity
        }

        influence = Math.max(seamInfluence, rimInfluence)
      }

      if (influence <= 1e-4) {
        continue
      }

      // Recover straight colour. GPU bake stores straight colour with alpha 1; the
      // CPU bake stores colour premultiplied by the mask weight, so divide it out.
      // Then apply the per-view gain so this view's tone matches the others.
      const invSrcAlpha = 1 / srcAlpha
      const srcR = Math.min(255, Math.round(layer.pixelData[j]     * invSrcAlpha * gainR))
      const srcG = Math.min(255, Math.round(layer.pixelData[j + 1] * invSrcAlpha * gainG))
      const srcB = Math.min(255, Math.round(layer.pixelData[j + 2] * invSrcAlpha * gainB))
      const [blendR, blendG, blendB] = blendRgbByMode(
        layer.blendMode, outputData[j], outputData[j + 1], outputData[j + 2], srcR, srcG, srcB
      )

      outputData[j]     = Math.round(outputData[j]     * (1 - influence) + blendR * influence)
      outputData[j + 1] = Math.round(outputData[j + 1] * (1 - influence) + blendG * influence)
      outputData[j + 2] = Math.round(outputData[j + 2] * (1 - influence) + blendB * influence)
      outputData[j + 3] = 255
    }

    // Commit this layer's coverage so later layers treat it as owned, and record the
    // best confidence seen so far per texel (the reference the next layer's seam blend
    // weighs itself against).
    const layerConf = layer.confidenceMap
    for (let i = 0; i < pixelCount; i += 1) {
      if (layer.coverageMask[i]) {
        committed[i] = 1
        const c = layerConf ? (layerConf[i] || 0) : 1
        if (c > committedConf[i]) committedConf[i] = c
        const e = edgeFade ? edgeFade[i] : 1
        if (e > committedEdge[i]) committedEdge[i] = e
      }
    }
  }

  // Pad the resolved colours past every covered island edge so display-time
  // bilinear sampling cannot pull the unpainted base across UV seams (the thin
  // white "wireframe" lines). Pass UV occupancy so the pad only fills genuinely
  // empty gutter texels and never bleeds a view's colour across a thin gutter onto
  // a neighbouring island (front colour leaking onto back-of-mesh charts).
  dilateProjectionGutter(outputData, committed, width, height, undefined, uvOccupancyMask)
}

// Seam smoothing post-process — SILHOUETTE seams, not UV seams.
//
// The visible seams in a projection bake are the SILHOUETTE joins where one
// view's projection meets another ON THE 3D SURFACE (e.g. the front view of an
// arm meeting the side view down the arm). ComfyUI colours each view slightly
// differently, so a tonal step shows at that join. The goal of this pass is to
// blend that step away smoothly.
//
// The previous implementation reasoned entirely in UV/texel space: it seeded a
// "seam" wherever two image-adjacent texels had different owners, and blurred
// with a CSS blur() over the atlas. But in a UV atlas, image-adjacent texels
// routinely belong to DIFFERENT islands (different parts of the mesh), so it
// (a) flagged every UV-island border as a false seam and (b) smeared colours
// across islands — the source of the heavy artifacts. It smoothed UV seams, the
// exact opposite of what is wanted.
//
// This version is 3D / surface-aware (like fillHolesPostProcessing): it
// rasterizes the mesh to recover each covered texel's world position, treats an
// ownership change as a seam ONLY when the two texels are close in 3D (a real
// silhouette join — UV-island borders are far in 3D and get rejected), grows
// the band ALONG the surface, and smooths each band texel by a 3D gaussian
// average over a small world-space ball. The ball straddles the silhouette seam
// (so the average is a clean blend of the two views) but is far too small to
// reach a different UV island, so no cross-island leakage. Needs the mesh; if it
// is unavailable the pass is a no-op (better than smearing UV seams).
export async function applySeamPostProcessing(textureCanvas, layerSnapshots, seamWidth, blurRadius, strength, texturableMesh, onProgress) {
  const w = textureCanvas.width
  const h = textureCanvas.height
  const pixelCount = w * h
  if (!pixelCount || !Array.isArray(layerSnapshots) || layerSnapshots.length === 0) return

  // 1. Reconstruct ownership: first layer (in order) to cover a texel owns it.
  const owner = new Int32Array(pixelCount).fill(-1)
  let distinctOwners = 0
  for (let li = 0; li < layerSnapshots.length; li++) {
    const cov = layerSnapshots[li]?.coverageMask
    if (!cov) continue
    let used = false
    for (let i = 0; i < pixelCount; i++) {
      if (owner[i] < 0 && cov[i]) { owner[i] = li; used = true }
    }
    if (used) distinctOwners++
  }
  if (distinctOwners < 2) return  // single view: no inter-view silhouette joins

  // The mesh is what lets us tell a real silhouette join from a UV-atlas border.
  const meshes = []
  if (texturableMesh?.root) {
    texturableMesh.root.traverse(obj => {
      if (obj.isMesh && obj.geometry && obj.geometry.attributes?.uv) meshes.push(obj)
    })
  }
  if (meshes.length === 0) return

  const ctx = textureCanvas.getContext('2d', { willReadFrequently: true }) || textureCanvas.getContext('2d')
  const origData = ctx.getImageData(0, 0, w, h)
  const srcPix = origData.data

  // 2. Rasterize every UV triangle → per covered texel: 3D world position and the
  //    local world-units-per-texel scale. This is what moves the rest of the pass
  //    out of UV space and onto the surface.
  const posMap = new Float32Array(pixelCount * 3)
  const texelSize = new Float32Array(pixelCount)
  const hasPos = new Uint8Array(pixelCount)
  const textureConfig = texturableMesh.textureConfig

  const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3()
  const eAB = new THREE.Vector3(), eAC = new THREE.Vector3(), cross = new THREE.Vector3()
  const uvA = new THREE.Vector2(), uvB = new THREE.Vector2(), uvC = new THREE.Vector2()

  for (let mi = 0; mi < meshes.length; mi++) {
    const mesh = meshes[mi]
    mesh.updateWorldMatrix(true, false)
    const matrixWorld = mesh.matrixWorld
    const geom = mesh.geometry
    const posAttr = geom.attributes.position
    const uvAttr = geom.attributes.uv
    const indexAttr = geom.index
    const triCount = indexAttr ? indexAttr.count / 3 : posAttr.count / 3

    for (let t = 0; t < triCount; t++) {
      const base = t * 3
      const i0 = indexAttr ? indexAttr.getX(base) : base
      const i1 = indexAttr ? indexAttr.getX(base + 1) : base + 1
      const i2 = indexAttr ? indexAttr.getX(base + 2) : base + 2

      vA.fromBufferAttribute(posAttr, i0).applyMatrix4(matrixWorld)
      vB.fromBufferAttribute(posAttr, i1).applyMatrix4(matrixWorld)
      vC.fromBufferAttribute(posAttr, i2).applyMatrix4(matrixWorld)

      uvA.set(uvAttr.getX(i0), uvAttr.getY(i0))
      uvB.set(uvAttr.getX(i1), uvAttr.getY(i1))
      uvC.set(uvAttr.getX(i2), uvAttr.getY(i2))
      const pA = mapUvToCanvasPoint(uvA, w, h, textureConfig)
      const pB = mapUvToCanvasPoint(uvB, w, h, textureConfig)
      const pC = mapUvToCanvasPoint(uvC, w, h, textureConfig)
      const x0 = pA.x, y0 = pA.y, x1 = pB.x, y1 = pB.y, x2 = pC.x, y2 = pC.y

      const denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
      if (Math.abs(denom) < 1e-10) continue

      // World-units per texel for this triangle = sqrt(3D area / UV-pixel area).
      eAB.subVectors(vB, vA); eAC.subVectors(vC, vA)
      const area3D = 0.5 * cross.crossVectors(eAB, eAC).length()
      const areaPx = 0.5 * Math.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
      const wpt = Math.sqrt(area3D / Math.max(areaPx, 1e-9))

      const invDenom = 1 / denom
      const minPx = Math.max(0, Math.floor(Math.min(x0, x1, x2)))
      const maxPx = Math.min(w - 1, Math.ceil(Math.max(x0, x1, x2)))
      const minPy = Math.max(0, Math.floor(Math.min(y0, y1, y2)))
      const maxPy = Math.min(h - 1, Math.ceil(Math.max(y0, y1, y2)))
      const baryEps = -1e-3

      for (let py = minPy; py <= maxPy; py++) {
        const rowBase = py * w
        for (let px = minPx; px <= maxPx; px++) {
          const idx = rowBase + px
          if (owner[idx] < 0 || hasPos[idx]) continue
          const fx = px + 0.5, fy = py + 0.5
          const wa = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) * invDenom
          const wb = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) * invDenom
          const wc = 1 - wa - wb
          if (wa < baryEps || wb < baryEps || wc < baryEps) continue
          const m = idx * 3
          posMap[m]     = wa * vA.x + wb * vB.x + wc * vC.x
          posMap[m + 1] = wa * vA.y + wb * vB.y + wc * vC.y
          posMap[m + 2] = wa * vA.z + wb * vB.z + wc * vC.z
          texelSize[idx] = wpt
          hasPos[idx] = 1
        }
      }
    }
  }
  if (onProgress) onProgress(0.4)

  // Mean world-per-texel — used for grid sizing.
  let sizeSum = 0, sizeN = 0
  for (let i = 0; i < pixelCount; i++) if (hasPos[i]) { sizeSum += texelSize[i]; sizeN++ }
  if (sizeN === 0) return
  const avgWpt = sizeSum / sizeN

  // On-surface test for two image-adjacent texels: close in 3D ⇒ the same surface
  // region (a real seam can cross here); far ⇒ a UV-atlas island border (reject).
  const ADJ = 4
  const onSurface = (a, b) => {
    if (!hasPos[a] || !hasPos[b]) return false
    const ma = a * 3, mb = b * 3
    const dx = posMap[ma] - posMap[mb]
    const dy = posMap[ma + 1] - posMap[mb + 1]
    const dz = posMap[ma + 2] - posMap[mb + 2]
    const lim = ADJ * Math.max(texelSize[a], texelSize[b])
    return dx * dx + dy * dy + dz * dz <= lim * lim
  }

  // 3. Seed seam texels: covered, with an ON-SURFACE 4-neighbour owned by a
  //    DIFFERENT view. UV-island borders fail onSurface, so they never seed —
  //    this is the fix for "smooths UV seams instead of silhouette seams".
  let frontier = []
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      const o = owner[i]
      if (o < 0 || !hasPos[i]) continue
      if ((x > 0     && owner[i - 1] >= 0 && owner[i - 1] !== o && onSurface(i, i - 1))
       || (x < w - 1 && owner[i + 1] >= 0 && owner[i + 1] !== o && onSurface(i, i + 1))
       || (y > 0     && owner[i - w] >= 0 && owner[i - w] !== o && onSurface(i, i - w))
       || (y < h - 1 && owner[i + w] >= 0 && owner[i + w] !== o && onSurface(i, i + w))) {
        frontier.push(i)
      }
    }
  }
  if (frontier.length === 0) return  // no real silhouette joins to smooth

  // 4. BFS the band ALONG the surface (only crossing on-surface edges), capped at
  //    the band radius. "Seam width" (0..1) maps to that radius, resolution-scaled.
  const bandRadius = Math.max(1, Math.min(64, Math.round(seamWidth * Math.max(w, h) / 64)))
  const dist = new Int32Array(pixelCount).fill(-1)
  for (let k = 0; k < frontier.length; k++) dist[frontier[k]] = 0
  for (let d = 0; d < bandRadius && frontier.length > 0; d++) {
    const next = []
    for (let k = 0; k < frontier.length; k++) {
      const i = frontier[k]
      const x = i % w, y = (i / w) | 0
      if (x > 0)     { const n = i - 1; if (owner[n] >= 0 && dist[n] < 0 && onSurface(i, n)) { dist[n] = d + 1; next.push(n) } }
      if (x < w - 1) { const n = i + 1; if (owner[n] >= 0 && dist[n] < 0 && onSurface(i, n)) { dist[n] = d + 1; next.push(n) } }
      if (y > 0)     { const n = i - w; if (owner[n] >= 0 && dist[n] < 0 && onSurface(i, n)) { dist[n] = d + 1; next.push(n) } }
      if (y < h - 1) { const n = i + w; if (owner[n] >= 0 && dist[n] < 0 && onSurface(i, n)) { dist[n] = d + 1; next.push(n) } }
    }
    frontier = next
  }
  if (onProgress) onProgress(0.55)

  // 5. Spatial hash of every covered texel (by 3D position) for fast ball queries.
  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
  for (let i = 0; i < pixelCount; i++) {
    if (!hasPos[i]) continue
    const m = i * 3
    const x = posMap[m], y = posMap[m + 1], z = posMap[m + 2]
    if (x < minX) minX = x; if (x > maxX) maxX = x
    if (y < minY) minY = y; if (y > maxY) maxY = y
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
  }
  const bbox = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6)
  const blurR = Math.max(1, Math.min(64, Math.round(blurRadius)))
  // Cell ≈ one blur radius wide, but floored so the grid never overflows the key
  // packing (≤ ~400 cells/axis keeps each index < 1024 for the 10-bit pack below).
  const cellSize = Math.max(blurR * avgWpt, bbox / 400)
  const invCell = 1 / cellSize
  const hashCell = (cx, cy, cz) => ((cx + 512) << 20) | ((cy + 512) << 10) | (cz + 512)
  const grid = new Map()
  for (let i = 0; i < pixelCount; i++) {
    if (!hasPos[i]) continue
    const m = i * 3
    const cx = Math.floor((posMap[m] - minX) * invCell)
    const cy = Math.floor((posMap[m + 1] - minY) * invCell)
    const cz = Math.floor((posMap[m + 2] - minZ) * invCell)
    const key = hashCell(cx, cy, cz)
    let cell = grid.get(key)
    if (!cell) { cell = []; grid.set(key, cell) }
    cell.push(i)
  }

  // 6. Smooth each band texel by a 3D gaussian average of nearby covered texels.
  //    The ball spans the silhouette seam (blending the two views) but is tiny in
  //    world space, so it cannot reach a different UV island. Feather the blend by
  //    distance-from-seam so interiors are untouched.
  const out = new Uint8ClampedArray(srcPix)
  let processed = 0
  let lastYield = performance.now()

  for (let i = 0; i < pixelCount; i++) {
    const d = dist[i]
    if (d < 0 || !hasPos[i]) continue

    const m = i * 3
    const qx = posMap[m], qy = posMap[m + 1], qz = posMap[m + 2]
    const radius = Math.max(blurR * texelSize[i], cellSize * 0.5)
    const r2 = radius * radius
    const sigma = radius * 0.6
    const twoSigma2 = 2 * sigma * sigma
    const baseCx = Math.floor((qx - minX) * invCell)
    const baseCy = Math.floor((qy - minY) * invCell)
    const baseCz = Math.floor((qz - minZ) * invCell)
    const cr = Math.max(1, Math.ceil(radius * invCell))

    let sumR = 0, sumG = 0, sumB = 0, sumW = 0
    for (let dz = -cr; dz <= cr; dz++) {
      for (let dy = -cr; dy <= cr; dy++) {
        for (let dx = -cr; dx <= cr; dx++) {
          const cell = grid.get(hashCell(baseCx + dx, baseCy + dy, baseCz + dz))
          if (!cell) continue
          for (let c = 0; c < cell.length; c++) {
            const s = cell[c]
            const sm = s * 3
            const ddx = posMap[sm] - qx, ddy = posMap[sm + 1] - qy, ddz = posMap[sm + 2] - qz
            const dd = ddx * ddx + ddy * ddy + ddz * ddz
            if (dd > r2) continue
            const wgt = Math.exp(-dd / twoSigma2)
            const sj = s * 4
            sumR += srcPix[sj] * wgt
            sumG += srcPix[sj + 1] * wgt
            sumB += srcPix[sj + 2] * wgt
            sumW += wgt
          }
        }
      }
    }
    if (sumW <= 1e-6) continue

    const t = 1 - d / bandRadius            // 1 at the seam → 0 at the band edge
    const feather = t * t * (3 - 2 * t)
    const f = feather * strength
    if (f <= 1e-3) continue

    const j = i * 4
    out[j]     = Math.round(srcPix[j]     * (1 - f) + (sumR / sumW) * f)
    out[j + 1] = Math.round(srcPix[j + 1] * (1 - f) + (sumG / sumW) * f)
    out[j + 2] = Math.round(srcPix[j + 2] * (1 - f) + (sumB / sumW) * f)

    processed++
    if ((processed & 0x3FFF) === 0) {
      const now = performance.now()
      if (now - lastYield > 30) {
        await new Promise(r => setTimeout(r, 0))
        lastYield = now
      }
    }
  }

  if (onProgress) onProgress(0.98)
  ctx.putImageData(new ImageData(out, w, h), 0, 0)
  if (onProgress) onProgress(1)
}

// 3D-aware hole filling. UV-space proximity is unreliable for AI-generated
// meshes where neighbouring UV islands can come from opposite sides of the
// body, so we fill each uncovered texel from the K nearest covered samples
// in 3D world space instead.
export async function fillHolesPostProcessing(textureCanvas, layerSnapshots, texturableMesh, smoothness, onProgress) {
  const w = textureCanvas.width
  const h = textureCanvas.height
  const pixelCount = w * h

  // Build coverage union across all snapshots
  const anyCoverage = new Uint8Array(pixelCount)
  for (let li = 0; li < layerSnapshots.length; li++) {
    const layer = layerSnapshots[li]
    if (!layer?.coverageMask) continue
    for (let i = 0; i < pixelCount; i++) {
      if (layer.coverageMask[i]) anyCoverage[i] = 1
    }
  }

  // Gather textured meshes (every mesh under root with a uv attribute)
  const meshes = []
  if (texturableMesh?.root) {
    texturableMesh.root.traverse(obj => {
      if (obj.isMesh && obj.geometry && obj.geometry.attributes?.uv) {
        meshes.push(obj)
      }
    })
  }
  if (meshes.length === 0) return

  const textureConfig = texturableMesh.textureConfig
  const ctx = textureCanvas.getContext('2d')
  const origData = ctx.getImageData(0, 0, w, h)
  const outData = new Uint8ClampedArray(origData.data)

  const vA = new THREE.Vector3()
  const vB = new THREE.Vector3()
  const vC = new THREE.Vector3()
  const uvA = new THREE.Vector2()
  const uvB = new THREE.Vector2()
  const uvC = new THREE.Vector2()

  // ── PASS 1: build samples (3D position + texture colour) from covered triangles ──
  const samples = []  // flat: [x, y, z, r, g, b, ...]

  for (let mi = 0; mi < meshes.length; mi++) {
    const mesh = meshes[mi]
    mesh.updateWorldMatrix(true, false)
    const matrixWorld = mesh.matrixWorld
    const geom = mesh.geometry
    const posAttr = geom.attributes.position
    const uvAttr = geom.attributes.uv
    const indexAttr = geom.index
    const triCount = indexAttr ? indexAttr.count / 3 : posAttr.count / 3

    for (let t = 0; t < triCount; t++) {
      const base = t * 3
      const i0 = indexAttr ? indexAttr.getX(base) : base
      const i1 = indexAttr ? indexAttr.getX(base + 1) : base + 1
      const i2 = indexAttr ? indexAttr.getX(base + 2) : base + 2

      vA.fromBufferAttribute(posAttr, i0).applyMatrix4(matrixWorld)
      vB.fromBufferAttribute(posAttr, i1).applyMatrix4(matrixWorld)
      vC.fromBufferAttribute(posAttr, i2).applyMatrix4(matrixWorld)

      uvA.set(uvAttr.getX(i0), uvAttr.getY(i0))
      uvB.set(uvAttr.getX(i1), uvAttr.getY(i1))
      uvC.set(uvAttr.getX(i2), uvAttr.getY(i2))
      const pA = mapUvToCanvasPoint(uvA, w, h, textureConfig)
      const pB = mapUvToCanvasPoint(uvB, w, h, textureConfig)
      const pC = mapUvToCanvasPoint(uvC, w, h, textureConfig)

      const ucx = Math.floor((pA.x + pB.x + pC.x) / 3)
      const ucy = Math.floor((pA.y + pB.y + pC.y) / 3)
      if (ucx < 0 || ucx >= w || ucy < 0 || ucy >= h) continue

      const centroidIdx = ucy * w + ucx
      if (!anyCoverage[centroidIdx]) continue

      const j = centroidIdx * 4
      samples.push(
        (vA.x + vB.x + vC.x) / 3,
        (vA.y + vB.y + vC.y) / 3,
        (vA.z + vB.z + vC.z) / 3,
        origData.data[j],
        origData.data[j + 1],
        origData.data[j + 2]
      )
    }
  }

  const sampleCount = samples.length / 6
  if (sampleCount === 0) return

  // ── Build a spatial hash grid for fast 3D nearest-neighbour queries ──
  let minX = Infinity, maxX = -Infinity
  let minY = Infinity, maxY = -Infinity
  let minZ = Infinity, maxZ = -Infinity
  for (let i = 0; i < sampleCount; i++) {
    const x = samples[i * 6], y = samples[i * 6 + 1], z = samples[i * 6 + 2]
    if (x < minX) minX = x; if (x > maxX) maxX = x
    if (y < minY) minY = y; if (y > maxY) maxY = y
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
  }
  const bboxSize = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6)
  const cellsPerDim = Math.max(4, Math.min(48, Math.round(Math.cbrt(sampleCount) * 1.4)))
  const cellSize = bboxSize / cellsPerDim
  const invCellSize = 1 / cellSize

  // Pack (cx, cy, cz) into a single int key: assumes |c| < 1024
  const hashCell = (cx, cy, cz) => ((cx + 512) << 20) | ((cy + 512) << 10) | (cz + 512)

  const grid = new Map()
  for (let i = 0; i < sampleCount; i++) {
    const cx = Math.floor((samples[i * 6]     - minX) * invCellSize)
    const cy = Math.floor((samples[i * 6 + 1] - minY) * invCellSize)
    const cz = Math.floor((samples[i * 6 + 2] - minZ) * invCellSize)
    const key = hashCell(cx, cy, cz)
    let cell = grid.get(key)
    if (!cell) { cell = []; grid.set(key, cell) }
    cell.push(i)
  }

  const K = Math.max(1, Math.min(32, Math.round(smoothness)))

  // Returns [r, g, b] for a query 3D point
  function findFillColor(qx, qy, qz) {
    const baseCx = Math.floor((qx - minX) * invCellSize)
    const baseCy = Math.floor((qy - minY) * invCellSize)
    const baseCz = Math.floor((qz - minZ) * invCellSize)

    let candidates = []
    let radius = 0
    const maxRadius = cellsPerDim * 2 + 4

    while (candidates.length < K && radius <= maxRadius) {
      if (radius === 0) {
        const cell = grid.get(hashCell(baseCx, baseCy, baseCz))
        if (cell) candidates.push(...cell)
      } else {
        // Add shells of cells at exactly this radius
        for (let dz = -radius; dz <= radius; dz++) {
          for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
              if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== radius) continue
              const cell = grid.get(hashCell(baseCx + dx, baseCy + dy, baseCz + dz))
              if (cell) candidates.push(...cell)
            }
          }
        }
      }
      radius++
    }

    if (candidates.length === 0) {
      for (let i = 0; i < sampleCount; i++) candidates.push(i)
    }

    // Compute distances and pick K nearest
    const dArr = new Array(candidates.length)
    for (let c = 0; c < candidates.length; c++) {
      const sIdx = candidates[c]
      const dx = samples[sIdx * 6]     - qx
      const dy = samples[sIdx * 6 + 1] - qy
      const dz = samples[sIdx * 6 + 2] - qz
      dArr[c] = [sIdx, dx * dx + dy * dy + dz * dz]
    }
    dArr.sort((a, b) => a[1] - b[1])

    const numK = Math.min(K, dArr.length)
    let sumR = 0, sumG = 0, sumB = 0, sumW = 0
    for (let k = 0; k < numK; k++) {
      const [sIdx, d2] = dArr[k]
      const weight = 1 / (d2 + 1e-6)
      sumR += samples[sIdx * 6 + 3] * weight
      sumG += samples[sIdx * 6 + 4] * weight
      sumB += samples[sIdx * 6 + 5] * weight
      sumW += weight
    }
    return [sumR / sumW, sumG / sumW, sumB / sumW]
  }

  // Per-vertex colour cache (vertex world-pos → fill colour)
  const vertexColorCache = new Map()
  function getVertexFillColor(x, y, z) {
    const kx = Math.round(x * 1e4)
    const ky = Math.round(y * 1e4)
    const kz = Math.round(z * 1e4)
    const key = `${kx},${ky},${kz}`
    let c = vertexColorCache.get(key)
    if (!c) {
      c = findFillColor(x, y, z)
      vertexColorCache.set(key, c)
    }
    return c
  }

  // Total triangle count for progress
  let totalTris = 0
  for (const m of meshes) {
    const g = m.geometry
    totalTris += (g.index ? g.index.count : g.attributes.position.count) / 3
  }

  // ── PASS 2: rasterise every triangle, fill uncovered texels by barycentric vertex-colour interpolation ──
  let trisDone = 0
  let lastYield = performance.now()

  for (let mi = 0; mi < meshes.length; mi++) {
    const mesh = meshes[mi]
    const matrixWorld = mesh.matrixWorld
    const geom = mesh.geometry
    const posAttr = geom.attributes.position
    const uvAttr = geom.attributes.uv
    const indexAttr = geom.index
    const triCount = indexAttr ? indexAttr.count / 3 : posAttr.count / 3

    for (let t = 0; t < triCount; t++) {
      const base = t * 3
      const i0 = indexAttr ? indexAttr.getX(base) : base
      const i1 = indexAttr ? indexAttr.getX(base + 1) : base + 1
      const i2 = indexAttr ? indexAttr.getX(base + 2) : base + 2

      vA.fromBufferAttribute(posAttr, i0).applyMatrix4(matrixWorld)
      vB.fromBufferAttribute(posAttr, i1).applyMatrix4(matrixWorld)
      vC.fromBufferAttribute(posAttr, i2).applyMatrix4(matrixWorld)

      uvA.set(uvAttr.getX(i0), uvAttr.getY(i0))
      uvB.set(uvAttr.getX(i1), uvAttr.getY(i1))
      uvC.set(uvAttr.getX(i2), uvAttr.getY(i2))
      const pA = mapUvToCanvasPoint(uvA, w, h, textureConfig)
      const pB = mapUvToCanvasPoint(uvB, w, h, textureConfig)
      const pC = mapUvToCanvasPoint(uvC, w, h, textureConfig)

      const x0 = pA.x, y0 = pA.y
      const x1 = pB.x, y1 = pB.y
      const x2 = pC.x, y2 = pC.y

      const denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
      if (Math.abs(denom) < 1e-10) { trisDone++; continue }

      const minPx = Math.max(0, Math.floor(Math.min(x0, x1, x2)))
      const maxPx = Math.min(w - 1, Math.ceil(Math.max(x0, x1, x2)))
      const minPy = Math.max(0, Math.floor(Math.min(y0, y1, y2)))
      const maxPy = Math.min(h - 1, Math.ceil(Math.max(y0, y1, y2)))

      // Fast skip if every pixel in the bbox is already covered
      let hasUncovered = false
      for (let py = minPy; py <= maxPy && !hasUncovered; py++) {
        const row = py * w
        for (let px = minPx; px <= maxPx; px++) {
          if (!anyCoverage[row + px]) { hasUncovered = true; break }
        }
      }
      if (!hasUncovered) { trisDone++; continue }

      const cA = getVertexFillColor(vA.x, vA.y, vA.z)
      const cB = getVertexFillColor(vB.x, vB.y, vB.z)
      const cC = getVertexFillColor(vC.x, vC.y, vC.z)

      const invDenom = 1 / denom
      const baryEps = -1e-3

      for (let py = minPy; py <= maxPy; py++) {
        const row = py * w
        for (let px = minPx; px <= maxPx; px++) {
          const pixelIdx = row + px
          if (anyCoverage[pixelIdx]) continue

          const fx = px + 0.5
          const fy = py + 0.5
          const wa = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) * invDenom
          const wb = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) * invDenom
          const wc = 1 - wa - wb
          if (wa < baryEps || wb < baryEps || wc < baryEps) continue

          const j = pixelIdx * 4
          outData[j]     = Math.round(wa * cA[0] + wb * cB[0] + wc * cC[0])
          outData[j + 1] = Math.round(wa * cA[1] + wb * cB[1] + wc * cC[1])
          outData[j + 2] = Math.round(wa * cA[2] + wb * cB[2] + wc * cC[2])
          outData[j + 3] = 255
        }
      }

      trisDone++

      if ((trisDone & 0xFF) === 0) {
        const now = performance.now()
        if (now - lastYield > 30) {
          if (onProgress) onProgress(trisDone / totalTris)
          await new Promise(r => setTimeout(r, 0))
          lastYield = now
        }
      }
    }
  }

  // ── PASS 3: UV-gutter edge-padding ──
  // Texels just outside an island border are inside no triangle, so PASS 2
  // leaves them transparent. Bilinear texture sampling then reaches past the
  // island edge and pulls those transparent texels in as white seam lines.
  // Pad covered/filled colour a few texels into the transparent gutter so the
  // bilinear tap stays on real colour (the same edge-pad the main projection
  // composite applies; the hole-fill output lacked it).
  const PAD_PIXELS = 4
  const alphaMask = new Uint8Array(pixelCount)
  for (let i = 0; i < pixelCount; i++) alphaMask[i] = outData[i * 4 + 3] > 0 ? 1 : 0

  for (let pass = 0; pass < PAD_PIXELS; pass++) {
    const ring = []
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const i = py * w + px
        if (alphaMask[i]) continue
        // Average the colour of opaque 8-neighbours committed in a prior ring
        // (alphaMask only updates after the full pass → clean ring growth, no
        // intra-pass cascade bias).
        let sr = 0, sg = 0, sb = 0, n = 0
        for (let dy = -1; dy <= 1; dy++) {
          const ny = py + dy
          if (ny < 0 || ny >= h) continue
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue
            const nx = px + dx
            if (nx < 0 || nx >= w) continue
            const ni = ny * w + nx
            if (!alphaMask[ni]) continue
            const nj = ni * 4
            sr += outData[nj]; sg += outData[nj + 1]; sb += outData[nj + 2]; n++
          }
        }
        if (n > 0) {
          const j = i * 4
          outData[j]     = Math.round(sr / n)
          outData[j + 1] = Math.round(sg / n)
          outData[j + 2] = Math.round(sb / n)
          outData[j + 3] = 255
          ring.push(i)
        }
      }
    }
    if (ring.length === 0) break
    for (let k = 0; k < ring.length; k++) alphaMask[ring[k]] = 1
  }

  if (onProgress) onProgress(1)
  ctx.putImageData(new ImageData(outData, w, h), 0, 0)
}
