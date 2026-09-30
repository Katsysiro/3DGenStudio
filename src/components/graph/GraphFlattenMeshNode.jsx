import { memo } from 'react'
import FlattenParameterFields from '../FlattenParameterFields'
import GraphMeshToolNode from './GraphMeshToolNode'

// Flatten to Albedo node: takes a single connected mesh, bakes every material's
// whole PBR look into ONE lit albedo texture (the Export dialog's "Flatten to one
// lit albedo (mobile)", via src/utils/meshFlatten.js) and saves the result as a
// new VERSION of the connected mesh. The node shell is shared with the other
// single-mesh tools (GraphMeshToolNode).
const FLATTEN_TOOL = {
  className: 'graph-node--flattenMesh',
  defaultName: 'FLATTEN TO ALBEDO',
  badge: 'FLAT',
  icon: 'texture',
  inputTitle: 'Mesh to flatten',
  actionName: 'Flatten',
  panelTitle: 'FLATTEN TO ALBEDO',
  runLabel: 'RUN FLATTEN',
  runningLabel: 'FLATTENING…',
  runningMeta: 'Flattening…',
  emptyMeta: 'Connect a mesh input, then run Flatten.',
  outputPending: 'after flattening',
  lockNote: 'Parameters locked while flattening',
  versionSuffix: 'flattened',
  namePlaceholder: 'Flattened mesh name',
  describeRun: sourceAsset => `Flattens ${sourceAsset.name} into one albedo and saves the result as a new version of it`,
  serviceHint: 'Bakes normal detail, occlusion, roughness and metal into one colour texture under a neutral studio light, with a lit Cycles bake on the Mesh Tools service. The UV islands are repacked rather than re-unwrapped, so a rig and its animation clips come through unchanged.',
  renderFields: ({ draft, setField, disabled }) => (
    <FlattenParameterFields
      options={draft}
      onChange={setField}
      disabled={disabled}
      controlClassName="nodrag"
    />
  )
}

const GraphFlattenMeshNode = memo(function GraphFlattenMeshNode({ data }) {
  return <GraphMeshToolNode data={data} tool={FLATTEN_TOOL} />
})

export default GraphFlattenMeshNode
