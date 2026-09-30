import { memo } from 'react'
import AutoRigParameterFields from '../AutoRigParameterFields'
import GraphMeshToolNode from './GraphMeshToolNode'

// Rig Mesh node: takes a single connected mesh, runs it through the SkinTokens /
// TokenRig rigging service (the same Auto Rig the Mesh Editor uses) and saves the
// rigged GLB as a new VERSION of the connected mesh. The node shell is shared
// with the other single-mesh tools (GraphMeshToolNode).
const RIG_TOOL = {
  className: 'graph-node--rigMesh',
  defaultName: 'RIG MESH',
  badge: 'RIG',
  icon: 'accessibility_new',
  inputTitle: 'Mesh to rig',
  actionName: 'Auto Rig',
  panelTitle: 'AUTO RIG',
  runLabel: 'RUN AUTO RIG',
  runningLabel: 'RIGGING…',
  runningMeta: 'Rigging…',
  emptyMeta: 'Connect a mesh input, then run Auto Rig.',
  outputPending: 'after rigging',
  lockNote: 'Parameters locked while rigging',
  versionSuffix: 'rigged',
  namePlaceholder: 'Rigged mesh name',
  describeRun: sourceAsset => `Rigs ${sourceAsset.name} and saves the result as a new version of it`,
  serviceHint: 'Auto Rig runs on the SkinTokens rigging service (Settings → Rigging). Needs an NVIDIA GPU.',
  renderFields: ({ draft, setField, disabled }) => (
    <AutoRigParameterFields
      options={draft}
      onChange={setField}
      disabled={disabled}
      controlClassName="nodrag"
    />
  )
}

const GraphRigMeshNode = memo(function GraphRigMeshNode({ data }) {
  return <GraphMeshToolNode data={data} tool={RIG_TOOL} />
})

export default GraphRigMeshNode
