import { memo, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Handle, Position, useUpdateNodeInternals } from '@xyflow/react'
import {
  DEFAULT_INPUT_ID,
  DEFAULT_OUTPUT_ID,
  buildMeshEditorPath,
  getAssetPreviewUrl,
  getConnectorPosition,
  getConnectorTypeMeta
} from '../../utils/graphHelpers'
import LastActionInfo from './LastActionInfo'

// The shell shared by the graph's single-mesh tool nodes (Rig Mesh, Flatten to
// Albedo): one mesh input, a tool that runs on it, and the result saved as a new
// VERSION of the connected mesh. The output connector only appears once a
// result has been saved, so downstream nodes consume the processed mesh.
//
// Presentational: the tool's options live in the node's parameter draft and the
// run itself is driven by GraphPage (data.onRunNodeAction). `tool` describes
// what differs between nodes — labels, icon, and the option fields, rendered by
// `renderFields({ draft, setField, disabled })`.
const GraphMeshToolNode = memo(function GraphMeshToolNode({ data, tool }) {
  const navigate = useNavigate()
  const updateNodeInternals = useUpdateNodeInternals()
  const draft = data.actionDraft
  const isProcessing = data.status === 'processing'
  const isDraftCollapsed = Boolean(data.isDraftCollapsed)
  // The panel stays mounted during a run (locked, not unmounted) so the options
  // the user typed are still there afterwards.
  const fieldsDisabled = isProcessing
  const inputConnectors = useMemo(() => (
    data.inputConnectors || [{ id: DEFAULT_INPUT_ID, type: 'mesh', isConnected: false }]
  ), [data.inputConnectors])
  const inputSource = (data.inputSources || []).find(source => source.type === 'mesh') || null
  const sourceAsset = inputSource?.asset || null
  const outputConnector = data.outputConnector || { id: DEFAULT_OUTPUT_ID, type: 'mesh' }
  const outputMeta = getConnectorTypeMeta('mesh')
  const hasOutputAsset = Boolean(data.asset?.id)
  const previewUrl = getAssetPreviewUrl(data.asset?.thumbnail || null)
  const nodeDisplayName = data.name || data.asset?.name || tool.defaultName
  const errorMessage = data.metadata?.error || ''
  const meshEditorPath = data.asset?.id
    ? buildMeshEditorPath({
        asset: data.asset,
        projectId: data.projectId,
        nodeId: data.id,
        returnTo: `/projects/${data.projectId}`
      })
    : ''
  const metaLabel = isProcessing
    ? (Number.isFinite(data.progress) ? `${data.progress}%` : tool.runningMeta)
    : sourceAsset
      ? `Input mesh · ${sourceAsset.name || inputSource?.label || 'Mesh'}`
      : tool.emptyMeta

  useEffect(() => {
    updateNodeInternals(String(data.id))
  }, [data.id, inputConnectors, hasOutputAsset, outputConnector.id, updateNodeInternals])

  const setField = (field, value) => data.onDraftFieldChange?.(data.id, field, value)

  const openMeshPreview = () => {
    if (data.asset?.filename) {
      data.onOpenMeshPreview?.(data.asset)
    }
  }

  return (
    <div className={`graph-node ${tool.className}`}>
      {inputConnectors.map((connector, index) => {
        const connectorMeta = getConnectorTypeMeta(connector.type || 'mesh')

        return (
          <div
            key={connector.id}
            className="graph-node__connector graph-node__connector--input"
            style={getConnectorPosition(index, inputConnectors.length)}
          >
            <Handle
              type="target"
              id={connector.id}
              position={Position.Left}
              className="graph-node__handle graph-node__handle--input"
              style={{ borderColor: connectorMeta.color }}
            />
            <span
              className="graph-node__connector-badge font-label"
              style={{
                color: connectorMeta.color,
                background: connectorMeta.background,
                borderColor: connectorMeta.color
              }}
              title={tool.inputTitle}
            >
              {connectorMeta.letter}
            </span>
          </div>
        )
      })}

      <div className={`graph-node__card image-card ${isProcessing ? 'image-card--loading image-card--locked' : ''}`}>
        <div className="image-card__actions">
          <button
            type="button"
            className="image-card__action-btn image-card__delete nodrag"
            onClick={() => data.onDelete?.(data.id)}
            title="Delete node"
          >
            <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>delete</span>
          </button>
        </div>

        <div className="image-card__thumb graph-node__thumb">
          {hasOutputAsset ? (
            <div className="image-card__thumb-item nodrag" style={{ position: 'relative', cursor: 'pointer' }} onClick={openMeshPreview}>
              {previewUrl ? (
                <img src={previewUrl} alt={data.asset?.name || nodeDisplayName} className="image-card__thumb-image" />
              ) : (
                <div className="image-card__thumb-placeholder">
                  <span className="material-symbols-outlined" style={{ fontSize: '32px', color: 'rgba(172,137,255,0.5)' }}>{tool.icon}</span>
                </div>
              )}
              <button
                type="button"
                className="image-card__edit-action-btn nodrag"
                style={{ position: 'absolute', bottom: '8px', right: '8px' }}
                onClick={event => { event.stopPropagation(); openMeshPreview() }}
                title="Open 3D preview"
              >
                <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>play_arrow</span>
                3D
              </button>
            </div>
          ) : (
            <div className="image-card__thumb-placeholder">
              <span className="material-symbols-outlined" style={{ fontSize: '32px', color: 'rgba(172,137,255,0.2)' }}>{tool.icon}</span>
            </div>
          )}

          <div className="image-card__edit-preview-indicator font-label">
            {sourceAsset ? `INPUT • ${sourceAsset.name}` : 'INPUT • MESH'}
          </div>
        </div>

        <div className="image-card__info">
          <div className="image-card__row">
            <input
              type="text"
              className="image-card__name graph-node__name-input nodrag"
              value={nodeDisplayName}
              placeholder={tool.defaultName}
              onChange={event => data.onNodeNameChange?.(data.id, event.target.value)}
              onBlur={event => data.onNodeNameCommit?.(data.id, event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  event.currentTarget.blur()
                }
              }}
            />
            <div className="image-card__badges">
              {data.metadata?.lastActionParams && (
                <LastActionInfo lastActionParams={data.metadata.lastActionParams} />
              )}
              <span
                className="image-card__source"
                style={{
                  color: outputMeta.color,
                  background: outputMeta.background
                }}
              >
                {tool.badge}
              </span>
            </div>
          </div>

          <p className="image-card__meta font-label">{metaLabel}</p>

          {isProcessing && data.progressDetail && (
            <p className="image-card__meta font-label">{data.progressDetail}</p>
          )}

          {isProcessing && Number.isFinite(data.progress) && (
            <div className="image-card__progress graph-node__progress" aria-hidden="true">
              <div
                className="image-card__progress-bar"
                style={{ width: `${Math.max(0, Math.min(100, data.progress || 0))}%` }}
              />
            </div>
          )}

          {!isProcessing && errorMessage && (
            <p className="image-card__meta font-label graph-node__error">{errorMessage}</p>
          )}

          {!isProcessing && !errorMessage && data.metadata?.detail && (
            <p className="image-card__meta font-label">{data.metadata.detail}</p>
          )}

          <div className="graph-node__ports-summary font-label">
            <span className="graph-node__port-label">Input · {sourceAsset ? 'Mesh connected' : 'empty'}</span>
            <span className="graph-node__port-label graph-node__port-label--output">
              Output · {hasOutputAsset ? outputMeta.label : tool.outputPending}
            </span>
          </div>

          <div className="image-card__attributes graph-node__actions-panel">
            <div className="image-card__edit-actions">
              <div className="graph-node__primary-actions">
                <button
                  className="image-card__edit-action-btn nodrag"
                  onClick={() => data.onToggleDraftCollapsed?.(data.id)}
                  title={isDraftCollapsed ? 'Show the parameters' : 'Hide the parameters'}
                  aria-expanded={!isDraftCollapsed}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                    {isDraftCollapsed ? 'expand_more' : 'expand_less'}
                  </span>
                  {isDraftCollapsed ? 'Params' : 'Hide'}
                </button>
                {meshEditorPath && (
                  <button className="image-card__edit-action-btn graph-node__edit-action nodrag" onClick={() => navigate(meshEditorPath)}>
                    <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>edit_square</span>
                    Edit
                  </button>
                )}
              </div>

              {isProcessing && (
                <div className="graph-node__panel-lock-note font-label">
                  <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>lock</span>
                  {tool.lockNote}
                </div>
              )}

              {!isDraftCollapsed && draft && (
                <div className={`image-card__edit-panel nodrag${isProcessing ? ' image-card__edit-panel--locked' : ''}`}>
                  <span className="graph-node__panel-title font-label">{tool.panelTitle}</span>

                  <div className="params-card__field">
                    <label className="params-card__label font-label">Name</label>
                    <input
                      type="text"
                      className="params-card__input nodrag"
                      placeholder={sourceAsset ? `${sourceAsset.name} (${tool.versionSuffix})` : tool.namePlaceholder}
                      value={draft.name || ''}
                      disabled={fieldsDisabled}
                      onChange={event => setField('name', event.target.value)}
                    />
                    <span className="image-card__param-hint">Name of the version saved on the input mesh</span>
                  </div>

                  {tool.renderFields({ draft, setField, disabled: fieldsDisabled })}

                  <div className="graph-node__linked-input font-label">
                    {sourceAsset
                      ? tool.describeRun(sourceAsset)
                      : `Connect a mesh output to this node to enable ${tool.actionName}`}
                  </div>

                  <button
                    className="gen-btn nodrag"
                    onClick={() => data.onRunNodeAction?.(data.id)}
                    disabled={!sourceAsset || isProcessing}
                  >
                    <span className="material-symbols-outlined">{isProcessing ? 'progress_activity' : tool.icon}</span>
                    {isProcessing ? tool.runningLabel : tool.runLabel}
                  </button>

                  <span className="image-card__param-hint">{tool.serviceHint}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {hasOutputAsset && (
        <div className="graph-node__connector graph-node__connector--output" style={getConnectorPosition(0, 1)}>
          <span
            className="graph-node__connector-badge font-label"
            style={{
              color: outputMeta.color,
              background: outputMeta.background,
              borderColor: outputMeta.color
            }}
            title={outputMeta.label}
          >
            {outputMeta.letter}
          </span>
          <Handle
            type="source"
            id={outputConnector.id}
            position={Position.Right}
            className="graph-node__handle graph-node__handle--output"
            style={{ borderColor: outputMeta.color }}
          />
        </div>
      )}
    </div>
  )
})

export default GraphMeshToolNode
