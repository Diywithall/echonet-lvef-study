import React, { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, FileText, ChevronRight, ChevronLeft,
  CheckCircle2, AlertCircle, Loader2, X, ZoomIn, Heart
} from 'lucide-react'
import './App.css'

const API_BASE = 'http://localhost:8000'

/* ─── ECG line — single PQRST complex, purely decorative ─── */
const ECGWave = ({ className }) => (
  <svg className={className} viewBox="0 0 500 52" preserveAspectRatio="none" fill="none">
    <g>
      <animateTransform 
        attributeName="transform" 
        type="translate" 
        from="0 0" 
        to="-500 0" 
        dur="2.8s" 
        repeatCount="indefinite"
      />
      <path
        d="M0,26 H70 Q76,23 82,26 H90 L94,29 98,4 102,48 106,26 H112 Q120,19 128,26 H500 M500,26 H570 Q576,23 582,26 H590 L594,29 598,4 602,48 606,26 H612 Q620,19 628,26 H1000"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </g>
  </svg>
)

/* ─── Extract frames from video via Canvas ─── */
const extractVideoFrames = (file, numFrames = 8) =>
  new Promise(resolve => {
    const video = document.createElement('video')
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const url = URL.createObjectURL(file)
    const frames = []
    let idx = 0
    let times = []

    video.preload = 'auto'
    video.muted = true
    video.playsInline = true
    video.crossOrigin = 'anonymous'
    video.src = url

    const seekNext = () => {
      if (idx >= times.length) {
        URL.revokeObjectURL(url)
        resolve(frames)
        return
      }
      video.currentTime = times[idx]
    }

    video.onseeked = () => {
      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        frames.push(canvas.toDataURL('image/jpeg', 0.85))
      } catch (e) { /* tainted canvas, skip */ }
      idx++
      seekNext()
    }

    video.onloadedmetadata = () => {
      canvas.width = Math.min(video.videoWidth || 320, 320)
      canvas.height = Math.round(canvas.width * (video.videoHeight || 240) / (video.videoWidth || 320))
      const dur = video.duration
      if (!dur || !isFinite(dur)) { URL.revokeObjectURL(url); return resolve([]) }
      // Start at 0.01s so seeking is always non-trivial and onseeked fires reliably
      times = Array.from({ length: numFrames }, (_, i) =>
        0.01 + (i / (numFrames - 1)) * (dur * 0.96 - 0.01)
      )
      seekNext()
    }

    video.onerror = () => { URL.revokeObjectURL(url); resolve([]) }
    setTimeout(() => { URL.revokeObjectURL(url); resolve(frames) }, 30_000)
  })

/* ─── Extract contour polygon points string ─── */
const getContourPoints = (result, i, total) => {
  if (!result?.contours?.length || !total) return null
  const cIdx = Math.min(Math.floor((i / total) * result.contours.length), result.contours.length - 1)
  const pts = result.contours[cIdx]
  if (!pts || pts.length < 3) return null
  return pts.map(p => `${p[0]},${p[1]}`).join(' ')
}

/* ─── Beat-by-Beat Cardiac Rhythm Chart Component ─── */
const BeatAnalysisCard = ({ result }) => {
  if (!result || !result.beat_ef_values || result.beat_ef_values.length === 0) return null

  const beats = result.beat_ef_values
  const outliers = new Set(result.outlier_beats || [])
  const meanEf = result.mean_ef ?? (beats.reduce((a, b) => a + b, 0) / beats.length).toFixed(1)
  const stdEf = result.ef_std ?? 0.0
  const consistency = result.consistency || 'HIGH'

  // SVG Chart Geometry
  const svgWidth = 480
  const svgHeight = 160
  const padL = 42
  const padR = 25
  const padT = 25
  const padB = 35

  const chartW = svgWidth - padL - padR
  const chartH = svgHeight - padT - padB

  const minVal = Math.max(0, Math.floor(Math.min(...beats, meanEf) - 8))
  const maxVal = Math.min(100, Math.ceil(Math.max(...beats, meanEf) + 8))
  const yRange = Math.max(12, maxVal - minVal)

  const getX = idx => {
    if (beats.length === 1) return padL + chartW / 2
    return padL + (idx / (beats.length - 1)) * chartW
  }

  const getY = val => padT + chartH - ((val - minVal) / yRange) * chartH

  const meanY = getY(meanEf)
  const polylinePoints = beats.map((v, i) => `${getX(i)},${getY(v)}`).join(' ')

  const getConsistencyInfo = c => {
    if (c === 'HIGH') {
      return {
        label: 'HIGH CONSISTENCY',
        color: 'var(--green)',
        bg: 'rgba(16, 185, 129, 0.12)',
        desc: 'Low variability across detected beats — consistent rhythm.'
      }
    }
    if (c === 'MODERATE') {
      return {
        label: 'MODERATE CONSISTENCY',
        color: 'var(--amber)',
        bg: 'rgba(245, 158, 11, 0.12)',
        desc: 'Moderate beat-to-beat LVEF variability detected.'
      }
    }
    return {
      label: 'LOW CONSISTENCY',
      color: 'var(--brand)',
      bg: 'rgba(244, 63, 94, 0.15)',
      desc: 'High variability across detected beats (possible irregular rhythm).'
    }
  }

  const info = getConsistencyInfo(consistency)

  return (
    <div className="card beat-card" style={{ marginTop: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
        <div>
          <p className="lbl" style={{ marginBottom: '0.2rem' }}>Beat-by-Beat Cardiac Rhythm & LVEF</p>
          <p style={{ fontSize: '0.78rem', color: 'var(--t2)', margin: 0 }}>
            Per-cycle ejection fraction modeling across {beats.length} detected beat{beats.length > 1 ? 's' : ''}
          </p>
        </div>
        <span style={{
          fontSize: '0.68rem',
          fontWeight: 700,
          color: info.color,
          backgroundColor: info.bg,
          padding: '0.25rem 0.65rem',
          borderRadius: '9999px',
          border: `1px solid ${info.color}44`,
          letterSpacing: '0.04em'
        }}>
          {info.label}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 210px', gap: '1.25rem', alignItems: 'center' }}>
        {/* SVG Chart */}
        <div style={{
          position: 'relative',
          width: '100%',
          background: 'var(--s1)',
          borderRadius: 'var(--r)',
          padding: '0.5rem',
          border: '1px solid var(--b0)'
        }}>
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            {/* Horizontal Grid lines */}
            {[minVal, Math.round((minVal + maxVal) / 2), maxVal].map((gVal, i) => {
              const gy = getY(gVal)
              return (
                <g key={i}>
                  <line x1={padL} y1={gy} x2={svgWidth - padR} y2={gy} stroke="var(--b1)" strokeDasharray="3 3" />
                  <text x={padL - 6} y={gy + 3} fill="var(--t3)" fontSize="9" textAnchor="end">{gVal}%</text>
                </g>
              )
            })}

            {/* Mean EF Line */}
            <line x1={padL} y1={meanY} x2={svgWidth - padR} y2={meanY} stroke="var(--brand)" strokeWidth="1.2" strokeDasharray="4 4" opacity="0.75" />
            <text x={svgWidth - padR + 3} y={meanY + 3} fill="var(--brand)" fontSize="9" fontWeight="600">Mean</text>

            {/* Connecting polyline */}
            {beats.length > 1 && (
              <polyline points={polylinePoints} fill="none" stroke="var(--t1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.85" />
            )}

            {/* Beat nodes */}
            {beats.map((val, idx) => {
              const cx = getX(idx)
              const cy = getY(val)
              const isOutlier = outliers.has(idx)

              return (
                <g key={idx}>
                  {/* Outlier warning glow */}
                  {isOutlier && (
                    <circle cx={cx} cy={cy} r="10" fill="rgba(245, 158, 11, 0.2)" stroke="var(--amber)" strokeWidth="1.2">
                      <animate attributeName="r" values="8;12;8" dur="1.8s" repeatCount="indefinite" />
                    </circle>
                  )}

                  {/* Beat node circle */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isOutlier ? "5.5" : "4.5"}
                    fill={isOutlier ? "var(--amber)" : "var(--brand)"}
                    stroke="#ffffff"
                    strokeWidth="1.5"
                  />

                  {/* EF Value Label */}
                  <text x={cx} y={cy - 9} fill={isOutlier ? "var(--amber)" : "var(--t1)"} fontSize="10" fontWeight="700" textAnchor="middle">
                    {val}%
                  </text>

                  {/* Beat Number X Label */}
                  <text x={cx} y={svgHeight - 8} fill="var(--t3)" fontSize="9" fontWeight="600" textAnchor="middle">
                    Beat {idx + 1}
                  </text>

                  {/* Outlier warning label */}
                  {isOutlier && (
                    <text x={cx} y={cy + 16} fill="var(--amber)" fontSize="9" fontWeight="bold" textAnchor="middle">
                      ⚠ Outlier
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
        </div>

        {/* Summary Panel */}
        <div style={{
          background: 'var(--s1)',
          padding: '0.85rem 1rem',
          borderRadius: 'var(--r)',
          border: '1px solid var(--b0)'
        }}>
          <p style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--t3)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.65rem' }}>
            Rhythm Summary
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <div>
              <span style={{ fontSize: '0.68rem', color: 'var(--t3)', display: 'block' }}>Mean EF</span>
              <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--t1)' }}>{meanEf}%</span>
            </div>
            <div>
              <span style={{ fontSize: '0.68rem', color: 'var(--t3)', display: 'block' }}>Std Dev</span>
              <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--t1)' }}>±{stdEf}%</span>
            </div>
          </div>

          <div style={{
            fontSize: '0.76rem',
            color: 'var(--t2)',
            lineHeight: '1.35',
            background: 'var(--s2)',
            padding: '0.55rem 0.65rem',
            borderRadius: '6px',
            borderLeft: `3px solid ${info.color}`
          }}>
            {info.desc}
          </div>
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════ */
export default function App() {
  const [uploads, setUploads] = useState([])
  const [activeTaskId, setActiveTaskId] = useState(null)
  const [result, setResult] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [frames, setFrames] = useState([])
  const [dragging, setDragging] = useState(false)
  const [lightbox, setLightbox] = useState(null)

  const fileRef = useRef(null)
  const dragCount = useRef(0)

  /* ── Keyboard Nav ── */
  useEffect(() => {
    if (lightbox === null || frames.length === 0) return
    const onKey = e => {
      if (e.key === 'Escape') setLightbox(null)
      if (e.key === 'ArrowLeft') setLightbox(p => Math.max(0, p - 1))
      if (e.key === 'ArrowRight') setLightbox(p => Math.min(frames.length - 1, p + 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightbox, frames.length])

  /* ── Poll backend ── */
  useEffect(() => {
    if (!activeTaskId) return
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`${API_BASE}/results/${activeTaskId}`).then(r => r.json())
        if (d.status === 'completed') {
          setResult(d.result)
          setActiveTaskId(null)
          setUploads(p => p.map(u => u.id === activeTaskId ? { ...u, s: 'ok' } : u))
          clearInterval(iv)
        } else if (d.status === 'failed') {
          alert('Analysis failed: ' + d.error)
          setActiveTaskId(null)
          setUploads(p => p.map(u => u.id === activeTaskId ? { ...u, s: 'err' } : u))
          clearInterval(iv)
        }
      } catch { /* keep polling */ }
    }, 2000)
    return () => clearInterval(iv)
  }, [activeTaskId])

  /* ── Upload ── */
  const handleFiles = useCallback(async (files) => {
    if (!files?.length) return
    setIsUploading(true)
    try {
      for (const file of Array.from(files)) {
        if (!file.type.startsWith('video/') && !/\.(mp4|avi)$/i.test(file.name)) continue
        setFrames([])
        const framesP = extractVideoFrames(file, 8)
        const form = new FormData()
        form.append('file', file)
        const up = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
        if (!up.ok) throw new Error()
        const { file_id } = await up.json()
        const an = await fetch(`${API_BASE}/analyze?file_id=${file_id}`, { method: 'POST' })
        const { task_id } = await an.json()
        setUploads(p => [{ id: task_id, name: file.name, s: 'proc' }, ...p])
        setActiveTaskId(task_id)
        setResult(null)
        setFrames(await framesP)
      }
    } catch {
      alert('Could not reach the backend server on port 8001.')
    } finally {
      setIsUploading(false)
    }
  }, [])

  /* ── Global drag-and-drop ── */
  useEffect(() => {
    const over = e => e.preventDefault()
    const enter = e => {
      e.preventDefault()
      if (e.dataTransfer?.types?.includes('Files')) { dragCount.current++; setDragging(true) }
    }
    const leave = e => {
      e.preventDefault()
      if (--dragCount.current <= 0) { dragCount.current = 0; setDragging(false) }
    }
    const drop = e => {
      e.preventDefault(); dragCount.current = 0; setDragging(false)
      if (e.dataTransfer?.files?.length) handleFiles(e.dataTransfer.files)
    }
    document.addEventListener('dragover', over)
    document.addEventListener('dragenter', enter)
    document.addEventListener('dragleave', leave)
    document.addEventListener('drop', drop)
    return () => {
      document.removeEventListener('dragover', over)
      document.removeEventListener('dragenter', enter)
      document.removeEventListener('dragleave', leave)
      document.removeEventListener('drop', drop)
    }
  }, [handleFiles])

  /* ── Lightbox keyboard nav ── */
  useEffect(() => {
    const fn = e => {
      if (lightbox === null) return
      if (e.key === 'Escape') setLightbox(null)
      if (e.key === 'ArrowRight') setLightbox(i => Math.min(i + 1, frames.length - 1))
      if (e.key === 'ArrowLeft') setLightbox(i => Math.max(i - 1, 0))
    }
    window.addEventListener('keydown', fn)
    return () => window.removeEventListener('keydown', fn)
  }, [lightbox, frames.length])

  const statusClass = d => {
    if (!d) return ''
    return d.includes('Normal') ? 'ok' : d.includes('Mild') ? 'mild' : 'sev'
  }

  const EF_STROKE = ef => ef >= 55 ? '#4ade80' : ef >= 40 ? '#fbbf24' : '#f87171'
  const C = 2 * Math.PI * 40   // r = 40

  return (
    <div className="root">
      {/* ── Complex Background Animations ── */}
      <div className="bg-aurora"></div>
      <div className="bg-particles"></div>
      <div className="bg-radar"></div>
      <div className="vertical-grids"></div>
      <div className="tech-ring ring-1"></div>
      <div className="tech-ring ring-2"></div>
      <div className="tech-ring ring-3"></div>

      {/* ── Drop overlay ── */}
      <AnimatePresence>
        {dragging && (
          <motion.div className="drop-mask"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="drop-mask-body">
              <ECGWave className="drop-ecg" />
              <p>Release to begin analysis</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ══ Page ══ */}
      <div className="page">

        {/* Header */}
        <header className="hdr">
          <div className="brand">
            <Heart className="brand-logo" size={26} />
            <span className="brand-name">POCUS</span>
            <span className="brand-rule" />
            <div className="brand-meta">
              <span className="brand-full">Point of Care Cardiac Ultrasound</span>
            </div>
          </div>
          <div className="hdr-right">
            <ECGWave className="hdr-ecg" />
            <span className="status-chip">
              <span className="status-dot" />
              System Active
            </span>
          </div>
        </header>

        {/* Body */}
        <div className="layout">

          {/* Sidebar */}
          <aside className="sidebar">
            {/* Upload */}
            <div className="card">
              <p className="lbl">New Scan</p>
              <div
                className={`dz${isUploading ? ' dz-busy' : ''}`}
                onClick={() => !isUploading && fileRef.current?.click()}
              >
                <span className="dz-ico">
                  {isUploading ? <Loader2 size={20} className="spin" /> : <Upload size={20} />}
                </span>
                <p className="dz-main">{isUploading ? 'Uploading…' : 'Upload Echocardiogram'}</p>
                <p className="dz-sub">MP4 · AVI · or drag anywhere</p>
                <input ref={fileRef} type="file" multiple hidden accept=".mp4,.avi"
                  onChange={e => handleFiles(e.target.files)} />
              </div>
            </div>

            {/* History */}
            <div className="card">
              <p className="lbl">Scan History</p>
              <div className="hist-list">
                {uploads.length === 0
                  ? <p className="hist-empty">No scans yet.</p>
                  : uploads.map(u => (
                    <div key={u.id} className={`hist-row${u.id === activeTaskId ? ' hist-row-active' : ''}`}>
                      <span className="hist-ico">
                        {u.id === activeTaskId
                          ? <Loader2 size={12} className="spin" />
                          : u.s === 'err'
                            ? <AlertCircle size={12} className="col-red" />
                            : <CheckCircle2 size={12} className="col-green" />}
                      </span>
                      <span className="hist-info">
                        <span className="hist-name">{u.name}</span>
                        <span className="hist-id">#{u.id.slice(0, 8)}</span>
                      </span>
                      <ChevronRight size={11} className="hist-arr" />
                    </div>
                  ))
                }
              </div>
            </div>
          </aside>

          {/* Main */}
          <main className="main">
            <AnimatePresence mode="wait">

              {/* Processing */}
              {activeTaskId && (
                <motion.div key="proc" className="card proc-card"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <ECGWave className="proc-wave" />
                  <p className="lbl" style={{ marginTop: '1.5rem' }}>Analysing</p>
                  <h2 className="proc-title">Running EchoNet Inference</h2>
                  <p className="proc-sub">
                    Measuring left ventricular wall motion and ejection fraction…
                  </p>
                  {frames.length > 0 && (
                    <div className="proc-thumbs">
                      {frames.map((f, i) => <img key={i} src={f} alt="" className="proc-th" />)}
                    </div>
                  )}
                </motion.div>
              )}

              {/* Results */}
              {!activeTaskId && result && (
                <motion.div key="res" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <div className="res-stack">

                    {/* Diagnosis + EF */}
                    <div className="card diag-card">
                      <div className="diag-row">
                        <div className="diag-left">
                          <p className="lbl">LVEF Assessment</p>
                          <h2 className="diag-title">{result.diagnosis}</h2>
                          <span className={`verdict verdict-${statusClass(result.diagnosis)}`}>
                            {result.diagnosis?.includes('Normal')
                              ? <CheckCircle2 size={11} /> : <AlertCircle size={11} />}
                            Automated Verdict
                          </span>
                          <p className="diag-body">{result.explanation}</p>
                        </div>
                        <div className="ef-col">
                          <p className="lbl">AI-Predicted LVEF (Direct Video Regression)</p>
                          <div className="ef-gauge">
                            <svg viewBox="0 0 100 100" className="ef-svg">
                              <circle cx="50" cy="50" r="40" className="ef-track" />
                              <circle
                                cx="50" cy="50" r="40"
                                className="ef-arc"
                                stroke={EF_STROKE(result.ef)}
                                strokeDasharray={`${(C * result.ef / 100).toFixed(1)} ${C.toFixed(1)}`}
                              />
                            </svg>
                            <div className="ef-center">
                              <span className="ef-val">{Math.round(result.ef)}</span>
                              <span className="ef-unit">% EF</span>
                            </div>
                          </div>
                          <p className={`ef-status ef-status-${statusClass(result.diagnosis)}`}>
                            {result.diagnosis}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Frame strip */}
                    <div className="card">
                      <div className="frames-hdr">
                        <p className="lbl" style={{ marginBottom: 0 }}>Frame Playback</p>
                        <span className="frames-hint">Click to enlarge · ← → to navigate</span>
                      </div>
                      <div className="frames-strip">
                        {frames.length > 0
                          ? frames.map((fr, i) => {
                            const pts = getContourPoints(result, i, frames.length)
                            return (
                              <button key={i} className="frame" onClick={() => setLightbox(i)}>
                                <img src={fr} alt={`Frame ${i + 1}`} />
                                {pts && (
                                  <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                                    <polygon points={pts} fill="rgba(244, 63, 94, 0.25)" stroke="#f43f5e" strokeWidth="1.8" />
                                  </svg>
                                )}
                                <span className="frame-n">{i + 1}</span>
                                <span className="frame-zoom"><ZoomIn size={10} /></span>
                              </button>
                            )
                          })
                          : Array.from({ length: 8 }, (_, i) => (
                            <div key={i} className="frame frame-ph">
                              <span className="frame-n">{i + 1}</span>
                            </div>
                          ))
                        }
                      </div>
                    </div>

                    {/* Beat-by-Beat Cardiac Rhythm Chart */}
                    <BeatAnalysisCard result={result} />

                    {/* Metrics */}
                    <div className="card">
                      <p className="lbl">Estimated Volumetric Metrics (Uncalibrated — Research Prototype)</p>
                      <div className="metrics">
                        {[
                          {
                            k: 'End-Diastolic Vol (EDV)',
                            v: result.edv != null ? `${result.edv} mL` : '—',
                          },
                          {
                            k: 'End-Systolic Vol (ESV)',
                            v: result.esv != null ? `${result.esv} mL` : '—',
                          },
                          {
                            k: 'Stroke Volume (SV)',
                            v: result.sv != null ? `${result.sv} mL` : '—',
                          },
                        ].map(m => (
                          <div key={m.k} className="metric">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <p className="metric-k">{m.k}</p>
                              <span style={{
                                fontSize: '0.6rem',
                                fontWeight: 700,
                                color: 'var(--amber)',
                                backgroundColor: 'rgba(245, 158, 11, 0.12)',
                                padding: '0.12rem 0.4rem',
                                borderRadius: '4px',
                                border: '1px solid rgba(245, 158, 11, 0.25)',
                                letterSpacing: '0.03em'
                              }}>
                                EXPLORATORY
                              </span>
                            </div>
                            <p className="metric-v">{m.v}</p>
                          </div>
                        ))}
                      </div>

                      <div style={{
                        marginTop: '0.85rem',
                        fontSize: '0.74rem',
                        color: 'var(--amber)',
                        background: 'rgba(245, 158, 11, 0.08)',
                        padding: '0.55rem 0.75rem',
                        borderRadius: '6px',
                        borderLeft: '3px solid var(--amber)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.45rem'
                      }}>
                        <AlertCircle size={13} style={{ flexShrink: 0 }} />
                        <span>Exploratory estimate — uncalibrated to physical scale, not a validated clinical measurement.</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Empty */}
              {!activeTaskId && !result && (
                <motion.div key="empty" className="card empty-card"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <ECGWave className="empty-wave" />
                  <h2 className="empty-title">No Active Scan</h2>
                  <p className="empty-body">
                    Upload an echocardiogram to begin automated cardiac analysis.
                    Drop a file anywhere on the screen or use the panel on the left.
                  </p>
                  <div className="empty-tags">
                    <span className="tag">Ejection Fraction</span>
                    <span className="tag">LV Segmentation</span>
                    <span className="tag">Wall Motion</span>
                  </div>
                </motion.div>
              )}

            </AnimatePresence>
          </main>
        </div>
      </div>

      {/* Lightbox / Frame Playback */}
      <AnimatePresence>
        {lightbox !== null && frames.length > 0 && (
          <motion.div className="lb-overlay"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={() => setLightbox(null)}>
            <div className="lb-content" onClick={e => e.stopPropagation()}>
              <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
                <img src={frames[lightbox]} alt="Frame" className="lb-img" />
                {(() => {
                  const pts = getContourPoints(result, lightbox, frames.length)
                  return pts ? (
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', borderRadius: 'var(--r2)' }}>
                      <polygon points={pts} fill="rgba(244, 63, 94, 0.25)" stroke="#f43f5e" strokeWidth="1.5" />
                    </svg>
                  ) : null
                })()}
              </div>
              <button className="lb-close" onClick={() => setLightbox(null)}><X size={24} /></button>
              <div className="lb-nav">
                <button className="lb-btn" disabled={lightbox === 0} onClick={() => setLightbox(p => p - 1)}>
                  <ChevronLeft size={24} />
                </button>
                <div className="lb-count">{lightbox + 1} / {frames.length}</div>
                <button className="lb-btn" disabled={lightbox === frames.length - 1} onClick={() => setLightbox(p => p + 1)}>
                  <ChevronRight size={24} />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
