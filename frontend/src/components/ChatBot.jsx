//frontend/src/components/ChatBot.js
import { useState, useEffect, useRef } from 'react'

const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_API_KEY}`

const SYSTEM_PROMPT = `You are a research assistant for a clinical trials database focused on 
immune checkpoint inhibitor (ICI) trials. The database contains 159 clinical trials covering 
cancer types such as Melanoma, NSCLC, Bladder, Colorectal, Gastric/GEJ, Head and Neck, 
Renal cell, and more. Trials involve ICI agents including Pembrolizumab, Nivolumab, 
Atezolizumab, Ipilimumab, Durvalumab, Avelumab, Cemiplimab, and Tremelimumab, classified 
as PD-1, PD-L1, or CTLA-4 inhibitors. Trials are Phase 2 or Phase 3 RCTs.
Your role is to answer research questions about these trials clearly and concisely. 
If asked something outside this domain, politely redirect the user to clinical trial topics.
Keep answers focused, factual, and suitable for a medical research audience.
Do not make up trial data — only answer based on general knowledge of ICI trials.`

// ─── Gemini API call ──────────────────────────────────────────────────────────
async function callGemini(userMessage, conversationHistory) {
  if (!GEMINI_API_KEY) throw new Error('No Gemini API key found. Add VITE_GEMINI_API_KEY to your .env file.')
  const contents = [
    { role: 'user', parts: [{ text: SYSTEM_PROMPT }] },
    { role: 'model', parts: [{ text: 'Understood. I am ready to assist with clinical trial research questions.' }] },
    ...conversationHistory,
    { role: 'user', parts: [{ text: userMessage }] }
  ]
  const response = await fetch(GEMINI_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents, generationConfig: { temperature: 0.7, maxOutputTokens: 1024 } })
  })
  if (!response.ok) {
    const err = await response.json()
    throw new Error(err?.error?.message || `Gemini API error ${response.status}`)
  }
  const data = await response.json()
  return data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response received.'
}

// ─── Hardcoded SQL queries — commented out, reserved for team LLM pipeline ───
/*
const HARDCODED_QUERIES = [
  {
    keywords: ['colorectal', 'multikinase'],
    label: 'Colorectal trials · ≥3 arms · Multikinase inhibitor control',
    sql: `SELECT "NCT", "Author", "Year", "Originial publication or Follow-up",
                 "Number of arms", "Control arm", "Cancer type", "Trial phase"
          FROM clinical_trials
          WHERE "Cancer type" = 'Colorectal'
            AND "Number of arms" >= 3
            AND "Control arm" = 'Multikinase inhibitor'`,
  },
]
function matchQuery(message) {
  const lower = message.toLowerCase()
  return HARDCODED_QUERIES.find(q =>
    q.keywords.every(kw => lower.includes(kw.toLowerCase()))
  ) || null
}
*/

// ─── Autocomplete helpers ─────────────────────────────────────────────────────
async function loadSeedQuestions() {
  try {
    const res = await fetch('/seed_questions.json')
    const data = await res.json()
    return data.map(d => d.original_question)
  } catch {
    return []
  }
}

function findSuggestion(input, questions) {
  if (!input || input.length < 4) return ''
  const lower = input.toLowerCase()
  const match = questions.find(q => q.toLowerCase().startsWith(lower))
  return match || ''
}

// ─── Satisfaction meter ───────────────────────────────────────────────────────
function computeSatisfaction(ratings) {
  if (!ratings.length) return null
  const score = ratings.reduce((acc, r) => {
    if (r === 'good') return acc + 1
    if (r === 'neutral') return acc + 0.5
    return acc
  }, 0)
  return score / ratings.length
}

function SatisfactionBar({ ratings }) {
  if (!ratings.length) return null
  const score = computeSatisfaction(ratings)
  const pct = Math.round(score * 100)
  const color = score >= 0.7 ? '#22c55e' : score >= 0.4 ? '#f59e0b' : '#ef4444'
  const label = score >= 0.7 ? 'Satisfied' : score >= 0.4 ? 'Mixed' : 'Unsatisfied'
  const pulseClass = score >= 0.7 ? 'pulse-green' : score < 0.4 ? 'pulse-red' : 'pulse-amber'
  return (
    <div className={`satisfaction-bar-wrap ${pulseClass}`}>
      <div className="satisfaction-bar-label">
        <span className="satisfaction-text">{label}</span>
        <span className="satisfaction-pct">{pct}%</span>
      </div>
      <div className="satisfaction-track">
        <div className="satisfaction-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="satisfaction-counts">
        <span className="sc-good">✔ {ratings.filter(r => r === 'good').length}</span>
        <span className="sc-neutral">● {ratings.filter(r => r === 'neutral').length}</span>
        <span className="sc-bad">✕ {ratings.filter(r => r === 'bad').length}</span>
      </div>
    </div>
  )
}

// ─── Feedback buttons ─────────────────────────────────────────────────────────
function FeedbackButtons({ messageId, onRate, onEdit, messageText }) {

  const [rating, setRating] = useState(null)
  const [animating, setAnimating] = useState(null)

  const buttons = [
    { value: 'good',    label: 'Helpful',    symbol: '✓', activeColor: '#22c55e' },
    { value: 'neutral', label: 'Okay',       symbol: '●', activeColor: '#d97706' },
    { value: 'bad',     label: 'Unhelpful',  symbol: '✕', activeColor: '#ef4444' },
  ]

const handleRate = (value) => {
    if (rating) return
    setAnimating(value)
    setTimeout(() => {
      setRating(value)
      setAnimating(null)
      onRate(messageId, value)
      if (value === 'good')    console.log('User liked the output')
      if (value === 'neutral') { console.log('User is okay with the output'); onEdit(messageId, messageText) }
      if (value === 'bad')     console.log('User did not like the output')
    }, 350)
  }

  if (rating) {
    const chosen = buttons.find(b => b.value === rating)
    return (
      <div className="feedback-thankyou" style={{ color: chosen.activeColor }}>
        <span className="feedback-thankyou-icon">{chosen.symbol}</span>
        <span>Thanks for the feedback</span>
      </div>
    )
  }

  return (
    <div className="feedback-buttons">
      <span className="feedback-prompt">Was this helpful?</span>
      {buttons.map(btn => (
        <button
          key={btn.value}
          className={`feedback-btn fb-${btn.value} ${animating === btn.value ? 'fb-animating' : ''}`}
          onClick={() => handleRate(btn.value)}
          title={btn.label}
        >
          {btn.symbol}
        </button>
      ))}
    </div>
  )
}

// ─── Main ChatBot component ───────────────────────────────────────────────────
export default function ChatBot({ onClose, db }) {
  const [messages, setMessages] = useState([
    {
      id: 0,
      from: 'bot',
      text: `Hi! I'm the test-project research assistant. Ask me anything about immune checkpoint inhibitor clinical trials — cancer types, ICI agents, trial phases, endpoints, and more.`,
      showFeedback: false,
    }
  ])
  const [input, setInput] = useState('')
  const [typing, setTyping] = useState(false)
  const [ratings, setRatings] = useState([])
  const [showSatisfaction, setShowSatisfaction] = useState(false)

  const [editingId, setEditingId] = useState(null)
  const [editDraft, setEditDraft] = useState('')

  // Autocomplete state
  const [seedQuestions, setSeedQuestions] = useState([])
  const [suggestion, setSuggestion] = useState('')

  const historyRef = useRef([])
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const idRef = useRef(1)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, typing])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    if (ratings.length > 0) setShowSatisfaction(true)
  }, [ratings])

  // Load seed questions on mount
  useEffect(() => {
    loadSeedQuestions().then(setSeedQuestions)
  }, [])

  const addMessage = (from, payload) => {
    const msg = { id: idRef.current++, from, ...payload }
    setMessages(prev => [...prev, msg])
    return msg
  }

  const handleRate = (messageId, value) => {
    setRatings(prev => [...prev, value])
  }

const handleInputChange = (e) => {
    const val = e.target.value
    setInput(val)
    setSuggestion(findSuggestion(val, seedQuestions))
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
  }

  const handleKeyDown = (e) => {
    // Tab accepts suggestion
    if (e.key === 'Tab' && suggestion) {
      e.preventDefault()
      setInput(suggestion)
      setSuggestion('')
      return
    }
    // Escape clears suggestion
    if (e.key === 'Escape') {
      setSuggestion('')
      return
    }
    // Enter sends
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const send = async () => {
    const text = input.trim()
    if (!text || typing) return
    setSuggestion('')
    addMessage('user', { text, showFeedback: false })
    setInput('')
    setTyping(true)
    try {
      const reply = await callGemini(text, historyRef.current)
      historyRef.current = [
        ...historyRef.current,
        { role: 'user', parts: [{ text }] },
        { role: 'model', parts: [{ text: reply }] }
      ]
      if (historyRef.current.length > 20) {
        historyRef.current = historyRef.current.slice(-20)
      }
      addMessage('bot', { text: reply, showFeedback: true })
    } catch (err) {
      addMessage('bot', { text: `❌ ${err.message}`, isError: true, showFeedback: false })
    } finally {
      setTyping(false)
    }
  }

const renderBubble = (msg) => {
    if (msg.isError) return <div className="chat-bubble error-bubble">{msg.text}</div>

    // Editing mode — yellow circle was clicked
    if (editingId === msg.id) {
      return (
        <div className="edit-bubble-wrap">
          <textarea
            className="edit-bubble-textarea"
            value={editDraft}
            onChange={e => setEditDraft(e.target.value)}
            rows={4}
            autoFocus
          />
          <div className="edit-bubble-actions">
            <button className="edit-save-btn" onClick={() => {
              setMessages(prev => prev.map(m =>
                m.id === msg.id ? { ...m, text: editDraft } : m
              ))
              setEditingId(null)
              setEditDraft('')
            }}>
              Save
            </button>
            <button className="edit-cancel-btn" onClick={() => {
              setEditingId(null)
              setEditDraft('')
            }}>
              Cancel
            </button>
          </div>
        </div>
      )
    }

    return (
      <div className="chat-bubble">
        {msg.text.split('\n').map((line, i, arr) => (
          <span key={i}>{line}{i < arr.length - 1 && <br />}</span>
        ))}
      </div>
    )
  }
  return (
    <div className="chatbot-panel">

      {/* ── Header ── */}
      <div className="chatbot-header">
        <div className="chatbot-header-left">
          <div className="chatbot-avatar">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="white">
              <path d="M8 1a5 5 0 00-5 5v1H2a1 1 0 000 2h1v1a5 5 0 0010 0v-1h1a1 1 0 000-2h-1V6a5 5 0 00-5-5z"/>
            </svg>
          </div>
          <div>
            <div className="chatbot-title">test-project Assistant</div>
            <div className="chatbot-status">
              <span className="status-dot" />
              {GEMINI_API_KEY ? 'Gemini connected' : 'No API key'}
            </div>
          </div>
        </div>
        <button className="chatbot-close-btn" onClick={onClose} title="Close">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M1 1l12 12M13 1L1 13"/>
          </svg>
        </button>
      </div>

      {/* ── Satisfaction bar ── */}
      {showSatisfaction && <SatisfactionBar ratings={ratings} />}

      {/* ── No API key warning ── */}
      {!GEMINI_API_KEY && (
        <div className="chatbot-warning">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 1L1 14h14L8 1zm0 3l4.5 8h-9L8 4zm-1 3v2h2V7H7zm0 3v2h2v-2H7z"/>
          </svg>
          Add <code>VITE_GEMINI_API_KEY</code> to your <code>.env</code> file and restart.
        </div>
      )}

      {/* ── Messages ── */}
      <div className="chatbot-messages">
        {messages.map(msg => (
          <div key={msg.id} className={`chat-message ${msg.from}`}>
            {msg.from === 'bot' && (
              <div className="bot-icon">
                <svg width="12" height="12" viewBox="0 0 16 16" fill="white">
                  <path d="M8 1a5 5 0 00-5 5v1H2a1 1 0 000 2h1v1a5 5 0 0010 0v-1h1a1 1 0 000-2h-1V6a5 5 0 00-5-5z"/>
                </svg>
              </div>
            )}
            <div className="msg-content-wrap">
              {msg.from === 'user'
                ? <div className="chat-bubble user-bubble">{msg.text}</div>
                : renderBubble(msg)
              }
              {msg.from === 'bot' && msg.showFeedback && (
                <FeedbackButtons
                  messageId={msg.id}
                  onRate={handleRate}
                  onEdit={(id, text) => { setEditingId(id); setEditDraft(text) }}
                  messageText={msg.text}
                />
              )}
            </div>
          </div>
        ))}

        {typing && (
          <div className="chat-message bot">
            <div className="bot-icon">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="white">
                <path d="M8 1a5 5 0 00-5 5v1H2a1 1 0 000 2h1v1a5 5 0 0010 0v-1h1a1 1 0 000-2h-1V6a5 5 0 00-5-5z"/>
              </svg>
            </div>
            <div className="chat-bubble typing-bubble">
              <span className="dot"/><span className="dot"/><span className="dot"/>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

{/* ── Input with autocomplete ── */}
      <div className="chatbot-input-area">

        {/* Suggestion chip — appears above input when match found */}
        {suggestion && (
          <div className="suggestion-chip" onClick={() => { setInput(suggestion); setSuggestion(''); inputRef.current?.focus() }}>
            <div className="suggestion-chip-inner">
              <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" style={{flexShrink:0, opacity:0.5}}>
                <path d="M8 1a7 7 0 110 14A7 7 0 018 1zm0 1.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM7 5h2v4H7V5zm0 5h2v2H7v-2z"/>
              </svg>
              <span className="suggestion-text">{suggestion}</span>
            </div>
            <kbd className="suggestion-tab-key">Tab ↵</kbd>
          </div>
        )}

        <div className="chatbot-input-row">
          <textarea
            ref={inputRef}
            className="chatbot-input"
            placeholder={GEMINI_API_KEY ? 'Ask about the trials…' : 'API key required…'}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={!GEMINI_API_KEY || typing}
          />
          <button
            className={`chatbot-send-btn ${input.trim() && !typing ? 'active' : ''}`}
            onClick={send}
            disabled={!input.trim() || typing || !GEMINI_API_KEY}
          >
            {typing ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{animation:'spin 0.8s linear infinite'}}>
                <path d="M12 2a10 10 0 0110 10" strokeLinecap="round"/>
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M1 1l14 7-14 7V9.5l10-1.5-10-1.5V1z"/>
              </svg>
            )}
          </button>
        </div>

      </div>
    </div>
  )
}