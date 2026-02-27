import { useState, useEffect, useRef } from 'react'

// ─── Gemini API config ────────────────────────────────────────────────────────
const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_API_KEY}`

// System prompt — tells Gemini it's a clinical trials research assistant
// TODO: Update this prompt when your team's custom LLM pipeline is ready
// TODO: Replace with your project name and context
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

// ─── Call Gemini API ──────────────────────────────────────────────────────────
async function callGemini(userMessage, conversationHistory) {
  if (!GEMINI_API_KEY) {
    throw new Error('No Gemini API key found. Add VITE_GEMINI_API_KEY to your .env file.')
  }

  // Build conversation contents for multi-turn chat
  const contents = [
    // Inject system prompt as the first user turn (Gemini Flash supports this pattern)
    {
      role: 'user',
      parts: [{ text: SYSTEM_PROMPT }]
    },
    {
      role: 'model',
      parts: [{ text: 'Understood. I am ready to assist with clinical trial research questions.' }]
    },
    // Add previous conversation turns
    ...conversationHistory,
    // Add the new user message
    {
      role: 'user',
      parts: [{ text: userMessage }]
    }
  ]

  const response = await fetch(GEMINI_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents,
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1024,
      }
    })
  })

  if (!response.ok) {
    const err = await response.json()
    throw new Error(err?.error?.message || `Gemini API error ${response.status}`)
  }

  const data = await response.json()
  return data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response received.'
}

// ─── Hardcoded SQL queries (commented out — will be replaced by team LLM pipeline) ──
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

// ─── Component ────────────────────────────────────────────────────────────────
export default function ChatBot({ onClose, db }) {
  // onQueryResult and onClearQuery props kept for when LLM pipeline is ready
  // and will power the table updates — not used yet

  const [messages, setMessages] = useState([
    {
      id: 0,
      from: 'bot',
      text: `Hi! I'm the test-project research assistant. Ask me anything about immune checkpoint inhibitor clinical trials — cancer types, ICI agents, trial phases, endpoints, and more.`,
    }
  ])
  const [input, setInput] = useState('')
  const [typing, setTyping] = useState(false)
  const [error, setError] = useState(null)

  // Conversation history for multi-turn context — stored as Gemini-formatted turns
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

  const addMessage = (from, payload) => {
    const msg = { id: idRef.current++, from, ...payload }
    setMessages(prev => [...prev, msg])
    return msg
  }

  const send = async () => {
    const text = input.trim()
    if (!text || typing) return

    addMessage('user', { text })
    setInput('')
    setTyping(true)
    setError(null)

    try {
      const reply = await callGemini(text, historyRef.current)

      // Save this turn to history for multi-turn context
      historyRef.current = [
        ...historyRef.current,
        { role: 'user', parts: [{ text }] },
        { role: 'model', parts: [{ text: reply }] }
      ]

      // Keep history from growing too large (last 10 turns = 20 entries)
      if (historyRef.current.length > 20) {
        historyRef.current = historyRef.current.slice(-20)
      }

      addMessage('bot', { text: reply })
    } catch (err) {
      setError(err.message)
      addMessage('bot', { text: `❌ ${err.message}`, isError: true })
    } finally {
      setTyping(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const renderBubble = (msg) => {
    if (msg.isError) {
      return (
        <div className="chat-bubble error-bubble">
          {msg.text}
        </div>
      )
    }
    // Render plain text preserving newlines
    return (
      <div className="chat-bubble">
        {msg.text.split('\n').map((line, i, arr) => (
          <span key={i}>
            {line}
            {i < arr.length - 1 && <br />}
          </span>
        ))}
      </div>
    )
  }

  return (
    <div className="chatbot-panel">
      {/* Header */}
      <div className="chatbot-header">
        <div className="chatbot-header-left">
          <div className="chatbot-avatar">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="white">
              <path d="M8 1a5 5 0 00-5 5v1H2a1 1 0 000 2h1v1a5 5 0 0010 0v-1h1a1 1 0 000-2h-1V6a5 5 0 00-5-5z"/>
            </svg>
          </div>
          <div>
            {/* TODO: Update name when your team's LLM is integrated */}
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

      {/* No API key warning */}
      {!GEMINI_API_KEY && (
        <div className="chatbot-warning">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 1L1 14h14L8 1zm0 3l4.5 8h-9L8 4zm-1 3v2h2V7H7zm0 3v2h2v-2H7z"/>
          </svg>
          Add <code>VITE_GEMINI_API_KEY</code> to your <code>.env</code> file and restart the dev server.
        </div>
      )}

      {/* Messages */}
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
            {msg.from === 'user'
              ? <div className="chat-bubble user-bubble">{msg.text}</div>
              : renderBubble(msg)
            }
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

      {/* Input */}
      <div className="chatbot-input-area">
        <textarea
          ref={inputRef}
          className="chatbot-input"
          placeholder={GEMINI_API_KEY ? 'Ask about the trials…' : 'API key required…'}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          rows={1}
          disabled={!GEMINI_API_KEY || typing}
        />
        <button
          className={`chatbot-send-btn ${input.trim() && !typing ? 'active' : ''}`}
          onClick={send}
          disabled={!input.trim() || typing || !GEMINI_API_KEY}
        >
          {typing ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" style={{animation:'spin 1s linear infinite'}}>
              <path d="M7 1a6 6 0 11-4.24 1.76"/>
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M1 1l14 7-14 7V9.5l10-1.5-10-1.5V1z"/>
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}