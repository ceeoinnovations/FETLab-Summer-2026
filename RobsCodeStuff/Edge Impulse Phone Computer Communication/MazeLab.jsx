import { useState, useEffect, useRef } from "react";

// ── CEEO relay constants ──────────────────────────────────────────────
const RELAY = "wss://@chrisrogers.pyscriptapps.com/talking-on-a-channel/api/channels/hackathon";
const STALE_MS = 2500;

function toPred(legoTopic) {
  const cmd = (legoTopic || "/LEGO").trim().replace(/\/+$/, "") || "/LEGO";
  const t = cmd.replace(/^\/LEGO\b/, "/PRED");
  return t === cmd ? "/PRED" : t;
}

// ── Colour tokens (matching maze.css palette) ─────────────────────────
const C = {
  bg:       "#0f1117",
  surface:  "#1a1d27",
  border:   "#2a2d3a",
  ink:      "#e2e8f0",
  muted:    "#64748b",
  accent:   "#3b82f6",
  signal:   "#22d3ee",
  recv:     "#a78bfa",
  green:    "#22c55e",
  red:      "#ef4444",
};

const css = (obj) => Object.entries(obj).map(([k,v])=>`${k}:${v}`).join(";");

// ── Library data ──────────────────────────────────────────────────────
const LIBRARY = [
  { label: "General", entries: [
    { fn: "wait(seconds)",  desc: "Pause for this many seconds" },
    { fn: "print(message)", desc: "Show a message in the Output box" },
  ]},
  { label: "Channels — channel", entries: [
    { fn: "channel.msg",            desc: "The latest message from the phone on this channel" },
    { fn: "channel.send('go')",     desc: "Send a message on the channel" },
    { fn: "channel.wait_for('go')", desc: "Pause until 'go' arrives" },
    { fn: "channel.clear()",        desc: "Forget the last message" },
  ]},
  { label: "Double Motor — dm", entries: [
    { fn: "dm.run()",             desc: "Drive straight until stop()" },
    { fn: "dm.run_time(ms)",      desc: "Drive for this many milliseconds" },
    { fn: "dm.turn_left(degrees)",desc: "Turn left in place (both wheels)" },
    { fn: "dm.turn_right(degrees)",desc:"Turn right in place (both wheels)" },
    { fn: "dm.set_speed(speed)",  desc: "Set speed 0–100" },
    { fn: "dm.stop()",            desc: "Stop both motors" },
  ]},
  { label: "Single Motor — sm", entries: [
    { fn: "sm.run()",            desc: "Run until stop()" },
    { fn: "sm.stop()",           desc: "Stop the motor" },
    { fn: "sm.set_speed(speed)", desc: "Set speed 0–100" },
  ]},
  { label: "Color Sensor — cs", entries: [
    { fn: "cs.detect_color()",      desc: "Returns color e.g. 'Red'" },
    { fn: "cs.detect_rgb()",        desc: "Returns (R, G, B) values" },
    { fn: "cs.detect_reflection()", desc: "Reflection 0–100" },
  ]},
  { label: "Controller — c", entries: [
    { fn: "c.drive(dm)",         desc: "Drive dm with the sticks" },
    { fn: "c.left_position()",   desc: "Left stick −100 to 100" },
    { fn: "c.right_position()",  desc: "Right stick −100 to 100" },
  ]},
  { label: "Edge Impulse (phone)", entries: [
    { fn: "channel.msg", desc: "Top prediction label sent from the phone" },
  ], note: "Open the Edge Impulse phone page on your phone. Enter your EI API key and this channel, then tap Start. The phone classifies with your trained model and sends its top prediction here." },
];

const DEVICES = ["Double Motor (dm)", "Single Motor (sm)", "Color Sensor (cs)", "Controller (c)"];

// ── Sub-components ────────────────────────────────────────────────────

function TopBar() {
  return (
    <div style={{
      display:"flex", alignItems:"center", gap:"0.75rem",
      padding:"0.6rem 1rem", background:C.surface,
      borderBottom:`1px solid ${C.border}`, flexShrink:0,
    }}>
      <span style={{
        width:10, height:10, borderRadius:"50%", background:C.signal,
        boxShadow:`0 0 6px ${C.signal}`, animation:"pulse 2s infinite",
        flexShrink:0,
      }}/>
      <span style={{fontWeight:700, fontSize:"0.95rem", color:C.ink, letterSpacing:"0.02em"}}>
        Maze Lab
      </span>
      <span style={{color:C.muted, fontSize:"0.8rem"}}>drive · detect · navigate</span>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}`}</style>
    </div>
  );
}

function DeviceBar({ chips, onAdd }) {
  const [sel, setSel] = useState(DEVICES[0]);
  return (
    <div style={{
      display:"flex", flexWrap:"wrap", alignItems:"center", gap:"0.4rem",
      padding:"0.5rem 0.75rem", background:C.surface, borderBottom:`1px solid ${C.border}`,
      flexShrink:0,
    }}>
      <span style={{color:C.muted, fontSize:"0.75rem", marginRight:"0.25rem"}}>Add device</span>
      <select value={sel} onChange={e=>setSel(e.target.value)} style={{
        background:C.bg, border:`1px solid ${C.border}`, color:C.ink,
        borderRadius:4, padding:"0.2rem 0.4rem", fontSize:"0.8rem",
      }}>
        {DEVICES.map(d=><option key={d}>{d}</option>)}
      </select>
      <button onClick={()=>onAdd(sel)} style={{
        background:"none", border:`1px solid ${C.accent}`, color:C.accent,
        borderRadius:4, padding:"0.2rem 0.5rem", cursor:"pointer", fontSize:"0.8rem",
      }}>＋ Add Device</button>
      {chips.map((c,i)=>(
        <span key={i} style={{
          background:C.border, color:C.ink, borderRadius:12,
          padding:"0.15rem 0.6rem", fontSize:"0.75rem",
        }}>{c}</span>
      ))}
    </div>
  );
}

function EditorPane({ output }) {
  const [code, setCode] = useState(
`# Write your LEGO Python here
wait(2)
print("Hello from Maze Lab!")
dm.run_time(1000)
`);
  return (
    <div style={{display:"flex", flexDirection:"column", flex:1, minHeight:0}}>
      <div style={{
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"0.4rem 0.75rem", borderBottom:`1px solid ${C.border}`,
      }}>
        <span style={{fontSize:"0.75rem", color:C.muted, textTransform:"uppercase", letterSpacing:"0.08em"}}>Code</span>
        <div style={{display:"flex", gap:"0.4rem"}}>
          <button style={{background:C.green, border:"none", color:"#fff", borderRadius:4, padding:"0.2rem 0.7rem", cursor:"pointer", fontSize:"0.8rem"}}>▶ Run</button>
          <button style={{background:C.red,   border:"none", color:"#fff", borderRadius:4, padding:"0.2rem 0.7rem", cursor:"pointer", fontSize:"0.8rem"}}>■ Stop</button>
        </div>
      </div>
      <textarea value={code} onChange={e=>setCode(e.target.value)} spellCheck={false} style={{
        flex:1, background:C.bg, color:C.ink, border:"none", outline:"none",
        padding:"0.75rem", fontFamily:"'JetBrains Mono','Fira Code',monospace",
        fontSize:"0.82rem", resize:"none", lineHeight:1.6,
      }}/>
      <div style={{borderTop:`1px solid ${C.border}`}}>
        <div style={{padding:"0.3rem 0.75rem", fontSize:"0.7rem", color:C.muted, textTransform:"uppercase", letterSpacing:"0.08em"}}>Output</div>
        <div style={{
          fontFamily:"'JetBrains Mono','Fira Code',monospace", fontSize:"0.78rem",
          color:C.signal, padding:"0.4rem 0.75rem", minHeight:60, whiteSpace:"pre-wrap",
        }}>{output || ""}</div>
      </div>
    </div>
  );
}

function LibraryPanel() {
  const [open, setOpen] = useState(true);
  const [expanded, setExpanded] = useState({});
  const toggle = k => setExpanded(p=>({...p,[k]:!p[k]}));
  return (
    <div style={{borderBottom:`1px solid ${C.border}`}}>
      <button onClick={()=>setOpen(p=>!p)} style={{
        width:"100%", textAlign:"left", background:"none", border:"none",
        padding:"0.5rem 0.75rem", cursor:"pointer", color:C.ink,
        fontSize:"0.8rem", fontWeight:600, display:"flex", justifyContent:"space-between",
      }}>
        Library <span>{open?"▾":"▸"}</span>
      </button>
      {open && (
        <div style={{padding:"0 0.5rem 0.5rem"}}>
          {LIBRARY.map(cat=>(
            <div key={cat.label} style={{marginBottom:"0.15rem"}}>
              <button onClick={()=>toggle(cat.label)} style={{
                width:"100%", textAlign:"left", background:"none", border:"none",
                padding:"0.3rem 0.25rem", cursor:"pointer",
                color:C.accent, fontSize:"0.78rem", fontWeight:600,
                display:"flex", justifyContent:"space-between",
              }}>
                {cat.label} <span style={{color:C.muted}}>{expanded[cat.label]?"▾":"▸"}</span>
              </button>
              {expanded[cat.label] && (
                <div style={{paddingLeft:"0.5rem"}}>
                  {cat.note && <p style={{color:C.muted, fontSize:"0.72rem", margin:"0.2rem 0 0.4rem", lineHeight:1.5}}>{cat.note}</p>}
                  {cat.entries.map(e=>(
                    <div key={e.fn} style={{marginBottom:"0.35rem"}}>
                      <code style={{color:C.signal, fontSize:"0.75rem", display:"block"}}>{e.fn}</code>
                      <span style={{color:C.muted, fontSize:"0.7rem"}}>{e.desc}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EdgeImpulsePanel({ channelTopic, onChannelChange }) {
  const [open, setOpen] = useState(true);
  const [topClass, setTopClass] = useState("none");
  const [topConf, setTopConf]   = useState(0);
  const [bars, setBars]         = useState([]);
  const [status, setStatus]     = useState("No model running yet.");
  const [predTopic, setPredTopic] = useState(() => toPred(channelTopic));
  const [chInput, setChInput]   = useState(channelTopic);
  const lastPredRef = useRef(performance.now());
  const stateRef    = useRef("nomodel");
  const wsRef       = useRef(null);
  const cancelRef   = useRef(false);
  const reconnRef   = useRef(null);

  // Stale timer
  useEffect(() => {
    const t = setInterval(() => {
      if (stateRef.current !== "nomodel" && performance.now() - lastPredRef.current > STALE_MS) {
        stateRef.current = "nomodel";
        setBars([]); setTopClass("none"); setTopConf(0);
        setStatus("No model running yet.");
      }
    }, 1000);
    return () => clearInterval(t);
  }, []);

  // WebSocket connection
  useEffect(() => {
    cancelRef.current = false;
    setStatus("Connecting…");

    const connect = () => {
      if (cancelRef.current) return;
      const ws = new WebSocket(RELAY);
      wsRef.current = ws;
      ws.onopen = () => {
        if (stateRef.current === "nomodel") setStatus("Waiting for model on " + predTopic + " …");
      };
      ws.onclose = () => { reconnRef.current = setTimeout(connect, 2000); };
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === "data" && m.payload) {
            const inner = JSON.parse(m.payload);
            if ((inner.topic || "") === predTopic) {
              const list = JSON.parse(inner.value);
              if (!Array.isArray(list)) return;
              stateRef.current = "bars";
              lastPredRef.current = performance.now();
              setStatus("");
              setBars(list);
              const top = list.reduce((a,b) => b.p > a.p ? b : a, list[0]);
              setTopClass(top.c);
              setTopConf(Math.round(top.p * 100));
            }
          }
        } catch {}
      };
    };
    connect();
    return () => {
      cancelRef.current = true;
      clearTimeout(reconnRef.current);
      wsRef.current?.close();
    };
  }, [predTopic]);

  const doSwitch = () => {
    const t = (chInput || "/LEGO").trim().replace(/\/+$/, "") || "/LEGO";
    setChInput(t);
    setPredTopic(toPred(t));
    onChannelChange?.(t);
    setBars([]); setTopClass("none"); setTopConf(0);
    stateRef.current = "nomodel";
    setStatus("Switched — waiting for model on " + toPred(t) + " …");
  };

  return (
    <div style={{borderBottom:`1px solid ${C.border}`}}>
      <button onClick={()=>setOpen(p=>!p)} style={{
        width:"100%", textAlign:"left", background:"none", border:"none",
        padding:"0.5rem 0.75rem", cursor:"pointer", color:C.ink,
        fontSize:"0.8rem", fontWeight:600, display:"flex", justifyContent:"space-between",
      }}>
        Edge Impulse <span>{open?"▾":"▸"}</span>
      </button>
      {open && (
        <div style={{padding:"0 0.75rem 0.75rem"}}>
          {/* Instruction */}
          <p style={{color:C.muted, fontSize:"0.72rem", margin:"0 0 0.5rem", lineHeight:1.5}}>
            Open the <strong style={{color:C.accent}}>Edge Impulse phone page</strong> on your phone,
            enter your EI API key and this channel, then tap Start.
          </p>
          {/* Channel switch */}
          <div style={{display:"flex", gap:"0.3rem", marginBottom:"0.5rem"}}>
            <input value={chInput} onChange={e=>setChInput(e.target.value)}
              onKeyDown={e=>e.key==="Enter"&&doSwitch()}
              style={{flex:1, background:C.bg, border:`1px solid ${C.border}`,
                color:C.ink, borderRadius:4, padding:"0.25rem 0.4rem", fontSize:"0.78rem"}}/>
            <button onClick={doSwitch} style={{
              background:"none", border:`1px solid ${C.accent}`, color:C.accent,
              borderRadius:4, padding:"0.25rem 0.5rem", cursor:"pointer", fontSize:"0.78rem",
            }}>Switch</button>
          </div>
          {/* Status */}
          {status && <div style={{color:C.muted, fontSize:"0.72rem", marginBottom:"0.4rem"}}>{status}</div>}
          {/* Detected / Confidence */}
          <div style={{display:"flex", gap:"1rem", marginBottom:"0.4rem"}}>
            <span style={{fontSize:"0.75rem", color:C.muted}}>Detected <strong style={{color:C.ink}}>{topClass}</strong></span>
            <span style={{fontSize:"0.75rem", color:C.muted}}>Confidence <strong style={{color:C.ink}}>{topConf}%</strong></span>
          </div>
          {/* Bars */}
          {bars.map((b,i)=>(
            <div key={i} style={{display:"flex", alignItems:"center", gap:"0.4rem", marginBottom:"0.25rem"}}>
              <span style={{width:80, fontSize:"0.7rem", color:C.ink, flexShrink:0, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>{b.c}</span>
              <div style={{flex:1, background:C.border, borderRadius:3, height:10}}>
                <div style={{width:Math.round(b.p*100)+"%", height:"100%", background:C.recv, borderRadius:3, transition:"width 0.2s"}}/>
              </div>
              <span style={{width:32, textAlign:"right", fontSize:"0.7rem", color:C.muted}}>{Math.round(b.p*100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ChannelsPanel({ channelTopic, onChannelChange }) {
  const [open, setOpen]   = useState(true);
  const [topic, setTopic] = useState(channelTopic);
  const [msg, setMsg]     = useState("");
  const [log, setLog]     = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef  = useRef(null);
  const cancelRef = useRef(false);

  useEffect(() => {
    cancelRef.current = false;
    const connect = () => {
      if (cancelRef.current) return;
      const ws = new WebSocket(RELAY);
      wsRef.current = ws;
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === "data" && m.payload) {
            const inner = JSON.parse(m.payload);
            if ((inner.topic||"") === topic) {
              setLog(l => [...l.slice(-49), `← ${inner.value}`]);
            }
          }
        } catch {}
      };
    };
    connect();
    return () => { cancelRef.current = true; wsRef.current?.close(); };
  }, [topic]);

  const send = () => {
    if (!wsRef.current || wsRef.current.readyState !== 1) return;
    const payload = JSON.stringify({ topic, value: msg });
    wsRef.current.send(JSON.stringify({ type:"data", payload }));
    setLog(l => [...l.slice(-49), `→ ${msg}`]);
  };

  return (
    <div>
      <button onClick={()=>setOpen(p=>!p)} style={{
        width:"100%", textAlign:"left", background:"none", border:"none",
        padding:"0.5rem 0.75rem", cursor:"pointer", color:C.ink,
        fontSize:"0.8rem", fontWeight:600, display:"flex", justifyContent:"space-between", alignItems:"center",
      }}>
        <span style={{display:"flex", alignItems:"center", gap:"0.4rem"}}>
          <span style={{width:7, height:7, borderRadius:"50%",
            background: connected ? C.green : C.red,
            boxShadow: connected ? `0 0 5px ${C.green}` : "none",
          }}/>
          Channels
        </span>
        <span>{open?"▾":"▸"}</span>
      </button>
      {open && (
        <div style={{padding:"0 0.75rem 0.75rem"}}>
          <input placeholder="Channel" value={topic}
            onChange={e=>{setTopic(e.target.value); onChannelChange?.(e.target.value);}}
            style={{width:"100%", boxSizing:"border-box", background:C.bg, border:`1px solid ${C.border}`,
              color:C.ink, borderRadius:4, padding:"0.25rem 0.4rem", fontSize:"0.78rem", marginBottom:"0.3rem"}}/>
          <div style={{display:"flex", gap:"0.3rem", marginBottom:"0.4rem"}}>
            <input placeholder="Message" value={msg} onChange={e=>setMsg(e.target.value)}
              onKeyDown={e=>e.key==="Enter"&&send()}
              style={{flex:1, background:C.bg, border:`1px solid ${C.border}`,
                color:C.ink, borderRadius:4, padding:"0.25rem 0.4rem", fontSize:"0.78rem"}}/>
            <button onClick={send} style={{
              background:C.accent, border:"none", color:"#fff",
              borderRadius:4, padding:"0.25rem 0.6rem", cursor:"pointer", fontSize:"0.78rem",
            }}>Send</button>
          </div>
          <div style={{
            background:C.bg, border:`1px solid ${C.border}`, borderRadius:4,
            padding:"0.3rem 0.4rem", minHeight:50, maxHeight:100, overflowY:"auto",
            fontFamily:"monospace", fontSize:"0.7rem", color:C.signal,
          }}>
            {log.length === 0
              ? <span style={{color:C.muted}}>{connected ? "Connected — no messages yet." : "Connecting…"}</span>
              : log.map((l,i)=><div key={i}>{l}</div>)}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────
export default function MazeLab() {
  const [chips, setChips]     = useState([]);
  const [output]              = useState("");
  const [channelTopic, setChannelTopic] = useState("/LEGO");

  return (
    <div style={{
      display:"flex", flexDirection:"column", height:"100vh",
      background:C.bg, color:C.ink,
      fontFamily:"-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",
      fontSize:"14px",
    }}>
      <TopBar/>
      <DeviceBar chips={chips} onAdd={d=>setChips(p=>[...p,d])}/>
      <div style={{display:"flex", flex:1, minHeight:0}}>
        {/* Left — editor */}
        <div style={{flex:1, display:"flex", flexDirection:"column", borderRight:`1px solid ${C.border}`, minWidth:0}}>
          <EditorPane output={output}/>
        </div>
        {/* Right — panels */}
        <div style={{width:260, display:"flex", flexDirection:"column", overflowY:"auto", flexShrink:0}}>
          <LibraryPanel/>
          <EdgeImpulsePanel channelTopic={channelTopic} onChannelChange={setChannelTopic}/>
          <ChannelsPanel    channelTopic={channelTopic} onChannelChange={setChannelTopic}/>
        </div>
      </div>
    </div>
  );
}
