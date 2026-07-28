# Maze Lab — Edge Impulse Redesign (no custom phone page needed)

## New approach overview

The custom `/ei-phone` route is no longer needed. Delete it.

Instead, the user connects their phone directly to Edge Impulse using EI's own
hosted mobile client (`smartphone.edgeimpulse.com`), which runs the WASM model
in the phone's browser. The laptop then subscribes to EI's inference debug
stream WebSocket to receive live predictions, and forwards the top result into
the CEEO relay so `channel.msg` works in Python exactly as before.

**User workflow (3 steps, no custom page):**
1. In EI Studio → Devices → Connect a new device → Mobile phone → scan QR code
2. On phone: tap "Switch to classification mode"
3. On laptop Maze Lab: enter Project ID + API key → click Connect

---

## Changes to make

### 1. Delete the `/ei-phone` route entirely

Remove `src/pages/EiPhone.tsx` and its router registration. It is no longer needed.

---

### 2. Replace `EdgeImpulsePanel` with this new implementation

The panel now has three inputs (Project ID, API Key, Channel) and a Connect button.
On connect it:
1. Fetches the list of online devices for the project
2. Starts an inference debug stream on the first online device
3. Opens EI's WebSocket and receives live classification results
4. Forwards the top result into the CEEO relay so `channel.msg` works

Replace the full contents of `EdgeImpulsePanel.tsx` with:

```tsx
import { useState, useEffect, useRef } from "react";

const RELAY = "wss://@chrisrogers.pyscriptapps.com/talking-on-a-channel/api/channels/hackathon";
const EI_BASE = "https://studio.edgeimpulse.com/v1/api";

function toPred(legoTopic: string) {
  const cmd = (legoTopic || "/LEGO").trim().replace(/\/+$/, "") || "/LEGO";
  const t = cmd.replace(/^\/LEGO\b/, "/PRED");
  return t === cmd ? "/PRED" : t;
}

type Pred = { c: string; p: number };
type Status = "idle" | "connecting" | "live" | "error";

export function EdgeImpulsePanel() {
  const [open, setOpen]         = useState(true);
  const [projectId, setProjectId] = useState("");
  const [apiKey, setApiKey]     = useState("");
  const [channel, setChannel]   = useState("/LEGO");
  const [status, setStatus]     = useState<Status>("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [topClass, setTopClass] = useState("none");
  const [topConf, setTopConf]   = useState(0);
  const [bars, setBars]         = useState<Pred[]>([]);

  const eiWsRef     = useRef<WebSocket | null>(null);
  const relayWsRef  = useRef<WebSocket | null>(null);
  const keepAliveRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pingRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamIdRef = useRef<number | null>(null);
  const activeRef   = useRef(false);

  // Colours matching maze.css tokens
  const C = {
    ink: "#e2e8f0", muted: "#64748b", accent: "#3b82f6",
    signal: "#22d3ee", recv: "#a78bfa", border: "#2a2d3a",
    bg: "#0f1117", green: "#22c55e", red: "#ef4444",
  };

  const msg = (s: Status, m: string) => { setStatus(s); setStatusMsg(m); };

  async function connect() {
    if (!projectId || !apiKey) {
      msg("error", "Enter Project ID and API key first."); return;
    }
    activeRef.current = true;
    msg("connecting", "Fetching devices…");

    // 1. Get device list — find first online device
    let deviceId: string;
    try {
      const res = await fetch(`${EI_BASE}/${projectId}/devices`, {
        headers: { "x-api-key": apiKey },
      });
      const data = await res.json();
      if (!data.success) { msg("error", data.error || "Devices fetch failed"); return; }
      const online = (data.devices as any[]).find(d => d.remote_mgmt_connected);
      if (!online) {
        msg("error", "No phone connected. Open EI Studio → Devices → Connect → Mobile phone, scan QR, then tap 'Switch to classification mode'.");
        return;
      }
      deviceId = online.deviceId;
    } catch (e: any) {
      msg("error", "Network error: " + e.message); return;
    }

    // 2. Start inference debug stream
    msg("connecting", "Starting inference stream…");
    let socketToken: string;
    try {
      // Get socket token first
      const tokRes = await fetch(`${EI_BASE}/${projectId}/socket-token`, {
        headers: { "x-api-key": apiKey },
      });
      const tokData = await tokRes.json();
      if (!tokData.success) { msg("error", "Socket token failed: " + tokData.error); return; }
      socketToken = tokData.token.socketToken;

      // Start the stream
      const streamRes = await fetch(
        `${EI_BASE}/${projectId}/device/${deviceId}/debug-stream/inference/start`,
        { method: "POST", headers: { "x-api-key": apiKey } }
      );
      const streamData = await streamRes.json();
      if (!streamData.success) { msg("error", "Stream start failed: " + streamData.error); return; }
      streamIdRef.current = streamData.streamId;
    } catch (e: any) {
      msg("error", "Stream error: " + e.message); return;
    }

    // 3. Open EI WebSocket
    msg("connecting", "Opening EI stream…");
    const eiWs = new WebSocket(
      `wss://studio.edgeimpulse.com/socket.io/?token=${socketToken}&EIO=4&transport=websocket`
    );
    eiWsRef.current = eiWs;

    // Socket.IO handshake
    eiWs.onopen = () => {
      eiWs.send("40"); // Socket.IO connect packet
      // Ping every 25s to keep alive
      pingRef.current = setInterval(() => eiWs.readyState === 1 && eiWs.send("2"), 25000);
    };

    eiWs.onmessage = (ev) => {
      const raw: string = ev.data;
      if (raw.startsWith("0") || raw.startsWith("40")) return; // handshake
      if (raw === "2") { eiWs.send("3"); return; } // server ping → pong
      if (!raw.startsWith("42")) return;
      try {
        const [event, payload] = JSON.parse(raw.slice(2));
        if (event !== "inference-results" && event !== "inference") return;
        // Payload shape: { result: { classification: { label: value, ... } } }
        // or: { result: [ { label, value }, ... ] }
        let list: Pred[] = [];
        const r = payload?.result;
        if (r?.classification && typeof r.classification === "object") {
          list = Object.entries(r.classification as Record<string, number>)
            .map(([c, p]) => ({ c, p: Number(p) }))
            .sort((a, b) => b.p - a.p);
        } else if (Array.isArray(r)) {
          list = (r as any[])
            .map(x => ({ c: x.label, p: Number(x.value) }))
            .sort((a, b) => b.p - a.p);
        }
        if (!list.length) return;
        setBars(list);
        const top = list[0];
        setTopClass(top.c);
        setTopConf(Math.round(top.p * 100));
        msg("live", "");
        // Forward to CEEO relay
        publishToRelay(list);
      } catch {}
    };

    eiWs.onerror = () => msg("error", "EI WebSocket error");
    eiWs.onclose = () => { if (activeRef.current) msg("error", "EI stream closed"); };

    // 4. Open CEEO relay WebSocket for forwarding
    const relayWs = new WebSocket(RELAY);
    relayWsRef.current = relayWs;
    relayWs.onopen = () => msg("live", "Live");

    // 5. Keep-alive ping to EI every 9s
    keepAliveRef.current = setInterval(async () => {
      if (!activeRef.current || streamIdRef.current === null) return;
      try {
        await fetch(
          `${EI_BASE}/${projectId}/device/${deviceId}/debug-stream/inference/keepalive`,
          { method: "POST", headers: { "x-api-key": apiKey } }
        );
      } catch {}
    }, 9000);
  }

  function publishToRelay(list: Pred[]) {
    const ws = relayWsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const predTopic = toPred(channel);
    ws.send(JSON.stringify({
      type: "data",
      payload: JSON.stringify({ topic: predTopic, value: JSON.stringify(list) }),
    }));
  }

  function disconnect() {
    activeRef.current = false;
    clearInterval(keepAliveRef.current!);
    clearInterval(pingRef.current!);
    eiWsRef.current?.close();
    relayWsRef.current?.close();
    setBars([]); setTopClass("none"); setTopConf(0);
    msg("idle", "");
  }

  // Cleanup on unmount
  useEffect(() => () => { activeRef.current = false; disconnect(); }, []);

  const statusColor = status === "live" ? C.green : status === "error" ? C.red : C.muted;

  return (
    <div style={{ borderBottom: `1px solid ${C.border}` }}>
      <button onClick={() => setOpen(p => !p)} style={{
        width: "100%", textAlign: "left", background: "none", border: "none",
        padding: "0.5rem 0.75rem", cursor: "pointer", color: C.ink,
        fontSize: "0.8rem", fontWeight: 600, display: "flex", justifyContent: "space-between",
      }}>
        Edge Impulse <span>{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div style={{ padding: "0 0.75rem 0.75rem" }}>

          {/* Instructions */}
          <p style={{ color: C.muted, fontSize: "0.72rem", marginBottom: "0.6rem", lineHeight: 1.5 }}>
            1. In EI Studio → <strong style={{ color: C.ink }}>Devices</strong> → Connect → Mobile phone → scan QR<br />
            2. On phone tap <strong style={{ color: C.ink }}>"Switch to classification mode"</strong><br />
            3. Enter your project ID + API key below and click Connect.
          </p>

          {/* Inputs */}
          {(["Project ID", "API Key", "Channel"] as const).map((label, i) => {
            const val  = [projectId, apiKey, channel][i];
            const set  = [setProjectId, setApiKey, setChannel][i];
            return (
              <div key={label} style={{ marginBottom: "0.35rem" }}>
                <div style={{ fontSize: "0.68rem", color: C.muted, marginBottom: "0.15rem" }}>{label}</div>
                <input
                  type={label === "API Key" ? "password" : "text"}
                  value={val}
                  onChange={e => set(e.target.value)}
                  disabled={status === "live" || status === "connecting"}
                  style={{
                    width: "100%", background: C.bg, border: `1px solid ${C.border}`,
                    color: C.ink, borderRadius: 4, padding: "0.25rem 0.4rem",
                    fontSize: "0.78rem", boxSizing: "border-box" as const,
                  }}
                />
              </div>
            );
          })}

          {/* Connect / Disconnect */}
          <div style={{ display: "flex", gap: "0.4rem", margin: "0.5rem 0" }}>
            {status !== "live" && status !== "connecting" ? (
              <button onClick={connect} style={{
                flex: 1, background: C.accent, border: "none", color: "#fff",
                borderRadius: 4, padding: "0.3rem", cursor: "pointer", fontSize: "0.78rem",
              }}>Connect</button>
            ) : (
              <button onClick={disconnect} style={{
                flex: 1, background: C.red, border: "none", color: "#fff",
                borderRadius: 4, padding: "0.3rem", cursor: "pointer", fontSize: "0.78rem",
              }}>{status === "connecting" ? "Connecting…" : "Disconnect"}</button>
            )}
          </div>

          {/* Status */}
          {statusMsg && (
            <div style={{ fontSize: "0.72rem", color: statusColor, marginBottom: "0.4rem", lineHeight: 1.4 }}>
              {statusMsg}
            </div>
          )}

          {/* Detected / Confidence */}
          {bars.length > 0 && (
            <>
              <div style={{ display: "flex", gap: "1rem", marginBottom: "0.4rem" }}>
                <span style={{ fontSize: "0.75rem", color: C.muted }}>
                  Detected <strong style={{ color: C.ink }}>{topClass}</strong>
                </span>
                <span style={{ fontSize: "0.75rem", color: C.muted }}>
                  Confidence <strong style={{ color: C.ink }}>{topConf}%</strong>
                </span>
              </div>
              {bars.map((b, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.25rem" }}>
                  <span style={{
                    width: 80, fontSize: "0.7rem", color: C.ink, flexShrink: 0,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>{b.c}</span>
                  <div style={{ flex: 1, background: C.border, borderRadius: 3, height: 10 }}>
                    <div style={{
                      width: Math.round(b.p * 100) + "%", height: "100%",
                      background: C.recv, borderRadius: 3, transition: "width 0.2s",
                    }} />
                  </div>
                  <span style={{ width: 32, textAlign: "right", fontSize: "0.7rem", color: C.muted }}>
                    {Math.round(b.p * 100)}%
                  </span>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
```

---

### 3. Update `LibraryPanel.tsx`

Replace the Edge Impulse entry in `CATEGORIES` description with:

```
Edge Impulse (phone) ▶
Connect your phone via EI Studio → Devices → Mobile phone.
Tap "Switch to classification mode" on the phone.
Enter your Project ID and API key in the Edge Impulse panel and click Connect.
The top prediction is available as: channel.msg
```

---

### 4. Update `MazeLab.tsx`

- Remove the import of `setupTeachable` and its `useEffect` call — the new `EdgeImpulsePanel` manages its own WebSocket lifecycle internally.
- Remove the import of `TeachableMachinePanel` and replace it with `EdgeImpulsePanel`.
- Remove any reference to `setupTeachable`, `teachable.ts` is now unused.

---

### 5. Files that do NOT change

- `teachable.ts` — can be deleted (no longer used)
- `channel.ts` — untouched
- `maze.css` — untouched
- `ChannelsPanel` — untouched

---

## Important implementation note on the EI WebSocket message format

EI's WebSocket uses Socket.IO framing. Messages arriving as `42[event, payload]`
are the ones that matter. The event name for inference results may be
`"inference-results"` or `"inference"` — handle both. Log the raw messages to
the console on first connect so you can confirm the exact event name for this
project's device type if the bars don't update.

Add this temporarily in `eiWs.onmessage` for debugging:
```js
if (raw.startsWith("42")) console.log("EI WS:", raw);
```
Remove once confirmed working.
