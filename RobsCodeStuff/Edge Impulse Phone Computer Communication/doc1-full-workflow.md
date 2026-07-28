# Maze Lab — Full Project Workflow

## Goal
Replace Teachable Machine with Edge Impulse in a LEGO Python web app.
Phone classifies images → result sent to laptop → Python reads `channel.msg` → drives LEGO.

---

## Original Architecture (Teachable Machine)

```
Phone (Teachable Machine) → CEEO relay (WebSocket) → Laptop (subscribes to /PRED topic) → Python
```

**Relay:** `wss://@chrisrogers.pyscriptapps.com/talking-on-a-channel/api/channels/hackathon`  
**Message format:** `[{"c":"classname","p":0.95}, ...]` on a `/PRED` topic variant  
**Key function** (`teachable.ts`): `toPred("/LEGO")` → `"/PRED"` (or `"/PRED/suffix"`)

---

## Phase 1 — Loveable duplication + text swap
**What happened:** Loveable duplicated the original PyScript app and swapped "Teachable Machine" labels to "Edge Impulse."  
**Result:** Labels changed correctly. DOM IDs preserved (`tm-current-class`, `tm-current-confidence`, etc.).  
**Problem:** `/maze/ei-phone.html` was created as a static file but React's router intercepted the path — inaccessible on iPhone.

---

## Phase 2 — Custom phone page attempt (REST API approach)
**Approach:** Build a phone page that captures camera frames, POSTs to EI's REST API (`/v1/api/classify-image`), publishes results to CEEO relay.  
**Result:** Got HTTP 403. EI's hosted inference endpoint requires a paid/enterprise plan — not available on free tier.  
**Dead end.**

---

## Phase 3 — EI debug stream approach (no phone page)
**Approach:** Use EI Studio's own mobile client (`smartphone.edgeimpulse.com`) on the phone. Laptop connects to EI's WebSocket inference debug stream directly.  
**API flow:**
1. `GET /v1/api/{projectId}/devices` → find online device
2. `GET /v1/api/{projectId}/socket-token` → get WS token
3. `POST /v1/api/{projectId}/device/{deviceId}/debug-stream/inference/start`
4. Open `wss://studio.edgeimpulse.com/socket.io/?token=XXX&EIO=4&transport=websocket`

**Result:** CORS block. Browser can't call EI's API directly from a Loveable-hosted frontend. Would need a backend proxy (Supabase Edge Function).  
**Dead end with Loveable.**

---

## Phase 4 — Leaving Loveable
**Decision:** Loveable was bottlenecking progress (missed changes, routing problems, CORS walls, each fix requiring a new instruction file). Moved to self-contained HTML files.

---

## Phase 5 — Final architecture (WASM on phone)

```
Phone (ei-phone.html) → loads EI WASM model → camera inference on-device
    → publishes [{c,p}] to CEEO relay on /PRED topic
Laptop (maze-lab.html) → subscribes to relay → renders bars → channel.msg available to Python
```

**No API calls. No CORS. No backend. No proxy.**  
Only requirement: EI WASM files hosted on a static host (GitHub Pages).

### WASM setup (one-time per EI project)
1. EI Studio → Deployment → WebAssembly → Build → download zip
2. Upload `edge-impulse-standalone.js` + `edge-impulse-standalone.wasm` to GitHub Pages
3. Paste base URL into phone page (e.g. `https://yourname.github.io/model/`)

---

## Files produced

### `maze-lab.html` — laptop app
Open directly in browser, no server needed. Key sections:
- **Top bar** — "Maze Lab · drive · detect · navigate" with pulse dot
- **Device bar** — add dm/sm/cs/c chips
- **Code editor** — textarea + Run/Stop (PyScript hooks commented in)
- **Library panel** — collapsible function reference
- **Edge Impulse panel** — subscribes to `/PRED` topic, renders live bars
- **Channels panel** — pub/sub on CEEO relay, exposes `window.channelMsg`

Relay connection logic mirrors `teachable.ts` exactly:
```js
function toPred(legoTopic) {
  const cmd = (legoTopic || "/LEGO").trim().replace(/\/+$/, "") || "/LEGO";
  const t = cmd.replace(/^\/LEGO\b/, "/PRED");
  return t === cmd ? "/PRED" : t;
}
// Listens on relay; when inner.topic === predTopic, parses [{c,p}] and renders bars
```

### `ei-phone.html` — phone page
Host on GitHub Pages; open on iPhone. Key flow:
```js
// 1. Get camera
stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:"environment" }});

// 2. Load WASM (from user-supplied base URL)
await loadScript(wasmBase + "edge-impulse-standalone.js");
classifier = new EdgeImpulseClassifier();
await classifier.init();

// 3. Every 200ms: capture frame → classify → publish to relay
const features = []; // flat array of 0xRRGGBB values from 96×96 canvas
const result = await classifier.classify(features);
const list = result.results.map(r => ({ c: r.label, p: r.value })).sort((a,b)=>b.p-a.p);
ws.send(JSON.stringify({ type:"data", payload: JSON.stringify({ topic: predTopic, value: JSON.stringify(list) })}));
```

---

## Test results so far
| Test | Result |
|------|--------|
| Stage 1: Relay WebSocket connection | ✅ Connected |
| Stage 2: Channel pub/sub (two tabs) | Not yet run |
| Stage 3: EI phone → laptop bars | Blocked (REST API 403, then CORS) |
| WASM phone page end-to-end | **Not yet tested** |
