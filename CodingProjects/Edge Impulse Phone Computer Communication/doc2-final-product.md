# Maze Lab — Final Product & Outstanding Issues

## What exists right now

Two self-contained HTML files. No build step, no framework, no backend.

---

## File 1: `maze-lab.html` (laptop)
Open by double-clicking in Chrome or Safari. Works immediately.

**Full source:** see attached `maze-lab.html`

**What works:**
- UI renders correctly (top bar, device bar, editor, library, EI panel, channels)
- CEEO relay connects automatically on load
- Channels panel sends/receives messages over relay
- Edge Impulse panel listens on `/PRED` topic and will render bars when phone sends data
- `window.channelMsg` is updated whenever a prediction arrives (PyScript hook)

**PyScript (LEGO execution) is stubbed out.** The original `main.py` and `pyscript.toml` from the Loveable project need to be dropped alongside this file and the PyScript lines uncommented:
```html
<!-- Uncomment these two lines to enable real LEGO execution: -->
<script defer src="https://pyscript.net/releases/2024.1.1/core.js"></script>
<script type="py" src="/maze/main.py" config="/maze/pyscript.toml"></script>
```

---

## File 2: `ei-phone.html` (phone)
Must be hosted — open `file://` won't work for camera on iPhone.  
**Recommended host:** GitHub Pages (free, 2-minute setup).

**Full source:** see attached `ei-phone.html`

**What works (in theory — untested end-to-end):**
- Camera access via `getUserMedia`
- CEEO relay WebSocket connection
- Publishes in exact `[{c,p}]` format the laptop expects
- Reconnects relay automatically on drop

---

## Outstanding issues / untested items

### 1. 🔴 WASM classifier — UNTESTED
The `EdgeImpulseClassifier` API used in the phone page is based on EI's documented standalone JS format, but **has not been tested with a real WASM build.**

Specifically unverified:
- Whether `new EdgeImpulseClassifier()` is the correct constructor name
- Whether `.classify(features)` accepts a flat `0xRRGGBB` array at 96×96
- Whether `result.results[i].label` and `result.results[i].value` are the correct field names
- Whether the `.wasm` file loads correctly when `edge-impulse-standalone.js` is on a different origin than the page

**How to debug:** Open phone page in Safari on iPhone, connect to Mac via Safari → Develop → [your phone] → inspect. Watch console for errors on model load and first classify call.

**Likely fix if classify format is wrong:** EI's standalone JS may expect raw float features (0.0–1.0) instead of packed 0xRRGGBB integers. If bars don't appear, change the feature packing in the phone page:
```js
// Current (packed RGB):
features.push((r << 16) | (g << 8) | b);

// Alternative (normalised float, try if above fails):
features.push(r / 255, g / 255, b / 255);
```

### 2. 🟡 Stage 2 relay pub/sub — NOT RUN
Confirmed relay connects (Stage 1 ✅) but two-tab message passing hasn't been verified. Run this before testing the phone:
- Open `maze-lab.html` in two browser tabs
- Send a message from the Channels panel in tab 1
- Confirm it appears in tab 2's log

### 3. 🟡 PyScript / LEGO execution — STUBBED
Run/Stop buttons print a placeholder message. Real LEGO execution requires:
- `public/maze/main.py` from the original project placed alongside `maze-lab.html`
- `public/maze/pyscript.toml` likewise
- The two commented-out `<script>` lines in `maze-lab.html` uncommented
- The file served from a local server (PyScript won't load over `file://`)

Quick local server: `cd` to the folder containing `maze-lab.html` and run:
```bash
python3 -m http.server 8080
# then open http://localhost:8080/maze-lab.html
```

### 4. 🟡 GitHub Pages CORS for WASM
The `.wasm` file must be served with `Content-Type: application/wasm`. GitHub Pages does this correctly by default — but if using a different static host, verify this header or the browser will refuse to load it.

### 5. 🟢 Channel field removed from EI panel
Earlier versions had a redundant "Channel" input inside the Edge Impulse panel alongside the Channels panel below. The current `maze-lab.html` removes this — the EI panel derives the `/PRED` topic automatically from whatever channel is set in the Channels panel.  
**Status: fixed in current build.**

---

## Next test sequence

1. Run Stage 2 (two-tab relay test) in `maze-lab.html`
2. Build EI WASM, host on GitHub Pages, open `ei-phone.html` on iPhone
3. Watch browser console on phone for classifier errors
4. If bars appear on laptop → end-to-end works
5. Uncomment PyScript lines and serve locally to test LEGO execution
