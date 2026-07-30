## Arduino UNO Q — App Lab

### Hardware
- **MCU**: STM32U585, runs Zephyr OS, executes Arduino sketch (C++). Owns GPIO, PWM, sensors, real-time control.
- **MPU**: Qualcomm QRB2210, runs Debian Linux, executes Python script. Owns networking, AI, heavy compute.
- Communication: RPC via `Arduino_RouterBridge` over an internal serial link (never open `/dev/ttyHS1` or `Serial1` directly).

### App File Structure
```
my_app/
├── app.yaml              # Linux-side manifest (required)
├── python/
│   ├── main.py           # Python entry point (required)
│   └── requirements.txt  # pip deps (auto-installed on run)
└── sketch/
    ├── sketch.ino         # Arduino sketch (required)
    └── sketch.yaml        # MCU build config (required)
```
Filenames and directory names are fixed — the runtime looks for them exactly.

### app.yaml
Declares project metadata and lists Bricks. Bricks run as Docker containers on the MPU.
```yaml
name: My App
description: "What it does"
version: "1.0.0"
ports: []
bricks: []   # App Lab populates this when you add Bricks via UI
```

### python/main.py boilerplate
```python
from arduino.app_utils import App, Bridge
import time

def loop():
    time.sleep(1)
    result = Bridge.call("my_mcu_func", arg1, arg2)

App.run(user_loop=loop)  # required — starts Bridge and event loop
```
- `arduino.app_utils` is pre-installed on the board.
- `App.run()` is mandatory; don't run as a plain script.
- Use `Bridge.on("func_name", handler)` to receive notifications from the MCU.

### sketch/sketch.ino boilerplate
```cpp
#include "Arduino_RouterBridge.h"

bool my_mcu_func(bool state) {
    digitalWrite(LED_BUILTIN, state ? LOW : HIGH);
    return state;
}

void setup() {
    Bridge.begin();                          // required
    Bridge.provide("my_mcu_func", my_mcu_func);  // expose to Python
}

void loop() {
    // Bridge.update() is handled automatically
}
```
- `Bridge.begin()` in `setup()` is mandatory.
- `Bridge.provide()` registers functions Python can call via `Bridge.call()`.
- `Bridge.provide_safe()` variant is thread-safe (called from main loop thread via `update_safe()`).
- RGB LEDs 1-2 are MPU-controlled (active low); LEDs 3-4 are MCU-controlled.

### sketch/sketch.yaml
```yaml
profiles:
  default:
    fqbn: arduino:zephyr:unoq
    platforms:
      - platform: arduino:zephyr
    libraries:
      - MsgPack (0.4.2)       # RouterBridge dependency
      - DebugLog (0.8.4)
      - ArxContainer (0.7.0)
      - ArxTypeTraits (0.3.1)
      # Add other libraries here, spelling must be exact
default_profile: default
```
Check exact library names in App Lab's library manager before adding.

### Bridge API summary
| Side | Call | Effect |
|------|------|--------|
| Python | `Bridge.call("func", args...)` | Synchronous RPC to MCU; blocks until response |
| Python | `Bridge.on("func", handler)` | Register handler for MCU notifications |
| Sketch | `Bridge.provide("func", fn)` | Expose function to Python |
| Sketch | `Bridge.call("func", args...).result(var)` | Synchronous RPC to Python |
| Sketch | `Bridge.notify("func", args...)` | Fire-and-forget to Python |

RPC round-trip latency ~8 ms.

### Bricks
Pre-packaged Docker services (AI vision, web UI, time-series DB, etc.) declared in `app.yaml` and imported in Python:
```python
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.video_objectdetection import VideoObjectDetection
ui = WebUI()  # serves on port 7000 by default
```
First run downloads container images; subsequent runs use cache.

### Running / deploying
```bash
# Via App Lab GUI: press Run button
# Via CLI on the board:
arduino-app-cli app start ~/ArduinoApps/my_app
arduino-app-cli app logs  ~/ArduinoApps/my_app
arduino-app-cli app stop  ~/ArduinoApps/my_app
```
Apps live in `~/ArduinoApps/` on the board. Python logs go to a log file, not stdout.
