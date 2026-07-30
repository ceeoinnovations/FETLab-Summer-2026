# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture
This is a single-page web application that uses **PyScript** (Pyodide) to run Python code (`app.py`) directly in the browser. It interfaces with **Web Bluetooth** via `bluetooth.js` to control external hardware (LEGO DoubleMotors).

- **`index.html`**: Contains the UI structure and CSS styles. The UI is split into three phases: Sequence Builder, Attacker Parameters, and Hardware Execution.
- **`app.py`**: The core Python logic. It manages the UI state, user-defined move sequences, and orchestrates calls to the Web Bluetooth API via `window.legoBluetooth`.
- **`bluetooth.js`**: Provides the JavaScript implementation for interacting with LEGO Bluetooth devices.

## Development and Running
This is a static site intended to be served from a web server or GitHub Pages.

- **Local Development**: Since this uses `pyodide` and `Web Bluetooth`, you must serve the files via a local HTTP server to avoid CORS issues and to ensure `Web Bluetooth` APIs are available (which requires a secure context/HTTPS, or `localhost`).
- **Debugging**: The UI includes a `[ System_Logs ]` panel. Errors in `app.py` or JavaScript will be printed here or to the browser's developer console.
- **State**: The application persists user settings and sequences to `window.localStorage`. If you need to reset the app state during development, you can clear the browser's local storage for the site.
