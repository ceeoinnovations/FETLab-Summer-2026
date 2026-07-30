from pyodide.ffi import create_proxy, to_js
from pyscript import document, window
import random, asyncio, time

is_connected = False
move_id_counter = 0

def normalize_angle(angle): return angle - 360 if angle > 180 else (angle + 360 if angle < -180 else angle)

def print_term(msg, color="lime"):
    t = document.querySelector("#terminal")
    if t: t.innerHTML += f"<span style='color:{color};'>{msg}</span><br>"; t.scrollTop = t.scrollHeight

def set_status(msg, color="lime"):
    t = document.querySelector("#terminal")
    if t: t.innerHTML = f"<span style='color:{color};'>{msg}</span><br>"

def update_status():
    lb = document.querySelector("#move-listbox")
    if not is_connected: set_status("Connect a motor to start the sequence", "yellow")
    elif lb and lb.children.length == 0: set_status("Add a move to begin...", "lime")
    else: set_status("Ready to execute!", "#00ffcc")
    if hasattr(window, 'syncProgrammerButtons'): window.syncProgrammerButtons()

def add_move(move):
    global move_id_counter
    lb = document.querySelector("#move-listbox")
    if not lb: return
    
    cfg = window.DIFF_CONFIG.to_py()[window.DIFFICULTY]
    if window.currentPhase == 1 and lb.children.length >= cfg["pMax"]:
        set_status(f"[PROGRAMMER ERROR] Max sequence length of {cfg['pMax']} reached!", "orange")
        return

    if window.currentPhase == 2:
        sel = document.querySelectorAll(".list-item.selected")
        if sel.length == 1:
            if hasattr(window, 'canHackerChange') and not window.canHackerChange(): return
            if hasattr(window, 'registerHackerChange'): window.registerHackerChange()
            if hasattr(window, 'saveHackerState'): window.saveHackerState()
            item = sel.item(0)
            item.innerText = move.lower()
            item.classList.remove('selected')
            if hasattr(window, 'logHackerAction'): window.logHackerAction(f"mem.ovr @ptr={item.getAttribute('data-move-id')} !val={move.upper()}")
            if hasattr(window, 'syncSettings'): window.syncSettings()
            if hasattr(window, 'syncHackerButtons'): window.syncHackerButtons()
            update_status()
        return

    item = document.createElement("div")
    item.className = "list-item"
    item.draggable = True
    item.innerText = move.lower()
    item.setAttribute("data-move-id", str(move_id_counter))
    move_id_counter += 1
    lb.appendChild(item)
    update_status()

def remove_selected(event):
    sel = document.querySelectorAll(".list-item.selected")
    if sel.length > 0:
        if window.currentPhase == 2:
            if hasattr(window, 'canHackerChange') and not window.canHackerChange(): return
            if hasattr(window, 'registerHackerChange'): window.registerHackerChange()
            if hasattr(window, 'saveHackerState'): window.saveHackerState()
        for i in range(sel.length):
            if window.currentPhase == 2 and hasattr(window, 'logHackerAction'):
                window.logHackerAction(f"mem.free @ptr={sel.item(i).getAttribute('data-move-id')}")
            sel.item(i).remove()
        if hasattr(window, 'syncSettings'): window.syncSettings()
        if hasattr(window, 'syncHackerButtons'): window.syncHackerButtons()
        update_status()

def clear_all(event):
    if window.currentPhase == 2: return 
    lb = document.querySelector("#move-listbox")
    if lb: lb.innerHTML = ""
    update_status()

async def connect_motor(event):
    global is_connected
    print_term("Triggering Web Bluetooth Pairing Menu...", "yellow")
    btn = document.querySelector("#btn-connect")
    btn.innerText = "CONNECTING..."
    btn.setAttribute("style", "width:auto; margin:0; padding:8px 20px; font-size:0.9em; color:var(--neon-cyan) !important; border-color:var(--neon-cyan) !important; opacity:0.8 !important;")
    device_name = await window.legoBluetooth.connectHub()
    
    if not device_name or device_name == False:
        print_term("Connection failed or cancelled.", "red")
        is_connected = False
        btn.innerText = "Connect Motor via Bluetooth"
        btn.removeAttribute("disabled")
        btn.setAttribute("style", "width:auto; margin:0; padding:8px 20px; font-size:0.9em; border-color:var(--neon-cyan); color:var(--neon-cyan); background:transparent;")
        document.querySelector("#btn-begin").setAttribute("disabled", "true")
        return

    is_connected = True
    print_term(f"Handshake complete! Bound to {device_name}.", "#00ffcc")
    await asyncio.sleep(1)
    document.querySelector("#btn-begin").removeAttribute("disabled")
    btn.innerText = "CONNECTED"
    btn.setAttribute("style", "width:auto; margin:0; padding:8px 20px; font-size:0.9em; color:var(--neon-cyan) !important; border-color:var(--neon-cyan) !important; opacity:0.5 !important; background:rgba(0,240,255,0.1) !important;")
    hw_id = document.getElementById("hardware-id")
    if hw_id: hw_id.innerText = str(device_name).upper(); hw_id.style.color = "var(--neon-cyan)"
    update_status()

async def run_sequence(event):
    if not is_connected: update_status(); return
    btn = document.querySelector("#btn-begin")
    btn.setAttribute("disabled", "true")
    btn.innerText = "EXECUTING..."
    btn.setAttribute("style", "color:var(--neon-cyan) !important; border-color:var(--neon-cyan) !important; opacity:0.8 !important; background:transparent !important;")

    for tid in ["#btn-diagnostics", "#btn-sensors", "#btn-deduction"]:
        if document.querySelector(tid): document.querySelector(tid).setAttribute("disabled", "true")

    lb = document.querySelector("#move-listbox")
    moves = [lb.children.item(i) for i in range(lb.children.length)]
    sets = { "forward_speed": document.querySelector("#fwd_spd").value, "backward_speed": document.querySelector("#bck_spd").value, "right_speed": document.querySelector("#rgt_spd").value, "left_speed": document.querySelector("#lft_spd").value, "right_angle": document.querySelector("#rgt_ang").value, "left_angle": document.querySelector("#lft_ang").value }

    fail_chance = window.DIFF_CONFIG.to_py()[window.DIFFICULTY]["failChance"]
    fails = [True if random.randint(1, 100) <= fail_chance else False for _ in moves]
    window.runtimeFailures = to_js(fails)
    print_term("Executing sequence...", "#00ffcc")
    
    if hasattr(window, 'startTelemetry'): window.startTelemetry(); window.startRecording()
    await asyncio.sleep(1)
    L_BIT, R_BIT = window.legoBluetooth.MOTOR_BITS_LEFT, window.legoBluetooth.MOTOR_BITS_RIGHT

    for idx, node in enumerate(moves):
        move = node.innerText.lower()
        if hasattr(window, 'getOriginalInstructionLabel'): window.currentMoveLabel = window.getOriginalInstructionLabel(node.getAttribute("data-move-id"))
        else: window.currentMoveLabel = f"Instr [{idx + 1}]"

        l_fail, r_fail = False, False
        if fails[idx]: 
            if random.randint(1, 2) == 1: l_fail = True
            else: r_fail = True

        try:
            if move in ["forward", "back"]:
                cmds = []
                spd = int(sets.get(f"{move}_speed", 100))
                l_tgt, r_tgt = (spd if not l_fail else 0), (spd if not r_fail else 0)
                if move == "back": l_tgt, r_tgt = -l_tgt, -r_tgt
                if hasattr(window, 'setCurrentTargetSpeeds'): window.setCurrentTargetSpeeds(l_tgt, r_tgt)
                
                fwd_dcw, fwd_dccw, fdeg, bdeg = 0, 1, 864, 900
                ldir = fwd_dcw if move == "forward" else fwd_dccw
                rdir = fwd_dccw if move == "forward" else fwd_dcw
                deg = fdeg if move == "forward" else bdeg

                if not l_fail: cmds.append({"bitMask": L_BIT, "speedPercent": spd, "direction": ldir, "degrees": deg})
                if not r_fail: cmds.append({"bitMask": R_BIT, "speedPercent": spd, "direction": rdir, "degrees": deg})
                if len(cmds) > 0: await window.legoBluetooth.runForDegreesSynced(to_js(cmds, dict_converter=window.Object.fromEntries))

            elif move in ["left", "right"]:
                ang = float(sets.get(f"{move}_angle", -90 if move=="left" else 90))
                spd = int(sets.get(f"{move}_speed", 10))
                b_mask = (L_BIT if not l_fail else 0) | (R_BIT if not r_fail else 0)
                
                if hasattr(window, 'setCurrentTargetSpeeds'): window.setCurrentTargetSpeeds((spd if not l_fail else 0), (spd if not r_fail else 0))
                if b_mask > 0:
                    start_a = window.legoBluetooth.getAngle()
                    if start_a is None: print_term("Error: IMU data unavailable.", "red")
                    else:
                        tgt_a = normalize_angle(start_a + ang)
                        await asyncio.ensure_future(window.legoBluetooth.runMotorContinuous(b_mask, spd, 0 if move == "left" else 1))
                        s_time = time.time()
                        while True:
                            curr = window.legoBluetooth.getAngle()
                            if curr is None or abs(normalize_angle(curr - tgt_a)) < 7.5: break
                            if time.time() - s_time > 15: print_term(f"Turn timeout!", "orange"); break
                            await asyncio.sleep(0.02)
                        await window.legoBluetooth.stopMotor(b_mask)
        except Exception as e: print_term(f"Command failed: {e}", "red")

        window.currentMoveLabel = "IDLE"
        if hasattr(window, 'setCurrentTargetSpeeds'): window.setCurrentTargetSpeeds(0, 0)
        await asyncio.sleep(0.5)

    window.currentMoveLabel = "IDLE"
    if hasattr(window, 'setCurrentTargetSpeeds'): window.setCurrentTargetSpeeds(0, 0)
    if hasattr(window, 'stopRecording'): window.stopRecording()
    print_term("Robot move sequence complete!")
    
    btn.innerText = "EXECUTED"
    btn.setAttribute("style", "color:var(--neon-cyan) !important; border-color:var(--neon-cyan) !important; opacity:0.5 !important; background:rgba(0,240,255,0.1) !important;")
    for tid in ["#btn-diagnostics", "#btn-sensors", "#btn-deduction"]:
        if document.querySelector(tid): document.querySelector(tid).removeAttribute("disabled")
    if hasattr(window, 'startPhase3Timer'): window.startPhase3Timer()

try:
    document.querySelector("#btn-forward").onclick = lambda e: add_move("forward")
    document.querySelector("#btn-back").onclick = lambda e: add_move("back")
    document.querySelector("#btn-left").onclick = lambda e: add_move("left")
    document.querySelector("#btn-right").onclick = lambda e: add_move("right")
    document.querySelector("#btn-remove").onclick = remove_selected
    document.querySelector("#btn-clear").onclick = clear_all
    document.querySelector("#btn-connect").addEventListener("click", create_proxy(lambda e: asyncio.ensure_future(connect_motor(e))))
    document.querySelector("#btn-begin").addEventListener("click", create_proxy(lambda e: asyncio.ensure_future(run_sequence(e))))
    update_status()
except Exception as e: print_term(f"Initialization Error: {e}", "red")
finally:
    if hasattr(window, 'triggerBootSequence'): window.triggerBootSequence()