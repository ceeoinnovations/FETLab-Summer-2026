window.openSensors = function() { document.getElementById('sensor-overlay').style.display = 'flex'; window.drawSensorGraph(); }
window.closeSensors = function() { document.getElementById('sensor-overlay').style.display = 'none'; }

var telemetryTimer = null;
var telemetryHistory = [];
var isRecording = false;
var recordStartMs = 0;
var graphMode = 'normal'; 
window.currentMoveLabel = "IDLE";
var t_leftSpeed = 0;
var t_rightSpeed = 0;

window.setCurrentTargetSpeeds = function(left, right) { t_leftSpeed = left; t_rightSpeed = right; };
window.toggleGraphMode = function() {
    if (window.DIFFICULTY === 'hard') {
        alert("ACCESS DENIED: Advanced Raw Telemetry is heavily encrypted on HARD difficulty.");
        return;
    }
    
    graphMode = graphMode === 'normal' ? 'advanced' : 'normal';
    const btn = document.getElementById('btn-toggle-graph');
    if (graphMode === 'normal') {
        btn.innerHTML = "Mode:<br>Normal (Simplified)"; btn.className = "btn-success btn-toggle-normal";
    } else {
        btn.innerHTML = "Mode:<br>Advanced (Raw Data)"; btn.className = "btn-success btn-toggle-advanced";
    }
    window.drawSensorGraph(); 
}

window.startRecording = function() { telemetryHistory = []; isRecording = true; recordStartMs = Date.now(); };
window.stopRecording = function() { isRecording = false; };

window.startTelemetry = function() {
    if (!telemetryTimer) {
        telemetryTimer = setInterval(() => {
            if(window.legoBluetooth) {
                let left = window.legoBluetooth.getMotorSpeed(1); let right = window.legoBluetooth.getMotorSpeed(2);
                let lVal = left !== null ? left : 0; let rVal = right !== null ? right : 0;
                
                if (isRecording) {
                    telemetryHistory.push({ t: Date.now() - recordStartMs, left: lVal, right: rVal, move: window.currentMoveLabel, tgtLeft: t_leftSpeed, tgtRight: t_rightSpeed });
                }
            }
        }, 50); 
    }
}

window.drawSensorGraph = function() {
    const canvas = document.getElementById('sensor-canvas');
    const ctx = canvas.getContext('2d');
    const h = 400; const pad = 40; const leftPad = 60; 
    
    let regions = [];
    let currentRegion = null;
    for (let pt of telemetryHistory) {
        if (currentRegion && currentRegion.move === pt.move) { currentRegion.end = pt.t; } 
        else {
            if (currentRegion) regions.push(currentRegion);
            currentRegion = { move: pt.move, start: pt.t, end: pt.t };
        }
    }
    if (currentRegion) regions.push(currentRegion);

    for (let r of regions) {
        if (r.move === 'IDLE' || r.move === 'END') continue;
        let pts = telemetryHistory.filter(p => p.t >= r.start && p.t <= r.end);
        let lastActive = pts.findLast(p => Math.abs(p.left) > 0 || Math.abs(p.right) > 0);
        if (lastActive) {
            r.visualEnd = Math.min(r.end, lastActive.t + 100);
        } else {
            r.visualEnd = Math.min(r.end, r.start + 300);
        }
    }

    let displayRegions = regions.filter(r => r.move !== 'IDLE' && r.move !== 'END');
    let targetWidth = 900;
    if (displayRegions.length > 6) { targetWidth = leftPad + pad + (displayRegions.length * 130); }
    
    if (canvas.width !== targetWidth) canvas.width = targetWidth;
    else ctx.clearRect(0, 0, canvas.width, h);
    const w = canvas.width;

    if (telemetryHistory.length === 0) {
        ctx.fillStyle = '#888';
        ctx.font = '16px Share Tech Mono'; ctx.textAlign = 'center';
        ctx.fillText("No data recorded yet. Execute Payload first.", w/2, h/2); return;
    }
    
    const maxT = Math.max(1, telemetryHistory[telemetryHistory.length - 1].t);
    
    function mapX(t, regIndex) {
        if (window.DIFFICULTY === 'hard' && regIndex !== undefined) {
            let segmentW = (w - leftPad - pad) / displayRegions.length;
            let reg = displayRegions[regIndex];
            let totalT = reg.visualEnd - reg.start;
            let progress = totalT <= 0 ? 0 : (t - reg.start) / totalT;
            if (progress < 0) progress = 0; if (progress > 1) progress = 1;
            return leftPad + (regIndex * segmentW) + (progress * segmentW);
        }
        return leftPad + (t / maxT) * (w - leftPad - pad);
    }
    
    ctx.font = '12px Share Tech Mono'; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
    for (let speed = -100; speed <= 100; speed += 20) {
        const y = h/2 - (speed / 100) * (h/2 - pad);
        ctx.strokeStyle = speed === 0 ? '#444' : 'rgba(255, 255, 255, 0.1)'; ctx.lineWidth = speed === 0 ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(leftPad, y); ctx.lineTo(w - pad, y); ctx.stroke();
        ctx.fillStyle = speed === 0 ? '#fff' : '#888';
        ctx.fillText(speed, leftPad - 10, y);
    }

    ctx.font = '14px Share Tech Mono'; ctx.textBaseline = 'bottom'; ctx.textAlign = 'center';
    
    for (let i = 0; i < displayRegions.length; i++) {
        let reg = displayRegions[i];
        let startX = mapX(reg.start, i);
        let endX = mapX(reg.visualEnd, i);

        ctx.strokeStyle = '#555'; ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(startX, pad); ctx.lineTo(startX, h - pad); 
        ctx.moveTo(endX, pad); ctx.lineTo(endX, h - pad); 
        ctx.stroke(); ctx.setLineDash([]);
        
        let arrowY = pad / 2 + 15; ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'; ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'; ctx.lineWidth = 1;

        if (endX - startX > 30) {
            ctx.beginPath();
            ctx.moveTo(startX + 10, arrowY); ctx.lineTo(endX - 10, arrowY); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(startX + 10, arrowY); ctx.lineTo(startX + 15, arrowY - 4); ctx.lineTo(startX + 15, arrowY + 4); ctx.fill();
            ctx.beginPath(); ctx.moveTo(endX - 10, arrowY); ctx.lineTo(endX - 15, arrowY - 4); ctx.lineTo(endX - 15, arrowY + 4); ctx.fill();
        }
        ctx.fillText(reg.move, (startX + endX) / 2, arrowY - 5);
    }

    if (graphMode === 'normal') {
        function drawSimplifiedLine(key, color) {
            ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.beginPath();
            ctx.moveTo(leftPad, h/2);
            
            for (let i = 0; i < displayRegions.length; i++) {
                let reg = displayRegions[i];
                let pts = telemetryHistory.filter(p => p.t >= reg.start && p.t <= reg.end);
                
                let maxSpd = 0;
                for (let p of pts) {
                    let tgt = key === 'left' ? p.tgtLeft : p.tgtRight;
                    if (Math.abs(tgt) > Math.abs(maxSpd)) maxSpd = tgt;
                }
                
                let amp = 0;
                if (maxSpd !== 0) {
                    // THE FIX: Treats the Speed Round graph identically to Easy!
                    if (window.DIFFICULTY === 'easy' || window.DIFFICULTY === 'speed') {
                        amp = (key === 'left') ? maxSpd : -maxSpd;
                    } else {
                        amp = (key === 'left') ? 100 : -100;
                    }
                }
                
                let activeDuration = reg.visualEnd - reg.start; 
                let ramp = window.DIFFICULTY === 'hard' ? 0 : Math.min(100, activeDuration * 0.2);

                let x1 = mapX(reg.start, i);
                ctx.lineTo(x1, h/2);
                
                let x2 = mapX(reg.start + ramp, i);
                let y2 = h/2 - (amp / 100) * (h/2 - pad);
                ctx.lineTo(x2, y2);
                
                let x3 = mapX(reg.visualEnd - ramp, i);
                ctx.lineTo(x3, y2);
                
                let x4 = mapX(reg.visualEnd, i);
                ctx.lineTo(x4, h/2);
            }
            
            ctx.lineTo(w - pad, h/2);
            ctx.stroke();
        }
        drawSimplifiedLine('left', '#00ff41'); drawSimplifiedLine('right', '#00f0ff');
    } else {
        function drawRawLine(key, color) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
            for (let i = 0; i < telemetryHistory.length; i++) {
                const pt = telemetryHistory[i];
                let regIdx = displayRegions.findIndex(r => pt.t >= r.start && pt.t <= r.end);
                if (regIdx === -1) regIdx = undefined;
                
                const x = mapX(pt.t, regIdx);
                const yOffset = (key === 'right') ? 4 : 0;
                const y = h/2 - (pt[key] / 100) * (h/2 - pad) + yOffset;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
        drawRawLine('left', '#00ff41'); drawRawLine('right', '#00f0ff');
    }
}