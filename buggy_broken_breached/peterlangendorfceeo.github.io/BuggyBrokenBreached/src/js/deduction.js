window.openDeduction = function() {
    document.getElementById('deduction-overlay').style.display = 'flex';
    const container = document.getElementById('deduction-list');
    const submitBtn = document.getElementById('btn-submit-report');
    
    if (container.children.length === 0) {
        deductionState = [];
        let moveIndex = 1;
        
        auditPairs.forEach((pair, index) => {
            // THE FIX: Set valid to TRUE by default
            deductionState.push({ valid: true, breached: false, broken: false });
            
            let isMove = pair.isMove;
            let displayValue = pair.pItem ? pair.pItem.value : "Unknown";
            let displayLabel = "Unknown";
            let isRealRobotMove = isMove && !pair.rItem.virtual;
            
            if (isMove) { displayLabel = `Instruction [${moveIndex}]`; moveIndex++; } 
            else { displayLabel = pair.pItem.label; }
            
            let div = document.createElement('div');
            div.className = 'deduction-row';
            
            let html = `<div class="deduction-label" style="width: 30%;">${displayLabel}<br><span style="font-size: 0.8em; color: #888;">${displayValue}</span></div>`;
            
            // THE FIX: Appended 'active-valid' default styling class to the VALID button so it appears highlighted immediately
            html += `<div class="deduction-group" style="flex: 1; display: flex; gap: 10px;">
                <button class="btn-deduct active-valid" id="btn-val-${index}" onclick="window.toggleDeduct(${index}, 'valid')">VALID</button>
                <button class="btn-deduct" id="btn-bre-${index}" onclick="window.toggleDeduct(${index}, 'breached')">BREACHED</button>`;
            
            if (isRealRobotMove) { html += `<button class="btn-deduct" id="btn-bro-${index}" onclick="window.toggleDeduct(${index}, 'broken')">BROKEN</button>`; } 
            else { html += `<div style="flex: 1; border: 1px dashed transparent;"></div>`; }
            
            html += `</div>`; div.innerHTML = html; container.appendChild(div);
        });
    }

    let allSelected = false;
    if (deductionState && deductionState.length > 0) {
        allSelected = true;
        for (let i = 0; i < deductionState.length; i++) {
            if (!deductionState[i].valid && !deductionState[i].breached && !deductionState[i].broken) { 
                allSelected = false; 
                break; 
            }
        }
    }
    
    if (submitBtn) { 
        submitBtn.disabled = !allSelected; 
        submitBtn.style.opacity = allSelected ? '1' : '0.5'; 
    }
}

window.closeDeduction = function() { document.getElementById('deduction-overlay').style.display = 'none'; }

window.toggleDeduct = function(index, type) {
    let state = deductionState[index];
    
    if (type === 'valid') { if (!state.valid) { state.valid = true; state.breached = false; state.broken = false; } else { state.valid = false; } } 
    else if (type === 'breached') { state.breached = !state.breached; if (state.breached) state.valid = false; } 
    else if (type === 'broken') { state.broken = !state.broken; if (state.broken) state.valid = false; }

    document.getElementById(`btn-val-${index}`).className = 'btn-deduct' + (state.valid ? ' active-valid' : '');
    document.getElementById(`btn-bre-${index}`).className = 'btn-deduct' + (state.breached ? ' active-breached' : '');
    
    let broBtn = document.getElementById(`btn-bro-${index}`);
    if (broBtn) { broBtn.className = 'btn-deduct' + (state.broken ? ' active-broken' : ''); }

    let allSelected = true;
    for (let i = 0; i < deductionState.length; i++) {
        if (!deductionState[i].valid && !deductionState[i].breached && !deductionState[i].broken) { allSelected = false; break; }
    }
    
    const submitBtn = document.getElementById('btn-submit-report');
    if (submitBtn) { submitBtn.disabled = !allSelected; submitBtn.style.opacity = allSelected ? '1' : '0.5'; }
}

window.submitDeduction = function() {
    if (window.stopTimer) window.stopTimer();

    let errors = [];
    let rIndexTracker = 0; 
    let doubleTroubleCatches = 0;
    
    for (let i = 0; i < auditPairs.length; i++) {
        let pair = auditPairs[i];
        let state = deductionState[i];
        
        if (!state.valid && !state.breached && !state.broken) { 
            state.valid = true; 
        }
        
        let trueBreached = false; let trueBroken = false; let label = "";
        let isRealRobotMove = pair.isMove && !pair.rItem.virtual;
        
        if (pair.isMove) {
            label = pair.pItem ? pair.pItem.label : "Instruction";
            let pVal = pair.pItem.value;
            let rVal = pair.rItem.value;
            trueBreached = (pVal !== rVal);

            if (isRealRobotMove) { trueBroken = (window.runtimeFailures && window.runtimeFailures[rIndexTracker] === true); rIndexTracker++; }
        } else {
            label = pair.pItem.label; trueBreached = (pair.pItem.value !== pair.rItem.value);
        }
        
        let trueValid = !trueBreached && !trueBroken;
        let rowCorrect = true;
        
        if (state.breached !== trueBreached) { errors.push(`[${label}] Breach status was incorrect. (Was actually ${trueBreached ? 'HACKED' : 'SAFE'})`); rowCorrect = false; }
        if (isRealRobotMove && state.broken !== trueBroken) { errors.push(`[${label}] Motor status was incorrect. (Was actually ${trueBroken ? 'FAILED' : 'WORKING'})`); rowCorrect = false; }
        if (state.valid !== trueValid && state.breached === trueBreached && state.broken === trueBroken) { errors.push(`[${label}] should have been marked VALID.`); rowCorrect = false; }

        if (trueBreached && trueBroken && rowCorrect) { doubleTroubleCatches++; }
    }
    
    window.closeDeduction(); document.getElementById('result-overlay').style.display = 'flex';
    const title = document.getElementById('result-title'); const desc = document.getElementById('result-desc');
    
    if (errors.length === 0) {
        title.innerText = "PROGRAMMER WINS!"; title.className = "programmer-win"; desc.style.borderColor = "var(--neon-green)";
        desc.innerHTML = "<span style='color: var(--neon-green); font-weight: bold;'>SYSTEM SECURED.</span>\n\nExcellent work! You correctly analyzed the Audit and Sensor logs to uncover the Hacker's payload and identify all hardware failures. The robot is safe!";
    } else {
        title.innerText = "HACKER WINS!"; title.className = "hacker-win"; desc.style.borderColor = "var(--neon-red)";
        let errorText = "<span style='color: var(--neon-red); font-weight: bold;'>BREACH SUCCESSFUL.</span>\nYour report was incorrect! The Hacker slipped through your defenses.\n\n<span style='color: white;'>ERRORS FOUND:</span>\n";
        errors.forEach(e => { errorText += "❌ " + e + "\n"; }); desc.innerHTML = errorText;
    }

    if (window.computeScore && window.playScoreTally) {
        const instructionCount = programmerState.filter(x => x.id.startsWith('MOVE_')).length;
        const scoreData = window.computeScore(instructionCount, errors.length, doubleTroubleCatches);
        window.playScoreTally(scoreData);
    }
}