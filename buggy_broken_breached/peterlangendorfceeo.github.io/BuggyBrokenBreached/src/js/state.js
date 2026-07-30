window.programmerState = [];
window.robotState = [];
window.deductionState = []; 
window.auditPairs = []; 
window.currentPhase = 1; 
window.p1Started = false;
window.isUndoing = false; 
window.hackerHistory = []; 

window.triggerBootSequence = function() {
    if (window.loadRules) window.loadRules();
    
    // Allows the loader to fade smoothly out, THEN triggers the Help Menu to avoid DOM racing
    setTimeout(() => {
        const loader = document.getElementById('loading-screen');
        if (loader) {
            loader.classList.add('fade-out');
            setTimeout(() => {
                loader.style.display = 'none';
                if (window.openHelp) window.openHelp();
            }, 800);
        }
    }, 3000); 
};

window.captureState = function() {
    let state = [];
    const listbox = document.getElementById('move-listbox');
    if (!listbox) return state;
    const moves = listbox.children;
    for(let i=0; i<moves.length; i++) {
        state.push({ id: 'MOVE_' + i, moveId: moves[i].getAttribute('data-move-id'), label: `Instruction [${i + 1}]`, value: moves[i].innerText.trim().toUpperCase() });
    }
    const settingsMap = [
        { id: 'FWD_SPD', el: 'fwd_spd', label: 'Forward Speed' }, { id: 'BCK_SPD', el: 'bck_spd', label: 'Backward Speed' },
        { id: 'RGT_SPD', el: 'rgt_spd', label: 'Right Speed' }, { id: 'LFT_SPD', el: 'lft_spd', label: 'Left Speed' },
        { id: 'RGT_ANG', el: 'rgt_ang', label: 'Right Angle' }, { id: 'LFT_ANG', el: 'lft_ang', label: 'Left Angle' }
    ];
    settingsMap.forEach(s => { let el = document.getElementById(s.el); if (el) state.push({ id: s.id, label: s.label, value: el.value }); });
    return state;
}

window.saveHackerState = function() {
    if (window.currentPhase !== 2) return;
    const listbox = document.getElementById('move-listbox');
    if (!listbox) return;
    window.hackerHistory.push({
        listHTML: listbox.innerHTML,
        settings: {
            fwd: document.getElementById('fwd_spd').value, bck: document.getElementById('bck_spd').value,
            rgtS: document.getElementById('rgt_spd').value, rgtA: document.getElementById('rgt_ang').value,
            lftS: document.getElementById('lft_spd').value, lftA: document.getElementById('lft_ang').value
        }
    });
};

window.undoHackerAction = function() {
    if (window.currentPhase !== 2 || window.hackerHistory.length === 0) return;
    window.isUndoing = true; 
    const last = window.hackerHistory.pop();
    
    document.getElementById('move-listbox').innerHTML = last.listHTML;
    document.getElementById('fwd_spd').value = last.settings.fwd;
    document.getElementById('bck_spd').value = last.settings.bck;
    document.getElementById('rgt_spd').value = last.settings.rgtS;
    document.getElementById('rgt_ang').value = last.settings.rgtA;
    document.getElementById('lft_spd').value = last.settings.lftS;
    document.getElementById('lft_ang').value = last.settings.lftA;
    
    if (window.syncSettings) window.syncSettings();
    if (window.syncHackerButtons) window.syncHackerButtons();
    if (window.logHackerAction) window.logHackerAction('sys.restore @mem_snapshot');
    if (window.hackerChangesCount > 0) window.hackerChangesCount--;
    setTimeout(() => { window.isUndoing = false; }, 50); 
};

window.syncProgrammerButtons = function() {
    if (window.currentPhase !== 1) return;
    const listbox = document.getElementById('move-listbox');
    if (!listbox) return;
    const count = listbox.children.length;
    
    if (!window.DIFF_CONFIG) return;
    const cfg = window.DIFF_CONFIG[window.DIFFICULTY];
    if (!cfg) return;
    
    const isMaxed = count >= cfg.pMax;
    const isMinMet = count >= cfg.pMin;
    
    ['btn-forward', 'btn-back', 'btn-left', 'btn-right'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.disabled = isMaxed;
            btn.style.opacity = isMaxed ? '0.3' : '1';
            btn.style.cursor = isMaxed ? 'not-allowed' : 'pointer';
        }
    });
    
    const nextBtn = document.getElementById('btn-phase1-next');
    if (nextBtn) {
        nextBtn.disabled = !isMinMet;
        nextBtn.style.opacity = !isMinMet ? '0.3' : '1';
        nextBtn.style.cursor = !isMinMet ? 'not-allowed' : 'pointer';
        if (!isMinMet) nextBtn.innerText = `ADD ${cfg.pMin - count} MORE ->`;
        else nextBtn.innerText = `NEXT PHASE ->`;
    }
};

window.syncHackerButtons = function() {
    if (window.currentPhase !== 2) return;
    const hasSelection = document.querySelectorAll('.list-item.selected').length > 0;
    ['btn-forward', 'btn-back', 'btn-left', 'btn-right', 'btn-remove'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.disabled = !hasSelection;
            btn.style.opacity = !hasSelection ? '0.3' : '1';
            btn.style.cursor = !hasSelection ? 'not-allowed' : 'pointer';
        }
    });
};

window.syncSettings = function() {
    const listbox = document.getElementById('move-listbox');
    if (!listbox) return;
    const moves = Array.from(listbox.children).map(c => c.innerText.trim().toLowerCase());
    const hasFwd = moves.includes('forward'); const hasBck = moves.includes('back');
    const hasLft = moves.includes('left'); const hasRgt = moves.includes('right');

    const fwdSpd = document.getElementById('fwd_spd'); if (fwdSpd) { fwdSpd.disabled = !hasFwd; if (!hasFwd) fwdSpd.value = "100"; }
    const bckSpd = document.getElementById('bck_spd'); if (bckSpd) { bckSpd.disabled = !hasBck; if (!hasBck) bckSpd.value = "100"; }
    const lftSpd = document.getElementById('lft_spd'); if (lftSpd) { lftSpd.disabled = !hasLft; if (!hasLft) lftSpd.value = "10"; }
    const lftAng = document.getElementById('lft_ang'); if (lftAng) { lftAng.disabled = !hasLft; if (!hasLft) lftAng.value = "-90"; }
    const rgtSpd = document.getElementById('rgt_spd'); if (rgtSpd) { rgtSpd.disabled = !hasRgt; if (!hasRgt) rgtSpd.value = "10"; }
    const rgtAng = document.getElementById('rgt_ang'); if (rgtAng) { rgtAng.disabled = !hasRgt; if (!hasRgt) rgtAng.value = "90"; }
};

window.buildAlignment = function() {
    let pMoves = window.programmerState.filter(x => x.id.startsWith('MOVE_'));
    let rMoves = window.robotState.filter(x => x.id.startsWith('MOVE_'));
    let pSettings = window.programmerState.filter(x => !x.id.startsWith('MOVE_'));
    let rSettings = window.robotState.filter(x => !x.id.startsWith('MOVE_'));
    window.auditPairs = [];

    pMoves.forEach(pItem => {
        let rItem = rMoves.find(r => r.moveId === pItem.moveId);
        if (!rItem) rItem = { id: 'BLANK_' + pItem.moveId, label: '[ DELETED ]', value: '[ DELETED ]', virtual: true };
        window.auditPairs.push({ pItem: pItem, rItem: rItem, isMove: true });
    });
    for(let s=0; s<pSettings.length; s++) { window.auditPairs.push({ pItem: pSettings[s], rItem: rSettings[s], isMove: false }); }
}

window.getOriginalInstructionLabel = function(moveId) {
    if (!window.programmerState) return "[ UNKNOWN ]";
    let pMoves = window.programmerState.filter(x => x.id.startsWith('MOVE_'));
    for (let i = 0; i < pMoves.length; i++) {
        if (pMoves[i].moveId === String(moveId)) return `Instr [${i + 1}]`; 
    }
    return "[ UNKNOWN ]";
};

window.startPhase1 = function() {
    window.p1Started = true;
    document.getElementById('p1-start-overlay').style.display = 'none';
    const settingsBtn = document.getElementById('btn-settings');
    if (settingsBtn) settingsBtn.style.display = 'none';
    window.startTimer(window.DIFF_CONFIG[window.DIFFICULTY].tP1, () => { window.goToPhase2(null, true); }, 'var(--neon-green)');
};

window.startPhase2 = function() {
    document.getElementById('p2-start-overlay').style.display = 'none';
    window.startTimer(window.DIFF_CONFIG[window.DIFFICULTY].tP2, () => { window.goToPhase3(null); }, 'var(--neon-red)');
};

window.startPhase3 = function() {
    document.getElementById('p3-start-overlay').style.display = 'none';
};