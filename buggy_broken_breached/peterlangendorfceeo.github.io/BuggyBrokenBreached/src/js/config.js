const isSpeedRound = window.location.href.includes('speed_round');
window.DIFFICULTY = isSpeedRound ? 'speed' : 'easy';

// THE FIX: Dynamically update the UI if Speed Round is detected!
if (isSpeedRound) {
    document.title = "Buggy, Broken, or Breached! - Speed Round";
    
    // We use a tiny timeout to ensure the HTML elements exist before we change them
    setTimeout(() => {
        const titleEl = document.querySelector('.game-title');
        if (titleEl) {
            titleEl.innerHTML = 'Buggy, Broken, or Breached! <span style="color: var(--neon-red); display: block; font-size: 0.6em; margin-top: 5px; text-shadow: 0 0 15px var(--neon-red);">[ SPEED ROUND ]</span>';
        }
        const settingsBtn = document.getElementById('btn-settings');
        if (settingsBtn) settingsBtn.style.display = 'none';
    }, 50);
}

window.DIFF_CONFIG = {
    easy: { failChance: 5, pMin: 5, pMax: 10, hMax: 5, checks: 5, mult: 1, tP1: 120, tP2: 120, tP3: 300, desc: "Loading..." },
    medium: { failChance: 10, pMin: 9, pMax: 14, hMax: 7, checks: 4, mult: 2, tP1: 90, tP2: 90, tP3: 240, desc: "Loading..." },
    hard: { failChance: 20, pMin: 12, pMax: 16, hMax: 999, checks: 3, mult: 3, tP1: 60, tP2: 60, tP3: 180, desc: "Loading..." },
    speed: { failChance: 15, pMin: 1, pMax: 5, hMax: 3, checks: 2, mult: 2, tP1: 60, tP2: 60, tP3: 180, 
        desc: "<div style='color:#888; font-style:italic; margin-bottom:15px; border-left: 2px solid #555; padding-left: 10px;'>- Speed is the essence of war.</div><strong>[ SPEED ROUND MECHANICS ]</strong><br><br>- 1m / 1m / 3m Timers.<br>- Programmer constraint: Max 5 instructions.<br>- Hacker limit: 3 malicious edits.<br>- Memory Audit allows 2 Hash Checks.<br>- High-fidelity Sensor Graph.<br>- 2x Score Multiplier." 
    }
};

window.openSettings = function() {
    if (isSpeedRound) return; // Completely locked in speed round!
    document.getElementById('settings-overlay').style.display = 'flex';
    window.updateSettingsUI();
};

window.closeSettings = function() {
    document.getElementById('settings-overlay').style.display = 'none';
};

window.setDifficulty = function(level) {
    if (isSpeedRound) return;
    window.DIFFICULTY = level;
    window.updateSettingsUI();
    if (window.syncProgrammerButtons) window.syncProgrammerButtons();
};

window.updateSettingsUI = function() {
    ['easy', 'medium', 'hard'].forEach(level => {
        let btn = document.getElementById(`btn-diff-${level}`);
        if (!btn) return;
        if (level === window.DIFFICULTY) {
            btn.classList.add('active-valid');
            btn.style.borderColor = 'var(--neon-green)';
            btn.style.color = 'var(--neon-green)';
        } else {
            btn.classList.remove('active-valid');
            btn.style.borderColor = '#888';
            btn.style.color = '#888';
        }
    });
    
    const descEl = document.getElementById('diff-desc');
    if (descEl) descEl.innerHTML = window.DIFF_CONFIG[window.DIFFICULTY].desc;

    const cfg = window.DIFF_CONFIG[window.DIFFICULTY];
    const formatTime = (secs) => {
        let m = Math.floor(secs / 60); let s = secs % 60;
        if (s === 0) return `${m} minute${m !== 1 ? 's' : ''}`;
        return `${m} minute${m !== 1 ? 's' : ''} and ${s} seconds`;
    };
    
    let t1 = document.getElementById('time-p1'); if (t1) t1.innerText = formatTime(cfg.tP1);
    let t2 = document.getElementById('time-p2'); if (t2) t2.innerText = formatTime(cfg.tP2);
    let t3 = document.getElementById('time-p3'); if (t3) t3.innerText = formatTime(cfg.tP3);
    
    if (isSpeedRound) {
        let btnSettings = document.getElementById('btn-settings');
        if (btnSettings) btnSettings.style.display = 'none';
    }
};

window.hackerChangesCount = 0;
window.canHackerChange = function() {
    let max = window.DIFF_CONFIG[window.DIFFICULTY].hMax;
    if (window.hackerChangesCount >= max) {
        let term = document.getElementById('terminal');
        if (term) {
            term.innerHTML += `<span style='color:var(--neon-red);'>[ERROR] HACKER LIMIT REACHED: You are only permitted ${max} modifications on this difficulty.</span><br>`;
            term.scrollTop = term.scrollHeight;
        }
        return false;
    }
    return true;
};
window.registerHackerChange = function() { window.hackerChangesCount++; };

async function loadRules() {
    try {
        const fetchText = async (url) => { const res = await fetch(url); return res.ok ? await res.text() : ""; };
        const pText = await fetchText('resources/p_instructions.txt');
        if (pText) document.getElementById('rules-programmer').innerHTML = pText.split('\n').filter(l=>l.trim()).map(l => `<li>${l.trim()}</li>`).join('');
        const hText = await fetchText('resources/h_instructions.txt');
        if (hText) document.getElementById('rules-hacker').innerHTML = hText.split('\n').filter(l=>l.trim()).map(l => `<li>${l.trim()}</li>`).join('');
        const eText = await fetchText('resources/exec_instructions.txt');
        if (eText) document.getElementById('rules-investigation').innerHTML = eText.split('\n').filter(l=>l.trim()).map(l => `<li>${l.trim()}</li>`).join('');
        const dEasy = await fetchText('resources/diff_easy.txt');
        if (dEasy) window.DIFF_CONFIG['easy'].desc = dEasy.replace(/\n/g, '<br>');
        const dMed = await fetchText('resources/diff_medium.txt');
        if (dMed) window.DIFF_CONFIG['medium'].desc = dMed.replace(/\n/g, '<br>');
        const dHard = await fetchText('resources/diff_hard.txt');
        if (dHard) window.DIFF_CONFIG['hard'].desc = dHard.replace(/\n/g, '<br>');
    } catch (e) {
        document.getElementById('rules-programmer').innerHTML = "<li>Failed to load rules. Local server required.</li>";
        document.getElementById('rules-hacker').innerHTML = "<li>Failed to load rules. Local server required.</li>";
        document.getElementById('rules-investigation').innerHTML = "<li>Failed to load rules. Local server required.</li>";
    } finally {
        window.updateSettingsUI();
    }
}
window.loadRules = loadRules;