window.timerInterval = null;

window.startTimer = function(totalSeconds, timeoutCallback, colorClass) {
    if (window.timerInterval) clearInterval(window.timerInterval);
    let timerEl = document.getElementById('global-timer');
    if (!timerEl) return;
    
    timerEl.style.display = 'block';
    timerEl.style.color = colorClass;
    timerEl.style.borderColor = colorClass;
    let remaining = totalSeconds;
    
    function updateDisplay() {
        let m = Math.floor(remaining / 60).toString().padStart(2, '0');
        let s = (remaining % 60).toString().padStart(2, '0');
        timerEl.innerText = `[ TIME REMAINING: ${m}:${s} ]`;
        
        if (remaining <= 10) {
            timerEl.style.color = '#fff';
            timerEl.style.backgroundColor = 'rgba(255,0,0,0.8)';
            timerEl.style.borderColor = 'rgba(255,0,0,1)';
        } else {
            timerEl.style.color = colorClass;
            timerEl.style.backgroundColor = 'rgba(0,0,0,0.8)';
            timerEl.style.borderColor = colorClass;
        }
    }
    
    updateDisplay();
    
    window.timerInterval = setInterval(() => {
        remaining--;
        updateDisplay();
        if (remaining <= 0) {
            clearInterval(window.timerInterval);
            timerEl.style.display = 'none';
            if (timeoutCallback) timeoutCallback();
        }
    }, 1000);
}

window.stopTimer = function() {
    if (window.timerInterval) clearInterval(window.timerInterval);
    let timerEl = document.getElementById('global-timer');
    if (timerEl) timerEl.style.display = 'none';
}

window.startPhase3Timer = function() {
    window.startTimer(window.DIFF_CONFIG[window.DIFFICULTY].tP3, () => {
        if (document.getElementById('result-overlay').style.display === 'none') {
            window.submitDeduction();
        }
    }, 'var(--neon-cyan)');
}