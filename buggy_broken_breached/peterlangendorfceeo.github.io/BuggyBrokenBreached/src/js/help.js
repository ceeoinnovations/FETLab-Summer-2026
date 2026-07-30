window.currentHelpStep = 1;
window.TOTAL_HELP_STEPS = 7;

window.openHelp = function() {
    document.getElementById('help-overlay').style.display = 'flex';
    window.currentHelpStep = 1;
    window.updateHelpUI();
};

window.closeHelp = function() {
    document.getElementById('help-overlay').style.display = 'none';
    if (window.currentPhase === 1 && !window.p1Started) {
        document.getElementById('p1-start-overlay').style.display = 'flex';
    }
};

window.nextHelp = function() {
    if (window.currentHelpStep < window.TOTAL_HELP_STEPS) {
        window.currentHelpStep++;
        window.updateHelpUI();
    }
};

window.prevHelp = function() {
    if (window.currentHelpStep > 1) {
        window.currentHelpStep--;
        window.updateHelpUI();
    }
};

window.updateHelpUI = function() {
    for (let i = 1; i <= window.TOTAL_HELP_STEPS; i++) {
        const step = document.getElementById('help-step-' + i);
        if (step) step.style.display = (i === window.currentHelpStep) ? 'block' : 'none';
    }
    
    const prevBtn = document.getElementById('btn-help-prev');
    const nextBtn = document.getElementById('btn-help-next');
    const finishBtn = document.getElementById('btn-help-finish');
    const dots = document.querySelectorAll('.help-dot');
    
    dots.forEach((dot, index) => {
        if (index + 1 === window.currentHelpStep) dot.classList.add('active');
        else dot.classList.remove('active');
    });
    
    if (window.currentHelpStep === 1) prevBtn.disabled = true;
    else prevBtn.disabled = false;
    
    if (window.currentHelpStep === window.TOTAL_HELP_STEPS) {
        nextBtn.disabled = true;
        finishBtn.disabled = false;
        finishBtn.style.opacity = '1';
        finishBtn.style.cursor = 'pointer';
    } else {
        nextBtn.disabled = false;
        finishBtn.disabled = true;
        finishBtn.style.opacity = '0.5';
        finishBtn.style.cursor = 'not-allowed';
    }
};

document.addEventListener('keydown', function(event) {
    if (event.key === "Escape") {
        const helpOverlay = document.getElementById('help-overlay');
        if (helpOverlay && helpOverlay.style.display !== 'none') window.closeHelp();
        const settingsOverlay = document.getElementById('settings-overlay');
        if (settingsOverlay && settingsOverlay.style.display !== 'none') window.closeSettings();
    }
});