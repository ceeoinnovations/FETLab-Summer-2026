window.animTimeouts = [];
window.p1_flicker = null; window.p1_spray = null; window.p2_flicker = null;

window.setAnimTimeout = function(callback, ms) {
    let id = setTimeout(callback, ms);
    window.animTimeouts.push(id);
    return id;
};

window.clearAnimTimeouts = function() {
    window.animTimeouts.forEach(clearTimeout);
    window.animTimeouts = [];
};

window.doTransition = function(callback, event) {
    const wipe = document.getElementById('transition-wipe');
    if (event) { document.documentElement.style.setProperty('--wipe-x', event.clientX + 'px'); document.documentElement.style.setProperty('--wipe-y', event.clientY + 'px'); } 
    else { document.documentElement.style.setProperty('--wipe-x', '50%'); document.documentElement.style.setProperty('--wipe-y', '50%'); }
    document.body.classList.add('exploding'); 
    wipe.style.transition = 'transform 0.6s cubic-bezier(0.9, 0.03, 0.05, 0.9)'; wipe.classList.add('wipe-in');
    setTimeout(() => { callback(); document.body.classList.remove('exploding'); setTimeout(() => { wipe.classList.remove('wipe-in'); }, 100); }, 600); 
};

window.skipToPhase2 = function() {
    window.stopTimer(); 
    window.clearAnimTimeouts();
    if (window.p1_flicker) clearInterval(window.p1_flicker); if (window.p1_spray) clearInterval(window.p1_spray);
    document.getElementById('hijack-overlay').style.display = 'none'; document.body.classList.remove('exploding');
    const wipe = document.getElementById('transition-wipe'); wipe.style.transition = 'none'; wipe.classList.remove('wipe-in');
    document.getElementById('step-builder').style.display = 'none';
    const footer = document.getElementById('footer-status');
    if(footer) footer.style.display = 'flex';
    
    window.programmerState = window.captureState(); window.currentPhase = 2;
    document.getElementById('phase2-container').style.display = 'flex';
    document.getElementById('phase2-list-container').appendChild(document.getElementById('shared-editor'));
    document.body.classList.add('hacker-mode'); 
    document.getElementById('btn-clear').style.display = 'none';
    document.getElementById('btn-undo').style.display = 'block';
    
    window.syncSettings(); window.syncHackerButtons(); 
    if (window.clearMatrix) window.clearMatrix('#120004'); window.matrixSettings.color = 'rgba(255, 0, 60, 0.7)'; window.matrixSettings.fade = 'rgba(18, 0, 4, 0.05)'; 
    if (window.preRenderMatrix) window.preRenderMatrix(); if (window.applyMatrixSettings) window.applyMatrixSettings();
    document.getElementById('p2-start-overlay').style.display = 'flex';
};

window.goToPhase2 = function(event, forced = false) {
    let pMoves = document.getElementById('move-listbox').children.length;
    let cfg = window.DIFF_CONFIG[window.DIFFICULTY];
    if (!forced) {
        if (pMoves < cfg.pMin) {
            let term = document.getElementById('terminal');
            if (term) { term.innerHTML += `<span style='color:orange;'>[SYSTEM ERROR] Sequence is too short! Must contain at least ${cfg.pMin} instructions in ${window.DIFFICULTY.toUpperCase()} mode.</span><br>`; term.scrollTop = term.scrollHeight; }
            return;
        }
    }
    window.stopTimer(); 
    const footer = document.getElementById('footer-status');
    if(footer) footer.style.display = 'none';
    
    const wipe = document.getElementById('transition-wipe');
    if (event) { document.documentElement.style.setProperty('--wipe-x', event.clientX + 'px'); document.documentElement.style.setProperty('--wipe-y', event.clientY + 'px'); }
    else { document.documentElement.style.setProperty('--wipe-x', '50%'); document.documentElement.style.setProperty('--wipe-y', '50%'); }
    
    document.body.classList.add('exploding'); wipe.style.transition = 'transform 0.6s cubic-bezier(0.9, 0.03, 0.05, 0.9)'; wipe.classList.add('wipe-in');
    
    window.setAnimTimeout(() => {
        document.body.classList.remove('exploding'); document.getElementById('step-builder').style.display = 'none';
        const overlay = document.getElementById('hijack-overlay'); const scene = document.getElementById('hijack-scene');
        overlay.style.display = 'flex'; scene.style.opacity = '1';
        const packet = document.getElementById('h-packet'); const hacker = document.getElementById('h-hacker');

        packet.style.transition = 'none'; packet.style.left = '50px'; packet.style.transform = 'translate3d(-50%, -50%, 0) scale(0)'; packet.style.filter = 'none'; 
        hacker.style.transition = 'none'; hacker.style.top = '-150px'; hacker.style.opacity = '0';
        wipe.style.transition = 'none'; wipe.classList.remove('wipe-in');
        
        const binaryRows = document.querySelectorAll('.binary-row');
        window.p1_flicker = setInterval(() => {
            binaryRows.forEach(row => {
                let charArray = row.innerText.split('');
                for(let k=0; k<2; k++) { let idx = Math.floor(Math.random() * charArray.length); charArray[idx] = charArray[idx] === '0' ? '1' : '0'; }
                row.innerText = charArray.join('');
            });
        }, 150);

        window.setAnimTimeout(() => {
            packet.style.transition = 'left 3.0s linear, transform 0.3s cubic-bezier(0.1, 0.8, 0.3, 1)';
            packet.style.left = '50%'; packet.style.transform = 'translate3d(-50%, -50%, 0) scale(1)';

            window.setAnimTimeout(() => {
                hacker.style.transition = 'top 0.3s cubic-bezier(0.8, 0, 0.2, 1), opacity 0.3s';
                hacker.style.top = '120px'; hacker.style.opacity = '1';

                window.setAnimTimeout(() => {
                    document.body.classList.add('exploding'); 
                    packet.style.filter = 'drop-shadow(0 0 20px #ff003c) sepia(1) hue-rotate(-50deg) saturate(5)';
                    
                    let sprayCount = 0;
                    window.p1_spray = setInterval(() => {
                        for(let i=0; i<3; i++) {
                            let p = document.createElement('div'); p.className = 'h-particle'; p.innerText = Math.random() > 0.5 ? '1' : '0';
                            
                            // THE FIX: Origin is now exactly the center collision point!
                            p.style.left = '400px'; 
                            p.style.top = '200px';
                            
                            let angle = Math.random() * Math.PI * 2; let distance = 50 + Math.random() * 200;
                            p.style.transition = 'all 0.6s cubic-bezier(0.1, 0.9, 0.2, 1)'; document.getElementById('hijack-scene').appendChild(p); void p.offsetWidth; 
                            p.style.transform = `translate3d(${Math.cos(angle) * distance}px, ${Math.sin(angle) * distance}px, 0) scale(1.5)`; p.style.opacity = '0';
                            window.setAnimTimeout(() => p.remove(), 600);
                        }
                        sprayCount++; if(sprayCount > 40) clearInterval(window.p1_spray); 
                    }, 25);

                    window.setAnimTimeout(() => { document.body.classList.remove('exploding'); }, 400);

                    window.setAnimTimeout(() => {
                        clearInterval(window.p1_flicker); scene.style.transition = 'opacity 0.5s'; scene.style.opacity = '0';
                        
                        window.setAnimTimeout(() => {
                            wipe.style.transition = 'none'; document.documentElement.style.setProperty('--wipe-x', '50%'); document.documentElement.style.setProperty('--wipe-y', '50%'); wipe.classList.add('wipe-in');
                            
                            window.programmerState = window.captureState(); window.currentPhase = 2;
                            document.getElementById('phase2-container').style.display = 'flex';
                            document.getElementById('phase2-list-container').appendChild(document.getElementById('shared-editor'));
                            document.body.classList.add('hacker-mode'); 
                            document.getElementById('btn-clear').style.display = 'none';
                            document.getElementById('btn-undo').style.display = 'block';
                            
                            window.syncSettings(); window.syncHackerButtons(); 
                            if (window.clearMatrix) window.clearMatrix('#120004'); window.matrixSettings.color = 'rgba(255, 0, 60, 0.7)'; window.matrixSettings.fade = 'rgba(18, 0, 4, 0.05)'; 
                            if (window.preRenderMatrix) window.preRenderMatrix(); if (window.applyMatrixSettings) window.applyMatrixSettings();

                            overlay.style.display = 'none';
                            window.setAnimTimeout(() => { 
                                wipe.style.transition = 'transform 0.6s cubic-bezier(0.9, 0.03, 0.05, 0.9)'; wipe.classList.remove('wipe-in'); 
                                if(footer) footer.style.display = 'flex';
                                document.getElementById('p2-start-overlay').style.display = 'flex';
                            }, 50);
                        }, 1000); 
                    }, 1200); 
                }, 300); 
            }, 2700); 
        }, 500); 
    }, 600); 
}

window.skipToPhase3 = function() {
    window.stopTimer(); window.clearAnimTimeouts();
    if (window.p2_flicker) clearInterval(window.p2_flicker);
    document.getElementById('deploy-overlay').style.display = 'none'; document.getElementById('phase2-container').style.display = 'none';
    document.body.classList.remove('exploding');
    const wipe = document.getElementById('transition-wipe'); wipe.style.transition = 'none'; wipe.classList.remove('wipe-in');
    
    const footer = document.getElementById('footer-status');
    if(footer) footer.style.display = 'flex';
    
    window.currentPhase = 3; window.robotState = window.captureState(); window.deductionState = []; document.getElementById('deduction-list').innerHTML = '';
    window.buildAlignment(); window.renderChecksumLists();
    
    document.getElementById('step-execution').style.display = 'flex'; document.body.classList.remove('hacker-mode');
    if (window.clearMatrix) window.clearMatrix('#0d0d12'); window.matrixSettings.color = 'rgba(0, 255, 65, 0.4)'; window.matrixSettings.fade = 'rgba(13, 13, 18, 0.08)';
    if (window.preRenderMatrix) window.preRenderMatrix(); if (window.applyMatrixSettings) window.applyMatrixSettings();
    document.getElementById('p3-start-overlay').style.display = 'flex';
};

window.goToPhase3 = function(event) {
    window.stopTimer(); 
    const footer = document.getElementById('footer-status');
    if(footer) footer.style.display = 'none';
    
    const wipe = document.getElementById('transition-wipe');
    if (event) { document.documentElement.style.setProperty('--wipe-x', event.clientX + 'px'); document.documentElement.style.setProperty('--wipe-y', event.clientY + 'px'); }
    else { document.documentElement.style.setProperty('--wipe-x', '50%'); document.documentElement.style.setProperty('--wipe-y', '50%'); }
    
    document.body.classList.add('exploding'); wipe.style.transition = 'transform 0.6s cubic-bezier(0.9, 0.03, 0.05, 0.9)'; wipe.classList.add('wipe-in');
    
    window.setAnimTimeout(() => {
        document.body.classList.remove('exploding'); document.getElementById('phase2-container').style.display = 'none';
        const overlay = document.getElementById('deploy-overlay'); const scene = document.getElementById('deploy-scene');
        overlay.style.display = 'flex'; scene.style.opacity = '1';
        
        const packet = document.getElementById('d-packet'); const bot = document.getElementById('d-bot');
        packet.style.transition = 'none'; packet.style.left = '50px'; packet.style.transform = 'translate3d(-50%, -50%, 0) scale(0)'; bot.style.filter = 'none';
        wipe.classList.remove('wipe-in');
        
        const binaryRows = document.querySelectorAll('#d-stream .binary-row');
        window.p2_flicker = setInterval(() => {
            binaryRows.forEach(row => {
                let chars = row.innerText.split('');
                for(let k=0; k<5; k++) { let idx = Math.floor(Math.random() * chars.length); chars[idx] = chars[idx] === '0' ? '1' : '0'; }
                row.innerText = chars.join('');
            });
        }, 100);

        window.setAnimTimeout(() => {
            packet.style.transition = 'left 1.5s cubic-bezier(0.4, 0, 1, 1), transform 0.2s ease-out';
            packet.style.left = '95%'; packet.style.transform = 'translate3d(-50%, -50%, 0) scale(1)';

            window.setAnimTimeout(() => {
                document.body.classList.add('exploding'); packet.style.opacity = '0'; bot.style.filter = 'drop-shadow(0 0 20px #ff003c) drop-shadow(0 0 30px #ff003c)';
                window.setAnimTimeout(() => { document.body.classList.remove('exploding'); }, 400);

                window.setAnimTimeout(() => {
                    scene.style.transition = 'opacity 0.5s'; scene.style.opacity = '0';
                    window.setAnimTimeout(() => {
                        // THE FIX: Hardened the transition trigger sequence
                        wipe.style.transition = 'none'; document.documentElement.style.setProperty('--wipe-x', '50%'); document.documentElement.style.setProperty('--wipe-y', '50%'); wipe.classList.add('wipe-in');
                        
                        window.currentPhase = 3; window.robotState = window.captureState(); window.deductionState = []; 
                        
                        const dedList = document.getElementById('deduction-list');
                        if(dedList) dedList.innerHTML = '';
                        
                        window.buildAlignment(); window.renderChecksumLists();
                        
                        const stepExec = document.getElementById('step-execution');
                        if(stepExec) stepExec.style.display = 'flex'; 
                        
                        document.body.classList.remove('hacker-mode');
                        if (window.clearMatrix) window.clearMatrix('#0d0d12'); window.matrixSettings.color = 'rgba(0, 255, 65, 0.4)'; window.matrixSettings.fade = 'rgba(13, 13, 18, 0.08)';
                        if (window.preRenderMatrix) window.preRenderMatrix(); if (window.applyMatrixSettings) window.applyMatrixSettings();
                        
                        overlay.style.display = 'none'; clearInterval(window.p2_flicker);
                        window.setAnimTimeout(() => { 
                            wipe.style.transition = 'transform 0.6s cubic-bezier(0.9, 0.03, 0.05, 0.9)'; wipe.classList.remove('wipe-in'); 
                            if(footer) footer.style.display = 'flex';
                            document.getElementById('p3-start-overlay').style.display = 'flex';
                        }, 50);
                    }, 1000);
                }, 800); 
            }, 1500); 
        }, 300);
    }, 600);
}