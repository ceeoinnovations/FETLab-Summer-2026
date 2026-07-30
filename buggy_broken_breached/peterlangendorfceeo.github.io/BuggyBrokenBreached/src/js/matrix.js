const mCanvas = document.getElementById('matrix-bg');
const mCtx = mCanvas.getContext('2d');
mCanvas.width = window.innerWidth; 
mCanvas.height = window.innerHeight;

const chars = '01'.split('');
const fontSize = 16;
let columns = Math.floor(mCanvas.width / fontSize);
let drops = [];

for (let x = 0; x < columns; x++) {
    drops[x] = Math.floor(Math.random() * (mCanvas.height / fontSize));
}

let matrixIntervalId;

window.matrixSettings = {
    color: 'rgba(0, 255, 65, 0.4)', 
    fade: 'rgba(13, 13, 18, 0.05)',
    speed: 120 
};

// Instantly wipe the canvas of previous colors
window.clearMatrix = function(bgColor) {
    mCtx.fillStyle = bgColor || '#0d0d12';
    mCtx.fillRect(0, 0, mCanvas.width, mCanvas.height);
};

// THE FIX: Extracted the 100-frame burst into a reusable function!
window.preRenderMatrix = function() {
    for (let x = 0; x < columns; x++) {
        drops[x] = Math.floor(Math.random() * (mCanvas.height / fontSize));
    }
    for(let i = 0; i < 100; i++) {
        mCtx.fillStyle = window.matrixSettings.fade;
        mCtx.fillRect(0, 0, mCanvas.width, mCanvas.height);
        mCtx.fillStyle = window.matrixSettings.color;
        mCtx.font = fontSize + 'px monospace';
        
        mCtx.shadowBlur = 5;
        mCtx.shadowColor = window.matrixSettings.color;

        for (let j = 0; j < drops.length; j++) {
            const text = chars[Math.floor(Math.random() * chars.length)];
            mCtx.fillText(text, j * fontSize, drops[j] * fontSize);
            if (drops[j] * fontSize > mCanvas.height && Math.random() > 0.975) {
                drops[j] = 0;
            }
            drops[j]++;
        }
        mCtx.shadowBlur = 0;
    }
};

window.clearMatrix('#0d0d12');
window.preRenderMatrix();

window.applyMatrixSettings = function() {
    if (matrixIntervalId) clearInterval(matrixIntervalId);
    matrixIntervalId = setInterval(() => {
        mCtx.fillStyle = window.matrixSettings.fade; 
        mCtx.fillRect(0, 0, mCanvas.width, mCanvas.height);
        mCtx.fillStyle = window.matrixSettings.color; 
        mCtx.font = fontSize + 'px monospace';
        
        // THE FIX: Live shadow glow on every drop!
        mCtx.shadowBlur = 5;
        mCtx.shadowColor = window.matrixSettings.color;

        for (let i = 0; i < drops.length; i++) {
            const text = chars[Math.floor(Math.random() * chars.length)];
            mCtx.fillText(text, i * fontSize, drops[i] * fontSize);
            if (drops[i] * fontSize > mCanvas.height && Math.random() > 0.975) { drops[i] = 0; }
            drops[i]++;
        }
        mCtx.shadowBlur = 0;
    }, window.matrixSettings.speed);
};

window.applyMatrixSettings();

window.addEventListener('resize', () => { 
    mCanvas.width = window.innerWidth; 
    mCanvas.height = window.innerHeight; 
    columns = Math.floor(mCanvas.width / fontSize);
    drops.length = 0;
    for (let x = 0; x < columns; x++) drops[x] = Math.floor(Math.random() * (mCanvas.height / fontSize));
    window.clearMatrix(window.currentPhase === 2 ? '#120004' : '#0d0d12');
    window.preRenderMatrix();
});