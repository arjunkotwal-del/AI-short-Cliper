document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const stopBtn = document.getElementById('stop-btn');
    const promptInput = document.getElementById('prompt-input');
    const logOutput = document.getElementById('log-output');
    
    // Config Inputs
    const numClipsInput = document.getElementById('num-clips-input');
    const minDurInput = document.getElementById('min-dur-input');
    const maxDurInput = document.getElementById('max-dur-input');
    const voiceoverInput = document.getElementById('voiceover-input');
    
    // Agent Cards
    const cards = {
        'Orchestrator': document.getElementById('card-orchestrator'),
        'Downloader': document.getElementById('card-downloader'),
        'Transcriber': document.getElementById('card-transcriber'),
        'Clipper': document.getElementById('card-clipper'),
    };

    function setActiveAgent(agentName) {
        // Remove active class from all
        Object.values(cards).forEach(card => card.classList.remove('active'));
        // Add to the current one
        if (agentName && cards[agentName]) {
            cards[agentName].classList.add('active');
        }
    }

    function appendLog(text, agentName) {
        const div = document.createElement('div');
        div.className = 'log-line';
        if (agentName) {
            div.classList.add(`agent-${agentName}`);
        }
        div.textContent = text;
        logOutput.appendChild(div);
        
        // Auto-scroll to bottom
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    function setControlsDisabled(disabled) {
        runBtn.disabled = disabled;
        promptInput.disabled = disabled;
        numClipsInput.disabled = disabled;
        minDurInput.disabled = disabled;
        maxDurInput.disabled = disabled;
        voiceoverInput.disabled = disabled;
        
        if (disabled) {
            runBtn.textContent = 'Running...';
            stopBtn.style.display = 'block';
        } else {
            runBtn.textContent = 'Run Agents';
            stopBtn.style.display = 'none';
        }
    }

    // Connect to SSE stream
    const eventSource = new EventSource('/stream');
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.text) {
            appendLog(data.text, data.agent);
            
            // Check for termination events
            const textLower = data.text.toLowerCase();
            if (
                textLower.includes('task complete') || 
                textLower.includes('critical error') || 
                textLower.includes('cancelled') || 
                textLower.includes('stopped')
            ) {
                setActiveAgent(null); // Turn off all pulsing
                setControlsDisabled(false);
            }
        }
        if (data.agent) {
            setActiveAgent(data.agent);
        }
    };

    runBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        const numClips = parseInt(numClipsInput.value) || 3;
        const minDur = parseFloat(minDurInput.value) || 10;
        const maxDur = parseFloat(maxDurInput.value) || 60;
        const voiceover = voiceoverInput.checked;

        // Reset UI and disable controls
        logOutput.innerHTML = '';
        appendLog('Initializing pipeline...', 'Orchestrator');
        setControlsDisabled(true);

        try {
            const res = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    num_clips: numClips,
                    min_duration: minDur,
                    max_duration: maxDur,
                    voiceover: voiceover
                })
            });
            
            if (!res.ok) throw new Error('Failed to start');
            
        } catch (err) {
            appendLog(`Error: ${err.message}`, null);
            setControlsDisabled(false);
        }
    });

    stopBtn.addEventListener('click', async () => {
        stopBtn.disabled = true;
        stopBtn.textContent = 'Stopping...';
        try {
            const res = await fetch('/api/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            if (!res.ok) throw new Error('Failed to stop');
        } catch (err) {
            appendLog(`Error: ${err.message}`, null);
        } finally {
            stopBtn.disabled = false;
            stopBtn.textContent = 'Stop Pipeline';
        }
    });
    
    // Allow enter key
    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') runBtn.click();
    });
});
