document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const promptInput = document.getElementById('prompt-input');
    const logOutput = document.getElementById('log-output');
    
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

    // Connect to SSE stream
    const eventSource = new EventSource('/stream');
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.text) {
            appendLog(data.text, data.agent);
        }
        if (data.agent) {
            setActiveAgent(data.agent);
        }
        
        if (data.text.includes('Task Complete')) {
            setActiveAgent(null); // Turn off all pulsing
        }
    };

    runBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        // Reset UI
        logOutput.innerHTML = '';
        appendLog('Initializing pipeline...', 'Orchestrator');
        runBtn.disabled = true;
        runBtn.textContent = 'Running...';

        try {
            const res = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });
            
            if (!res.ok) throw new Error('Failed to start');
            
        } catch (err) {
            appendLog(`Error: ${err.message}`, null);
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Run Agents';
        }
    });
    
    // Allow enter key
    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') runBtn.click();
    });
});
