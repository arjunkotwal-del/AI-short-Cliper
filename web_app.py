import sys
import queue
import threading
import json
from flask import Flask, request, jsonify, Response, send_from_directory

app = Flask(__name__)

# Queue to hold log lines for SSE stream
log_queue = queue.Queue()

class StreamIntercept:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, data):
        self.original_stdout.write(data)
        if data.strip():
            log_queue.put(data)

    def flush(self):
        self.original_stdout.flush()

# Intercept standard output globally so we can stream it to the web UI
sys.stdout = StreamIntercept(sys.stdout)

@app.route("/")
def index():
    return send_from_directory('static', 'index.html')

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory('static', path)

@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                line = log_queue.get(timeout=1.0)
                
                # Heuristics to detect which agent is active based on the logs
                agent_name = None
                line_lower = line.lower()
                
                if "[orchestrator]" in line_lower: 
                    agent_name = "Orchestrator"
                elif "[downloader]" in line_lower or "[download/local]" in line_lower or "downloading" in line_lower: 
                    agent_name = "Downloader"
                elif "[transcriber]" in line_lower or "[transcribe" in line_lower or "whisper" in line_lower: 
                    agent_name = "Transcriber"
                elif "[clipper]" in line_lower or "[clip" in line_lower or "ffmpeg" in line_lower: 
                    agent_name = "Clipper"
                
                yield f"data: {json.dumps({'text': line, 'agent': agent_name})}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
                
    return Response(generate(), mimetype="text/event-stream")

from shorts_generator.cancel_token import reset_cancellation, cancel_pipeline

@app.route("/api/run", methods=["POST"])
def run_pipeline():
    reset_cancellation()
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
        
    num_clips = data.get("num_clips", 3)
    min_duration = data.get("min_duration", 10.0)
    max_duration = data.get("max_duration", 60.0)
    voiceover = data.get("voiceover", False)

    # Instruct Orchestrator to propagate configuration settings down to tools
    config_context = (
        f"\n\n[USER CONFIGURATION]\n"
        f"- Number of clips requested: {num_clips}\n"
        f"- Minimum duration per clip: {min_duration} seconds\n"
        f"- Maximum duration per clip: {max_duration} seconds\n"
        f"- Enable TTS voiceover audio overlay: {voiceover}\n"
        f"You MUST pass these parameters (`num_clips`, `min_duration`, `max_duration`, `voiceover`) "
        f"to the Clipper tools when calling them. Do not use default values."
    )

    def background_run():
        from agentic_main import orchestrator_agent
        from agent import run_agent_loop
        
        messages = [{"role": "user", "content": prompt + config_context}]
        try:
            print(f"[Orchestrator] Starting task: {prompt}\n", flush=True)
            run_agent_loop(orchestrator_agent, messages)
            print(f"\n[Orchestrator] Task Complete!\n", flush=True)
        except Exception as e:
            print(f"\n[Orchestrator] Critical Error: {e}\n", flush=True)
            
    # Run agent loop in a background thread so the HTTP request completes instantly
    threading.Thread(target=background_run, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def stop_pipeline():
    cancel_pipeline()
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    from shorts_generator.config import require_openai_key
    require_openai_key()
    print("Starting Web Dashboard on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
