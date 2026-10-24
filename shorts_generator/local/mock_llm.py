"""Offline stand-in for the LLM, used when MOCK_LLM=true.

Answers each prompt the pipelines send with a reply of the right shape, so the default
clipping pipeline and the agentic MathSolver both run without an API key:

  content-type prompt  -> {"content_type", "density"}
  highlight prompt     -> {"highlights": [...]} picked evenly from the transcript
  social-copy prompt   -> {"caption", "hashtags"}
  anything else        -> the MathSolver demo script
"""
import json
import re
from typing import Dict, List

_TRANSCRIPT_LINE = re.compile(r"^\[(\d+(?:\.\d+)?)s\]\s*(.+)$", re.MULTILINE)
_CLIP_TITLE = re.compile(r"^Clip title:\s*(.*)$", re.MULTILINE)
_MOCK_CLIP_SECONDS = 30.0

MATH_SOLVER_SCRIPT = """
```python
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Solving the differential equation: dy/dx + 2*y = x with y(0) = 5")
print("This is a first-order linear ordinary differential equation.")
print("Steps:")
print("1. Standard form: dy/dx + 2*y = x")
print("2. Integrating factor: I(x) = exp(integral(2 dx)) = exp(2x)")
print("3. Multiply both sides by I(x): exp(2x)*dy/dx + 2*exp(2x)*y = x*exp(2x)")
print("4. Integrate both sides: exp(2x)*y = integral(x*exp(2x) dx) = (x/2 - 1/4)*exp(2x) + C")
print("5. General solution: y(x) = C*exp(-2x) + x/2 - 1/4")
print("6. Apply initial condition y(0) = 5: 5 = C - 1/4 => C = 21/4 = 5.25")
print("7. Particular solution: y(x) = 5.25*exp(-2x) + x/2 - 1/4")

x_sym = sp.Symbol('x')
y_func = sp.Function('y')
eq = sp.Eq(y_func(x_sym).diff(x_sym) + 2*y_func(x_sym), x_sym)
sol = sp.dsolve(eq, y_func(x_sym), ics={y_func(0): 5})
print(f"SymPy Solution: {sol}")

# Plotting
fig, ax = plt.subplots(facecolor='#0f172a')
ax.set_facecolor('#1e293b')
x_vals = np.linspace(-1, 3, 400)
# y(x) = 5.25 * exp(-2x) + x/2 - 0.25
y_vals = 5.25 * np.exp(-2 * x_vals) + x_vals / 2.0 - 0.25
ax.plot(x_vals, y_vals, color='#38bdf8', linewidth=2.5, label='y(x)')
ax.plot(0, 5, 'ro', markersize=8, label='y(0) = 5')
ax.grid(True, color='#334155', linestyle='--')
ax.spines['bottom'].set_color('#475569')
ax.spines['left'].set_color('#475569')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors='#94a3b8')
ax.set_title("Particular Solution of dy/dx + 2y = x", color='#f8fafc', fontsize=12, pad=10)
ax.set_xlabel("x", color='#94a3b8')
ax.set_ylabel("y", color='#94a3b8')
ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc')
plt.savefig("static/math_solution.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("Plot successfully saved to static/math_solution.png")
```
"""


def _mock_highlights(prompt: str) -> List[Dict]:
    lines = [(float(t), text.strip()) for t, text in _TRANSCRIPT_LINE.findall(prompt)]
    if not lines:
        return []
    duration = lines[-1][0] + 10.0
    picks: List[Dict] = []
    for fraction in (0.2, 0.5, 0.8):
        start, text = min(lines, key=lambda line: abs(line[0] - duration * fraction))
        if any(abs(start - p["start_time"]) < _MOCK_CLIP_SECONDS for p in picks):
            continue
        words = text.split()
        picks.append({
            "title": " ".join(words[:5]).lower().strip(".,!?") or "mock highlight",
            "start_time": round(start, 1),
            "end_time": round(min(start + _MOCK_CLIP_SECONDS, duration), 1),
            "hook_score": 70, "flow_score": 70, "value_score": 70, "trend_score": 50,
            "score": 66,
            "hook_sentence": " ".join(words[:20]),
            "virality_reason": "mock highlight chosen at an even position in the transcript",
        })
    return picks


def mock_response(prompt: str) -> str:
    if "classify the content type" in prompt:
        return json.dumps({"content_type": "other", "density": "medium"})
    if '{"highlights"' in prompt:
        return json.dumps({"highlights": _mock_highlights(prompt)})
    if '"caption"' in prompt and '"hashtags"' in prompt:
        match = _CLIP_TITLE.search(prompt)
        title = (match.group(1).strip() if match else "") or "mock clip"
        return json.dumps({"caption": f"{title[:1].upper()}{title[1:]} (mock caption)", "hashtags": ["#shorts", "#mock"]})
    return MATH_SOLVER_SCRIPT
