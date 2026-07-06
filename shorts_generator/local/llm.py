"""Local LLM backend — calls OpenAI directly so no MuAPI account is needed."""
from ..config import OPENAI_MODEL, require_openai_key

_TIMEOUT = 120  # seconds per request
_MAX_RETRIES = 2


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend with timeout and retry."""
    import os
    if os.getenv("MOCK_LLM", "false").lower() == "true":
        # Return a valid Python script that solves the ODE y' + 2y = x, y(0)=5
        return """
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

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from e

    client = OpenAI(
        api_key=require_openai_key(),
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""
