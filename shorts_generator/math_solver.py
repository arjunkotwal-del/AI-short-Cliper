import os
import sys
import re
import subprocess
from shorts_generator.highlights import call_openai_llm

MATH_SOLVER_PROMPT_TEMPLATE = """
You are an expert mathematician and Python developer specializing in SymPy and matplotlib.
Your task is to write a self-contained Python script to solve a differential equation analytically or numerically, print the step-by-step explanation and final solution, and plot the solution.

Differential Equation Description: {description}

Requirements for the Python script:
1. Imports: Import sympy as sp, matplotlib.pyplot as plt, numpy as np.
2. Headless Matplotlib: You MUST call `import matplotlib; matplotlib.use('Agg')` BEFORE importing plt to prevent GUI issues.
3. Solving:
   - Define the independent variable (typically x or t) and dependent variable (typically y or u) as a sympy function: `x = sp.Symbol('x')`, `y = sp.Function('y')`.
   - Write the differential equation as a sympy Eq: `eq = sp.Eq(...)`.
   - Try to solve it analytically using `sp.dsolve(eq, y(x))`.
   - If initial conditions are specified (e.g. y(0) = 3), parse and apply them as the `ics` argument in `dsolve`. For example, `ics={{y(0): 3}}`.
   - If analytic solution fails, write fallback numerical integration code using `scipy.integrate.solve_ivp` or Euler's method to compute numerical points.
4. Printing:
   - Print a clear title and the equation.
   - Print a step-by-step mathematical explanation of how to solve this type of equation (e.g. separable, linear first-order, exact, second-order constant coefficients, etc.).
   - Print the final solution equation clearly (both general and particular if initial conditions are given). Use readable formatting (e.g. `y(x) = C1*exp(-2*x) + x/2 - 1/4`).
5. Plotting:
   - If a solution (analytical or numerical) is found, plot the solution function.
   - For a general solution, plot a family of curves (e.g., for 3 different values of the integration constant, like C1 = -1, C1 = 0, C1 = 1) to show the family of solutions.
   - For a particular solution, plot the single specific curve and highlight the initial condition point with a marker.
   - **Modern Aesthetic Style**:
     - Use a dark slate background matching a premium dashboard: `fig, ax = plt.subplots(facecolor='#0f172a')`, `ax.set_facecolor('#1e293b')`.
     - Customize the grid, spines, ticks to be clean and light-colored: e.g. `ax.grid(True, color='#334155', linestyle='--')`, label colors `#94a3b8`, title/labels `#f8fafc`.
     - Use vibrant colored lines for curves: e.g. `#38bdf8` (sky blue) or `#a855f7` (purple) or `#f43f5e` (rose).
     - Set limits properly so the curve is fully visible and clean.
     - Add a legend if multiple curves.
     - Save the plot to "static/math_solution.png" with `plt.savefig("static/math_solution.png", bbox_inches='tight', dpi=150)`.
     - Close the figure: `plt.close(fig)`.
6. Output:
   - The script must only print the explanation and output details.

Your response must be ONLY the Python code block wrapped in ```python and ```. Do not add any text before or after the code block.
"""

def run_math_solver(equation_description: str) -> str:
    print(f"[MathSolver] Analyzing equation: {equation_description}...", flush=True)
    prompt = MATH_SOLVER_PROMPT_TEMPLATE.format(description=equation_description)
    
    try:
        raw_response = call_openai_llm(prompt)
    except Exception as e:
        err_msg = f"Failed to call LLM: {str(e)}"
        print(f"[MathSolver] {err_msg}", flush=True)
        return err_msg

    # Extract the Python code block
    match = re.search(r"```python\s*(.*?)\s*```", raw_response, re.DOTALL)
    if not match:
        code = raw_response.strip()
    else:
        code = match.group(1).strip()
        
    temp_filename = "temp_math_solver.py"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(code)
        
    print("[MathSolver] Executing generated solver script...", flush=True)
    
    try:
        result = subprocess.run(
            [sys.executable, temp_filename],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        # Clean up the temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        if result.returncode == 0:
            output = result.stdout
            print(output, flush=True)
            return output
        else:
            err_msg = f"Solver execution failed:\n{result.stderr}"
            print(f"[MathSolver] {err_msg}", flush=True)
            return err_msg
            
    except Exception as e:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass
        err_msg = f"Execution error: {str(e)}"
        print(f"[MathSolver] {err_msg}", flush=True)
        return err_msg
