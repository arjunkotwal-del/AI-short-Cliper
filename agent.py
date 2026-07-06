import inspect
import json
from typing import Callable, Dict, List, Optional, Any
from openai import OpenAI

from shorts_generator.config import require_openai_key, OPENAI_MODEL


class Agent:
    def __init__(self, name: str, instructions: str, tools: List[Callable] = None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.tool_map = {tool.__name__: tool for tool in self.tools}

    def get_tool_schemas(self) -> List[Dict]:
        schemas = []
        for tool in self.tools:
            schemas.append(function_to_schema(tool))
        return schemas


def function_to_schema(func: Callable) -> Dict:
    """Convert a Python function to an OpenAI tool schema using inspect."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for name, param in sig.parameters.items():
        if name == "self":
            continue
            
        param_type = "string"
        if param.annotation == int:
            param_type = "integer"
        elif param.annotation == float:
            param_type = "number"
        elif param.annotation == bool:
            param_type = "boolean"
        elif param.annotation == dict:
            param_type = "object"
        elif param.annotation == list:
            param_type = "array"
            
        parameters["properties"][name] = {
            "type": param_type,
            "description": f"Parameter {name}"
        }
        
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)
            
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.split("\n")[0] if doc else f"Call {func.__name__}",
            "parameters": parameters
        }
    }


def get_mock_agent_response(agent: Agent, messages: List[Any]) -> Any:
    """Helper to mock OpenAI responses for offline testing."""
    class MockToolCallFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class MockToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            self.type = "function"
            self.function = MockToolCallFunction(name, arguments)

    class MockMessage:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []
            self.role = "assistant"
        def model_dump(self, **kwargs):
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in self.tool_calls
                ] if self.tool_calls else None
            }

    # Find last user query
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    query_lower = user_query.lower()
    

    
    if agent.name == "Orchestrator":
        has_finished = False
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("name") == "finish_task":
                has_finished = True
                break
                
        has_solved = False
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("name") == "solve_differential_equation":
                has_solved = True
                break
                
        if has_finished:
            return MockMessage(content="Successfully solved the differential equation and plotted the solution.")
        elif has_solved:
            sol_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "tool" and msg.get("name") == "solve_differential_equation":
                    sol_content = msg.get("content", "")
                    break
            import json
            return MockMessage(
                content=f"Task Complete! Here is the solution:\n\n{sol_content}",
                tool_calls=[MockToolCall("mock_tc_4", "finish_task", json.dumps({"message": "Successfully solved the differential equation and plotted the solution."}))]
            )
        elif any(term in query_lower for term in ["solve", "equation", "math", "dy/dx", "y'", "y''", "ode"]):

            return MockMessage(
                content="This is a mathematical problem. Transferring to the MathSolver agent.",
                tool_calls=[MockToolCall("mock_tc_1", "transfer_to_math_solver", "{}")]
            )
        else:
            return MockMessage(
                content="Mock Orchestrator: Please provide a mathematical equation to solve (e.g. solve dy/dx + 2y = x, y(0)=5)!"
            )
            
    elif agent.name == "MathSolver":
        has_solved = False
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("name") == "solve_differential_equation":
                has_solved = True
                break
        

        if not has_solved:
            import json
            return MockMessage(
                content="I will solve this differential equation using SymPy and plot it.",
                tool_calls=[MockToolCall("mock_tc_2", "solve_differential_equation", json.dumps({"equation_description": user_query}))]
            )
        else:
            return MockMessage(
                content="I have solved the equation and saved the plot to static/math_solution.png. Task complete!",
                tool_calls=[MockToolCall("mock_tc_3", "transfer_to_orchestrator", "{}")]
            )
            
    return MockMessage(content=f"Mock response for agent {agent.name}")


def run_agent_loop(agent: Agent, messages: List[Dict]) -> Dict:
    """Runs the agent loop until it returns a final string or switches agents."""
    import os
    mock_enabled = os.getenv("MOCK_LLM", "false").lower() == "true"
    
    if not mock_enabled:
        client = OpenAI(api_key=require_openai_key())
    
    # Prepend instructions
    messages = [{"role": "system", "content": agent.instructions}] + messages
    
    while True:
        # Prepare tools
        tools = agent.get_tool_schemas()
        
        if mock_enabled:
            message = get_mock_agent_response(agent, messages)
        else:
            kwargs = {
                "model": OPENAI_MODEL,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
                
            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
        
        messages.append(message.model_dump(exclude_unset=True) if hasattr(message, "model_dump") else message)
        
        if not message.tool_calls:
            # Done! Agent replied with text
            return {"agent": agent, "messages": messages, "response": message.content}
            
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            print(f"\n[{agent.name}] Calling tool: {func_name}({args})", flush=True)
            
            if func_name in agent.tool_map:
                func = agent.tool_map[func_name]
                try:
                    result = func(**args)
                    # Handoff pattern: If a tool returns an Agent, we switch control
                    if isinstance(result, Agent):
                        print(f"[{agent.name}] Handing off to {result.name}...", flush=True)
                        # Add a fake tool response to keep message history valid
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": f"Transferred to {result.name}"
                        })
                        # IMPORTANT: When switching agents, we re-run the loop with the new agent.
                        # We must strip the old system prompt and let the new agent inject its own.
                        # Since we prepended the system prompt at the start of THIS function, 
                        # we pop it off before delegating.
                        messages.pop(0)
                        return run_agent_loop(result, messages)
                    
                    # Normal tool result
                    content = str(result)
                except Exception as e:
                    content = f"Error: {e}"
                    print(f"[{agent.name}] Tool error: {e}")
            else:
                content = f"Error: Tool {func_name} not found."
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": content
            })
