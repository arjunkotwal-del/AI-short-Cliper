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


def run_agent_loop(agent: Agent, messages: List[Dict]) -> Dict:
    """Runs the agent loop until it returns a final string or switches agents."""
    client = OpenAI(api_key=require_openai_key())
    
    # Prepend instructions
    messages = [{"role": "system", "content": agent.instructions}] + messages
    
    while True:
        # Prepare tools
        tools = agent.get_tool_schemas()
        
        kwargs = {
            "model": OPENAI_MODEL,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        
        messages.append(message.model_dump(exclude_unset=True))
        
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
