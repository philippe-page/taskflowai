from pydantic import BaseModel
from pydantic.fields import Field
from typing import Callable, Optional, Union, Dict, List, Any, Set, Tuple
from datetime import datetime
import json, re, hashlib

COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "LABEL": "\033[96m",  # Cyan
    "TOOL_NAME": "\033[1;34m",  # Bold Blue for tool names
    "PARAM_VALUE": "\033[1;35m",  # Bold Magenta (Pink) for parameter values
    "RESULT_VALUE": "\033[32m",  # Green for result values
    "WARNING": "\033[1;33m",  # Bold Yellow for warnings
    "ERROR": "\033[1;31m",  # Bold Red for tool errors
}

class Task(BaseModel):
    role: str = Field(..., description="The role or type of agent performing the task")
    goal: str = Field(..., description="The objective or purpose of the task")
    attributes: Optional[str] = Field(None, description="Additional attributes or characteristics of the agent or expected responses")
    context: Optional[str] = Field(None, description="The background information or setting for the task")
    instruction: str = Field(..., description="Specific directions for completing the task")
    llm: Callable = Field(..., description="The language model function to be called")
    image_data: Optional[Union[List[str], str]] = Field(None, description="Optional base64-encoded image data")
    agent: Optional[Any] = Field(None, description="The agent associated with this task")
    tools: Optional[Set[Callable]] = Field(default=None, description="Optional set of tool functions")
    temperature: Optional[float] = Field(default=0.7, description="Temperature setting for the language model")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum number of tokens for the language model response")
    require_json_output: bool = Field(default=False, description="Whether to request JSON output from the LLM")
    model_config = {
        "arbitrary_types_allowed": True
    }

    @classmethod
    def create(cls, agent: Optional[Any] = None,
                role: Optional[str] = None, 
                goal: Optional[str] = None,
                attributes: Optional[str] = None, 
                context: Optional[str] = None,
                instruction: Optional[str] = None, 
                llm: Optional[Callable] = None,
                tools: Optional[Set[Callable]] = None, 
                image_data: Optional[Union[List[str], str]] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                require_json_output: bool = False,
                callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Union[str, Exception]:
        """
        Create and execute a task with flexible parameter assignment.

        This method allows for two primary ways of task creation:
        1. By assigning an agent, which provides default values for most parameters.
        2. By directly specifying individual parameters.

        The core components of a task are:
        - agent: An optional agent object that provides default values.
        - context: Optional background information for the task.
        - instruction: Specific directions for the task.
        - require_json_output: Optional boolean to require JSON output from the LLM.

        Additional parameters can be specified to override agent settings or when no agent is provided:
        - role: The role or type of agent performing the task.
        - goal: The objective of the task.
        - attributes: Additional characteristics of the agent or expected responses.
        - llm: The language model function to be used.
        - tools: A set of tool functions available for the task.
        - image_data: Optional image data for image-based tasks.
        - temperature: Temperature setting for the language model.
        - max_tokens: Maximum number of tokens for the language model response.
        - callback: An optional function to handle task execution updates.

        Returns:
            Union[str, Exception]: The result of task execution or an Exception if an error occurs.
        """
        try:
            task_data = {
                "role": role or (agent.role if agent else None),
                "goal": goal or (agent.goal if agent else None),
                "attributes": attributes or (agent.attributes if agent else None),
                "context": context,
                "instruction": instruction,
                "llm": llm or (agent.llm if agent else None),
                "tools": tools or (agent.tools if agent else None),
                "image_data": image_data,
                "temperature": temperature or (agent.temperature if agent else 0.7),
                "max_tokens": max_tokens or (agent.max_tokens if agent else 4000),
                "require_json_output": require_json_output,
                "agent": agent
            }
            task = cls.model_validate(task_data)
            return task.execute(callback)
        except Exception as e:
            if callback:
                callback({"type": "error", "content": str(e)})
            return e

    def execute(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Union[str, Exception]:
        tools_to_use = self.agent.tools if self.agent and self.agent.tools else self.tools
        if tools_to_use:
            tool_usage_history = self._execute_tool_loop(callback)
            if isinstance(tool_usage_history, Exception):
                return tool_usage_history
            return self._execute_final_task(tool_usage_history, callback)
        else:
            llm_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "image_data": self.image_data,
                "require_json_output": self.require_json_output
            }

            llm_result = self.llm(
                self.system_prompt(), 
                self.user_prompt(), 
                **llm_params
            )
            
            if isinstance(llm_result, tuple) and len(llm_result) == 2:
                response, error = llm_result
            elif isinstance(llm_result, dict):
                response = json.dumps(llm_result)
                error = None
            elif isinstance(llm_result, str):
                response = llm_result
                error = None
            else:
                error = ValueError(f"Unexpected result type from LLM: {type(llm_result)}")
                response = ""

            if error:
                if callback:
                    callback({"type": "error", "content": str(error)})
                return error
            if callback:
                callback({"type": "final_response", "content": response})
            return response

    def system_prompt(self) -> str:
        attributes = f" {self.attributes}" if self.attributes else ""
        article = "an" if self.role.lower()[0] in "aeiou" else "a"
        return f"You are {article} {self.role}.{attributes} Your goal is to {self.goal}.\n"

    def user_prompt(self) -> str:
        prompt = ""
        if self.context:
            prompt += f"{self.context}\n\n"
        prompt += self.instruction
        return prompt


    def _execute_tool_loop(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Union[str, Exception]:
        tool_descriptions = "\nYou can call the following functions to gather information or perform actions:\n" + "\n".join([f"- {func.__name__}: {func.__doc__}" for func in self.tools]).rstrip()
        tool_usage_history = ""
        previous_calls = {}  # Reset this dictionary for each new task execution

        max_iterations = 6
        for _ in range(max_iterations):
            
            # Set these variables based on whether there's been any tool usage
            more = "more " if tool_usage_history else ""
            additional = "additional " if tool_usage_history else ""

            # Build the context string
            context_parts = []

            if self.context:
                context_parts.append(self.context)

            context_parts.append(tool_descriptions)

            if tool_usage_history:
                context_parts.append(f"Tool Usage History:\n{tool_usage_history}")

            context = "\n-----\n".join(context_parts).strip()

            tool_loop_task = Task(
                role=f"{self.role}",
                goal=f"{self.goal}",
                attributes=f"You are {self.attributes}. You do not call functions when you have sufficient information, or have completed all necessary actions and function calls. You understand tools cost money and time. You avoid at all costs repeating function calls with the exact same parameters." if self.attributes else "",
                instruction=f"""
=====
The original task instruction:
{self.instruction}
=====

You are now determining if you need to call {more}tools to gather {more}information or perform {additional}actions to complete the given task, or if the task is complete and you are ready to provide your final response. 

Now respond with a JSON object in one of these formats to either request tool calls, or to indicate you are ready to provide your final response:

If tool calls are still needed for information gathering or task execution:
{{
    "tool_calls": [
        {{
            "tool": "tool_name",
            "params": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
    ]
}}

If no more tool calls are required:
{{
    "status": "READY"
}}

If you need to make tool calls, consider whether you need to make tool calls successively or all at once. If the result of one tool call is required as input for another tool, make your calls one at a time. If multiple tool calls can be made independently, you may request them all at once. Successive tool calls can also be made one after the other, allowing you to wait for the result of one tool call before making another if needed. Now respond with a JSON object that *either requests tool calls, or indicates you are ready** to provide your final response. Do NOT comment before or after the JSON; return *only a valid JSON* in any case.
""",
                context=context,
                llm=self.llm,
                tools=self.tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            try:
                llm_result = tool_loop_task.llm(
                    tool_loop_task.system_prompt(),
                    tool_loop_task.user_prompt(),
                    require_json_output=True
                )
                
                if isinstance(llm_result, tuple) and len(llm_result) == 2:
                    response, error = llm_result
                elif isinstance(llm_result, dict):
                    response = json.dumps(llm_result)
                    error = None
                elif isinstance(llm_result, str):
                    response = llm_result
                    error = None
                else:
                    raise ValueError(f"Unexpected result type from LLM: {type(llm_result)}")

                if error:
                    if callback:
                        callback({"type": "error", "content": str(error)})
                    return error
                
                try:
                    # Find the first occurrence of '{' and the last occurrence of '}'
                    start = response.find('{') 
                    end = response.rfind('}') + 1
                    if start != -1 and end != -1:
                        json_str = response[start:end]
                        tool_requests = json.loads(json_str)
                    else:
                        raise json.JSONDecodeError("No JSON object found", response, 0)

                    if isinstance(tool_requests, dict):
                        if 'tool_calls' in tool_requests:
                            tool_calls = tool_requests['tool_calls']
                            if not tool_calls:  # Empty tool_calls array
                                print(f"{COLORS['LABEL']}Status: {COLORS['PARAM_VALUE']}READY{COLORS['RESET']}")
                                print() 
                                print(COLORS['RESET'], end='')
                                return tool_usage_history
                            
                            for tool_request in tool_calls:
                                tool_name = tool_request["tool"].lower()
                                tool_params = tool_request.get("params", {})

                                # Generate a hash for this tool call
                                call_hash = hashlib.md5(f"{tool_name}:{json.dumps(tool_params, sort_keys=True)}".encode()).hexdigest()

                                # Increment the call count before checking
                                previous_calls[call_hash] = previous_calls.get(call_hash, 0) + 1

                                # Debug print to check the state of previous_calls
                                #print(f"DEBUG: previous_calls before increment: {previous_calls}")

                                if previous_calls[call_hash] == 3:
                                    # Third occurrence (second repeat), warn the LLM
                                    warning_message = (
                                        f"\n\nWarning: You've repeated the exact same tool call for '{tool_name}'."
                                        "**Please try a different tool, different parameters, or consider that you may be ready to proceed with your final response."
                                    )
                                    tool_usage_history += warning_message
                                    tool_loop_task.context += warning_message
                                    print(f"{COLORS['WARNING']}Warning: Tool call for '{tool_name}' has been repeated verbatim three times. "
                                          f"Skipping execution and notifying LLM.{COLORS['RESET']}")
                                    continue
                                elif previous_calls[call_hash] == 4:
                                    # Fourth occurrence (third repeat), exit the loop
                                    print(f"{COLORS['WARNING']}Tool call for '{tool_name}' has been repeated verbatim four times. "
                                          f"Exiting loop to proceed to the final response.{COLORS['RESET']}")
                                    return tool_usage_history

                                print(f"{COLORS['LABEL']}Tool Use: {COLORS['TOOL_NAME']}{tool_name}{COLORS['RESET']}")
                                print(f"{COLORS['LABEL']}Parameters:")
                                for key, value in tool_params.items():
                                    print(f"  {COLORS['LABEL']}{key}: {COLORS['PARAM_VALUE']}{value}{COLORS['RESET']}")
                                print() 
                                print(COLORS['RESET'], end='')

                                # Create a dictionary of tools with function names as keys
                                tools_dict = {func.__name__.lower(): func for func in self.tools}
                                
                                if tool_name in tools_dict:
                                    try:
                                        result = tools_dict[tool_name](**tool_params)
                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        tool_result = f"\n\nYou just used the tool: '{tool_name}' with the parameters: {json.dumps(tool_params, indent=2)}\n\nThe following is the result of the tool's use:\n{result}"
                                    except Exception as e:
                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        tool_result = f"\n\nYou just used the tool: '{tool_name}' with the parameters: {json.dumps(tool_params, indent=2)}\nAn error occurred during the tool's use: {str(e)}"
                                        print(f"{COLORS['WARNING']}[{timestamp}] Error executing tool '{tool_name}': {str(e)}{COLORS['RESET']}")
                                    
                                    tool_usage_history += tool_result
                                    tool_loop_task.context += tool_result

                                    # Print a snippet of the result
                                    result_snippet = str(result)[:400] + "..." if len(str(result)) > 400 else str(result)
                                    print(f"{COLORS['LABEL']}Result: {COLORS['RESULT_VALUE']}{result_snippet}{COLORS['RESET']}")
                                    print()  # Add a newline for  separation

                                    if callback:
                                        callback({
                                            "type": "tool_call",
                                            "tool": tool_name,
                                            "params": tool_params,
                                            "result": result
                                        })

                                else:
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    tool_result = f"\n\nYou just attempted to use the tool: '{tool_name}' with the parameters: {json.dumps(tool_params, indent=2)}\nError: Tool '{tool_name}' not found."
                                    tool_usage_history += tool_result
                                    tool_loop_task.context += tool_result
                                    print(f"{COLORS['WARNING']}[{timestamp}] Tool Error: Tool '{tool_name}' not found.{COLORS['RESET']}")

                        elif 'status' in tool_requests and tool_requests['status'] == 'READY':
                            print(f"{COLORS['LABEL']}Status: {COLORS['PARAM_VALUE']}READY{COLORS['RESET']}")
                            print() 
                            print(COLORS['RESET'], end='')
                            return tool_usage_history
                        
                        elif not tool_requests:
                            print(f"{COLORS['LABEL']}Status: {COLORS['PARAM_VALUE']}READY{COLORS['RESET']}")
                            print()
                            print(COLORS['RESET'], end='')
                            return tool_usage_history
                        
                        else:
                            raise ValueError("Invalid response format")
                    elif isinstance(tool_requests, list) and not tool_requests:
                        print(f"{COLORS['LABEL']}Status: {COLORS['PARAM_VALUE']}READY{COLORS['RESET']}")
                        print()
                        print(COLORS['RESET'], end='')
                        return tool_usage_history
                    else:
                        raise ValueError("Invalid response format")

                except (json.JSONDecodeError, ValueError) as e:
                    error_message = f"\n\nError: {str(e)} Please provide a valid JSON object."
                    tool_usage_history += error_message
                    tool_loop_task.context += error_message
                    print(f"{COLORS['WARNING']}Tool Error: {str(e)} Please provide a valid JSON object.{COLORS['RESET']}")

            except Exception as e:
                error_message = f"\n\nError during tool loop execution: {str(e)}"
                tool_usage_history += error_message
                tool_loop_task.context += error_message
                print(f"{COLORS['ERROR']}Error during tool loop execution: {str(e)}{COLORS['RESET']}")
                if callback:
                    callback({"type": "error", "content": str(e)})
                return e

        warning_message = "\n\nWarning: Maximum iterations of tool use loop reached without completion."
        print(f"{COLORS['ERROR']}Warning: Maximum iterations of tool use loop reached without completion.{COLORS['RESET']}")
        tool_usage_history += warning_message
        return tool_usage_history
    
    def _execute_final_task(self, tool_usage_history: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Union[str, Dict, Exception]:
        if tool_usage_history:
            final_context = f"{self.context or ''}\n-----------\nYou've just performed the following actions:\n{tool_usage_history}"
        else:
            final_context = self.context or ''

        instruction = f"Now focus on addressing the instruction:\n{self.instruction}"
        if self.require_json_output:
            instruction += "\nDo NOT comment before or after the JSON. Provide only a valid JSON object as your response."

        updated_task = Task(
            role=self.role,
            goal=self.goal,
            attributes=self.attributes,
            context=final_context,
            instruction=instruction,
            llm=self.llm,
            tools=self.tools,
            image_data=self.image_data,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            require_json_output=self.require_json_output
        )
        
        try:
            llm_params = {
                "temperature": updated_task.temperature,
                "max_tokens": updated_task.max_tokens,
                "image_data": updated_task.image_data,
            }
            # Only add require_json_output if it's explicitly set to True
            if updated_task.require_json_output:
                llm_params["require_json_output"] = True

            llm_result = updated_task.llm(updated_task.system_prompt(), updated_task.user_prompt(), **llm_params)
            
            if isinstance(llm_result, tuple) and len(llm_result) == 2:
                response, error = llm_result
            elif isinstance(llm_result, dict):
                response = json.dumps(llm_result)
                error = None
            elif isinstance(llm_result, str):
                response = llm_result
                error = None
            else:
                error = ValueError(f"Unexpected result type from LLM: {type(llm_result)}")
                response = ""

            if error:
                if callback:
                    callback({"type": "error", "content": str(error)})
                return error
            
            if self.require_json_output:
                if isinstance(response, str):
                    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            response = json.loads(json_str)
                        except json.JSONDecodeError:
                            error = ValueError(f"Failed to parse JSON from LLM response: {response}")
                            if callback:
                                callback({"type": "error", "content": str(error)})
                            return error
                    else:
                        error = ValueError(f"No JSON object found in LLM response: {response}")
                        if callback:
                            callback({"type": "error", "content": str(error)})
                        return error
                elif not isinstance(response, dict):
                    error = ValueError(f"Expected JSON object, got {type(response)}: {response}")
                    if callback:
                        callback({"type": "error", "content": str(error)})
                    return error

            if callback:
                callback({"type": "final_response", "content": response})

            return response
        
        except Exception as e:
            if callback:
                callback({"type": "error", "content": str(e)})
            return e