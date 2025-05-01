"""Tool agents implementation."""

import json
from typing import List

from dotenv import load_dotenv
from groq import Groq

from tool import validate_arguments, Tool, Dict, Any
from utils.completions import completions_create, build_prompt_structure, ChatHistory
from utils.extraction import extract_tag_content, TagContentResult

load_dotenv()

TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.
For each function call return a json object with function name and arguments within <tool_call></tool_call>
XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>,  "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>
"""


class ToolAgent:
    """
    The ToolAgent class represents an agent that can interact with a language model and use tools
    to assist with user queries. It generates function calls based on user input, validates arguments,
    and runs the respective tools.

    Attributes:
        tools (Tool | list[Tool]): A list of tools available to the agent.
        model (str): The model to be used for generating tool calls and responses.
        client (Groq): The Groq client used to interact with the language model.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool objects.
    """

    def __init__(
        self,
        tools: Tool | List[Tool],
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.client: Groq = Groq()
        self.model: str = model
        self.tools: List[Tool] = tools if isinstance(tools, list) else [tools]
        self.tools_dict: Dict[str, Tool] = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        """
        Collects the function signatures of all available tools.

        Returns:
            str: A concatenated string of all tool function signatures in JSON format.
        """
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> Dict[str, Any]:
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_calls_content (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """
        observations: Dict[str, Any] = {}
        for tool_call_str in tool_calls_content:
            tool_call: Dict = json.loads(tool_call_str)
            tool_name: str = tool_call["name"]
            tool: Tool = self.tools_dict[tool_name]

            # Validate and execute the tool call
            validated_tool_call = validate_arguments(
                tool_call=tool_call, tool_signature=json.loads(tool.fn_signature)
            )
            # Store the result using the tool call ID
            observations[validated_tool_call["id"]] = tool.run(
                **validated_tool_call["arguments"]
            )

        return observations

    def run(
        self,
        user_msg: str,
    ) -> str:
        """
        Handles the full process of interacting with the language model and executing a tool based on user input.

        Args:
            user_msg (str): The user's message that prompts the tool agent to act.

        Returns:
            str: The final output after executing the tool and generating a response from the model.
        """
        user_prompt: Dict[str, str] = build_prompt_structure(
            prompt=user_msg, role="user"
        )

        tool_chat_history: List[Dict[str, str]] = ChatHistory(
            [
                build_prompt_structure(
                    prompt=TOOL_SYSTEM_PROMPT % self.add_tool_signatures(),
                    role="system",
                ),
                user_prompt,
            ]
        )
        agent_chat_history: List[Dict[str, str]] = ChatHistory([user_prompt])

        tool_call_response: str = completions_create(
            self.client, messages=tool_chat_history, model=self.model
        )
        tool_calls: TagContentResult = extract_tag_content(
            text=str(tool_call_response), tag="tool_call"
        )

        if tool_calls.found:
            observations: Dict[str, Any] = self.process_tool_calls(tool_calls.content)
            agent_chat_history.append(
                build_prompt_structure(
                    prompt=f'f"Observation: {observations}"', role="user"
                )
            )
        return completions_create(self.client, agent_chat_history, self.model)
