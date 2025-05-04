"""Reflect and act agent implementation"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from dotenv import load_dotenv
from groq import Groq

from tool_pattern.tool import validate_arguments, Tool, Dict, Any
from utils.completions import (
    completions_create,
    build_prompt_structure,
    ChatHistory,
    List,
)
from utils.extraction import extract_tag_content, TagContentResult

load_dotenv()

REACT_SYSTEM_PROMPT: str = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools>
%s
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""


class ReactAgent:
    """
    A class that represents an agent using the ReAct logic that interacts with tools to process
    user inputs, make decisions, and execute tool calls. The agent can run interactive sessions,
    collect tool signatures, and process multiple tool calls in a given round of interaction.

    Attributes:
        client (Groq): The Groq client used to handle model-based completions.
        model (str): The name of the model used for generating responses. Default is "llama-3.3-70b-versatile".
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool instances.
    """

    def __init__(
            self,
            tools: Tool | List[Tool],
            model: str = "llama-3.3-70b-versatile",
            system_prompt: str = '',
    ) -> None:
        self.client: Groq = Groq()
        self.model: str = model
        self.system_prompt: str = system_prompt
        inner_tools: List[tools] = tools if isinstance(tools, list) else [tools]
        self.tools_dict: Dict[str, Tool] = {tool.name: tool for tool in inner_tools}

    @property
    def tool_signatures(self) -> str:
        """
        Collects the function signatures of all available tools.

        Returns:
            str: A concatenated string of all tool function signatures in JSON format.
        """
        return "\n".join(map(lambda tool: tool.fn_signature, self.tools_dict.values()))

    def process_tool_calls(self, tool_calls_content: List[str]) -> Dict[str, Any]:
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_calls_content (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """

        def process_single_tool(tool_call_str: str) -> Tuple[str, Any]:
            """Perform a single tool call."""
            tool_call: Dict[str, Any] = json.loads(s=tool_call_str)
            tool_name: str = tool_call["name"]
            tool: Tool = self.tools_dict[tool_name]
            # Validate and execute the tool call
            validated_tool_call: Dict[str, str | Dict[str, Any]] = validate_arguments(
                tool_call=tool_call, tool_signature=json.loads(s=tool.fn_signature)
            )
            return validated_tool_call["id"], tool.run(
                **validated_tool_call["arguments"]
            )

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            return dict(ex.map(process_single_tool, tool_calls_content))

    def run(
            self,
            user_msg: str,
            max_rounds: int = 10,
    ) -> str:
        """
        Executes a user interaction session, where the agent processes user input, generates responses,
        handles tool calls, and updates chat history until a final response is ready or the maximum
        number of rounds is reached.

        Args:
            user_msg (str): The user's input message to start the interaction.
            max_rounds (int): Maximum number of interaction rounds the agent should perform. Default is 10.

        Returns:
            str: The final response generated by the agent after processing user input and any tool calls.
        """
        user_prompt: Dict[str, str] = build_prompt_structure(
            prompt=user_msg, role="user", tag="question"
        )
        if self.tools_dict:
            self.system_prompt += "\n" + REACT_SYSTEM_PROMPT % self.tool_signatures

        chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.system_prompt,
                    role="system",
                ),
                user_prompt,
            ]
        )

        if self.tools_dict:
            # Run the ReAct loop for max_rounds if the tool collection is not empty
            for _ in range(max_rounds):
                completion: str = completions_create(
                    self.client, chat_history, self.model
                )

                response: TagContentResult = extract_tag_content(
                    text=str(completion), tag="response"
                )
                if response.found:
                    return response.content[0]
                tool_calls: TagContentResult = extract_tag_content(
                    text=str(completion), tag="tool_call"
                )
                chat_history.append(msg=build_prompt_structure(prompt=completion, role="assistant"))
                if tool_calls.found:
                    observations: Dict[str, Any] = self.process_tool_calls(
                        tool_calls_content=tool_calls.content
                    )
                    chat_history.append(msg=build_prompt_structure(prompt=f"{observations}", role="user")
                                        )

        return completions_create(
            client=self.client, messages=chat_history, model=self.model
        )
