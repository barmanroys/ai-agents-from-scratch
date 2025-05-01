"""Generate an essay based on the reflection pattern."""

from dotenv import load_dotenv
from groq import Groq

from utils.completions import (
    completions_create,
    build_prompt_structure,
    FixedFirstChatHistory,
)

load_dotenv()

BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change, output this: <OK>
"""


class ReflectionAgent:
    """
    A class that implements a Reflection Agent, which generates responses and reflects
    on them using the LLM to iteratively improve the interaction. The agent first generates
    responses based on provided prompts and then critiques them in a reflection step.

    Attributes:
        model (str): The model name used for generating and reflecting on responses.
        client (Groq): An instance of the Groq client to interact with the language model.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", client: Groq = Groq()):
        self.model: str = model
        self.client: Groq = client

    def run(
        self,
        user_msg: str,
        generation_system_prompt: str = "",
        reflection_system_prompt: str = "",
        n_steps: int = 10,
    ) -> str:
        """
        Runs the ReflectionAgent over multiple steps, alternating between generating a response
        and reflecting on it for the specified number of steps.

        Args:
            user_msg (str): The user message or query that initiates the interaction.
            generation_system_prompt (str, optional): The system prompt for guiding the generation process.
            reflection_system_prompt (str, optional): The system prompt for guiding the reflection process.
            n_steps (int, optional): The number of generate-reflect cycles to perform. Default to 3.

        Returns:
            str: The final generated response after all cycles are completed.
        """
        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        # Given the iterative nature of the Reflection Pattern, we might exhaust the LLM context (or
        # make it really slow). That's the reason I'm limiting the chat history to three messages.
        # The `FixedFirstChatHistory` is a very simple class, that creates a Queue that always keeps
        # fixed the first message. I thought this would be useful for maintaining the system prompt
        # in the chat history.
        total_length: int = 3
        generation_history: FixedFirstChatHistory = FixedFirstChatHistory(
            messages=[
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_msg, role="user"),
            ],
            total_length=total_length,
        )
        reflection_history: FixedFirstChatHistory = FixedFirstChatHistory(
            messages=[
                build_prompt_structure(prompt=reflection_system_prompt, role="system")
            ],
            total_length=total_length,
        )
        counter: int = 0
        while True:
            # Generate the response
            generation: str = completions_create(
                client=self.client, messages=generation_history, model=self.model
            )
            generation_history.append(
                msg=build_prompt_structure(prompt=generation, role="assistant")
            )
            reflection_history.append(
                msg=build_prompt_structure(prompt=generation, role="assistant")
            )
            # Generate the critique
            critique: str = completions_create(
                client=self.client, messages=reflection_history, model=self.model
            )
            if "<OK>" in critique or counter == n_steps:
                return generation
            counter += 1
            generation_history.append(
                msg=build_prompt_structure(prompt=critique, role="user")
            )
            reflection_history.append(
                msg=build_prompt_structure(prompt=critique, role="assistant")
            )
