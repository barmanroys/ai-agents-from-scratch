"""Helper functions and classes."""

from typing import List, Optional, Dict

from groq import Groq


def completions_create(client: Groq, messages: list, model: str) -> str:
    """
    Sends a request to the client's `completions.create` method to interact with the language model.

    Args:
        client (Groq): The Groq client object
        messages (list[dict]): A list of message objects containing chat history for the model.
        model (str): The model to use for generating tool calls and responses.

    Returns:
        str: The content of the model's response.
    """
    return str(client.chat.completions.create(messages=messages, model=model))


def build_prompt_structure(prompt: str, role: str, tag: str = "") -> dict:
    """
    Builds a structured prompt that includes the role and content.

    Args:
        prompt (str): The actual content of the prompt.
        role (str): The role of the speaker (e.g. user, assistant).
        tag: (str): Possible HTML or mark-down tags

    Returns:
        dict: A dictionary representing the structured prompt.
    """
    if tag:
        prompt = f"<{tag}>{prompt}</{tag}>"
    return {"role": role, "content": prompt}


def update_chat_history(history: list, msg: str, role: str):
    """
    Updates the chat history by appending the latest response.

    Args:
        history (list): The list representing the current chat history.
        msg (str): The message to append.
        role (str): The role type (e.g. 'user', 'assistant', 'system')
    """
    history.append(build_prompt_structure(prompt=msg, role=role))


class ChatHistory(List):
    """Implement the chat history class."""

    def __init__(self, messages: Optional[List[Dict]] = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        messages = messages or []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: Dict) -> None:
        """Add a message to the queue.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(0)  # Remove the earliest message in case of buffer overflow
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: Optional[List[Dict]] = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        super().__init__(messages, total_length)

    def append(self, msg: Dict):
        """Add a message to the queue. The zero-th messaage will always stay fixed.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(
                1
            )  # Keep the zero-th message but pop the next one in case of buffer overlow
        super().append(msg)
