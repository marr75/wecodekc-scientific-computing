from __future__ import annotations

import functools
from typing import Callable, Literal

from llm_easy_tools import get_tool_defs, process_response, ToolResult
from openai import OpenAI
from openai.types import chat as chat_types
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from dotenv import load_dotenv


load_dotenv()
client = OpenAI()


# Dictionary to map roles to colors for pretty printing
role_to_color = {
    "system": "red",
    "user": "green",
    "assistant": "blue",
    "function": "magenta",
}


def pretty_print_conversation(messages: list[dict]) -> None:
    """
    Pretty prints the conversation with colors based on roles.

    Args:
        messages (list[dict]): List of message dictionaries.
    """
    for message in messages:
        if hasattr(message, "to_dict"):
            message = message.to_dict()
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("tool_calls"):
            print(colored(f"assistant: {message['tool_calls']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))


# Default GPT model to be used
GPT_MODEL = "gpt-4o-mini"


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages: list[chat_types.ChatCompletionMessageParam],
    tools: list[chat_types.ChatCompletionToolParam] = None,
    tool_choice: Literal["none", "auto", "required"] = "auto",
    model: str = GPT_MODEL,
) -> chat_types.ChatCompletion:
    """
    Requests a chat completion from the OpenAI API with retry logic.

    Args:
        messages (list[ChatCompletionMessageParam]): List of messages in the conversation.
        tools (list[ChatCompletionToolParam], optional): List of tools available for the conversation.
        tool_choice (Literal["none", "auto", "required"], optional): How tools should be used. Defaults to "auto".
        model (str, optional): The model to use. Defaults to GPT_MODEL.

    Returns:
        ChatCompletion: The response from the API.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tools and tool_choice or "none",
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")


class Conversation:
    """
    A class to manage and interact with a chat conversation.
    """

    conversation_history: list[chat_types.ChatCompletionMessageParam]
    tool_calls: list[ToolResult]
    tools: list[Callable]
    mode: str

    @functools.cached_property
    def tool_defs(self):
        """
        Retrieves the definitions for the tools.

        Returns:
            list: Definitions of the tools.
        """
        return get_tool_defs(self.tools)

    def __init__(self, system_message: str, tools: list[Callable], model: str = "gpt-3.5-turbo") -> None:
        """
        Initializes the conversation with a system message and tools.

        Args:
            system_message (str): The initial system message.
            tools (list[Callable]): List of tools available for the conversation.
            model (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        """
        self.system_message = system_message
        self.conversation_history = []
        self.add_message("system", system_message)
        self.tools = tools
        self.tool_calls = []
        self.model = model

    def add_message(
        self,
        role: str | None = None,
        content: str | None = None,
        name: str | None = None,
    ) -> None:
        """
        Adds a message to the conversation history.

        Args:
            role (str, optional): The role of the sender of the message.
            content (str, optional): The content of the message.
            name (str, optional): The name of the tool (if applicable).
        """
        message: chat_types.ChatCompletionMessageParam = {"role": role, "content": content}
        if name:
            message["name"] = name
        self.conversation_history.append(message)

    def add_message_from_response(self, response_message: chat_types.ChatCompletionMessage) -> None:
        """
        Adds a message from a response to the conversation history.

        Args:
            response_message (ChatCompletionMessage): The response message to add.
        """
        self.conversation_history.append(response_message)

    def add_message_from_tool_result(self, tool_call: ToolResult) -> None:
        """
        Adds a message from a tool result to the conversation history.

        Args:
            tool_call (ToolResult): The tool result to add.
        """
        self.conversation_history.append(tool_call.to_message())

    def display_conversation(self) -> None:
        """
        Displays the conversation history with pretty printing.
        """
        pretty_print_conversation(self.conversation_history)

    def process_chat_completion(self) -> None:
        """
        Processes the chat completion, managing the conversation and tool calls.
        """
        finish_reason = None
        while finish_reason != "stop" and finish_reason != "length":
            response = chat_completion_request(
                messages=self.conversation_history, tools=self.tool_defs, model=self.model
            )
            response_message = response.choices[0].message
            self.add_message_from_response(response_message)
            tool_call_results = process_response(response, self.tools)
            if tool_call_results:
                self.tool_calls += tool_call_results
                for tool_call_result in tool_call_results:
                    self.add_message_from_tool_result(tool_call_result)
            finish_reason = response.choices[0].finish_reason
        # Append the tool call results to the conversation history
        self.display_conversation()
