"""
This is a warmup exercise. The instructions, documentation, and implementations are missing. Let's complete the code
together. We will implement a Rock-Paper-Scissors game. The user will play against the computer. The computer will
randomly choose between Rock, Paper, and Scissors. The user will provide their choice, and the winner will be
determined. The game will continue until the user decides to stop playing.
"""

import random
from typing import Literal

import utils

Choice = Literal["Rock", "Paper", "Scissors"]


def get_computer_choice() -> Choice:
    """ """
    ...


def determine_winner(user_choice: Choice, computer_choice: Choice) -> str:
    """ """
    ...


system_message = ""
tools = [get_computer_choice, determine_winner]

if __name__ == "__main__":
    conversation = utils.Conversation(system_message=system_message, tools=tools)
    user_message = "I want to play Rock-Paper-Scissors."
    conversation.add_message(role="user", content=user_message)
    conversation.process_chat_completion()
    conversation.add_message(role="user", content="Rock")
    conversation.process_chat_completion()
    conversation.add_message(role="user", content="Paper")
    conversation.process_chat_completion()
    conversation.add_message(role="user", content="Scissors")
    conversation.process_chat_completion()
