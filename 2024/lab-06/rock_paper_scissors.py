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
    """
    You choose rock, paper, or scissors.
    """
    return random.choice(["Rock", "Paper", "Scissors"])


def determine_winner(user_choice: Choice, computer_choice: Choice) -> str:
    """
    Correctly determines the winner and loser of a Rock-Paper-Scissors game, returns winner.
    """
    # Strategy map pattern to pick winners and losers
    possibilities = {
        "Rock": {"Rock": "Tie", "Paper": "Computer", "Scissors": "User"},
        "Paper": {"Rock": "User", "Paper": "Tie", "Scissors": "Computer"},
        "Scissors": {"Rock": "Computer", "Paper": "User", "Scissors": "Tie"},
    }
    return possibilities[user_choice][computer_choice]


system_message = (
    "You're playing rock paper scissors. "
    "Use the tools to make a choice, compare it to the user, and determine the winner. "
    "Don't choose or reveal choice until the user has made their choice. "
    "Congratulate or console the users. "
    "You can stop when the user says so (i.e. 'exit'). "
)
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
