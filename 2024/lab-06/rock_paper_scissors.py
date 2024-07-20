"""

"""

import random
from typing import Literal

import utils

Choice = Literal["Rock", "Paper", "Scissors"]


def get_computer_choice() -> Choice:
    return random.choice(["Rock", "Paper", "Scissors"])


def determine_winner(user_choice: Choice, computer_choice: Choice) -> str:
    if user_choice == computer_choice:
        return "It's a tie!"
    if (
        (user_choice == "Rock" and computer_choice == "Scissors")
        or (user_choice == "Paper" and computer_choice == "Rock")
        or (user_choice == "Scissors" and computer_choice == "Paper")
    ):
        return "User wins!"
    return "Computer wins!"


system_message = (
    "Play Rock-Paper-Scissors with the user. "
    "Use the provided tools to make a choice and determine the winner. "
    "Ask the user for their choice, then make your choice, and determine the winner. "
    "Celebrate your victories and console the user when they win. "
)
tools = [get_computer_choice, determine_winner]
conversation = utils.Conversation(system_message=system_message, tools=tools, model="gpt-4o-mini")
user_message = "I want to play Rock-Paper-Scissors."
conversation.add_message(role="user", content=user_message)
conversation.process_chat_completion()
