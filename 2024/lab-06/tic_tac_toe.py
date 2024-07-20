import random
from typing import Annotated, Optional, Literal
import math

import utils

Mark = Literal["X", "O", "-"]
board_state: list[list[Mark]] = [
    ["-", "-", "-"],
    ["-", "-", "-"],
    ["-", "-", "-"],
]


def add_mark(col: int, row: int, mark: Mark) -> str:
    """
    Add a mark to the board at the given column and row.
    """
    global board_state
    if board_state[row][col] != "-":
        raise ValueError("Cell is already occupied")
    board_state[row][col] = mark
    return peek_at_board()


def peek_at_board() -> str:
    """
    Display the current state of the board.
    """
    global board_state
    return "\n".join([" | ".join(row) for row in board_state])


def clear_board() -> None:
    """
    Clear the board.
    """
    global board_state
    board_state = [
        ["-", "-", "-"],
        ["-", "-", "-"],
        ["-", "-", "-"],
    ]


victories = {
    "X": 0,
    "O": 0,
}


def record_victory(mark: Mark) -> str:
    """
    Record a victory for the given mark.
    """
    global victories
    victories[mark] += 1
    message = f"{mark} wins! The score is now X: {victories['X']} - O: {victories['O']}"
    return message


def check_victory(mark: Mark) -> bool:
    """
    Check if the given mark has won the game.
    """
    global board_state
    for row in board_state:
        if all(cell == mark for cell in row):
            return True
    for col in range(3):
        if all(board_state[row][col] == mark for row in range(3)):
            return True
    if all(board_state[i][i] == mark for i in range(3)):
        return True
    if all(board_state[i][2 - i] == mark for i in range(3)):
        return True
    return False


def coin_flip() -> str:
    """
    Flip a coin to determine which player is which mark. If you get "X", you go first. If you get "O", the user goes first.
    """
    return random.choice(["X", "O"])


system_message = (
    "Play tic-tac-toe with the user. "
    "You will be one of the players. "
    "Use the provided tools to interact with the board. "
    "The coin flip decides which mark you'll have and who goes first. "
    "Tell the user the result of the coin flip. "
    "When it is your turn, go ahead and play. "
    "When it is the user's turn, ask them for a move. "
    "Analyze the board and record victories when appropriate. "
    "Celebrate your victories and console the user when they win. "
)
tools = [add_mark, peek_at_board, clear_board, record_victory, coin_flip]
conversation = utils.Conversation(system_message=system_message, tools=tools, model="gpt-4o-mini")
user_message = "I want to play tic-tac-toe."
conversation.add_message(role="user", content=user_message)
conversation.process_chat_completion()
