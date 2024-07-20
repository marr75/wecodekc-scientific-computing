from typing import Annotated, Optional, Literal
import math

import utils


operations = Literal["+", "-", "/", "*", "^", "sqrt", "log", "sin", "cos", "tan", "cot", "sec", "csc"]


def calculate(x: float, y: Optional[float], operation: operations) -> float:
    """
    Perform a mathematical operation on 1 or 2 numbers.
    """
    if operation == "+":
        return x + y
    elif operation == "-":
        return x - y
    elif operation == "/":
        return x / y
    elif operation == "*":
        return x * y
    elif operation == "^":
        return x**y
    elif operation == "sqrt":
        return x ** (1 / 2)
    elif operation == "log":
        return x ** (1 / y)
    elif operation == "sin":
        return math.sin(x)
    elif operation == "cos":
        return math.cos(x)
    elif operation == "tan":
        return math.tan(x)
    elif operation == "cot":
        return 1 / math.tan(x)
    elif operation == "sec":
        return 1 / math.cos(x)
    elif operation == "csc":
        return 1 / math.sin(x)
    else:
        raise ValueError(f"Invalid operation: {operation}")


memory_stack = []


def memory_store(value: float) -> str:
    """
    Store a value in memory (a stack).
    """
    memory_stack.append(value)
    return "Value stored in memory."


def memory_recall() -> float:
    """
    Recall the value stored in memory (a stack).
    """
    return memory_stack.pop()


system_message = "Calculate arbitrary math expressions for the user. ALWAYS Store final values for later use."
conversation = utils.Conversation(
    system_message=system_message, tools=[calculate, memory_store, memory_recall], model="gpt-4o"
)
user_message = "What's 12 * 3 + 15 / 3?"
conversation.add_message(role="user", content=user_message)
conversation.process_chat_completion()
