"""

"""

import sympy as sp
import numpy as np
import plotly.express as px

import utils


def validate_equation(equation: str) -> bool:
    """ """
    ...


def plot_equation(equation: str) -> None:
    """ """
    ...


system_message = ""
tools = [validate_equation, plot_equation]

if __name__ == "__main__":
    conversation = utils.Conversation(system_message=system_message, tools=tools)
    user_message = "Plot the equation y = x^2"
    conversation.add_message(role="user", content=user_message)
    conversation.process_chat_completion()
    conversation.main_loop()
