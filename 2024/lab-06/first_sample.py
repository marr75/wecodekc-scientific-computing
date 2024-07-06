# %%
from typing import Annotated

from llm_easy_tools import get_tool_defs, process_response
import utils
from pprint import pprint


# %%
# Define a tool function. The `: str` part is a type hint for the function argument.
# These type hints are used to make functions easier to read and maintain.
# They also allow the AI to understand how to call the function.
def contact_user(name: str, city: str) -> str:
    return f"User {name} from {city} was contacted"


# %%
response = utils.chat_completion_request(
    messages=[
        {"role": "user", "content": "Contact John. John lives in Warsaw"},
    ],
    tools=get_tool_defs([contact_user]),
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [contact_user])

pprint(results[0].output)


# %%
def annotated_contact_user(
    name: Annotated[str, "Name of the user"],
    email: Annotated[str, "Email of the user"],
    phone: Annotated[str, "Phone number of the user"],
) -> str:
    """
    Contact the user with the given name, email, and phone number.
    """
    pass


pprint(get_tool_defs([annotated_contact_user]))

# %%
conversation = utils.Conversation(system_message="Hello", tools=[contact_user])
conversation.add_message(role="user", content="Contact John. John lives in Warsaw")
conversation.process_chat_completion()
