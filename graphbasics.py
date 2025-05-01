from typing import Annotated, Sequence, TypeVar
from langgraph.graph import Graph, StateGraph
from typing import Dict, TypedDict

# Define state type
class State(TypedDict):
    input: str
    output: str

# Node 1: Input processor
def get_user_input(state: State) -> State:
    user_input = input("Enter 'foo', 'bar', or 'baz': ")
    state["input"] = user_input.lower()
    return state

# Node 2: Foo handler
def handle_foo(state: State) -> State:
    state["output"] = "You chose foo! Here's a foo response."
    return state

# Node 3: Bar/Baz handler
def handle_bar_baz(state: State) -> State:
    if state["input"] == "bar":
        state["output"] = "You chose bar! Here's a bar response."
    else:
        state["output"] = "You chose baz! Here's a baz response."
    return state

# Router function
def router(state: State) -> str:
    if state["input"] == "foo":
        return "handle_foo"
    return "handle_bar_baz"

# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("get_input", get_user_input)
workflow.add_node("handle_foo", handle_foo)
workflow.add_node("handle_bar_baz", handle_bar_baz)

# Create edges
workflow.set_entry_point("get_input")
workflow.add_conditional_edges(
    "get_input",
    router,
    {
        "handle_foo": "handle_foo",
        "handle_bar_baz": "handle_bar_baz"
    }
)
workflow.set_finish_point("handle_foo")
workflow.set_finish_point("handle_bar_baz")

# Compile the graph
app = workflow.compile()

# Run the graph
if __name__ == "__main__":
    state = {"input": "", "output": ""}
    result = app.invoke(state)
    print(result["output"])
