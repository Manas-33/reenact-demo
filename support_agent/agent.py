"""A LangGraph support agent: retrieve help articles, then answer grounded in them.

A real think -> act loop (RAG): the model searches the help center, applies one
category label, posts a reply grounded in what it retrieved, and summarizes. Four
tools - two read-only (`search_docs` / `read_doc`) and two mutating (`label_ticket`
/ `post_reply`). Recorded through the reenact LangChain callback handler, so a run
captures the LLM calls, the tool calls, and the LangGraph node boundaries at once.

`build_support_graph` takes any object the agent node can `.invoke` - a real
tool-bound `ChatAnthropic` when recording, a scripted fake model when testing
offline - so the same graph records real trajectories and runs in CI with no key.

This is the demo's *grounding* agent: its differentiated regression is a reply that
invents a feature the retrieved articles do not support, which a plain assertion
cannot see but an evidence-backed criterion (reply_grounded / faithfulness) can.
"""

# LangGraph/LangChain ship incomplete type stubs; run this module in basic mode so
# their partially-unknown generics do not trip type checking.
# pyright: basic

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from reenact.record.langchain import ReenactCallbackHandler
from reenact.schema import SideEffect, ToolCallEvent, Trajectory

from support_agent.fixtures import DOCS


@tool
def search_docs(query: str) -> str:
    """Search the help center and return the most relevant article's text."""
    words = set(query.lower().split())
    best = max(DOCS.items(), key=lambda item: len(words & set(item[1].lower().split())))
    return best[1]


@tool
def read_doc(name: str) -> str:
    """Read a help-center article by name, e.g. 'billing.md'."""
    return DOCS.get(name, f"article not found: {name}")


@tool
def label_ticket(ticket_id: str, label: str) -> str:
    """Apply a category label to a ticket (mutating)."""
    return f"labeled ticket #{ticket_id} as {label}"


@tool
def post_reply(ticket_id: str, body: str) -> str:
    """Post a public reply on a ticket (mutating)."""
    return f"posted reply to ticket #{ticket_id}"


TOOLS = [search_docs, read_doc, label_ticket, post_reply]
READ_ONLY_TOOLS = frozenset({"search_docs", "read_doc"})
MUTATING_TOOLS = frozenset({"label_ticket", "post_reply"})

SUPPORT_SYSTEM = (
    "You are a support assistant for a SaaS product. For the ticket you are given: "
    "search the help center for relevant articles, apply exactly one category label "
    "(one of: account, billing, api), and post a short reply. Ground the reply in "
    "the articles you retrieved - do not invent features, tiers, or policies that "
    "the articles do not state. Finish with a one-sentence summary of what you did."
)


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_support_graph(model: Any, tools: list[Any] | None = None) -> Any:
    """Compile the support graph around ``model`` (already tool-bound, or a fake)."""
    active_tools = tools if tools is not None else TOOLS

    def agent(state: _State) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    def route(state: _State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(_State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(active_tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


def label_side_effects(trajectory: Trajectory) -> Trajectory:
    """Set each recorded tool call's side-effect class by name.

    The blanket callback handler records every tool as UNKNOWN; the demo knows which
    tools mutate, so this stamps the real class onto the committed cassette - what
    makes ``no_mutating_tool_reexecuted`` meaningful on these recordings.
    """
    for event in trajectory.events:
        if isinstance(event, ToolCallEvent):
            if event.name in MUTATING_TOOLS:
                event.side_effect = SideEffect.MUTATING
            elif event.name in READ_ONLY_TOOLS:
                event.side_effect = SideEffect.READ_ONLY
    return trajectory


def support_trajectory(
    model: Any,
    ticket: str,
    *,
    tools: list[Any] | None = None,
    system: str | None = None,
    name: str | None = None,
) -> Trajectory:
    """Run the support agent on ``ticket`` and return the recorded trajectory.

    ``system`` defaults to the shipped prompt; a break-me variant passes a degraded
    prompt (one that drops the "do not invent" guardrail) to seed a regression.
    """
    handler = ReenactCallbackHandler()
    graph = build_support_graph(model, tools)
    graph.invoke(
        {
            "messages": [
                SystemMessage(content=system if system is not None else SUPPORT_SYSTEM),
                HumanMessage(content=ticket),
            ]
        },
        config={"callbacks": [handler]},
    )
    trajectory = handler.recorder.trajectory
    if name is not None:
        trajectory.name = name
    return label_side_effects(trajectory)
