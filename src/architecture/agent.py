## imports
import functools
import os
from typing import Any, Generator, Literal, Optional
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (ChatDatabricks,UCFunctionToolkit,VectorSearchRetrieverTool,)
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (ChatAgentChunk,ChatAgentMessage,ChatAgentResponse,ChatContext,)
from pydantic import BaseModel

## Genie Agent
GENIE_SPACE_ID = "01efff2d9bc11284a752c17ad9222d8a"
genie_agent_description = "This agent can answer questions the startups_catalog table. This table contains crucial information on various startups, serving as a catalog for new businesses. It provides key details such as city, region, website, and primary sector. This data is essential for business analysis, market research, and collaboration opportunities within the startup ecosystem, offering insights into startup diversity, growth stages, and industries. The table streamlines decision-making processes for investors, accelerators, and potential collaborators seeking engagement with startups."

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description=genie_agent_description,
    client=WorkspaceClient(
        host=os.getenv("DATABRICKS_HOST") or os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    ),
)

LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
assert LLM_ENDPOINT_NAME is not None
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


## Vector Search Agent
tools_agent1 = []

vector_search_documentation_tool = [
    VectorSearchRetrieverTool(
        index_name="dts_proves_pre.startups_documentacio.documentacio_docs_chunked_index",
        num_results=5,
        query_type="HYBRID", 
        tool_name="vector_search_info_retriever",
        tool_description="Proporciona informació sobre idees de negoci, consells per a emprenedors i opcions de finançament per a startups.",
    )
]

tools_agent1.extend(vector_search_documentation_tool)

info_agent_description = (
    "The  agent specializes in provides information about business ideas, advice for entrepreneurs, and funding options for startups.",
)
info_agent = create_react_agent(llm, tools=tools_agent1)

## Supervisor agent

MAX_ITERATIONS = 3 # max number of iterations between supervisor and worker nodes before returning to the user

worker_descriptions = {
    "Genie": genie_agent_description,
    "Informer": info_agent_description,
}

formatted_descriptions = "\n".join(
    f"- {name}: {desc}" for name, desc in worker_descriptions.items()
)

system_prompt = f"Decide between routing between the following workers or ending the conversation if an answer is provided. \n{formatted_descriptions}"
options = ["FINISH"] + list(worker_descriptions.keys())
FINISH = {"next_node": "FINISH"}

def supervisor_agent(state):
    count = state.get("iteration_count", 0) + 1
    if count > MAX_ITERATIONS:
        return FINISH
    
    class nextNode(BaseModel):
        next_node: Literal[tuple(options)]

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
    next_node = supervisor_chain.invoke(state).next_node
    
    # if routed back to the same node, exit the loop
    if state.get("next_node") == next_node:
        return FINISH
    return {
        "iteration_count": count,
        "next_node": next_node
    }


## Multiagent graph architecture

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
                "name": name,
            }
        ]
    }


def final_answer(state):
    prompt = "Using only the content in the messages, respond to the previous user question using the answer given by the other assistant messages."
    preprocessor = RunnableLambda(
        lambda state: state["messages"] + [{"role": "user", "content": prompt}]
    )
    final_answer_chain = preprocessor | llm
    return {"messages": [final_answer_chain.invoke(state)]}


class AgentState(ChatAgentState):
    next_node: str
    iteration_count: int


informer_node = functools.partial(agent_node, agent=info_agent, name="Informer")
genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")

workflow = StateGraph(AgentState)
workflow.add_node("Genie", genie_node)
workflow.add_node("Informer", informer_node)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("final_answer", final_answer)
workflow.set_entry_point("supervisor")
# We want our workers to ALWAYS "report back" to the supervisor when done
for worker in worker_descriptions.keys():
    workflow.add_edge(worker, "supervisor")


workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {**{k: k for k in worker_descriptions.keys()}, "FINISH": "final_answer"},
)
workflow.add_edge("final_answer", END)
multi_agent = workflow.compile()

# Wrap our multi-agent in ChatAgent
class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg})
                    for msg in node_data.get("messages", [])
                )


mlflow.langchain.autolog()
AGENT = LangGraphChatAgent(multi_agent)
mlflow.models.set_model(AGENT)
