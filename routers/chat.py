from fastapi import APIRouter, HTTPException

from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain import hub
from langchain.agents import initialize_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

from config.constant import embedding, llm, prompt

from dotenv import load_dotenv
load_dotenv()

router = APIRouter(
    prefix="/v1/chat",
    tags=["Chat"],
)

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@router.post("/chat")
async def chat(question: str):
    path_chroma = "./kb/chroma"
    vector_db = Chroma(embedding_function=embedding, persist_directory=path_chroma)
    retriever = vector_db.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="retriever_tool",
        description="Search documents to answer questions about current events",
    )

    search = SerpAPIWrapper()
    search_tool = Tool(
        name="online_search_tool",
        description="Search online to answer questions about current events",
        func=search.run,
    )

    # prompt = hub.pull("hwchase17/react")
    # print(prompt)

    tools = [
        retriever_tool,
        # search_tool
    ]

    # agent_instructions = "Always use 'retriever_tool' tool first, Use the other tools if these don't work."

    # agent = initialize_agent(
    #     tools,
    #     llm,
    #     agent_instructions=agent_instructions,
    #     agent="zero-shot-react-description",
    #     handle_parsing_errors=True,
    #     verbose=True
    # )

    # agent = create_react_agent(llm, tools, prompt)
    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     handle_parsing_errors=True,
    #     verbose=True
    # )
    
    # q = question
    # c = f"Always use the retriever_tool first, {q}. If cannot get the answer, use other tool"
    # # res = agent_executor.invoke({"input": c})
    # res = agent.run(c)

    # return res

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    q = question
    # c = f"Use the 'retriever_tool' first to answer: {q}. If cannot get the answer, use other tool"
    c = f"Use the 'retriever_tool' to answer: {q}. If cannot get the answer, answer 'I don't know'."
    res = agent_with_chat_history.invoke(
        {"input": c},
        config={"configurable": {"session_id": "123"}},
    )
    
    return res['output']