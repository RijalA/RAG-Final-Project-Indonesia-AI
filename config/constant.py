from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))

# embedding
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name=os.getenv('HUGGINGFACE_EMBEDDING_MODEL'))

# llm
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

model_name = os.getenv('HUGGINGFACE_LLM_MODEL')

endpoint = f"https://api-inference.huggingface.co/models/{model_name}"

llm = HuggingFaceEndpoint(
    endpoint_url=endpoint,
    task='text-generation',
    temperature=0.1,
)

# prompt
from langchain_core.prompts import PromptTemplate

template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'chat_history', 'tool_names', 'tools'],
    template=template
)