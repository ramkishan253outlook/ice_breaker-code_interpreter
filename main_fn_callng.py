from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

@tool
def multiply(x: float, y: float) -> float:
    """Multiple 'x' times 'y"""
    return x * y

def main_fn_callng():

    print("Hello tool calling")
    tools = [TavilySearchResults(), multiply]
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages(
          [
            ("system","You're are helpful assitant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
          ]
    )

    tc_agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=tc_agent, tools=tools)

    res = agent_executor.invoke(input={"input": "what is the weather in dubai right now? compare it with San Fransisco, output should be in celsious"})

    print(res)

if __name__ == "__main__":
    main_fn_callng()



