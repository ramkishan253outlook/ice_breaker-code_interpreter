from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor 
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# REPL - READ EVALUATE PRINT LOOP

load_dotenv()


def main():

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to python REPL, which you use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of the code to answer the question.
    You might know the answer without running the code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    agent = create_react_agent(
        prompt=prompt, llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"), tools=tools
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke(
        input={
            "input": """Generate and save 5 qr codes in current working directory that point to www.google.com.
            You have qr code package installed already.
            """
        }
    )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    csv_agent.invoke("How many columns are tehre in the episode_info csv file")

    csv_agent.invoke(
        input={
            "input": "print the seasons by ascending order of the number of episodes they have"
        }
    )

if __name__ == "__main__":
    main()
