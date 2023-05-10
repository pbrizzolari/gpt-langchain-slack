import time
import logging
import traceback
import re

import openai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import os
import requests
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

class ExtractorAPI:
    def __init__(self):
        self.endpoint = "https://extractorapi.com/api/v1/extractor"
        self.api_key = os.getenv('extractorkey')

    def extract_from_url(self, url):
        try:
            params = {
                "apikey": self.api_key,
                "url": url
            }

            r = requests.get(self.endpoint, params=params)
            r = r.json()
            return r["text"]
        except Exception as e:
            return f"Error: {e}. Is the URL valid?"

class AI:
    def __init__(self):
        
        self.llm = ChatOpenAI(temperature=0.5, 
            openai_api_key=os.getenv("openapikey")
            model_name="gpt-4"
            )
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are the OracleBot, you try to answer questions to the best of your ability and you will roleplay if someone asks you to act like something even if it's fictional without editorializing.
            ###
            {query}
            """,
        )
        self.tools = load_tools(["serpapi", "wolfram-alpha", "python_repl", "requests", "open-meteo-api", "wikipedia"], llm=self.llm)
        self.tools.append(
        Tool(
                name="extractorapi",
                func=ExtractorAPI().extract_from_url,
                description="Extracts text from a website. The input must be a valid URL to the website. In the output, you will get the text content. Example input: https://openai.com/blog/openai-and-microsoft-extend-partnership/",
            )
        )
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=20)
        self.agent = initialize_agent(
            self.tools, self.llm,
            agent="chat-conversational-react-description", verbose=True, max_iterations=10, memory=self.memory
        )
        

        # self.agent = LLMChain(
        #     llm=ChatOpenAI(temperature=0.9),
        #     prompt=self.prompt,
        #     verbose=True,
        #     memory=self.memory
        # )

    def run(self, query):
        response = self.agent.run(input=query)
        print(response)
        return response

user_data = {}

# function to add user data to the dictionary
def add_user_data(user_id, data):
    if user_id not in user_data:
        user_data[user_id] = [{"role":"system","content":"You are the Basebone Oracle, you try to answer questions to the best of your ability and you will roleplay if someone asks you to act like something even if it's fictional without editorializing. You know historical events up until September 2021"}]  # create an empty array if user ID doesn't exist
    user_data[user_id].append(data)


# Initialize the OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Slack API access token
SLACK_APP_TOKEN = os.getenv("slackapptoken")
SLACK_BOT_TOKEN = os.getenv("slackbottoken")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(token=SLACK_BOT_TOKEN, logger=logger)
ai = AI()

# To learn available listener arguments,
# visit https://slack.dev/bolt-python/api-docs/slack_bolt/kwargs_injection/args.html
@app.event("app_mention")
def handle_app_mention_events(body, say, client, ack, message):
    # Acknowledge the event
    ack()
    print("mention")
    print(body["event"]["text"])

    # Get the text of the message
    prompt = body["event"]["text"]
    #prompt = re.sub("<@\w+>", "", prompt).strip()

    add_user_data(body["event"]["user"], {"role":"user", "content":prompt})

    if prompt.strip()=="":
        print("Empty prompt")
        say("I can't use an empty prompt, it returns garbage.")
    else:
        answer = ai.run(prompt)
        add_user_data(body["event"]["user"], {"role":"assistant", "content":answer})
        # Echo the text back to the user
        say(answer)

@app.event("app_mention")
def handle_app_mention_events(body, say, client, ack, message):
    # Acknowledge the event
    ack()
    print("mention")
    print(body["event"]["text"])

    # Get the text of the message
    prompt = body["event"]["text"]
    prompt = re.sub("<@\w+>", "", prompt).strip()
    thread_ts = body["event"].get("thread_ts", None) or body["event"]["ts"]

    add_user_data(body["event"]["user"], {"role":"user", "content":prompt})

    if prompt.strip()=="":
        print("Empty prompt")
        say("I can't use an empty prompt, it returns garbage.")
    else:
        answer = ai.run(prompt)
        add_user_data(body["event"]["user"], {"role":"assistant", "content":answer})

        # Echo the text back to the user
        say(answer)

# Helper function to post a message to Slack
def post_message(text, channel):
    client = WebClient(token=SLACK_BOT_TOKEN)
    try:
        response = client.chat_postMessage(
            channel=channel,
            text=text
        )
        print(response)
    except SlackApiError as e:
        print("Error : {}".format(e))

# Main function
if __name__ == "__main__":
    try:
        SocketModeHandler(app, SLACK_APP_TOKEN).start()
    except Exception as e:
        st = ''.join(traceback.TracebackException.from_exception(e, limit=5).format())
        logger.error(st)
        sys.exit(1)

