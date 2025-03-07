# with no debugging

import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

import json

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

def vector_datastore():
    # print("Initializing Vector Datastore...")  # Debugging print
    log_dir = "logs_astraDB"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"astraDB_log_{timestamp}.txt")

    # print(f"Creating log file: {log_filename}")  # Debugging print
    with open(log_filename, 'w') as log_file:
        embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        # print("Created OpenAI Embeddings")  # Debugging print

        vstore = AstraDBVectorStore(
            collection_name="risk_assessment_01",
            embedding=embedding,
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        # print("Connected to AstraDB for risk assessment")  # Debugging print

        vstore2 = AstraDBVectorStore(
            collection_name="uploaded",
            embedding=embedding,
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        # print("Connected to AstraDB for user uploads")  # Debugging print
        

        retriever = vstore.as_retriever(search_kwargs={"k": 3})
        retriever_tool = create_retriever_tool(retriever, "contract_info",
                                "Search for information about contracts and risks associated with different contracts.")
        # print("Created retriever tool 1: ", retriever_tool)  # Debugging print

        retriever02 = vstore2.as_retriever(search_kwargs={"k": 3})
        retriever_tool2 = create_retriever_tool(retriever02, "user_uploaded_contract",
                        "Search for information about the user's uploaded specific contract and use it for compliance and risk analysis")
        # print("Created retriever tool 2")  # Debugging print

        tools = [retriever_tool, retriever_tool2]
        log_file.write(f"TOOLS:\n{tools}\n")
    
    # print("Vector Datastore initialization complete.")  # Debugging print
    return tools


class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('./scratchpad_prompt.txt', 'r') as file:
            scratchpad_prompt = file.read().strip()

        # scratchpad_prompt = ""


        #call the function for vectorDB....
        self.tools = vector_datastore()
        # contract_info = self.tools[0]
        # user_uploaded_contract = self.tools[1]


        # with open('scratchpad_prompt.json', 'r') as file2:
        #     scratchpad_prompt = json.load(file2)  # Load scratchpad prompt

        # self.scratchpad_prompt = scratchpad_prompt['ChainOfThought']  # Store for later use


        # Mapping placeholders to tool names
        # tool_map = {
        #             "{user_uploaded_contract}": self.tools[1],  # First tool in list
        #             "{contract_info}": self.tools[0],  # Second tool in list
        #     }

        # # Replace placeholders in JSON
        # for entry in self.scratchpad_prompt:
        #     for key, value in entry.items():
        #         if isinstance(value, str):  # Ensure it's a string before replacing
        #             for placeholder, actual_tool in tool_map.items():
        #                 value = value.replace(f"{{{placeholder}}}", actual_tool.__class__.__name__)  # Replace placeholder
        #             entry[key] = value  # Update the dictionary with the replaced value

        system_prompt = f"""
        Role:
        You are Luna, a legal compliance and risk assessment assistant specializing in analyzing legal documents. You help users understand legal risks and compliance requirements with clarity and accuracy.

        How You Work:
        -The user provides a legal document for analysis.
        -The user also gives a specific prompt or question related to the document.
        -You retrieve relevant legal knowledge from a vector database containing compliance guidelines, case laws, and regulatory references.
        -You analyze the document in context and generate a concise, fact-based response.

        Response Guidelines:
        -Keep responses under 20 words, unless the user asks for more details.
        -Maintain a friendly, approachable, and conversational tone with natural fillers.
        -Use clear legal insights and cite references when applicable.
        -Do not speculate, provide personal opinions, or generate content beyond retrieved information.
        -Do not generate code.

        Objective:
        Make compliance and risk assessment simple, engaging, and insightful for the user by delivering precise, fact-based legal insights with a warm, conversational approach.

        You have access to the following tools:
        -contract_info: Search for information about contracts and risks associated with different contracts.
        -user_uploaded_contract: Search for information about the user's uploaded specific contract and use it for compliance and risk analysis.

        Answer the following questions as best you can. 

        Use the following format:
        [Question]: the input question you must answer
        [Thought]: you should always think about what to do
        [Action]: the action to take, should be one of [contract_info, user_uploaded_contract]
        [Action] Input: the input to the action
        [Observation]: the result of the action... (this Thought/Action/Action Input/Observation can repeat N times)
        [Thought]: I now know the final answer
        [Final Answer]: the final answer to the original input question

        Refer the below examples:
        {scratchpad_prompt}

        """

        
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            # SystemMessagePromptTemplate.from_template("You are a friendly chat-bot. Interact with the user like a friend."),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        # self.conversation = LLMChain(
        #     llm=self.llm,
        #     prompt=self.prompt,
        #     memory=self.memory
        # )

        

        # Create Agent
        self.agent = create_openai_tools_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)

        # Create Agent Executor
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, return_intermediate_steps=False)

        

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()
        response = {
            "output" : ""
        }

        try:
            # print(f"agent_executor Query received: {text}")
            # Use AgentExecutor for the response
            response = self.agent_executor.invoke({
                "input": text,
                "chat_history": self.memory.chat_memory.messages,  # Pass the chat history
                "agent_scratchpad": ""  # Initialize or manage agent_scratchpad
                # "agent_scratchpad": self.scratchpad_prompt  # Pass scratchpad prompt
            })

            print("response :", response)
            # print('debug response =',response)  # Add this to debug and view the structure of the response,

            # # Extract intermediate steps if available
            # if "intermediate_steps" in response:
            #     print("\nIntermediate Steps:")
            #     for i, step in enumerate(response["intermediate_steps"]):
            #         print(f"Step {i + 1}: {step}")
            


        except KeyError as e:
            print(f"Missing input variable: {e}")
            response = {"text": "Sorry, I couldn't process your request due to a configuration error."}

        except Exception as e:
            # Fallback to normal conversation if tools are not applicable
            print(f"Agent execution failed: {e}. Falling back to conversation.")
            # response = self.conversation.invoke({"text": text})
        
        end_time = time.time()

        try:
            self.memory.chat_memory.add_ai_message(response['output'])  # Add AI response to memory
        except KeyError as e:
            print(f"KeyError while adding AI message to memory: {e}")
            self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['output']}")
        return response['output']

class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-luna-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        # DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-nodisp", "-f", "wav", "-"]

        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            if r.status_code != 200:
                print(f"Request failed with status code: {r.status_code}")
                print("Response Text:", r.text)
                return
            
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:  # Check if this is the first chunk received
                        first_byte_time = time.time()  # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                        # print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        print("Initializing Deepgram Client...")
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        print("Creating WebSocket connection...")
        dg_connection = deepgram.listen.asynclive.v("1")
        print ("Listening...")

        print("Setting up event listener...")

        async def on_message(self, result, **kwargs):
            # print(f"DEBUG: Received from Deepgram: {result}")
            # print("Received a message from Deepgram!")
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2-conversationalai",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        # print("Starting Deepgram WebSocket...")
        await dg_connection.start(options)

        # # Open a microphone stream on the default input device
        # print("Initializing microphone...")
        microphone = Microphone(dg_connection.send)
        print('[Listening...]')
        microphone.start()


        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()
        # print('mic stop')

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                # print("conversation completed. You can start a new session.")
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""


async def run_conversation():
    manager = ConversationManager()
    await manager.main()

if __name__ == "__main__":
    asyncio.run(run_conversation())