import asyncio
import json
from groq import AsyncGroq
import os

import groq
from state_manager import StateManager, State
from jinja2 import Environment, FileSystemLoader
from vector_db import VectorDB
import torch
import logging

logger = logging.getLogger(__name__)


class AIReceptionist:
    def __init__(self):
        self.state_manager = StateManager()
        self.client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.conversation_history = []
        self.jinja_env = Environment(loader=FileSystemLoader("templates/prompts"))
        self.vector_db = VectorDB()
        self.state_history = []
        # this is for device for encoding qdrant query
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eta_mentioned = False
        self.data_avl = False
        self.instructions = None
        self.db_result = {
            "available": False,
            "data": None,
        }

    def reset_context(self):
        self.conversation_history = []
        self.state_history = []
        self.eta_mentioned = False
        self.data_avl = False
        self.db_result = {
            "available": False,
            "data": None,
        }
        self.state_manager.reset()
        self.data_avl = False
        logger.info("Context has been reset.")

    async def process_input(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})

        response = await self.generate_response(user_input)
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    async def generate_response(self, user_input: str) -> str:
        try:
            system_prompt = self.jinja_env.get_template("system_prompt.j2").render(
                current_state=self.state_manager.state,
                context=self.state_manager.get_context(),
                state_history=self.state_history,
                data_available=self.data_avl,
                instructions=self.db_result['data'],
            )

            # Incorporate database response into the context
            if self.db_result["available"] and self.db_result["data"]:
                self.state_manager.update_context(db_instructions=self.db_result["data"])

            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history,
                {"role": "user", "content": user_input},
            ]

            print(f"Current state: {self.state_manager.state}")
            print(f"Current context: {self.state_manager.get_context()}")
            print(f"State history: {self.state_history}")
            print(f"Data available?: {self.data_avl}")

            print(f"User input: {user_input}")
            print("Sending request to Groq API...")
            response_text = ""
            parsed_response = {}
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model="llama-3.1-70b-versatile",
                        messages=messages,
                        temperature=0.2,
                        max_tokens=1024,
                        top_p=0.95,
                        frequency_penalty=0.1,
                        presence_penalty=0.1,
                        stop=None,
                    )
                    print("Received response from Groq API")

                    if not response.choices or not response.choices[0].message:
                        raise Exception("No response received from the API")

                    response_text = response.choices[0].message.content

                    # Log the raw response text
                    logger.info(f"Raw response text: {response_text}")

                    # Try to parse the response as JSON
                    try:
                        parsed_response = json.loads(response_text)
                        break  # Exit the retry loop if JSON parsing is successful
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to decode JSON response: {response_text}"
                        )
                        if attempt < max_retries - 1:
                            logger.info("Reprompting the language model...")
                        else:
                            raise ValueError(
                                "Failed to generate a valid JSON response after multiple attempts"
                            )

                except groq.BadRequestError as e:
                    logger.error(f"An error occurred in generate_response: {e}")
                    logger.error(f"Error details: {e.response.json()}")
                    return "I'm sorry, an error occurred while processing your request. Please try again later."

            # Process the AI response
            try:
                print(f"Parsed AI response: {json.dumps(parsed_response, indent=2)}")

                new_state = State[parsed_response.get("new_state", "INITIAL")]
                state_transition_message = self.state_manager.transition_to(new_state)
                print(state_transition_message)
                self.state_history.append(new_state.name)

                if "context_updates" in parsed_response:
                    context_update_message = self.state_manager.update_context(
                        **parsed_response["context_updates"]
                    )
                    print(context_update_message)
                if new_state == State.LOCATION:
                    emergency_type = parsed_response.get("context_updates", {}).get(
                        "emergency_type"
                    )
                    if emergency_type:
                        logger.info(f"Emergency type detected: {emergency_type}")

                        asyncio.create_task(
                            self.fetch_db_result(emergency_type, self.db_result)
                        )

                if new_state == State.INTERMEDIARY:
                    logger.info(f"State: Intermediary reached")
                    print(f"Database result: {self.db_result['data']}")
                    print(f"Database available: {self.db_result['available']}")
                    if self.db_result["available"] == True:
                        parsed_response[
                            "response"
                        ] += f"Please follow this: {self.db_result['data']}"
                    else:
                        parsed_response[
                            "response"
                        ] += "Please wait, getting emergency information in a moment."

                if new_state == State.MESSAGE:
                    self.state_manager.clear_context()

                final_response = parsed_response.get(
                    "response", "I'm sorry, I couldn't generate a proper response."
                )
                print(f"Final response: {final_response}")
                return final_response

            except json.JSONDecodeError:
                print(f"Failed to parse AI response as JSON. Returning raw response.")
                return response_text  # Return the raw response if it's not in the expected JSON format

        except groq.BadRequestError as e:
            logger.error(f"An error occurred in generate_response: {e}")
            logger.error(f"Error details: {e.response.json()}")
            return "I'm sorry, an error occurred while processing your request. Please try again later."

    def get_state_context(self) -> dict:
        return self.state_manager.get_context()

    async def fetch_db_result(self, emergency_type, db_result):
        await asyncio.sleep(15)
        result = await self.get_instructions_from_db(emergency_type)
        instructions = result
        logger.info(f"Database result: {result}")
        db_result["data"] = result["response"]
        db_result["available"] = True
        self.data_avl = True
        self.instructions = instructions


    async def get_instructions_from_db(self, query: str) -> dict:
        logger.info(f"Querying vector database for: {query}")
        search_result = self.vector_db.search(query)
        if search_result:
            logger.info(f"Vector database returned result for: {query}")
            return search_result
        logger.info(f"No result found in vector database for: {query}")
        return {
            "source": "fallback",
            "response": "I'm sorry, I couldn't find any specific instructions for that situation in my database.",
        }
