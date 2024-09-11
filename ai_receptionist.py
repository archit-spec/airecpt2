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
import time
import functools
from groq import InternalServerError, RateLimitError

logger = logging.getLogger(__name__)

def log_function_call(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"Calling function: {func.__name__}")
        logger.info(f"Arguments: {args}, {kwargs}")
        result = await func(*args, **kwargs)
        logger.info(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper

class TokenBucket:
    def __init__(self, tokens_per_minute, tokens_per_day):
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_day = tokens_per_day
        self.tokens = tokens_per_minute
        self.last_refill = time.time()
        self.daily_tokens = tokens_per_day
        self.day_start = time.time()

    async def consume(self, tokens):
        while True:
            self.refill()
            if self.tokens >= tokens and self.daily_tokens >= tokens:
                self.tokens -= tokens
                self.daily_tokens -= tokens
                return
            await asyncio.sleep(1)

    def refill(self):
        now = time.time()
        if now - self.day_start >= 86400:  # 24 hours
            self.daily_tokens = self.tokens_per_day
            self.day_start = now
        
        time_passed = now - self.last_refill
        self.tokens = min(self.tokens_per_minute, self.tokens + time_passed * (self.tokens_per_minute / 60))
        self.last_refill = now

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
        self.token_bucket = TokenBucket(20000, 1000000)  # 20,000 tokens per minute, 1,000,000 tokens per day
        self.db_query_made = False

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
        self.db_query_made = False
        logger.info("Context has been reset.")
    
    async def process_input(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})

        response_dict = await self.generate_response(user_input)
        
        response_text = response_dict.get("response", "I'm sorry, I couldn't generate a proper response.")

        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text

    async def generate_response(self, user_input: str) -> dict:
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Estimate token usage (you may need to implement a more accurate estimation)
                estimated_tokens = len(user_input.split()) + 100  # rough estimate
                await self.token_bucket.consume(estimated_tokens)

                system_prompt = self.jinja_env.get_template("system_prompt.j2").render(
                    current_state=self.state_manager.state,
                    context=self.state_manager.get_context(),
                    state_history=self.state_history,
                    data_available=self.data_avl,
                    instructions=self.instructions,
                )

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
                    if not isinstance(parsed_response, dict):
                        raise json.JSONDecodeError("Parsed JSON is not a dictionary", response_text, 0)
                except json.JSONDecodeError:
                    # If it's not JSON or not a dictionary, use the raw text as the response
                    parsed_response = {"response": response_text, "new_state": "INITIAL"}

                # Ensure new_state is a valid State enum value
                new_state = parsed_response.get("new_state", "INITIAL")
                if new_state not in State.__members__:
                    new_state = "INITIAL"

                parsed_response["new_state"] = new_state
                parsed_response["response"] = parsed_response.get("response", response_text)

                # Process the AI response
                print(f"Parsed AI response: {json.dumps(parsed_response, indent=2)}")

                state_transition_message = self.state_manager.transition_to(State[new_state])
                print(state_transition_message)
                self.state_history.append(new_state)

                if "context_updates" in parsed_response:
                    context_update_message = self.state_manager.update_context(
                        **parsed_response["context_updates"]
                    )
                    print(context_update_message)

                # Check if we need to make a database query
                if not self.db_query_made and (new_state == State.LOCATION or new_state == State.EMERGENCY):
                    emergency_type = self.state_manager.get_context().get("emergency_type")
                    if emergency_type:
                        self.db_query_made = True
                        db_result = self.get_instructions_from_db(emergency_type)
                        self.db_result["data"] = db_result["response"]
                        self.db_result["available"] = True
                        self.data_avl = True
                        self.instructions = db_result["response"]
                        print(f"Database query made for {emergency_type}")
                        print(f"Database result: {self.db_result['data']}")
                        
                        # Transition to INTERMEDIARY state
                        new_state = State.INTERMEDIARY
                        state_transition_message = self.state_manager.transition_to(new_state)
                        print(state_transition_message)
                        self.state_history.append(new_state.name)
                        
                        # Update the response with the database instructions
                        parsed_response["response"] += f" Please follow these instructions: {self.db_result['data']}"
                        parsed_response["new_state"] = new_state.name

                if new_state == State.MESSAGE:
                    self.state_manager.clear_context()

                return parsed_response

            except (InternalServerError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    return {"response": "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later.", "new_state": "INITIAL"}

            except Exception as e:
                logger.error(f"An unexpected error occurred: {str(e)}")
                return {"response": "I'm sorry, an unexpected error occurred. Please try again later.", "new_state": "INITIAL"}

    def get_state_context(self) -> dict:
        return self.state_manager.get_context()

    def get_instructions_from_db(self, query: str) -> dict:
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
