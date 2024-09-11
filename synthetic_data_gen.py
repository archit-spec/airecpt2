import asyncio
import json
import random
import time
from ai_receptionist import AIReceptionist, log_function_call
from state_manager import State
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    def __init__(self):
        self.receptionist = AIReceptionist()
        self.emergency_types = [
            "Heart Attack", "Stroke", "Severe Bleeding", "Choking", "Allergic Reaction",
            "Burns", "Fracture", "Poisoning", "Seizure", "Asthma Attack"
        ]
        self.locations = [
            "Home", "Office", "Park", "Shopping Mall", "Restaurant", "Gym",
            "School", "Beach", "Highway", "Public Transportation"
        ]

    def generate_scenario(self):
        emergency = random.choice(self.emergency_types)
        location = random.choice(self.locations)
        return f"A person is experiencing a {emergency} at {location}."

    @log_function_call
    async def generate_conversation(self, scenario):
        conversation = []
        user_input = scenario
        max_turns = 10  # Limit the conversation to 10 turns to prevent infinite loops

        for _ in range(max_turns):
            # Simulate user input
            conversation.append({"role": "user", "content": user_input})
            
            # Get AI response
            ai_response = await self.receptionist.generate_response(user_input)
            
            # Add raw JSON response to conversation
            conversation.append({"role": "function", "name": "generate_response", "content": json.dumps(ai_response)})
            
            # Add human-readable response to conversation
            conversation.append({"role": "assistant", "content": ai_response["response"]})

            # Check if conversation should end
            if self.receptionist.state_manager.state == State.MESSAGE:
                break

            # Generate next user input
            user_input = await self.simulate_user_response(ai_response["response"])

            # Add a natural pause between questions
            await asyncio.sleep(random.uniform(1, 3))

        return conversation

    @log_function_call
    async def simulate_user_response(self, ai_response):
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                prompt = f"Based on the AI's last response: '{ai_response}', generate a short, realistic user response that continues the emergency conversation:"
                simulated_response = await self.receptionist.generate_response(prompt)
                
                logger.info(f"Function simulate_user_response returned: {simulated_response}")
                
                return simulated_response.get("response", "I'm not sure what to say next.")

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    return "I'm sorry, I'm having trouble continuing the conversation right now."

    @log_function_call
    async def generate_dataset(self, num_samples):
        dataset = []
        for i in range(num_samples):
            scenario = self.generate_scenario()
            conversation = await self.generate_conversation(scenario)
            dataset.append({
                "scenario": scenario,
                "conversation": conversation
            })
            self.receptionist.reset_context()
            logger.info(f"Completed scenario {i+1}/{num_samples}")
            
            # Append the new scenario to the JSON file
            self.append_to_json(dataset[-1], "synthetic_emergency_dataset.json")
        
        return dataset

    def append_to_json(self, new_data, filename):
        try:
            with open(filename, 'r+') as file:
                file.seek(0, 2)  # Move to the end of the file
                position = file.tell() - 1
                file.seek(position)
                file.write(',\n')
                json.dump(new_data, file, indent=2)
                file.write(']')
        except FileNotFoundError:
            with open(filename, 'w') as file:
                json.dump([new_data], file, indent=2)
        
        logger.info(f"Appended new scenario to {filename}")

async def main():
    generator = SyntheticDataGenerator()
    num_samples = 1  # Adjust as needed
    await generator.generate_dataset(num_samples)
    logger.info(f"Generated {num_samples} synthetic conversations and appended to synthetic_emergency_dataset.json")

if __name__ == "__main__":
    asyncio.run(main())
