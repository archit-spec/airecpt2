import asyncio
import os
from ai_receptionist import AIReceptionist
from vector_db import VectorDB
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

receptionist = AIReceptionist()

vector_db = VectorDB()
logger.info("Available collections: %s", vector_db.get_collections())

# Initialize the collection and load data only if it doesn't exist
if "emergency_instructions" not in [
    c.name for c in vector_db.get_collections().collections
]:
    vector_db.initialize_collection()
    vector_db.load_data("emergency_instructions.json")
    logger.info("Initialized 'emergency_instructions' collection and loaded data.")
else:
    logger.info(
        "Collection 'emergency_instructions' already exists. Skipping initialization."
    )

async def main():
    print("Welcome to the AI Receptionist Terminal Interface!")
    print("Type 'exit' to quit the program.")
    
    while True:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
        
        if user_input.lower() == 'exit':
            print("Thank you for using AI Receptionist. Goodbye!")
            break
        
        try:
            response = await receptionist.process_input(user_input)
            print(f"AI: {response}")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            print("AI: I'm sorry, an error occurred. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())
