import time
import random
from enum import Enum
import os
from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from groq import AsyncGroq

class State(Enum):
    INITIAL = 0
    EMERGENCY = 1
    MESSAGE = 2
    LOCATION = 3
    FINAL = 4

class EmergencyDatabase:
    def __init__(self):
        self.client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

    async def get_instructions(self, emergency: str) -> str:
        prompt = f"Provide concise first aid instructions for a {emergency} emergency. Include only the most critical steps."
        
        start_time = time.time()
        messages = [
            {"role": "system", "content": "You are a medical expert providing emergency first aid instructions."},
            {"role": "user", "content": prompt}
        ]
        
        response_content = ""
        stream = await self.client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=1,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                response_content += content

        elapsed_time = time.time() - start_time
        if elapsed_time < 15:
            await asyncio.sleep(15 - elapsed_time)
        
        return response_content

class AIReceptionist:
    def __init__(self):
        self.state = State.INITIAL
        self.emergency_db = EmergencyDatabase()
        self.user_input = ""
        self.emergency_type = ""
        self.location = ""
        self.message = ""
        self.client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.conversation_history = []

    async def process_input(self, user_input: str) -> str:
        self.user_input = user_input.lower()
        
        if self.state == State.INITIAL:
            return await self.handle_initial_state()
        elif self.state == State.EMERGENCY:
            return await self.handle_emergency_state()
        elif self.state == State.MESSAGE:
            return await self.handle_message_state()
        elif self.state == State.LOCATION:
            return await self.handle_location_state()
        elif self.state == State.FINAL:
            return await self.handle_final_state()
        else:
            return "I'm sorry, I don't understand. Could you please repeat that?"

    async def generate_response(self, prompt: str) -> str:
        messages = []
        for chat in self.conversation_history:
            messages.append({"role": 'user', "content": chat[0]})
            messages.append({"role": 'assistant', "content": chat[1]})

        messages.append({"role": "user", "content": prompt})

        response_content = ""
        stream = await self.client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=1,
            max_tokens=100,
            top_p=1,
            stop=None,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                response_content += content

        self.conversation_history.append((prompt, response_content))
        return response_content

    async def handle_initial_state(self) -> str:
        prompt = f"User input: '{self.user_input}'. Determine if this is an emergency or a message. If it's an emergency, ask for details. If it's a message, ask for the message content. If unclear, ask for clarification."
        response = await self.generate_response(prompt)
        
        if "emergency" in self.user_input:
            self.state = State.EMERGENCY
        elif "message" in self.user_input:
            self.state = State.MESSAGE
        
        return response

    async def handle_emergency_state(self) -> str:
        self.emergency_type = self.user_input
        self.state = State.LOCATION
        return "I am checking what you should do immediately. Meanwhile, can you tell me which area you are located right now?"

    async def handle_message_state(self) -> str:
        self.message = self.user_input
        self.state = State.FINAL
        return "Thanks for the message. We will forward it to Dr. Adrin."

    async def handle_location_state(self) -> str:
        self.location = self.user_input
        eta = random.randint(5, 30)
        response = f"Dr. Adrin will be coming to your location immediately. The estimated time of arrival is {eta} minutes."
        
        if "late" in self.user_input:
            instructions = await self.emergency_db.get_instructions(self.emergency_type)
            response += f" I understand that you are worried that Dr. Adrin will arrive too late. Meanwhile, we suggest that you {instructions}"
        
        self.state = State.FINAL
        return response

    async def handle_final_state(self) -> str:
        if self.emergency_type:
            return "Don't worry, please follow these steps. Dr. Adrin will be with you shortly."
        else:
            return "Is there anything else I can help you with?"

app = FastAPI()

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

receptionist = AIReceptionist()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = await receptionist.process_input(data)
        await websocket.send_text(response)

@app.get("/state")
async def get_state(request: Request):
    state_context = receptionist.get_state_context()
    return templates.TemplateResponse("state.html", {"request": request, "state_context": state_context})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
