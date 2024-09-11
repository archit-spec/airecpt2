
from groq import Groq
import os
import json
import time 
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_prompt = """
Always responsond in valid json objects 
{
    "new_state": "INITIAL|EMERGENCY|MESSAGE|LOCATION|INTERMEDIARY|FINAL",
    "context_updates": {
        "emergency_type": "string (if applicable)",
        "location": "string (if applicable)",
        "message": "string (if applicable)"
    },
    "response": "Your response to the user.",
}

"""

user_input = input("user:")

messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
starttime = time.time()
response =  client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.2,
                        max_tokens=1024,
                        top_p=0.95,
                        frequency_penalty=0.1,
                        presence_penalty=0.1,
                        stop=None,
        )

print(time.time() - starttime , response.choices)

response_text = response.choices[0].message.content


parsed_response = json.loads(response_text)

new_state = parsed_response.get("new_state", "INITIAL")

print(new_state)