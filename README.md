# to run

```shell
$ export GROQ_API_KEY="sk-..." # set api key

$ uv pip install -r requirements.txt && python app.py # install dependencies 

```


you can get  your groq  apikey here https://console.groq.com/keys

here is a sample video

[![Watch the video](https://img.shields.io/badge/Watch%20Video-Click%20Here-blue)](https://github.com/archit-spec/airecpt2/raw/main/smallest_ai.mp4)


note: only templates/prompts/system_prompt.j2 prompt is being actually used.


what its doing:

```
In the first node you need to confirm from the user if they are having an emergency or they would like to leave a message.
In the second node if its a message, you need to ask the user for the message. If its an emergency, you need to first confirm what is the emergency
If the given emergency is in your vector database (like qdrant or whatever you want), you need to, based on the emergency, provide an immediate next step to the user to do while the doctor calls the user back. Eg. If the patient is not breathing - then do CPR - <details of what CPR is>. This call to the database needs to be artificially slowed down by 15 seconds.
While the call to the database is happening, you need to continue the conversation saying, “I am checking what you should do immediately, meanwhile, can you tell me which area are you located right now?”
Once the user gives the area, then tell the user Dr. Adrin will be coming to their location immediately. Give a random estimated time of arrival.
If the user says that the arrival will be too late, then return the response from point 3 and tell them to follow those steps while the doctor comes so that the patient gets better. Eg. “I understand that you are worried that Dr. Adrin will arrive too late, meanwhile we would suggest that you start CPR, i.e. pushing against the chest of the patient and blowing air into his mouth in a constant rhythm”. If by the time you reach step 6, 15 secs is not complete in point 3, then wait for point 3 by saying “Please hold just a sec”
If in point 2 it was just a message, then ask the user for the message. And after receiving the message say “Thanks for the message, we will forward it to Dr. Adrin”
If it was an emergency, post point 6 say “Don’t worry, please follow these steps, Dr. Adrin will be with you shortly”
If at any point the user says something unrelated, say “I don’t understand that and repeat the question/statement”
```
