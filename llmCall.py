from litellm import completion
import os
import json
import sys

# PROMPT ENGINEERING

rules = """
You are a precise study assistant.

- If the question is casual (e.g. greetings), respond normally.
- Otherwise, use ONLY the context.
- If not found in context, say "I don't know".
- Do NOT make up information.
- Be concise and clear.
"""
ctx = "Darsh loves the color green. He also hates the color golden as he thinks it is too childish."
def setup():
    import litellm
    litellm.set_verbose = False
    del litellm

def getSettings(pathToAPIkey):
    try:
        with open(pathToAPIkey, "r") as f:
            settings = json.load(f)
    except Exception as e:
        print(f"An error occured with loading the API key. Error:\n{e}")

        sys.exit()
    return [settings["api"], settings["maxContext"]]

def getLLMresponse(prompt, model="openrouter/google/gemma-3n-e4b-it:free"):
    response = completion(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


r"""
 /$$      /$$  /$$$$$$  /$$$$$$ /$$   /$$
| $$$    /$$$ /$$__  $$|_  $$_/| $$$ | $$
| $$$$  /$$$$| $$  \ $$  | $$  | $$$$| $$
| $$ $$/$$ $$| $$$$$$$$  | $$  | $$ $$ $$
| $$  $$$| $$| $$__  $$  | $$  | $$  $$$$
| $$\  $ | $$| $$  | $$  | $$  | $$\  $$$
| $$ \/  | $$| $$  | $$ /$$$$$$| $$ \  $$
|__/     |__/|__/  |__/|______/|__/  \__/
"""

setup()
prompt = ""
history = []
api, maxContext = getSettings("settings.json")

os.environ["OPENROUTER_API_KEY"] = api

while True:
    userPrompt = input("You: ")

    if "/q" in userPrompt or "/e" in userPrompt:
        print("Bye!")
        break
        
    history.append(f"User: {userPrompt}")
    prompt = f"""
Rules:
{rules}

Context:
{ctx}

Previous Conversation:
{"\n".join(history)}
Question:
{userPrompt}

Answer:
"""
    response = getLLMresponse(prompt, "openrouter/google/gemma-3n-e4b-it:free")
    history.append(f"AI: {response}")
    history = history[-maxContext:]
    print(f"AI: {response}")

