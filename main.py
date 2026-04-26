from litellm import completion
from chunking import chunk_text
import os
import json
import sys
from retriever import retrieve

ctx = ""

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RED = "\033[31m"

def style(text, *codes):
    return f"{''.join(codes)}{text}{RESET}"

def panel(title, body, color=CYAN):
    lines = body.splitlines() or [""]
    width = max(len(title) + 2, *(len(line) for line in lines))
    top = f"╭─ {title} " + "─" * max(0, width - len(title) - 2)
    print(style(top, BOLD, color))
    for line in lines:
        print(style(f"│ {line.ljust(width)}", color))
    print(style(f"╰{'─' * (width + 1)}", color))

def print_banner():
    banner = """
.___________. __    __   __  .__   __.   _______ 
|           ||  |  |  | |  | |  \ |  |  /  _____|
`---|  |----`|  |__|  | |  | |   \|  | |  |  __  
    |  |     |   __   | |  | |  . `  | |  | |_ | 
    |  |     |  |  |  | |  | |  |\   | |  |__| | 
    |__|     |__|  |__| |__| |__| \__|  \______| 
    """.strip("\n")
    print(style(banner, BOLD, BLUE))
    print(style("╭────────────────────────────────────────────╮", DIM, BLUE))
    print(style("│  ✦ RAG chat ready. Type /q or /e to exit. │", CYAN))
    print(style("╰────────────────────────────────────────────╯", DIM, BLUE))

def setup():
    import litellm
    litellm.set_verbose = False
    os.environ["LITELLM_LOG"] = "ERROR"
    del litellm

def getSettings(pathToAPIkey):
    try:
        with open(pathToAPIkey, "r") as f:
            settings = json.load(f)
    except Exception as e:
        print(f"An error occured with loading the settings. Error:\n{e}")

        sys.exit()
    return settings

def getRules(pathToRules):
    try:
        with open(pathToRules, "r") as f:
            rules = json.load(f)
    except Exception as e:
        print(f"An error occured with loading the rules. Error:\n{e}")

        sys.exit()
    return rules

def getLLMresponse(prompt, model="openrouter/google/gemma-4-26b-a4b-it:free"):
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
print_banner()

prompt = ""
history = []

settings = getSettings("settings.json")

api = settings["api"]
maxContext = settings["maxContext"]
rules = "\n".join(settings["rules"])

os.environ["OPENROUTER_API_KEY"] = api

with open("Data/chemMole.txt", "r") as f:
    text = f.read()
chunks = chunk_text(text, chunk_size=80, overlap=20)

from embedding import embed_texts

chunk_embeds = embed_texts(chunks)

while True:
    userPrompt = input(style("\n❯ You: ", BOLD, GREEN))

    if "/q" in userPrompt or "/e" in userPrompt:
        print(style("\n✦ Bye!", BOLD, MAGENTA))
        break
        
    history.append(f"User: {userPrompt}")
    ctx = "\n".join(retrieve(userPrompt, chunks, chunk_embeds))
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
    response = getLLMresponse(prompt, "openrouter/tencent/hy3-preview:free")
    history.append(f"AI: {response}")
    history = history[-maxContext:]
    panel("Assistant", response, YELLOW)

