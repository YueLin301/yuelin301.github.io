---
title: Ollama Memo
date: 2024-07-16 12:30:00 +0800
categories: [Efficiency, Code Utils]
tags: [Tech, Efficiency, Code Utils, Llama, LLM]
math: True
---

## Installation


1. Download the app on [the website](https://ollama.com/download).
2. Install the app.
3. Run `ollama run llama3` in a terminal, then it will install `Llama3: 7B` by default, taking up 4.7G storage space.
4. Then we can query `Llama3` in the terminal.
   1. To exit: `Ctrl + d` or `/bye`
   2. To enter again: `ollama run llama3`

Once we open the app, the server will run at [http://localhost:11434](http://localhost:11434).

## Using in Command Line

My snippet in `Termius`:

```shell
source ~/.bash_profile
conda activate rlbasic
ollama run llama3
```

## Using by Python


### Way 1: Server
[[Official Guide]](https://github.com/ollama/ollama/blob/main/docs/api.md)

```python
import requests
import json

default_url = "http://127.0.0.1:11434/api/generate"


def query_model(text, server_url=default_url, use_stream=True):
    response = requests.post(
        server_url, json={"model": "llama3", "prompt": text}, stream=use_stream
    )

    results = ""
    for line in response.iter_lines():
        if line:
            json_response = json.loads(line.decode("utf-8"))
            response_i = json_response.get("response", "")
            if use_stream:
                print("Response_i:", response_i)
            results += response_i
            if json_response.get("done", False):
                break

    print("Response:", results)


while True:
    user_input = input("Input: ")
    if user_input.lower() == "quit":
        break
    query_model(user_input)
```

### Way 2: Python API

[[Official Guide]](https://pypi.org/project/ollama/)

```shell
pip install ollama
```

```python
import ollama

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": "Why is the sky blue?",
        },
    ],
)
print(response["message"]["content"])
```


```python
# Streaming
import ollama

stream = ollama.chat(
    model='llama3',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```