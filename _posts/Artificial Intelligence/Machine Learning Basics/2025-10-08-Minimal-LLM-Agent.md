---
title: Minimal LLM Agent
date: 2025-10-08 12:40:00 +0800
categories: [Artificial Intelligence, Machine Learning Basics]
tags: [Tech, Efficiency, Code_Utils]
math: True
---


> OpenAI 后面推出了个 `Response` 方法，要替代 `Chat Completion` 了，能自动记住上下文了。但是这个方法目前只有 OpenAI 系列模型能用，其他平台用不了，所以暂时先不考虑这个。
{: .prompt-info }


## Overview

```
pip install MinimalLLMAgent
```

- [Homepage](https://yuelin301.github.io/posts/Minimal-LLM-Agent): https://yuelin301.github.io/posts/Minimal-LLM-Agent
- [PyPI Page](https://pypi.org/project/MinimalLLMAgent/): https://pypi.org/project/MinimalLLMAgent/
- [GitHub Page](https://github.com/YueLin301/min_llm_agent): https://github.com/YueLin301/min_llm_agent


Features:
- simple and unified
- memory management
- a terminal simulation that allows for web-style interaction


Models & Pricing:
[[OpenAI](https://openai.com/api/pricing/), 
[Grok](https://docs.x.ai/docs/models), 
[DeepSeek](https://api-docs.deepseek.com/zh-cn/quick_start/pricing), 
[Gemini](https://ai.google.dev/gemini-api/docs/pricing?gad_campaignid=20860603089&gbraid=0AAAAACn9t65pzlA_HxdUpPvBVpGwkD-14&hl), 
[Ali](https://help.aliyun.com/zh/model-studio/models)]

```python
from min_llm_agent import *

print_all_supported_platforms()
print_all_supported_accessible_models()

# supported_platform_name_list = ["OpenAI", "Grok", "DeepSeek", "Gemini", "Ali"]
print_accessible_models("OpenAI", id_only=True)
```


## Examples

See the `demo` folder.

### 1: String Input

```python
from min_llm_agent import min_llm_agent_class

if __name__ == "__main__":

    llm_agent = min_llm_agent_class(platform_name="OpenAI", model_name="gpt-4o-mini")

    question = "What is the capital of France?"
    response = llm_agent(question)

    print(f"Question: {question}")
    print(f"Answer by model {llm_agent.model_name}: {response}")
    
    # llm_agent.print_memory(memory_item_separator="/")
    llm_agent.print_memory()
```

```
Question: What is the capital of France?
Answer by model gpt-4o-mini: The capital of France is Paris.
======================================
Memory:
--------------------------------------
[0] (user): What is the capital of France?
[1] (assistant): The capital of France is Paris.
======================================
```


### 2: Dict Input

```python
from min_llm_agent import min_llm_agent_class

if __name__ == "__main__":

    llm_agent = min_llm_agent_class(platform_name="OpenAI", model_name="gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "1+1=?",
        },
        {
            "role": "user",
            "content": "1+2=?",
        }
    ]
    response = llm_agent(messages)

    print(f"Messages: {messages}")
    print(f"Answer by model {llm_agent.model_name}: {response}")
    
    llm_agent.print_memory()
```


```
Messages: [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': '1+1=?'}, {'role': 'user', 'content': '1+2=?'}]
Answer by model gpt-4o-mini: 1 + 1 = 2 and 1 + 2 = 3.
======================================
Memory:
--------------------------------------
[0] (system): You are a helpful assistant.
[1] (user): 1+1=?
[2] (user): 1+2=?
[3] (assistant): 1 + 1 = 2 and 1 + 2 = 3.
======================================
```

### 3: Interact

```python
from min_llm_agent import min_llm_agent_class

if __name__ == "__main__":

    llm_agent = min_llm_agent_class(platform_name="OpenAI", model_name="gpt-4o-mini")
    llm_agent.interact()
```


```
==========================

This is Yue's minimal LLM agent, powered by the model "gpt-4o-mini".

- To submit a query: start a new line, type '/', and press Enter.
    - Line breaks are allowed and recognized as a part of the query.
- Query 'q' or 'Q' to exit.
- Query 'm' or 'M' to print the memory.

See more details on: https://github.com/YueLin301/min_llm_agent

>>>>>>>>>>>>>>>>>>>>>>>>>>
[0] Question:
> 1+1=
/

<<<<<<<<<<<<<<<<<<<<<<<<<<
[0] Answer by the model gpt-4o-mini:
1 + 1 = 2.


>>>>>>>>>>>>>>>>>>>>>>>>>>
[1] Question:
> how are you
/

<<<<<<<<<<<<<<<<<<<<<<<<<<
[1] Answer by the model gpt-4o-mini:
I'm just a computer program, so I don't have feelings, but I'm here and ready to help you! How can I assist you today?


>>>>>>>>>>>>>>>>>>>>>>>>>>
[2] Question:
> m
/
==========================
Memory:
--------------------------
[0] (user): 1+1=
[1] (assistant): 1 + 1 = 2.
[2] (user): how are you
[3] (assistant): I'm just a computer program, so I don't have feelings, but I'm here and ready to help you! How can I assist you today?
==========================
>>>>>>>>>>>>>>>>>>>>>>>>>>
[2] Question:
> q
/
```


### 1a: Memoryless Query

```python
from min_llm_agent import min_llm_agent_class

if __name__ == "__main__":

    llm_agent = min_llm_agent_class(platform_name="OpenAI", model_name="gpt-4o-mini")

    question = "What is the capital of France?"
    response = llm_agent(question, with_memory=False)

    print(f"Question: {question}")
    print(f"Answer by model {llm_agent.model_name}: {response}")

    llm_agent.print_memory()
```

```
Question: What is the capital of France?
Answer by model gpt-4o-mini: The capital of France is Paris.
======================================
Memory:
--------------------------------------
======================================
```

### 2a: More Keywords

- JSON mode
- temperature

See more detailed keywords on [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat).

```python
from min_llm_agent import min_llm_agent_class

if __name__ == "__main__":

    llm_agent = min_llm_agent_class(platform_name="OpenAI", model_name="gpt-4o-mini")

    messages = [
        {"role": "system", "content": "Extract the event information. Output in JSON format, including the event name, date, and participants."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ]

    response = llm_agent(messages, response_format={"type": "json_object"}, temperature=0.5)

    print(f"Messages: {messages}")
    print(f"Answer by model {llm_agent.model_name}: {response}")

    llm_agent.print_memory()
```

```
Messages: [{'role': 'system', 'content': 'Extract the event information. Output in JSON format, including the event name, date, and participants.'}, {'role': 'user', 'content': 'Alice and Bob are going to a science fair on Friday.'}]
Answer by model gpt-4o-mini: {
  "event_name": "Science Fair",
  "date": "Friday",
  "participants": ["Alice", "Bob"]
}
================================================
Memory:
------------------------------------------------
[0] (system): Extract the event information. Output in JSON format, including the event name, date, and participants.
[1] (user): Alice and Bob are going to a science fair on Friday.
[2] (assistant): {
  "event_name": "Science Fair",
  "date": "Friday",
  "participants": ["Alice", "Bob"]
}
================================================
```


### 4: Memory Management

```python
from min_llm_agent import min_llm_agent_class
from LyPythonToolbox import lyprint_separator
from pprint import pprint

if __name__ == "__main__":
    llm_agent = min_llm_agent_class(platform_name="OpenAI", model_name="gpt-4o-mini")

    response = llm_agent("1+1=?")
    response = llm_agent("1+2=?", with_memory=False)
    llm_agent.print_memory()
    
    lyprint_separator("|")
    llm_agent.reset_memory()
    print("Reset memory...")
    response = llm_agent("1+3=?")
    llm_agent.print_memory()
    
    lyprint_separator("|")
    print("Set memory...")
    llm_agent.set_memory([{"role": "system", "content": "You are a self-interested and rational player."}])
    llm_agent.print_memory()

    lyprint_separator("|")
    response = llm_agent("You are playing a coordination game. State your strategy in only a sentence.", role="user")
    print("Get memory and pprint...")
    memory = llm_agent.get_memory()
    pprint(memory)

    lyprint_separator("|")
    print("Append memory...")
    llm_agent.append_memory({"role": "user", "content": "1+4=?"})
    llm_agent.print_memory()
```


```
================================================
Memory:
------------------------------------------------
[0] (user): 1+1=?
[1] (assistant): 1 + 1 = 2.
================================================
||||||||||||||||||||||||||||||||||||||||||||||||
Reset memory...
================================================
Memory:
------------------------------------------------
[0] (user): 1+3=?
[1] (assistant): 1 + 3 = 4.
================================================
||||||||||||||||||||||||||||||||||||||||||||||||
Set memory...
================================================
Memory:
------------------------------------------------
[0] (system): You are a self-interested and rational player.
================================================
||||||||||||||||||||||||||||||||||||||||||||||||
Get memory and pprint...
[{'content': 'You are a self-interested and rational player.',
  'role': 'system'},
 {'content': 'You are playing a coordination game. State your strategy in only '
             'a sentence.',
  'role': 'user'},
 {'content': 'I will choose the strategy that aligns with the most commonly '
             'played option by other players to ensure mutual coordination and '
             'benefit.',
  'role': 'assistant'}]
||||||||||||||||||||||||||||||||||||||||||||||||
Append memory...
================================================
Memory:
------------------------------------------------
[0] (system): You are a self-interested and rational player.
[1] (user): You are playing a coordination game. State your strategy in only a sentence.
[2] (assistant): I will choose the strategy that aligns with the most commonly played option by other players to ensure mutual coordination and benefit.
[3] (user): 1+4=?
================================================
```



## How to Use

### Installation

```
pip install MinimalLLMAgent
```

### API Key

For security reasons, this project does not maintain any API key files. You need to configure the API key yourself in the **environment variables.** Check the following guidelines to see how it is done:
- [OpenAI guideline](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
- [Alibaba guideline](https://help.aliyun.com/zh/model-studio/configure-api-key-through-environment-variables)


**An Example Set Sp for MacOS Users:**

1. Append the following API configurations to the end of the `~/.zshrc` file.

```
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"

export XAI_API_KEY="xai-xxx"
export XAI_BASE_URL="https://api.x.ai/v1"

export DEEPSEEK_API_KEY="sk-xxx"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"

export GEMINI_API_KEY=""
export GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

export DASHSCOPE_API_KEY="sk-xxx"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

2. Run `source ~/.zshrc` to update.


## Resources
- [LLM Comparisons](https://artificialanalysis.ai/)
- OpenAI
    - [Models and pricing](https://openai.com/api/pricing/)
    - [How to create API keys](https://platform.openai.com/settings/organization/api-keys)
    - [Billing](https://platform.openai.com/settings/organization/billing/overview)
- Grok
    - [Models and pricing](https://docs.x.ai/docs/models)
    - [API Guides](https://docs.x.ai/docs/guides/responses-api)
    - [A query example](https://docs.x.ai/docs/tutorial#step-4-make-a-request-from-python-or-javascript)
    - [A query example - OpenAI lib](https://www.datacamp.com/tutorial/grok-3-api)
- DeepSeek
    - [Models and pricing](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
    - [API](https://platform.deepseek.com/api_keys)
    - [Billing](https://platform.deepseek.com/usage)
- Gemini
    - [Models and pricing](https://ai.google.dev/gemini-api/docs/pricing?gad_campaignid=20860603089&gbraid=0AAAAACn9t65pzlA_HxdUpPvBVpGwkD-14&hl)
    - [API](https://ai.google.dev/gemini-api/docs/api-key)
    - [A query example](https://ai.google.dev/gemini-api/docs/api-key?hl=zh-cn#provide-api-key-explicitly)
    - [A query example - OpenAI lib](https://ai.google.dev/gemini-api/docs/openai?hl=zh-cn#python)
    - [Billing](https://aistudio.google.com/usage)
- Alibaba
    - [Models and pricing](https://help.aliyun.com/zh/model-studio/models)
    - [API Guides](https://help.aliyun.com/zh/model-studio/get-api-key?scm=20140722.H_2712195._.OR_help-T_cn~zh-V_1)
    - [A query example](https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api)
    - [Dashboard](https://bailian.console.aliyun.com/?tab=model#/efm/model_experience_center/text)