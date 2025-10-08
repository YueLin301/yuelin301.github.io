---
title: Minimal LLM Agent
date: 2025-10-08 12:40:00 +0800
categories: [Artificial Intelligence, Machine Learning Basics]
tags: [Tech, Efficiency, Code_Utils]
math: True
---


> 要管理几个平台的API，包括测试基本的query啥的，顺手花了点时间写了个这个
{: .prompt-warning }


## Overview

```
pip install MinimalLLMAgent
```

- [Homepage](https://yuelin301.github.io/posts/Minimal-LLM-Agent/#3-interact): https://yuelin301.github.io/posts/Minimal-LLM-Agent
- [PyPI Page](https://pypi.org/project/MinimalLLMAgent/): https://pypi.org/project/MinimalLLMAgent/
- [GitHub Page](https://github.com/YueLin301/min_llm_agent): https://github.com/YueLin301/min_llm_agent


Features:
- simple and unified
- memory management
- a terminal simulation that allows for web-style interaction


Models & Pricing:

`supported_platform_name_list = ["OpenAI", "Grok", "DeepSeek", "Gemini", "Ali"]`

- [OpenAI](https://openai.com/api/pricing/)
- [Grok](https://docs.x.ai/docs/models)
- [DeepSeek](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
- [Gemini](https://ai.google.dev/gemini-api/docs/pricing?gad_campaignid=20860603089&gbraid=0AAAAACn9t65pzlA_HxdUpPvBVpGwkD-14&hl)
- [Ali](https://help.aliyun.com/zh/model-studio/models)

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
            "content": "What is the capital of France?",
        }
    ]
    response = llm_agent(messages)

    print(f"Messages: {messages}")
    print(f"Answer by model {llm_agent.model_name}: {response}")
    
    # llm_agent.print_memory(memory_item_separator="/")
    llm_agent.print_memory()
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

## How to Use

### API Key

For security reasons, this project does not maintain any API key files. You need to configure the API key yourself in the **environment variables.** Check the following guidelines to see how it is done:
- [OpenAI guideline](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
- [Alibaba guideline](https://help.aliyun.com/zh/model-studio/configure-api-key-through-environment-variables)


Resources:
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
    - [A query example](https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api)
    - [Dashboard](https://bailian.console.aliyun.com/?tab=model#/efm/model_experience_center/text)

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