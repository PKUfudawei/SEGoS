import json
import os
from segos.clients.agent import run_agent
from evaluation.token_usage import normalize_agent_usage, record_usage

def get_llm_response(messages, is_string=False, model=None):
    model = model or os.getenv("SEGOS_MODEL")
    text, usage = run_agent(messages, model)
    record_usage(normalize_agent_usage(usage), bucket="aux")
    if is_string:
        return text
    return json.loads(text.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
