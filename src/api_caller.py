import os
import requests
from typing import List, Dict, Any

OPENAI_API_BASE = "http://123.129.219.111:3000/v1"
# MODEL_NAME = "deepseek-v3.2"
MODEL_NAME = "gpt-4o"


class ReActTrajectoryCaller:
    def __init__(
        self,
        model: str = MODEL_NAME,
        api_base: str = OPENAI_API_BASE,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = ""
        self.task_prompt = ""

        api_key = os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("API_KEY environment variable is not set")

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        self.messages: List[Dict[str, str]] = []

    def add_system(self, content: str):
        self.system_prompt = content
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str):
        self.task_prompt = content
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_interpreter(self, content: str):
        """
        Add execution result returned by code interpreter / sandbox.
        This mirrors the <interpreter>...</interpreter> block in your prompt.
        """
        self.messages.append(
            {
                "role": "user",
                "content": f"<interpreter>\n{content}\n</interpreter>",
            }
        )

    def step(self) -> str:
        """
        Send current messages to the model and get the next assistant reply.
        """
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]

        self.add_assistant(reply)
        return reply

