import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai-next.com/v1"),
    api_key=os.environ["LLM_API_KEY"],
)
for m in sorted(client.models.list(), key=lambda m: m.id):
    print(m.id)