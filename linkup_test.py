import os
from linkup import LinkupClient
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LINKUP_API_KEY")
if not api_key:
    raise RuntimeError("LINKUP_API_KEY is missing. Check your .env file.")

client = LinkupClient(api_key=api_key)
response = client.search(
    query="What are common substitutes for double cream in cooking?",
    depth="fast",
    output_type="sourcedAnswer",
)

print("LINKUP WORKED")
print(response)
