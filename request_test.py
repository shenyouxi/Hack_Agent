from adaption import Adaption

client = Adaption()

response = client.responses.create(
    model="qwen3.5-397b",
    input="Write a one-sentence bedtime story about a unicorn.",
)

print(response.output_text)