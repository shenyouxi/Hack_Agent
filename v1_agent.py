# Improved version of v0: avoid hallucination 

import os, json, sys
from openai import OpenAI

<<<<<<< HEAD
=======
# client = OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.environ["GROQ_API_KEY"],
# )
# MODEL = "openai/gpt-oss-120b"
>>>>>>> 58514ca (update v1, add v2)

client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai-next.com/v1"),
    api_key=os.environ["LLM_API_KEY"],
    timeout=60,
    max_retries=3,
)
MODEL = os.environ["LLM_MODEL"] 

NOTES_DIR = os.path.abspath("notes")

# 1. 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "读取 notes 文件夹里的一个文本文件，返回全部内容。"
                           "当用户询问某个笔记文件的内容时使用。参数是文件名，例如 todo.txt",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "文件名"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "列出 notes 文件夹里所有文件的文件名。当用户想知道有哪些笔记，"
                           "或者不确定文件名时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# 2. 工具的真实实现
def run_tool(name, args):
    if name == "read_note":
        path = os.path.abspath(os.path.join(NOTES_DIR, args["filename"]))
        if os.path.commonpath([NOTES_DIR, path]) != NOTES_DIR:
            return "错误：只能读取 notes 文件夹内的文件"
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            return content if content.strip() else "（文件是空的，没有任何内容）"
        except Exception as e:
            return f"读取失败：{e}"
    if name == "list_notes":
        try:
            return "\n".join(os.listdir(NOTES_DIR)) or "文件夹是空的"
        except Exception as e:
            return f"读取失败：{e}"
    return "未知工具"

# 3. agent 循环
question = sys.argv[1] if len(sys.argv) > 1 else "总结一下 todo.txt 的内容"
messages = [
    {
        "role": "system", 
        "content": "Always use the user's language to answer. "
                   "Keep the answer concise and easy to understand. "
                   "Only use information returned by tools. If a file is empty or a tool returns nothing, say so plainly. Never invent or guess content."
    },
    {"role": "user", "content": question},
]

for _ in range(10):
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    msg = resp.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        print(msg.content)
        break

    for call in msg.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        output = run_tool(call.function.name, args)
        print(f"[调用工具] {call.function.name} {args}")
        print(f"[工具返回] {output[:80]}")
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": output,
        })
