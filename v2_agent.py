# v2: reusable agent template (OpenNext, OpenAI-compatible API)
# Usage: python agent.py "your task"
# Needs env vars: LLM_API_KEY, LLM_MODEL  (optional: LLM_BASE_URL)

import os, json, sys
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai-next.com/v1"),
    api_key=os.environ["LLM_API_KEY"],
    timeout=60,
    max_retries=3,
)
MODEL = os.environ["LLM_MODEL"]

NOTES_DIR = os.path.abspath("notes")    # agent can READ here
OUTPUT_DIR = os.path.abspath("output")  # agent can WRITE here
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_path(base, filename):
    """Resolve filename inside base; return None if it escapes base."""
    path = os.path.abspath(os.path.join(base, filename))
    if os.path.commonpath([base, path]) != base:
        return None
    return path


# ---------- 1. Tools: add a new tool = add one schema + one function ----------
def list_notes(args):
    try:
        return "\n".join(os.listdir(NOTES_DIR)) or "文件夹是空的"
    except Exception as e:
        return f"读取失败：{e}"


def read_note(args):
    path = safe_path(NOTES_DIR, args["filename"])
    if path is None:
        return "错误：只能读取 notes 文件夹内的文件"
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return content if content.strip() else "（文件是空的，没有任何内容）"
    except Exception as e:
        return f"读取失败：{e}"


def write_output(args):
    path = safe_path(OUTPUT_DIR, args["filename"])
    if path is None:
        return "错误：只能写入 output 文件夹内的文件"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return f"已写入 output/{args['filename']}（{len(args['content'])} 字符）"
    except Exception as e:
        return f"写入失败：{e}"


def schema(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOLS = {
    "list_notes": (list_notes, schema(
        "list_notes", "列出 notes 文件夹里所有文件名。不确定文件名时使用。", {}, [])),
    "read_note": (read_note, schema(
        "read_note", "读取 notes 文件夹里的一个文本文件，返回全部内容。",
        {"filename": {"type": "string", "description": "文件名，例如 todo.txt"}}, ["filename"])),
    "write_output": (write_output, schema(
        "write_output", "把内容写入 output 文件夹里的文件（覆盖已有文件）。用于保存最终结果。",
        {"filename": {"type": "string", "description": "输出文件名，例如 plan.md"},
         "content": {"type": "string", "description": "要写入的完整文本"}},
        ["filename", "content"])),
}
tool_schemas = [s for _, s in TOOLS.values()]


def run_tool(name, args):
    if name not in TOOLS:
        return f"未知工具：{name}"
    try:
        return TOOLS[name][0](args)
    except Exception as e:  # never crash the loop because of a tool
        return f"工具执行出错：{e}"


# ---------- 2. Agent loop ----------
SYSTEM_PROMPT = (
    "Always use the user's language to answer. Keep the answer concise. "
    "Only use information returned by tools. If a file is empty or a tool returns "
    "nothing, say so plainly. Never invent or guess content. "
    "When the task asks you to save a result, call write_output and then confirm "
    "which file you wrote."
)

question = sys.argv[1] if len(sys.argv) > 1 else "总结一下 todo.txt 的内容"
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
]

for step in range(10):
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tool_schemas)
    msg = resp.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        print(msg.content)
        break

    for call in msg.tool_calls:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        output = run_tool(call.function.name, args)
        print(f"[步骤{step + 1}] 调用工具 {call.function.name} {str(args)[:80]}")
        print(f"[工具返回] {output[:80]}")
        messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
else:
    print("（达到最大步数仍未完成）")