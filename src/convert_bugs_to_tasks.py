import os
import argparse
from pathlib import Path
from anthropic import Anthropic

# ⚙️ 配置
INPUT_DIR = "concurrency_bugs"   # 读取 txt 的文件夹
OUTPUT_DIR = "benchmark/concurrency/realworld_bug_scene" # 写入结果的文件夹
MODEL_NAME = "claude-sonnet-4-5"  # Claude 模型名称
PROMPT = """
Here is a commit message that records how a concurrency bug is fixed in a live repo. please craft a programming problem inspired by this bug fix. this programming problem should be a standalone task that could be implemented by a programmer as a runnable go program. It should reflect the concurrency bug but don't directly mention what the bug might be or how to fix it. please solely return the problem itself.\n
"""

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def process_file(client: Anthropic, input_path: Path, output_path: Path):
    # 读取文件内容
    with input_path.open("r", encoding="utf-8") as f:
        content = f.read()

    print(f"Sending to Claude: {input_path.name}")
    resp = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        temperature=0,
        messages=[
            {"role": "user", "content": PROMPT+content}
        ],
    )

    # 提取 Claude 回复文本
    reply_text = "\n".join(
        block.text for block in resp.content if block.type == "text"
    )

    # 写入输出
    with output_path.open("w", encoding="utf-8") as f:
        f.write(reply_text)

    print(f"Saved: {output_path}")

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理多少个 .txt 文件 (默认全部)")
    args = parser.parse_args()

    limit = args.limit

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("请先在环境变量里设置 ANTHROPIC_API_KEY")

    client = Anthropic(api_key=api_key)

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"⚠️ 在 {input_dir.resolve()} 下面没有找到 .txt 文件")
        return

    # 如果设了 limit，则截断
    if limit is not None:
        txt_files = txt_files[:limit]
        print(f"限制处理 {limit} 个文件\n")

    for input_path in txt_files:
        output_path = output_dir / input_path.name
        process_file(client, input_path, output_path)

    print("\n✅ 全部处理完成")

if __name__ == "__main__":
    main()
