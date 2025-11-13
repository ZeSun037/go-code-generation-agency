#!/usr/bin/env python3
import os
from go_generator import GoCodeSynthesisPipeline

# 路径设置（相对于 src/）
problems_dir = "../leetcode/problems"
programs_dir = "../leetcode/programs"
results_dir = "../leetcode/results"

# 确保输出目录存在
os.makedirs(programs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 初始化 pipeline
pipeline = GoCodeSynthesisPipeline(max_iterations=5, model="claude-haiku-4-5-20251001")

# 遍历所有题目文件
for filename in os.listdir(problems_dir):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(problems_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        task = f.read().strip()

    # 替换空格为下划线
    safe_name = filename.replace(" ", "_")
    program_file = os.path.join(programs_dir, safe_name)
    result_file = os.path.join(results_dir, safe_name)

    print(f"\nProcessing: {filename} -> {safe_name}")

    try:
        # 运行 pipeline
        final_code, success, history = pipeline.run(task)

        # 保存生成代码
        with open(program_file, "w", encoding="utf-8") as f:
            f.write(final_code)

        # 保存状态
        last_iteration = history[-1] if history else {}
        status_text = f"Status: {'✅ PASSED' if success else '⚠️  HAS ISSUES'}\n"
        status_text += f"Iterations: {len(history)}"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(status_text)

        print(f"Saved program: {program_file}")
        print(f"Saved result: {result_file}")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        # 即使出错也继续处理其他题目
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"❌ Error: {e}")
