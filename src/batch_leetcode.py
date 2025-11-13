#!/usr/bin/env python3
import os
from go_generator import GoCodeSynthesisPipeline


problems_dir = "../leetcode/problems"
programs_dir = "../leetcode/programs"
results_dir = "../leetcode/results"


os.makedirs(programs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


pipeline = GoCodeSynthesisPipeline(max_iterations=5, model="claude-haiku-4-5-20251001")

for filename in os.listdir(problems_dir):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(problems_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        task = f.read().strip()

    safe_name = filename.replace(" ", "_")
    program_file = os.path.join(programs_dir, safe_name)
    result_file = os.path.join(results_dir, safe_name)

    print(f"\nProcessing: {filename} -> {safe_name}")

    try:

        final_code, success, history = pipeline.run(task)

        with open(program_file, "w", encoding="utf-8") as f:
            f.write(final_code)

        #
        last_iteration = history[-1] if history else {}
        status_text = f"Status: {'PASSED' if success else 'HAS ISSUES'}\n"
        status_text += f"Iterations: {len(history)}"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(status_text)

        print(f"Saved program: {program_file}")
        print(f"Saved result: {result_file}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Error: {e}")
