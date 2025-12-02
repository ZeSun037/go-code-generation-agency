#!/usr/bin/env python3
"""
Enhanced Go Code Synthesis Pipeline
Supports multiple LLMs (Claude, Gemini, OpenAI, DeepSeek) and extended static analysis.

Batch mode with --taskfolder and --outputfolder.
For each .txt task file in taskfolder, run the pipeline and save a corresponding .json result file in outputfolder.
example usage: 
python3 src/leetcode_solver_multi_model.py \
    --taskfolder benchmark/coding/leetcode/ \
    --outputfolder benchmark/coding/results/ \
    --provider openai

Prerequisites (Python):
    pip install anthropic google-generativeai openai

Prerequisites (Go Tools):
    go install github.com/kisielk/errcheck@latest
    go install github.com/polyfloyd/go-errorlint@latest
    go install honnef.co/go/tools/cmd/staticcheck@latest
    go install github.com/mgechev/revive@latest
    go install github.com/gordonklaus/ineffassign@latest
    go install golang.org/x/vuln/cmd/govulncheck@latest
"""

import json
import os
import sys
import subprocess
import tempfile
import shutil
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm

# --- LLM Backend Abstraction ---

verbose = True

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass

class AnthropicBackend(LLMBackend):
    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.model_name = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text

class GeminiBackend(LLMBackend):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        self.system_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        combined_prompt = f"SYSTEM INSTRUCTION: {system_prompt}\n\nUSER TASK: {user_prompt}"
        response = self.model.generate_content(combined_prompt)
        return response.text

class AzureBackend(LLMBackend):
    def __init__(self, key:str, model: str):
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        endpoint = "https://models.github.ai/inference"
        self.model = model
        self.model_name = model
        token = key  
        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from azure.ai.inference.models import SystemMessage, UserMessage
        response = self.client.complete(
            messages=[
                SystemMessage(system_prompt),
                UserMessage(user_prompt),
            ],
            max_tokens=4000,
            model=self.model
        )

        return response.choices[0].message.content
        
class OpenAIBackend(LLMBackend):
    def __init__(self, key:str, model: str):
        from openai import OpenAI
        endpoint = "https://models.github.ai/inference"
        self.model = model
        self.model_name = model
        self.client = OpenAI(
            base_url=endpoint,
            api_key=key,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "developer",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model=self.model
        )

        return response.choices[0].message.content
# --- Analysis Data Structures ---

@dataclass
class AnalysisResult:
    """Result from running a single external tool (here: go test)."""
    tool_name: str
    passed: bool
    output: str
    errors: List[str]


class GoCodeSynthesisPipeline:
    """Go code synthesis loop driven by `go test` failures."""

    def __init__(
        self,
        llm_backend: LLMBackend,
        max_iterations: int = 5,
    ):
        self.llm = llm_backend
        self.max_iterations = max_iterations
        self.workspace: Optional[str] = None
        self._last_code: str = ""

    # ---------- basic env helpers ----------

    def _check_tool(self, tool: str) -> bool:
        """Check if a tool is available (only need `go`)."""
        try:
            subprocess.run([tool, "help"], capture_output=True, timeout=2)
            return True
        except Exception:
            return False

    def check_prerequisites(self) -> bool:
        """Verify required tools are installed (just `go`)."""
        tools = {
            "go": "Go compiler",
        }

        missing = []
        for tool, install in tools.items():
            if not self._check_tool(tool):
                cmd_str = "Install Go"
                missing.append(f"  • {tool}: {cmd_str}")

        if missing:
            print("❌ Missing required tools:")
            print("\n".join(missing))
            return False
        return True

    def setup_workspace(self) -> str:
        """Create temporary Go module workspace."""
        self.workspace = tempfile.mkdtemp(prefix="go_synthesis_")
        subprocess.run(
            ["go", "mod", "init", "synthesis"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
        )
        return self.workspace

    def cleanup_workspace(self):
        if self.workspace and os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
            self.workspace = None

    # ---------- LLM interaction ----------

    def generate_code(self, prompt: str, test_code: str, feedback: Optional[str] = None) -> Tuple[str, str]:
        """
        Ask the LLM to either generate initial code or fix code based on `go test` output.
        """

        system = """You are an expert Go programmer.

Rules:
1. Return ONLY valid Go source code (no Markdown code fences).
2. Preserve any given function signatures and types in the prompt.
3. Use the same package name as in the prompt (usually `package main`).
4. Ensure the code compiles and passes `go test` with the provided tests.
"""

        if feedback:
            message = f"""The previous Go code failed when running `go test ./...`. Please fix it.

TASK PROMPT:
{prompt}

PREVIOUS CODE:
{self._last_code}

GO TEST CODE:
{test_code}

GO TEST OUTPUT (stderr/stdout):
{feedback}

Generate corrected Go code that fixes the issues and makes all tests pass.
Return ONLY the raw Go source code (no ``` fences)."""
        else:
            # 首轮：根据 prompt 生成完整实现
            message = f"""You are given a partial Go program or a description of a task.

Complete or implement the Go code so that it compiles and passes the test suite
defined in a separate *_test.go file.

TASK PROMPT:
{prompt}
"""

        raw_response = self.llm.generate(system, message)

        # 清理掉 LLM 偶尔加的 ```go ``` 包裹
        code = raw_response.strip()
        if "```go" in code:
            code = code.split("```go", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]

        return code.strip(), message

    # ---------- filesystem / tool helpers ----------

    def write_to_workspace(self, filename: str, content: str) -> str:
        if self.workspace is None:
            raise RuntimeError("Workspace is not set up")
        filepath = os.path.join(self.workspace, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def run_tool(self, cmd: List[str], tool_name: str) -> AnalysisResult:
        """Run an external command (here: `go test`) inside workspace."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (result.stdout or "") + (result.stderr or "")
            passed = result.returncode == 0
            errors = [] if passed else [line.strip() for line in output.splitlines() if line.strip()]
            return AnalysisResult(tool_name, passed, output, errors)
        except subprocess.TimeoutExpired:
            return AnalysisResult(tool_name, False, "Timeout", ["Tool timed out after 120s"])
        except Exception as e:
            return AnalysisResult(tool_name, False, str(e), [f"Failed: {e}"])

    # ---------- main loop: generate -> go test -> fix ----------

    def run_go_mod_tidy(self):
        if verbose:
            print("\n  🔧 Running go mod tidy...")
        result = self.run_tool(["go", "mod", "tidy"], "go mod tidy")
        if not result.passed:
            print(f"  ⚠️ go mod tidy failed: {result.output}")
        return result.passed
    
    def run(
        self,
        task: str,
        test_setup: Optional[str] = None,
        example_test: Optional[str] = None,
    ) -> Tuple[str, bool, List[Dict[str, Any]]]:
        """
        Test-driven synthesis loop.

        Args:
            task:          textual prompt (可以是完整/部分 Go 代码，如你 task.json 里的 prompt)
            test_setup:    来自 task.json["test_setup"] 的内容 (包含 package / imports)
            example_test:  来自 task.json["example_test"] (或你想用的 test body)

        Returns:
            final_code: str
            success:   bool (go test 是否通过)
            rounds:    list of { "code", "passed", "errors": [ {verifier, error}, ... ] }
        """
        if verbose:
            print("=" * 70)
            print("🚀 Go Code Synthesis Pipeline (go test-based)")
            print("=" * 70)
            print(f"\n📝 Task: {task}\n")

        rounds: List[Dict[str, Any]] = []

        if not self.check_prerequisites():
            return "", False, rounds

        self.setup_workspace()
        if verbose:
            print(f"📁 Workspace: {self.workspace}\n")

        # 先把 tests 写进 workspace：test_setup + example_test
        test_code = ""
        if test_setup or example_test:
            pieces: List[str] = []
            if test_setup:
                pieces.append(test_setup.strip())
            if example_test:
                pieces.append(example_test.strip())
            test_code = "\n\n".join(pieces) + "\n"

            if verbose:
                print("🧪 Writing tests to main_test.go")
            self.write_to_workspace("main_test.go", test_code)

        feedback: Optional[str] = None
        final_code: Optional[str] = None
        success = False

        try:
            for iteration in range(1, self.max_iterations + 1):
                if verbose:
                    print("\n" + "=" * 70)
                    print(f"🔄 ITERATION {iteration}/{self.max_iterations}")
                    print("=" * 70)
                    print("\n  ✨ Generating Go code...")

                code, full_prompt = self.generate_code(task, test_code, feedback)
                self._last_code = code
                final_code = code

                #print(full_prompt)

                # 写当前解到 main.go
                self.write_to_workspace("main.go", code)

                # 跑 go test
                if verbose:
                    print("  🧪 Running `go test ./...` ...")
                self.run_go_mod_tidy()
                test_result = self.run_tool(["go", "test", "./..."], "go test")

                round_info: Dict[str, Any] = {
                    "code": code,
                    "passed": test_result.passed,
                    "errors": [],
                }
                if not test_result.passed:
                    round_info["errors"].append(
                        {
                            "verifier": "go test",
                            "error": test_result.output.strip(),
                        }
                    )
                rounds.append(round_info)

                if test_result.passed:
                    if verbose:
                        print("\n✅ SUCCESS! All tests passed!")
                    success = True
                    break

                # 为下一轮构造 feedback：直接用 go test 的 stdout+stderr
                truncated_out = test_result.output or ""
                if len(truncated_out) > 4000:
                    truncated_out = truncated_out[:4000] + "\n...[truncated]..."
                feedback = truncated_out

                if verbose:
                    print("\n  ⚠️ Tests failed. Retrying with feedback from `go test`...")

            if not success and verbose:
                print(f"\n❌ Max iterations ({self.max_iterations}) reached.")

            return final_code or "", success, rounds

        finally:
            self.cleanup_workspace()


def get_backend(provider: str, model: Optional[str]) -> LLMBackend:
    """Factory to create the appropriate LLM backend."""
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return AnthropicBackend(key, model or "claude-haiku-4-5")
    
    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        return GeminiBackend(key, model or "gemini-2.5-flash")
        
    elif provider == "openai":
        key = os.environ.get("GITHUB_TOKEN")
        if not key:
            raise ValueError("GITHUB_TOKEN not set")
        return OpenAIBackend(key, model or "openai/gpt-4o-mini")
        
    elif provider == "deepseek":
        key = os.environ.get("GITHUB_TOKEN")
        if not key:
            raise ValueError("GITHUB_TOKEN not set")
        return AzureBackend(
            key, 
            model or "deepseek/DeepSeek-R1-0528", 
        )
    
    elif provider == "meta":
        key = os.environ.get("GITHUB_TOKEN")
        if not key:
            raise ValueError("GITHUB_TOKEN not set")
        return AzureBackend(
            key, 
            model or "meta/Llama-4-Scout-17B-16E-Instruct", 
        )
        
    raise ValueError(f"Unknown provider: {provider}")


def main():
    parser = argparse.ArgumentParser(description="Go Code Synthesis Pipeline")
    parser.add_argument("task", nargs="?", help="The task description (single-task mode)")
    parser.add_argument("-f", "--file", help="Read task from file (single-task mode)")
    parser.add_argument("--provider", choices=["anthropic", "gemini", "openai", "deepseek", "meta"],
                        default="anthropic", help="LLM Provider")
    parser.add_argument("--model", help="Specific model name (optional)")
    parser.add_argument("--taskfolder", help="Folder containing .txt task files (batch mode)")
    parser.add_argument("--outputfolder", help="Folder to write .json results for tasks (batch mode)")
    parser.add_argument("--batchlimit", help="maximum number of files processed for testing purporse(batch mode)")
    parser.add_argument("--maxiter", help="maximum number of iterations per task")

    args = parser.parse_args()

    # Determine mode: batch or single-task
    batch_mode = args.taskfolder is not None

    max_iter = args.maxiter or "3"
    max_iter = int(max_iter)

    global verbose

    if batch_mode:
        verbose = False
        if not args.outputfolder:
            print("❌ In batch mode, --outputfolder is required.")
            sys.exit(1)

        taskfolder = args.taskfolder
        outputfolder = os.path.join(args.outputfolder, args.provider)

        if not os.path.isdir(taskfolder):
            print(f"❌ Task folder does not exist or is not a directory: {taskfolder}")
            sys.exit(1)

        os.makedirs(outputfolder, exist_ok=True)
        num_results_in_output_folder = len(
            [f for f in os.listdir(outputfolder) if os.path.isfile(os.path.join(outputfolder, f))]
        )

        backend = get_backend(args.provider, args.model)
        pipeline = GoCodeSynthesisPipeline(backend, max_iterations=max_iter)

        model_name = args.model or getattr(backend, "model_name", None)

        task_files = [
            f for f in sorted(os.listdir(taskfolder))
            if f.lower().endswith(".json")
        ]

        if args.batchlimit:
            try:
                limit = int(args.batchlimit)
                task_files = task_files[:limit]
            except ValueError:
                pass

        # 跳过已经生成结果的前 N 个任务
        task_files = task_files[num_results_in_output_folder:]

        total = len(task_files)
        print(f"📂 Found {total} task files. Starting batch processing...")

        for fname in tqdm(task_files, desc="Processing tasks", unit="task"):
            task_path = os.path.join(taskfolder, fname)

            # 默认值
            task_text: str = ""
            test_setup: Optional[str] = None
            example_test: Optional[str] = None

            if fname.lower().endswith(".json"):
                with open(task_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 对应你给的 task.json 结构
                task_text = (data.get("prompt") or "").strip()
                test_setup = data.get("test_setup")
                example_test = data.get("example_test")
            else:
                # 老的 .txt 模式：只用纯文本 task，没有测试
                with open(task_path, "r", encoding="utf-8") as f:
                    task_text = f.read().strip()

            if not task_text:
                print(f"\n⚠️  Skipping empty task file: {fname}")
                continue

            final_code, success, rounds = pipeline.run(
                task_text,
                test_setup=test_setup,
                example_test=example_test,
            )

            result_obj = {
                "task": fname,
                "provider": args.provider,
                "model": model_name,
                "rounds": rounds,
                "passed": success,
            }

            out_name = os.path.splitext(fname)[0] + ".json"
            out_path = os.path.join(outputfolder, out_name)
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(result_obj, out_f, indent=4)


if __name__ == "__main__":
    main()
