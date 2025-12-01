#!/usr/bin/env python3
"""
Enhanced Go Code Synthesis Pipeline
Supports multiple LLMs (Claude, Gemini, OpenAI, DeepSeek) and extended static analysis.

Batch mode with --taskfolder and --outputfolder.
For each .txt task file in taskfolder, run the pipeline and save a corresponding .json result file in outputfolder.
example usage: 
python3 src/go_generator_multi_model.py --taskfolder benchmark/concurrency/realworld_bug_scene/ --outputfolder benchmark/concurrency/results/ --provider anthropic --batchlimit 3

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
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
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
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        self.system_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        combined_prompt = f"SYSTEM INSTRUCTION: {system_prompt}\n\nUSER TASK: {user_prompt}"
        response = self.model.generate_content(combined_prompt)
        return response.text

class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.model_name = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

# --- Analysis Data Structures ---

@dataclass
class AnalysisResult:
    """Result from a single analysis tool."""
    tool_name: str
    passed: bool
    output: str
    errors: List[str]

class GoCodeSynthesisPipeline:
    """Complete pipeline for Go code generation and analysis."""

    def __init__(
        self,
        llm_backend: LLMBackend,
        max_iterations: int = 5,
    ):
        self.llm = llm_backend
        self.max_iterations = max_iterations
        self.workspace = None
        self._last_code = ""

    def _check_tool(self, tool: str) -> bool:
        """Check if a tool is available."""
        try:
            subprocess.run([tool, "--help"], capture_output=True, timeout=2)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                subprocess.run([tool, "version"], capture_output=True, timeout=2)
                return True
            except Exception:
                return False

    def check_prerequisites(self) -> bool:
        """Verify all required tools are installed."""
        tools = {
            "go": "Go compiler",
            "errcheck": "github.com/kisielk/errcheck@latest",
            "go-errorlint": "github.com/polyfloyd/go-errorlint@latest",
            "staticcheck": "honnef.co/go/tools/cmd/staticcheck@latest",
            "revive": "github.com/mgechev/revive@latest",
            "ineffassign": "github.com/gordonklaus/ineffassign@latest",
            "govulncheck": "golang.org/x/vuln/cmd/govulncheck@latest"
        }

        missing = []
        for tool, install in tools.items():
            if not self._check_tool(tool):
                cmd_str = (
                    f"go install {install}"
                    if "github" in install or "honnef" in install or "golang.org" in install
                    else "Install Go"
                )
                missing.append(f"  • {tool}: {cmd_str}")

        if missing:
            print("❌ Missing required tools:")
            print("\n".join(missing))
            return False
        return True

    def setup_workspace(self) -> str:
        """Create temporary Go workspace."""
        self.workspace = tempfile.mkdtemp(prefix="go_synthesis_")
        subprocess.run(
            ["go", "mod", "init", "synthesis"], cwd=self.workspace, capture_output=True
        )
        return self.workspace

    def cleanup_workspace(self):
        if self.workspace and os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
            self.workspace = None

    def generate_code(self, prompt: str, feedback: Optional[str] = None) -> Tuple[str, str]:
        """Generate Go code using the configured LLM backend."""
        
        system = """You are an expert Go programmer. Generate clean, single-file Go code.
                Strictly follow these rules:
                1. Return ONLY the Go code.
                2. Do not use Markdown formatting (no ```go ... ```).
                3. Include package main and func main().
                4. Ensure all imports are used.
                """

        if feedback:
            message = f"""The previous code had issues. Please fix them.

TASK:
{prompt}

PREVIOUS CODE:
{self._last_code}

ANALYSIS ERRORS:
{feedback}

Generate corrected Go code addressing all issues above. Return ONLY the raw code."""
        else:
            message = f"""Generate a complete, single-file Go program for this task:

{prompt}

Make it production-ready, concurrent (if applicable), and robust."""

        raw_response = self.llm.generate(system, message)
        
        # Cleanup response if LLM ignores instructions and adds markdown
        code = raw_response.strip()
        if "```go" in code:
            code = code.split("```go", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
        
        return code.strip(), message

    def write_to_workspace(self, filename: str, content: str) -> str:
        filepath = os.path.join(self.workspace, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def run_go_mod_tidy(self):
        if verbose:
            print("\n  🔧 Running go mod tidy...")
        result = self.run_tool(["go", "mod", "tidy"], "go mod tidy")
        if not result.passed:
            print(f"  ⚠️ go mod tidy failed: {result.output}")
        return result.passed

    def run_tool(self, cmd: List[str], tool_name: str) -> AnalysisResult:
        try:
            result = subprocess.run(
                cmd, cwd=self.workspace, capture_output=True, text=True, timeout=30
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            errors = [] if passed else [line.strip() for line in output.split("\n") if line.strip()]
            return AnalysisResult(tool_name, passed, output, errors)
        except subprocess.TimeoutExpired:
            return AnalysisResult(tool_name, False, "Timeout", ["Tool timed out after 30s"])
        except Exception as e:
            return AnalysisResult(tool_name, False, str(e), [f"Failed: {e}"])

    # --- Analysis Categories ---

    def analyze_concurrency(self, filepath: str) -> List[AnalysisResult]:
        if verbose:
            print("    → go build -race")
        results = [self.run_tool(["go", "build", "-race", "-o", "/dev/null", filepath], "go build -race")]
        if verbose:
            print("    → gosec")
        results.append(self.run_tool(["gosec", "."], "gosec"))
        if verbose:
            print("    → govulncheck")
        results.append(self.run_tool(["govulncheck", "."], "govulncheck"))
        return results

    # def analyze_code_quality(self, filepath: str) -> List[AnalysisResult]:
    #     print("    → go vet")
    #     results = [self.run_tool(["go", "vet", filepath], "go vet")]
    #     print("    → revive")
    #     results.append(self.run_tool(["revive", filepath], "revive"))
    #     print("    → ineffassign")
    #     results.append(self.run_tool(["ineffassign", filepath], "ineffassign"))
    #     return results

    def analyze_error_handling(self, filepath: str) -> List[AnalysisResult]:
        if verbose:
            print("    → errcheck")
        results = [self.run_tool(["errcheck", filepath], "errcheck")]
        if verbose:
            print("    → go-errorlint")
        results.append(self.run_tool(["go-errorlint", filepath], "go-errorlint"))
        return results

    def analyze_performance(self, filepath: str) -> List[AnalysisResult]:
        if verbose:
            print("    → staticcheck")
        return [self.run_tool(["staticcheck", filepath], "staticcheck")]

    def run_all_analyses(self, filepath: str) -> Dict[str, List[AnalysisResult]]:
        self.run_go_mod_tidy()
        if verbose:
            print("\n  🔍 Running Analysis Tools...")
        return {
            "concurrency_security": self.analyze_concurrency(filepath),
            # "quality_correctness": self.analyze_code_quality(filepath),
            "error_handling": self.analyze_error_handling(filepath),
            "performance": self.analyze_performance(filepath),
        }

    def format_feedback(self, analyses: Dict[str, List[AnalysisResult]]) -> Optional[str]:
        feedback = []
        has_errors = False
        for category, results in analyses.items():
            for result in results:
                if not result.passed:
                    has_errors = True
                    # Simplify output for LLM consumption
                    feedback.append(f"TOOL: {result.tool_name}\nERROR:\n{result.output[:500]}")
        return "\n\n".join(feedback) if has_errors else None

    def run(self, task: str) -> Tuple[str, bool, List[Dict[str, Any]]]:
        """
        Run the synthesis+analysis loop.

        Returns:
            final_code: str
            success: bool
            rounds: List[{
                "code": str,
                "passed": bool,
                "errors": List[{"verifier": str, "error": str}]
            }]
        """
        if verbose:
            print("=" * 70)
            print("🚀 Go Code Synthesis Pipeline (Enhanced)")
            print("=" * 70)
            print(f"\n📝 Task: {task}\n")

        rounds: List[Dict[str, Any]] = []

        if not self.check_prerequisites():
            return "", False, rounds

        self.setup_workspace()
        if verbose:
            print(f"📁 Workspace: {self.workspace}\n")

        feedback = None
        final_code = None
        success = False

        try:
            for iteration in range(1, self.max_iterations + 1):
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"🔄 ITERATION {iteration}/{self.max_iterations}")
                    print(f"{'='*70}")

                    print(f"\n  ✨ Generating Go code...")
                code, full_prompt = self.generate_code(task, feedback)
                self._last_code = code
                final_code = code

                filepath = self.write_to_workspace("main.go", code)
                # self.write_to_workspace(f"iter_{iteration}.go", code)

                analyses = self.run_all_analyses(filepath)
                
                # Check if all passed
                all_passed = all(r.passed for results in analyses.values() for r in results)
                
                # Collect per-round error details
                round_info = {
                    "code": code,
                    "passed": all_passed,
                    "errors": []
                }
                if verbose:
                    print("\n  📊 Analysis Summary:")
                for cat, results in analyses.items():
                    for result in results:
                        status = "✅" if result.passed else "❌"
                        if verbose:
                            print(f"    {status} {result.tool_name}")
                        if not result.passed:
                            round_info["errors"].append({
                                "verifier": result.tool_name,
                                "error": result.output.strip()
                            })

                rounds.append(round_info)

                if all_passed:
                    if verbose:
                        print("\n✅ SUCCESS! All analyses passed!")
                    success = True
                    break

                feedback = self.format_feedback(analyses)
                if verbose:
                    print(f"\n  ⚠️  Issues detected. Retrying...")

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
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIBackend(key, model or "gpt-4o")
        
    elif provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        # DeepSeek is API-compatible with OpenAI
        return OpenAIBackend(
            key, 
            model or "deepseek-coder", 
            base_url="https://api.deepseek.com"
        )
        
    raise ValueError(f"Unknown provider: {provider}")


def main():
    parser = argparse.ArgumentParser(description="Go Code Synthesis Pipeline")
    parser.add_argument("task", nargs="?", help="The task description (single-task mode)")
    parser.add_argument("-f", "--file", help="Read task from file (single-task mode)")
    parser.add_argument("--provider", choices=["anthropic", "gemini", "openai", "deepseek"],
                        default="anthropic", help="LLM Provider")
    parser.add_argument("--model", help="Specific model name (optional)")
    parser.add_argument("--taskfolder", help="Folder containing .txt task files (batch mode)")
    parser.add_argument("--outputfolder", help="Folder to write .json results for tasks (batch mode)")
    parser.add_argument("--batchlimit", help="maximum number of files processed for testing purporse(batch mode)")

    args = parser.parse_args()

    # Determine mode: batch or single-task
    batch_mode = args.taskfolder is not None
    global verbose

    if batch_mode:
        verbose = False
        # Batch mode: require outputfolder
        if not args.outputfolder:
            print("❌ In batch mode, --outputfolder is required.")
            sys.exit(1)

        taskfolder = args.taskfolder
        outputfolder = os.path.join(args.outputfolder, args.provider)

        if not os.path.isdir(taskfolder):
            print(f"❌ Task folder does not exist or is not a directory: {taskfolder}")
            sys.exit(1)

        os.makedirs(outputfolder, exist_ok=True)

        # Setup backend and pipeline once for batch
        backend = get_backend(args.provider, args.model)
        pipeline = GoCodeSynthesisPipeline(backend, max_iterations=3)

        model_name = args.model or getattr(backend, "model_name", None)
        task_files = [f for f in sorted(os.listdir(taskfolder)) if f.lower().endswith(".txt")]

        if args.batchlimit:
            try:
                limit = int(args.batchlimit)
                task_files = task_files[:limit]
            except ValueError:
                pass

        total = len(task_files)

        print(f"📂 Found {total} task files. Starting batch processing...")

        # Iterate over .txt files in taskfolder
        for fname in tqdm(task_files, desc="Processing tasks", unit="task"):
            if not fname.lower().endswith(".txt"):
                continue

            task_path = os.path.join(taskfolder, fname)
            with open(task_path, "r", encoding="utf-8") as f:
                task_text = f.read().strip()

            if not task_text:
                print(f"\n⚠️  Skipping empty task file: {fname}")
                continue
            if verbose:
                print(f"\n\n================ Processing task file: {fname} ================")
            final_code, success, rounds = pipeline.run(task_text)

            # Build JSON record
            result_obj = {
                "task": fname,  # task file name
                "provider": args.provider,
                "model": model_name,
                "rounds": rounds,
                "passed": success
            }

            out_name = os.path.splitext(fname)[0] + ".json"
            out_path = os.path.join(outputfolder, out_name)
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(result_obj, out_f, indent=4)
            if verbose:
                print(f"\n💾 Result JSON saved to: {out_path}")

        sys.exit(0)

    # ----- Single-task mode (original behavior) -----
    if args.file:
        with open(args.file, 'r', encoding="utf-8") as f:
            task = f.read().strip()
    elif args.task:
        task = args.task
    else:
        parser.print_help()
        sys.exit(1)

    try:
        backend = get_backend(args.provider, args.model)
        pipeline = GoCodeSynthesisPipeline(backend, max_iterations=5)
        final_code, success, rounds = pipeline.run(task)
        
        output_file = "generated_code.go"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_code)
            
        print(f"\n💾 Final code saved to: {output_file}")
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
