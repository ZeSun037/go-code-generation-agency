#!/usr/bin/env python3
"""
Enhanced Go Code Synthesis Pipeline
Supports multiple LLMs (Claude, Gemini, OpenAI, DeepSeek) and extended static analysis.

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

# --- LLM Backend Abstraction ---

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass

class AnthropicBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text

class GeminiBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.system_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Gemini handles system prompts at configuration or via concatenation
        # For simplicity in 1.5, we can prepend it or use system_instruction if supported by lib version
        combined_prompt = f"SYSTEM INSTRUCTION: {system_prompt}\n\nUSER TASK: {user_prompt}"
        response = self.model.generate_content(combined_prompt)
        return response.text

class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

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
                # Special handling for built-in go tools
                cmd_str = f"go install {install}" if "github" in install or "honnef" in install or "golang.org" in install else "Install Go"
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
            code = code.split("```go")[1]
        if "```" in code:
            code = code.split("```")[0]
        
        return code.strip(), message

    def write_to_workspace(self, filename: str, content: str) -> str:
        filepath = os.path.join(self.workspace, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def run_go_mod_tidy(self):
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
        print("    → go build -race")
        results = [self.run_tool(["go", "build", "-race", "-o", "/dev/null", filepath], "go build -race")]
        print("    → gosec")
        results.append(self.run_tool(["gosec", "-quiet", filepath], "gosec"))
        print("    → govulncheck")
        results.append(self.run_tool(["govulncheck", "."], "govulncheck"))
        return results

    def analyze_code_quality(self, filepath: str) -> List[AnalysisResult]:
        print("    → go vet")
        results = [self.run_tool(["go", "vet", filepath], "go vet")]
        print("    → revive")
        results.append(self.run_tool(["revive", filepath], "revive"))
        print("    → ineffassign")
        results.append(self.run_tool(["ineffassign", filepath], "ineffassign"))
        return results

    def analyze_error_handling(self, filepath: str) -> List[AnalysisResult]:
        print("    → errcheck")
        results = [self.run_tool(["errcheck", filepath], "errcheck")]
        print("    → go-errorlint")
        results.append(self.run_tool(["go-errorlint", filepath], "go-errorlint"))
        return results

    def analyze_performance(self, filepath: str) -> List[AnalysisResult]:
        print("    → staticcheck")
        return [self.run_tool(["staticcheck", filepath], "staticcheck")]

    def run_all_analyses(self, filepath: str) -> Dict[str, List[AnalysisResult]]:
        self.run_go_mod_tidy()
        print("\n  🔍 Running Analysis Tools...")
        return {
            "concurrency_security": self.analyze_concurrency(filepath),
            "quality_correctness": self.analyze_code_quality(filepath),
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

    def run(self, task: str) -> Tuple[str, bool, List[Dict]]:
        print("=" * 70)
        print("🚀 Go Code Synthesis Pipeline (Enhanced)")
        print("=" * 70)
        print(f"\n📝 Task: {task}\n")

        if not self.check_prerequisites():
            return "", False, []

        self.setup_workspace()
        print(f"📁 Workspace: {self.workspace}\n")

        history = []
        feedback = None
        final_code = None

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"🔄 ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*70}")

            print(f"\n  ✨ Generating Go code...")
            code, full_prompt = self.generate_code(task, feedback)
            self._last_code = code
            final_code = code

            filepath = self.write_to_workspace("main.go", code)
            self.write_to_workspace(f"iter_{iteration}.go", code)

            analyses = self.run_all_analyses(filepath)
            
            # Check if all passed
            all_passed = all(r.passed for results in analyses.values() for r in results)
            
            # Print Summary
            print("\n  📊 Analysis Summary:")
            for cat, results in analyses.items():
                for result in results:
                    status = "✅" if result.passed else "❌"
                    print(f"    {status} {result.tool_name}")

            history.append({"iteration": iteration, "passed": all_passed})

            if all_passed:
                print("\n✅ SUCCESS! All analyses passed!")
                return final_code, True, history

            feedback = self.format_feedback(analyses)
            print(f"\n  ⚠️  Issues detected. Retrying...")

        print(f"\n❌ Max iterations ({self.max_iterations}) reached.")
        return final_code, False, history


def get_backend(provider: str, model: str) -> LLMBackend:
    """Factory to create the appropriate LLM backend."""
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key: raise ValueError("ANTHROPIC_API_KEY not set")
        return AnthropicBackend(key, model or "claude-3-5-sonnet-20240620")
    
    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key: raise ValueError("GEMINI_API_KEY not set")
        return GeminiBackend(key, model or "gemini-1.5-pro")
        
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key: raise ValueError("OPENAI_API_KEY not set")
        return OpenAIBackend(key, model or "gpt-4o")
        
    elif provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key: raise ValueError("DEEPSEEK_API_KEY not set")
        # DeepSeek is API-compatible with OpenAI
        return OpenAIBackend(
            key, 
            model or "deepseek-coder", 
            base_url="https://api.deepseek.com"
        )
        
    raise ValueError(f"Unknown provider: {provider}")

def main():
    parser = argparse.ArgumentParser(description="Go Code Synthesis Pipeline")
    parser.add_argument("task", nargs="?", help="The task description")
    parser.add_argument("-f", "--file", help="Read task from file")
    parser.add_argument("--provider", choices=["anthropic", "gemini", "openai", "deepseek"], default="anthropic", help="LLM Provider")
    parser.add_argument("--model", help="Specific model name (optional)")
    
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r') as f:
            task = f.read().strip()
    elif args.task:
        task = args.task
    else:
        parser.print_help()
        sys.exit(1)

    try:
        backend = get_backend(args.provider, args.model)
        pipeline = GoCodeSynthesisPipeline(backend, max_iterations=5)
        final_code, success, _ = pipeline.run(task)
        
        output_file = "generated_code.go"
        with open(output_file, "w") as f:
            f.write(final_code)
            
        print(f"\n💾 Final code saved to: {output_file}")
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()