#!/usr/bin/env python3
"""
Single-File Go Code Synthesis Pipeline
Generates Go code using LLM and iteratively improves it with static analysis.

Usage:
    python pipeline.py "your task description here"

Environment:
    ANTHROPIC_API_KEY - Required for LLM API access

Example:
    export ANTHROPIC_API_KEY='your-key'
    # python pipeline.py "Create a concurrent web scraper with rate limiting"
    python pipeline.py -f taskfilename
    
"""

import json
import os
import sys
import subprocess
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


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
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
        max_iterations: int = 5,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY required. Set via env var or constructor."
            )

        self.model = model
        self.max_iterations = max_iterations
        self.workspace = None

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
        }

        missing = []
        missing.extend(
            (
                f"  • {tool}: go install {install}"
                if "github" in install or "honnef" in install
                else f"  • {tool}: https://golang.org/doc/install"
            )
            for tool, install in tools.items()
            if not self._check_tool(tool)
        )
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
        """Remove temporary workspace."""
        if self.workspace and os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)

    def generate_code(
        self, prompt: str, feedback: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generate Go code using LLM."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        system = """You are an expert Go programmer. Generate clean, single-file Go code.
                Return ONLY the Go code with no markdown formatting or explanations."""

        if feedback:
            message = f"""Previous code had issues. Fix them.

                        TASK:
                        {prompt}

                        PREVIOUS CODE:
                        {self._last_code if hasattr(self, '_last_code') else '(no previous code)'}

                        ANALYSIS ERRORS:
                        {feedback}

                        Generate corrected Go code addressing all issues above."""
        else:
            message = f"""Generate a complete, single-file Go program for this task:

                        {prompt}

                        Include package main and a main() function. Make it production-ready."""

        response = client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": message}],
            system=system,
        )

        code = response.content[0].text.strip()

        # Strip markdown if present
        if "```go" in code:
            code = code.split("```go")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip(), message

    def write_to_workspace(self, filename: str, content: str) -> str:
        """Write code to workspace."""
        filepath = os.path.join(self.workspace, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def run_go_mod_tidy(self):
        print("\n  🔧 Running go mod tidy...")
        result = self.run_tool(["go", "mod", "tidy"], "go mod tidy")
        if not result.passed:
            print("  ⚠️ go mod tidy failed")
        return result.passed

    def run_tool(self, cmd: List[str], tool_name: str) -> AnalysisResult:
        """Execute analysis tool and capture results."""
        try:
            result = subprocess.run(
                cmd, cwd=self.workspace, capture_output=True, text=True, timeout=30
            )

            output = result.stdout + result.stderr
            passed = result.returncode == 0
            errors = (
                []
                if passed
                else [line.strip() for line in output.split("\n") if line.strip()]
            )

            return AnalysisResult(tool_name, passed, output, errors)
        except subprocess.TimeoutExpired:
            return AnalysisResult(
                tool_name, False, "Timeout", ["Tool timed out after 30s"]
            )
        except Exception as e:
            return AnalysisResult(tool_name, False, str(e), [f"Failed: {e}"])

    def analyze_concurrency(self, filepath: str) -> List[AnalysisResult]:
        """Run concurrency analysis tools."""
        # go build -race
        print("    → go build -race")
        results = [
            self.run_tool(
                ["go", "build", "-race", "-o", "/dev/null", filepath],
                "go build -race",
            )
        ]
        # go vet
        print("    → go vet")
        results.append(self.run_tool(["go", "vet", filepath], "go vet"))
        # gosec
        print("    → gosec")
        results.append(self.run_tool(["gosec", filepath], "gosec"))

        return results

    def analyze_memory(self, filepath: str) -> List[AnalysisResult]:
        """Run memory analysis (static checks only)."""
        # Note: gops and pkg/profile are runtime tools, not static analyzers
        # Using go vet for static memory-related checks
        print("    → go vet (memory)")
        return [self.run_tool(["go", "vet", filepath], "go vet (memory checks)")]

    def analyze_error_handling(self, filepath: str) -> List[AnalysisResult]:
        """Run error handling analysis."""
        # errcheck
        print("    → errcheck")
        results = [self.run_tool(["errcheck", filepath], "errcheck")]
        # go-errorlint
        print("    → go-errorlint")
        results.append(self.run_tool(["go-errorlint", filepath], "go-errorlint"))

        return results

    def analyze_performance(self, filepath: str) -> List[AnalysisResult]:
        """Run performance analysis."""
        # staticcheck
        print("    → staticcheck")
        return [self.run_tool(["staticcheck", filepath], "staticcheck")]

    def run_all_analyses(self, filepath: str) -> Dict[str, List[AnalysisResult]]:
        """Execute all analysis categories."""
        self.run_go_mod_tidy()
        print("\n  🔍 Running Analysis Tools...")

        return {
            "concurrency": self.analyze_concurrency(filepath),
            "memory": self.analyze_memory(filepath),
            "error_handling": self.analyze_error_handling(filepath),
            "performance": self.analyze_performance(filepath),
        }

    def format_feedback(
        self, analyses: Dict[str, List[AnalysisResult]]
    ) -> Optional[str]:
        """Format analysis results into LLM feedback."""
        feedback = []
        has_errors = False

        for category, results in analyses.items():
            category_errors = []
            for result in results:
                if not result.passed:
                    has_errors = True
                    category_errors.append(f"[{result.tool_name}]\n{result.output}")

            if category_errors:
                feedback.append(
                    f"\n{'='*60}\n"
                    f"{category.upper().replace('_', ' ')} ISSUES\n"
                    f"{'='*60}\n" + "\n\n".join(category_errors)
                )

        return "\n".join(feedback) if has_errors else None

    def print_summary(self, analyses: Dict[str, List[AnalysisResult]]):
        """Print analysis summary."""
        print("\n  📊 Analysis Summary:")
        for results in analyses.values():
            for result in results:
                status = "✅" if result.passed else "❌"
                print(f"    {status} {result.tool_name}")
                if not result.passed:
                    print(f"      {result.output}")
                    if result.errors:
                        print(f"      Errors: {result.errors}")

    def run(self, task: str) -> Tuple[str, bool, List[Dict]]:
        """
        Main pipeline execution.

        Returns:
            (final_code, success, iteration_history)
        """
        print("=" * 70)
        print("🚀 Go Code Synthesis Pipeline")
        print("=" * 70)
        print(f"\n📝 Task: {task}\n")

        # Check tools
        if not self.check_prerequisites():
            raise RuntimeError("Missing required tools")

        # Setup
        self.setup_workspace()
        print(f"📁 Workspace: {self.workspace}\n")

        history = []
        feedback = None
        final_code = None

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"🔄 ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*70}")

            # Generate
            print(f"\n  ✨ Generating Go code...")
            code, full_prompt = self.generate_code(task, feedback)
            self._last_code = code
            final_code = code

            # Write
            filepath = self.write_to_workspace("main.go", code)
            print("  ✅ Code written")

            # Log
            self.write_to_workspace(f"iter_{iteration}.go", code)
            self.write_to_workspace(f"iter_{iteration}.txt", full_prompt)

            # Analyze
            analyses = self.run_all_analyses(filepath)
            self.print_summary(analyses)

            # Check results
            all_passed = all(r.passed for results in analyses.values() for r in results)

            # Record
            history.append(
                {
                    "iteration": iteration,
                    "code": code,
                    "analyses": analyses,
                    "passed": all_passed,
                }
            )

            if all_passed:
                print("\n" + "=" * 70)
                print("✅ SUCCESS! All analyses passed!")
                print("=" * 70)
                return final_code, True, history

            # Prepare feedback
            feedback = self.format_feedback(analyses)
            print(f"\n  ⚠️  Issues detected, preparing feedback...")

        print("\n" + "=" * 70)
        print(
            f"❌ Max iterations ({self.max_iterations}) reached with issues remaining"
        )
        print("=" * 70)
        return final_code, False, history


def main():
    """CLI interface."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Task description required")
        print("\nUsage:")
        print(
            '  python pipeline.py "Create a concurrent HTTP server with rate limiting"'
        )
        sys.exit(1)
    with open("tasks/tasks.txt", "r") as f:
        tasks = f.readlines()

    # task = " ".join(sys.argv[1:])
    if sys.argv[1] == "-f":
        task_file = sys.argv[2]
        with open(task_file, "r", encoding="utf-8") as f:
            task = f.read().strip()
    else:
        task = " ".join(sys.argv[1:])

    try:
        _extracted_from_main_14(tasks[0])
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


# TODO Rename this here and in `main`
def _extracted_from_main_14(task):
    # Run pipeline
    key = json.load(open("config.json")).get("API_KEY")
    pipeline = GoCodeSynthesisPipeline(max_iterations=5, api_key=key)
    final_code, success, history = pipeline.run(task)

    # Save output
    output_file = "generated_code.go"
    with open(output_file, "w") as f:
        f.write(final_code)

    # Print results
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Status: {'✅ PASSED' if success else '⚠️  HAS ISSUES'}")
    print(f"Iterations: {len(history)}")
    print(f"Code saved: {output_file}")
    print(f"Workspace: {pipeline.workspace}")

    if success:
        print("\n🎉 Code is ready to use!")
        print(f"\nTry it:")
        print(f"  cd {pipeline.workspace}")
        print("  go run main.go")
    else:
        print("\n⚠️  Code generated but some issues remain.")
        print("Review the output above for details.")

    # Optional: cleanup workspace
    # pipeline.cleanup_workspace()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
