#!/usr/bin/env python3
"""
Go Code Generator with Gobra Verification
Generates Go code from requirements using an LLM and verifies it with Gobra.
Automatically retries with feedback if verification fails.
"""

import os
import subprocess
import tempfile
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# For LLM API - using Anthropic as example, can be swapped for OpenAI
try:
    from anthropic import Anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    
# Alternative: OpenAI
# try:
#     from openai import OpenAI
# except ImportError:
#     print("Please install openai: pip install openai")


class RequirementType(Enum):
    FUNCTIONAL = "functional"
    CONCURRENCY = "concurrency"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class Requirements:
    """Container for different types of requirements"""
    functional: List[str]
    concurrency: List[str]
    security: List[str]
    performance: List[str]
    
    def to_prompt_section(self) -> str:
        """Convert requirements to a formatted prompt section"""
        sections = []
        
        if self.functional:
            sections.append("Functional Requirements:\n" + 
                          "\n".join(f"- {req}" for req in self.functional))
        
        if self.concurrency:
            sections.append("Concurrency Requirements:\n" + 
                          "\n".join(f"- {req}" for req in self.concurrency))
        
        if self.security:
            sections.append("Security Requirements:\n" + 
                          "\n".join(f"- {req}" for req in self.security))
        
        if self.performance:
            sections.append("Performance Requirements:\n" + 
                          "\n".join(f"- {req}" for req in self.performance))
        
        return "\n\n".join(sections)


class GoCodeGenerator:
    """Main class for generating and verifying Go code"""
    
    def __init__(self, api_key: str, max_retries: int = 3):
        """
        Initialize the generator
        
        Args:
            api_key: API key for the LLM service
            max_retries: Maximum number of verification retry attempts
        """
        self.api_key = api_key
        self.max_retries = max_retries
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)
        
        # For OpenAI:
        # self.client = OpenAI(api_key=api_key)
    
    def create_prompt(self, requirements: Requirements, 
                     verification_feedback: Optional[str] = None) -> str:
        """
        Create a prompt for the LLM to generate Go code
        
        Args:
            requirements: The requirements for the code
            verification_feedback: Optional feedback from previous verification failure
        
        Returns:
            Formatted prompt string
        """
        prompt = """Generate a Go program that meets the following requirements.
The code should include Gobra verification annotations (//@ annotations) for formal verification.

{}

Important Guidelines:
1. Include proper Gobra annotations for:
   - Function preconditions (//@ requires)
   - Function postconditions (//@ ensures)
   - Loop invariants (//@ invariant)
   - Assertions (//@ assert)
   - Ghost variables if needed
   
2. For concurrent code, include:
   - Proper synchronization annotations
   - Lock invariants
   - Channel contracts
   
3. Ensure the code is verifiable and follows Go best practices.

Output only the Go code with annotations, no explanations.
""".format(requirements.to_prompt_section())
        
        if verification_feedback:
            prompt += f"\n\nPrevious Verification Feedback:\n{verification_feedback}\n"
            prompt += "Please fix the issues mentioned above and regenerate the code.\n"
        
        return prompt
    
    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM API to generate Go code
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            Generated Go code
        """
        try:
            # Anthropic API call
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
            
            # OpenAI API call (alternative)
            # response = self.client.chat.completions.create(
            #     model="gpt-4",
            #     messages=[
            #         {"role": "system", "content": "You are an expert Go programmer with knowledge of Gobra verification."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=0.2,
            #     max_tokens=4000
            # )
            # return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            raise
    
    def extract_go_code(self, response: str) -> str:
        """
        Extract Go code from LLM response
        
        Args:
            response: Raw response from LLM
        
        Returns:
            Extracted Go code
        """
        # Try to extract code between ```go and ``` markers
        if "```go" in response:
            start = response.find("```go") + 5
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # Try to extract code between ``` markers
        if "```" in response:
            start = response.find("```") + 3
            # Skip language identifier if present
            if response[start:start+10].strip().startswith(('go', 'golang')):
                start = response.find("\n", start) + 1
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # Return as-is if no code blocks found
        return response.strip()
    
    def save_to_file(self, code: str, filepath: str) -> None:
        """Save Go code to a file"""
        with open(filepath, 'w') as f:
            f.write(code)
    
    def run_gobra(self, filepath: str) -> Tuple[bool, str]:
        """
        Run Gobra verification on the Go file
        
        Args:
            filepath: Path to the Go file to verify
        
        Returns:
            Tuple of (success: bool, output: str)
        """
        try:
            # Run Gobra command
            # Note: Assumes Gobra is installed and available in PATH
            result = subprocess.run(
                ["gobra", "--recursive", filepath],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            # Check if verification succeeded
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, "Gobra verification timed out after 60 seconds"
        except FileNotFoundError:
            return False, "Gobra not found. Please ensure Gobra is installed and in PATH"
        except Exception as e:
            return False, f"Error running Gobra: {str(e)}"
    
    def parse_verification_errors(self, gobra_output: str) -> str:
        """
        Parse Gobra output to extract meaningful error messages
        
        Args:
            gobra_output: Raw output from Gobra
        
        Returns:
            Parsed error feedback
        """
        lines = gobra_output.split('\n')
        errors = []
        
        for line in lines:
            # Look for error patterns in Gobra output
            if 'error' in line.lower() or 'failed' in line.lower():
                errors.append(line.strip())
            elif 'assertion' in line.lower() and 'might not hold' in line:
                errors.append(line.strip())
            elif 'precondition' in line.lower() or 'postcondition' in line.lower():
                errors.append(line.strip())
        
        if errors:
            return "Verification errors:\n" + "\n".join(errors[:10])  # Limit to 10 errors
        else:
            return "Verification failed. Full output:\n" + gobra_output[:1000]  # First 1000 chars
    
    def generate_and_verify(self, requirements: Requirements) -> Dict:
        """
        Main method to generate Go code and verify it with Gobra
        
        Args:
            requirements: The requirements for the code generation
        
        Returns:
            Dictionary with results including code, verification status, and attempts
        """
        results = {
            "success": False,
            "code": None,
            "verification_output": None,
            "attempts": 0,
            "history": []
        }
        
        verification_feedback = None
        
        for attempt in range(1, self.max_retries + 1):
            print(f"\n=== Attempt {attempt}/{self.max_retries} ===")
            results["attempts"] = attempt
            
            # Create prompt (with feedback if this is a retry)
            prompt = self.create_prompt(requirements, verification_feedback)
            
            # Generate code
            print("Generating Go code...")
            try:
                llm_response = self.call_llm(prompt)
                go_code = self.extract_go_code(llm_response)
            except Exception as e:
                print(f"Failed to generate code: {e}")
                results["history"].append({
                    "attempt": attempt,
                    "error": str(e),
                    "code": None
                })
                continue
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
                f.write(go_code)
                temp_filepath = f.name
            
            print(f"Code saved to: {temp_filepath}")
            
            # Run Gobra verification
            print("Running Gobra verification...")
            success, output = self.run_gobra(temp_filepath)
            
            # Store attempt history
            results["history"].append({
                "attempt": attempt,
                "code": go_code,
                "verification_success": success,
                "verification_output": output[:500]  # First 500 chars
            })
            
            if success:
                print("✓ Verification successful!")
                results["success"] = True
                results["code"] = go_code
                results["verification_output"] = output
                
                # Clean up temp file
                os.unlink(temp_filepath)
                break
            else:
                print("✗ Verification failed")
                verification_feedback = self.parse_verification_errors(output)
                print(f"Feedback: {verification_feedback[:200]}...")  # Show first 200 chars
                
                # Clean up temp file
                os.unlink(temp_filepath)
                
                if attempt < self.max_retries:
                    print("Retrying with feedback...")
                    time.sleep(1)  # Small delay between retries
        
        if not results["success"]:
            print(f"\nFailed to generate verifiable code after {self.max_retries} attempts")
        
        return results


def main():
    """Main function to demonstrate the generator"""
    
    # Example usage
    print("Go Code Generator with Gobra Verification")
    print("=" * 50)
    
    # Get API key from environment or prompt
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Enter your Anthropic API key: ").strip()
    
    # Define requirements
    requirements = Requirements(
        functional=[
            "Implement a thread-safe counter with increment and get operations",
            "The counter should start at 0",
            "Increment should add 1 to the counter",
            "Get should return the current value"
        ],
        concurrency=[
            "Must be safe for concurrent access by multiple goroutines",
            "Use mutex for synchronization",
            "No data races allowed"
        ],
        security=[
            "Counter value should not overflow",
            "Only authorized operations (increment, get) should be allowed"
        ],
        performance=[
            "Minimize lock contention",
            "Operations should be O(1)"
        ]
    )
    
    # Create generator and run
    generator = GoCodeGenerator(api_key=api_key, max_retries=3)
    
    print("\nRequirements:")
    print(requirements.to_prompt_section())
    print("\nStarting generation and verification process...")
    
    results = generator.generate_and_verify(requirements)
    
    # Display results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Success: {results['success']}")
    print(f"Total attempts: {results['attempts']}")
    
    if results["success"]:
        print("\nGenerated and verified Go code:")
        print("-" * 50)
        print(results["code"])
        print("-" * 50)
        
        # Save to file
        output_file = "verified_go_code.go"
        with open(output_file, 'w') as f:
            f.write(results["code"])
        print(f"\nCode saved to: {output_file}")
    else:
        print("\nFailed to generate verifiable code")
        print("Last attempt code:")
        if results["history"]:
            print(results["history"][-1]["code"])
    
    # Save full results to JSON
    results_file = "generation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {results_file}")


if __name__ == "__main__":
    main()