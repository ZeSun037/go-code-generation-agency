import json
import os
from pathlib import Path


def split_jsonl_to_json_files(jsonl_path, output_folder, prefix="task"):
    """
    Read a JSONL file and create separate JSON files for each line.
    
    Args:
        jsonl_path (str): Path to the input JSONL file
        output_folder (str): Folder where JSON files will be saved
        prefix (str): Prefix for output filenames (default: "item")
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Read JSONL file and create individual JSON files
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            try:
                # Parse JSON from line
                data = json.loads(line.strip())
                
                # Create filename
                filename = f"{prefix}_{idx}.json"
                filepath = os.path.join(output_folder, filename)
                
                # Write to individual JSON file
                with open(filepath, 'w', encoding='utf-8') as outfile:
                    json.dump(data, outfile, indent=2, ensure_ascii=False)
                
                print(f"Created: {filepath}")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {idx}: {e}")
                continue
    
    print(f"\nCompleted! Created {idx} JSON files in '{output_folder}'")


# Example usage
if __name__ == "__main__":
    # Specify your input file and output folder
    input_jsonl = "/home/fanbao/291P/humaneval-x/data/go/data/humaneval.jsonl"  # Change to your JSONL file path
    output_dir = "/home/fanbao/291P/go-code-generation-agency/benchmark/leetcode"    # Change to your desired output folder
    
    split_jsonl_to_json_files(input_jsonl, output_dir)