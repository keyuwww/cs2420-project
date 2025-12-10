import json

input_path = "claude_eval_enhanced_merged.jsonl"      # your source file
output_path = "null_evals.jsonl"  # your destination file

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        line = line.strip()
        if not line:
            continue
        
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip malformed lines
        
        # Check for evaluation == null
        if obj.get("evaluation") is None:
            outfile.write(json.dumps(obj) + "\n")

print("Done. Saved entries with evaluation=null to:", output_path)
