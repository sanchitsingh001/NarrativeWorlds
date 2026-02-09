import json
import sys

try:
    with open('Narrative/narrative.json', 'r') as f:
        content = f.read()
        data = json.loads(content)
        print("JSON is VALID")
except json.JSONDecodeError as e:
    print(f"JSON is INVALID: {e}")
except Exception as e:
    print(f"Error: {e}")
