import json
from pathlib import Path
from jsonschema import Draft4Validator

schema_path = Path("output/schema.json")
if not schema_path.exists():
    print("Error: schema.json not found in output/")
    exit(1)

schema = json.load(schema_path.open())
validator = Draft4Validator(schema)

output_dir = Path("output")
for json_file in sorted(output_dir.glob("file*.json")):
    data = json.load(json_file.open())
    errors = list(validator.iter_errors(data))
    if errors:
        print(f"❌ {json_file.name} is INVALID")
        for e in errors:
            print("   ", e.message)
    else:
        print(f"✅ {json_file.name} is valid")