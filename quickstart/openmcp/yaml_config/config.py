import yaml
from pathlib import Path

yaml_path = Path(__file__).parent / "template.yaml"  # adjust path as needed
yaml_text = yaml_path.read_text(encoding="utf-8")   # read as plain text
mcp_yaml_config = yaml.safe_load(yaml_text)  # parse YAML to dict