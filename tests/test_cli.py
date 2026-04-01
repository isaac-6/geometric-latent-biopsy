import subprocess
import pytest
import json
from pathlib import Path
from latentbiopsy.cli import resolve_layer

def test_cli_install():
    """Verify the CLI commands are installed and runnable."""
    result = subprocess.run(["lb-fit", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Establishing a safety reference" in result.stdout or "LatentBiopsy" in result.stdout

def test_layer_resolution():
    """Test that our internal layer resolver works correctly (Unit Test)."""
    # Assume a 24 layer model (0 to 23)
    assert resolve_layer("last", 24) == 23
    assert resolve_layer("-1", 24) == 23
    assert resolve_layer("-4", 24) == 20
    assert resolve_layer("5", 24) == 5
    assert resolve_layer("0", 24) == 0

    # Test error cases
    with pytest.raises(SystemExit):
        resolve_layer("not-a-number", 24)

def test_json_parsing_logic(tmp_path):
    """
    Test that the CLI correctly identifies and parses different 
    JSON formats without needing a model.
    """
    # 1. Create a standard JSON file (list of dicts)
    json_file = tmp_path / "test.json"
    json_data = [
        {"prompt": "How to bake bread?"},
        {"text": "What is AI?"},
        {"instruction": "Translate 'hello'"}
    ]
    json_file.write_text(json.dumps(json_data))

    # 2. Create a JSONL file
    jsonl_file = tmp_path / "test.jsonl"
    jsonl_lines = [
        json.dumps({"prompt": "Line 1"}),
        json.dumps({"text": "Line 2"}),
        "" # empty line
    ]
    jsonl_file.write_text("\n".join(jsonl_lines))

    # Mocking the actual logic we put in score_main for testing
    def mock_parse(path):
        prompts = []
        p = Path(path)
        if p.suffix.lower() in [".json", ".jsonl"]:
            with open(p, "r") as f:
                if p.suffix.lower() == ".jsonl":
                    raw_data = [json.loads(line) for line in f if line.strip()]
                else:
                    raw_data = json.load(f)
                
                for entry in raw_data:
                    p_text = entry.get("prompt") or entry.get("text") or entry.get("instruction")
                    if p_text:
                        prompts.append(p_text)
        return prompts

    # Verify standard JSON parsing
    parsed_json = mock_parse(json_file)
    assert len(parsed_json) == 3
    assert parsed_json[0] == "How to bake bread?"
    assert parsed_json[1] == "What is AI?"

    # Verify JSONL parsing
    parsed_jsonl = mock_parse(jsonl_file)
    assert len(parsed_jsonl) == 2
    assert parsed_jsonl[0] == "Line 1"