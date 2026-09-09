# argparse-schema

A library for bi-directional conversion between `argparse.ArgumentParser` and structured schemas (Python Dataclasses, YAML, and Dictionaries).

## Features

- **Argparse Introspection**: Extract argument metadata (type, default, help, nargs) from an `ArgumentParser`.
- **Bidirectional Conversion**:
    - `extract_default_args`: Parser -> Dict
    - `build_cmdline_args`: Dict -> List[str] (CLI args)
- **Code Generation**:
    - `generate_dataclass`: Generate a `compoconf`-compatible Python dataclass from parser metadata.
    - `generate_defaults_yaml`: Generate an annotated YAML configuration with defaults and help text.
    - `generate_cli_metadata_code`: Generate Python code for static metadata storage.
- **Megatron-LM Compatibility**: Special handling for list arguments (space or comma-separated) and optional union types for scalar/list defaults.

## Usage

### 1. Introspection and CLI Generation

```python
import argparse
from argparse_schema import (
    get_arg_metadata, 
    get_action_specs, 
    extract_default_args, 
    build_cmdline_args
)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--tags", nargs="+", default=["train"])

# Extract defaults
defaults = extract_default_args(parser)

# Generate CLI args from a modified dict
config = {"lr": 0.001, "tags": ["train", "experimental"]}
metadata = get_arg_metadata(parser)
specs = get_action_specs(parser)

# Standard space-separated tokens: ['--lr', '0.001', '--tags', 'train', 'experimental']
cli = build_cmdline_args(config, metadata, specs)

# Comma-joined token: ['--lr', '0.001', '--tags', 'train,experimental']
cli_comma = build_cmdline_args(config, metadata, specs, list_sep=",")

# Without metadata/specs, keys are turned into flags and values are emitted as strings.
# Booleans are treated as flags: True emits the flag, False/None emit nothing.
cli_fallback = build_cmdline_args({"dry_run": True, "epochs": 2}, None, None)
# ['--dry-run', '--epochs', '2']

# String values "True"/"False" are treated as normal strings and still emit a value token.
cli_bool_strings = build_cmdline_args({"flag": "True", "off": "False"}, None, None)
# ['--flag', 'True', '--off', 'False']
```

### 2. Schema Generation (Dataclass & YAML)

```python
from argparse_schema import generate_dataclass, generate_defaults_yaml

metadata = get_arg_metadata(parser)
defaults = extract_default_args(parser)

# Generate Python code
py_code = generate_dataclass(metadata, defaults, class_name="MyConfig")
with open("schema.py", "w") as f:
    f.write(py_code)

# Generate YAML defaults
yaml_text = generate_defaults_yaml(metadata, defaults)
with open("defaults.yaml", "w") as f:
    f.write(yaml_text)
```

## Testing

Run the tests using `pytest`:

```bash
PYTHONPATH=src pytest tests/
```

## License and Attribution

Copyright 2026 Korbinian Poeppel.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the [LICENSE](LICENSE)
file for the specific language governing permissions and limitations under the
License.

This library is derived from `oellm_autoexp/argparse_schema` in
[OpenEuroLLM/oellm-autoexp](https://github.com/OpenEuroLLM/oellm-autoexp),
Copyright 2026 OpenEuroLLM Consortium, also licensed under Apache 2.0. It is
maintained here as a standalone package and is periodically re-synced with
upstream.
