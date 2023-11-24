# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
# Run code quality checks
style_check:
	black --check .
	ruff .

style:
	black .
	ruff . --fix
