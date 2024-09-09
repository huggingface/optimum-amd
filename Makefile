# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
# Run code quality checks
SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)
DEFAULT_CLONE_URL := https://github.com/huggingface/optimum-amd.git
# If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))

style_check:
	black --check .
	ruff .

style:
	black .
	ruff . --fix

build_doc_docker_image:
	echo "REAL_CLONE_URL: $(REAL_CLONE_URL)"
	echo "COMMIT_SHA_SUBPACKAGE: $(COMMIT_SHA_SUBPACKAGE)"
	docker build -t doc_maker --build-arg commit_sha=$(COMMIT_SHA_SUBPACKAGE) --build-arg clone_url=$(REAL_CLONE_URL) ./docs

doc: build_doc_docker_image
	@test -n "$(BUILD_DIR)" || (echo "BUILD_DIR is empty." ; exit 1)
	@test -n "$(VERSION)" || (echo "VERSION is empty." ; exit 1)
	docker run -v $(CURRENT_DIR):/doc_folder --workdir=/doc_folder doc_maker \
	doc-builder build optimum.amd /optimum-amd/docs/source/ \
		--repo_name optimum-amd \
		--build_dir $(BUILD_DIR) \
		--version $(VERSION) \
		--version_tag_suffix "" \
		--html \
		--clean

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf optimum_amd.egg-info/

interact:
	docker run -it --rm \
			--shm-size 64G \
			--net=host \
			--cap-add=sys_nice \
			--volume $(CURRENT_DIR):/workspace \
			--volume /home/mohit/.cache/huggingface/hub:/data/hf_cache/ \
			--workdir /workspace \
			--entrypoint /bin/bash \
			optimum-amd-zentorch-mht:4.2.0

models = \
    "google/gemma-2-9b-it" \
    "EleutherAI/gpt-j-6B" \
    "meta-llama/Llama-2-7b-chat-hf" \
    "meta-llama/Llama-2-13b-chat-hf" \
    "meta-llama/Meta-Llama-3-8B-Instruct" \
    "mistralai/Mistral-7B-Instruct-v0.3" \
    "Qwen/Qwen2-7B-Instruct" \
    "Qwen/Qwen1.5-14B-Chat"

benchmark:
	for model in $(models); do \
		for i in {0..23}; do \
			start_core=$$((i * 8)); \
			end_core=$$((start_core + 7)); \
			if [ $$start_core -lt 96 ]; then \
				numa_node=0; \
			else \
				start_core=$$((start_core + 32)); \
				end_core=$$((end_core + 32)); \
				numa_node=1; \
			fi; \
			echo "Starting core $$start_core to core $$end_core on NUMA node $$numa_node with model $$model"; \
			python examples/benchmarks/epyc/benchmark_model.py --physcpubind $$start_core-$$end_core --membind $$numa_node --model_id $$model & \
		done; \
		wait; \
	done


benchmark2:
	for model in $(models); do \
			for i in {0..63}; do \
					start_core=$$((i * 8)); \
					end_core=$$((start_core + 7)); \
					if [ $$start_core -lt 128 ] || [ $$start_core -ge 256 -a $$start_core -lt 384 ]; then \
							numa_node=0; \
					else \
							numa_node=1; \
					fi; \
					echo "Starting core $$start_core to core $$end_core on NUMA node $$numa_node with model $$model"; \
					python examples/benchmarks/epyc/benchmark_model.py --physcpubind $$start_core-$$end_core --membind $$numa_node --model_id $$model & \
			done; \
			wait; \
	done


