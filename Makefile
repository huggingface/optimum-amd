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
			optimum-amd-zentorch-mht:5.0.0-rc6

models = \
    "google/gemma-2-9b-it" \
    "EleutherAI/gpt-j-6B" \
    "meta-llama/Llama-2-7b-chat-hf" \
    "meta-llama/Llama-2-13b-chat-hf" \
    "meta-llama/Meta-Llama-3-8B-Instruct" \
    "mistralai/Mistral-7B-Instruct-v0.3" \
    "Qwen/Qwen2-7B-Instruct" \
    "Qwen/Qwen1.5-14B-Chat"

models = "meta-llama/Meta-Llama-3.1-8B-Instruct"

CACHE_IMPLEMENTATION := static
REPO_ID := "optimum-amd/zendnn-benchmark"
VERSION := 5_rc7

BACKEND := zentorch
DTYPE := bfloat16
TASK := "text-generation"

BATCH_SIZES := 32
SEQUENCE_LENGTHS := 1024
DECODE_LENGTHS := 1024

CORE_COUNT := $(shell nproc)
SOCKET_COUNT := $(shell lscpu | grep 'Socket(s):' | awk '{print $$2}')
THREADS_PER_CORE := $(shell lscpu | grep 'Thread(s) per core:' | awk '{print $$4}')

NUMA_THRESHOLD := $(shell expr $(CORE_COUNT) / $(SOCKET_COUNT) / $(THREADS_PER_CORE))
CORE_COUNT := $(shell expr $(CORE_COUNT) / $(THREADS_PER_CORE))

benchmark-run-inner:
	@echo "Running benchmark with N_INSTANCES=$(N_INSTANCES), BATCH_SIZE=$(BATCH_SIZE), SEQUENCE_LENGTH=$(SEQUENCE_LENGTH), DECODE_LENGTH=$(DECODE_LENGTH)"
	@cores_per_instance=$$(($(CORE_COUNT) / $(N_INSTANCES))); \
	for model in $(models); do \
		for i in $$(seq 0 $$(($(N_INSTANCES) - 1))); do \
			start_core=$$((i * $$cores_per_instance)); \
			end_core=$$((start_core + $$cores_per_instance - 1)); \
			if [ $(N_INSTANCES) -eq 2 ] && [ $$i -eq 1 ] && [ "$(DEVICE)" = "turin" ]; then \
				numa_node=1; \
			elif [ $$start_core -lt $(NUMA_THRESHOLD) ] || [ $$start_core -ge 256 -a $$start_core -lt 384 ]; then \
				numa_node=0; \
			else \
				numa_node=1; \
			fi; \
			echo "Starting core $$start_core to core $$end_core on NUMA node $$numa_node with model $$model"; \
			python examples/benchmarks/epyc/benchmark_model.py \
				--physcpubind $$start_core-$$end_core \
				--membind $$numa_node \
				--model_id $$model \
				--batch_size $(BATCH_SIZE) \
				--sequence_length $(SEQUENCE_LENGTH) \
				--decode_length $(DECODE_LENGTH) \
				--backend $(BACKEND) \
				--dtype $(DTYPE) \
				--task $(TASK) \
				--device $(DEVICE) \
				--num_instances $(N_INSTANCES) \
				--cache_implementation $(CACHE_IMPLEMENTATION) \
				--repo_id $(REPO_ID) \
				--version $(VERSION) \
				--instance $$i & \
		done; \
		wait; \
	done

benchmark-run:
	$(MAKE) benchmark-run-inner N_INSTANCES=$(N_INSTANCES) BATCH_SIZE=$(BATCH_SIZE) SEQUENCE_LENGTH=$(SEQUENCE_LENGTH) DECODE_LENGTH=$(DECODE_LENGTH); \

run-benchmark:
	@echo "Running benchmark on device: $(DEVICE)"
	@echo "NUMA threshold: $(NUMA_THRESHOLD)"
	@for ninstances in $(N_INSTANCES); do \
		for batch_size in $(BATCH_SIZES); do \
			for seq_length in $(SEQUENCE_LENGTHS); do \
				for decode_length in $(DECODE_LENGTHS); do \
					echo "Running benchmark with N_INSTANCES=$$ninstances, BATCH_SIZE=$$batch_size, SEQUENCE_LENGTH=$$seq_length, DECODE_LENGTH=$$decode_length"; \
					$(MAKE) benchmark-run N_INSTANCES=$$ninstances BATCH_SIZE=$$batch_size SEQUENCE_LENGTH=$$seq_length DECODE_LENGTH=$$decode_length; \
				done; \
			done; \
		done; \
	done

benchmark-turin:
	$(MAKE) run-benchmark DEVICE=turin N_INSTANCES="2 4 8 16"

benchmark-genoa:
	$(MAKE) run-benchmark DEVICE=genoa N_INSTANCES="2 6 12"
