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

build-quark:
	docker build -t quark-mht docker/quantization-quark/

interact:
	docker run --rm -it   --entrypoint bash \
	--cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd \
	--device=/dev/dri --group-add video --ipc=host --shm-size 64g --net host \
	-v /home/amd/.cache/huggingface/hub:/data \
	-v $(PWD):/tgi \
	tgi-mht:2.5

interact-quark:
	docker run --rm -it   --entrypoint bash \
	--cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd \
	--device=/dev/dri --group-add video --ipc=host --shm-size 64g --net host \
	-v /home/amd/.cache/huggingface/hub:/data \
	-v $(PWD):/quark \
	-v $(PWD)/../transformers:/tr \
	quark-mht
