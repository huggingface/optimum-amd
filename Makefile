# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
# Run code quality checks
style_check:
	black --check .
	ruff .

style:
	black .
	ruff . --fix

build_doc_docker_image:
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
