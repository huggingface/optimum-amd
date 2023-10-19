# Install optimum-benchmark from source
pip install optimum-benchmark[peft,diffusers]@git+https://github.com/huggingface/optimum-benchmark.git

# Install transformers from source
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ..

# Clean up the previous results
rm -rf experiments

# Run the benchmarks
for file in benchmarks/configs/*.yaml; do
    config=$(basename $file .yaml)

    if [ "$config" = "base_config" ]; then
        continue
    fi

    echo "Running benchmark for $config"
    optimum-benchmark --config-dir benchmarks/configs --config-name $config --multirun
done

# Publish the results
python benchmarks/publish.py
