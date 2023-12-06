# Optimum-AMD documentation

1. Setup
```bash
pip install hf-doc-builder watchdog --upgrade
```

2. Local Development
```bash
doc-builder preview optimum.amd docs/source/
```
3. Build Docs
```bash
doc-builder build optimum.amd docs/source/ --build_dir build/ 
```
