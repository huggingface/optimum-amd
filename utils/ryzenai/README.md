# Utilities


## Generate NPU Baseline Operators

* Setup the Ryzen testing environment.

* Run all the following command to generate the Ryzen cache for the test models.

```bash
$env:RUN_SLOW=1; pytest -m "prequantized_model_test or quant_test" .\tests\ryzenai\
```

The tests will generate the `vitisai_ep_report.json` in `ryzen_cache` folder.


* Run the below script to generate baseline operators.

```bash
python .\utils\ryzenai\generate_operators_baseline.py .\ryzen_cache\ .\tests\ryzenai\operators_baseline.json
```