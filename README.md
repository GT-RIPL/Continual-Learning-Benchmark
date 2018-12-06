# Continual-Learning-Benchmark
Evaluate and compare continual learning algorithms with three learning scenarios.

Reference:
```
@article{Hsu18_EvalCL,
  title={Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines},
  author={Yen-Chang Hsu and Yen-Cheng Liu and Zsolt Kira},
  booktitle={NeurIPS Continual learning Workshop },
  year={2018},
  url={https://arxiv.org/abs/1810.12488}
}
```

## Preparation
```bash
pip install -r requirements.txt
```

## Demo
The scripts for reproducing the results of this paper are under the scripts folder.

- Example: Run all algorithms in the incremental domain scenario with split MNIST.
```bash
./scripts/split_MNIST_incremental_domain.sh 0
# The last number is gpuid
# Outputs will be saved in ./outputs
```

- Eaxmple outputs: Summary of repeats
```text
===Summary of experiment repeats: 3 / 3 ===
The regularization coefficient: 400.0
The last avg acc of all repeats: [90.517 90.648 91.069]
mean: 90.74466666666666 std: 0.23549144829955856
```

- Eaxmple outputs: The grid search for regularization coefficient
```text
reg_coef: 0.1 mean: 76.08566666666667 std: 1.097717733400629
reg_coef: 1.0 mean: 77.59100000000001 std: 2.100847606721314
reg_coef: 10.0 mean: 84.33933333333334 std: 0.3592671553160509
reg_coef: 100.0 mean: 90.83800000000001 std: 0.6913701372395712
reg_coef: 1000.0 mean: 87.48566666666666 std: 0.5440161353816179
reg_coef: 5000.0 mean: 68.99133333333333 std: 1.6824762174313899

```

## Usage
- Enable the grid search for the regularization coefficient: Use the option with a list of values, ex: -reg_coef 0.1 1 10 100 ...
- Repeat the experiment N times: Use the option -repeat N

Lookup available options:
```bash
python iBatchLearn.py -h
```

## More
The supports of more continual learning algorithms are comming soon.