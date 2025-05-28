# My Paper Title

This repository is the official implementation of "Dissecting Causal Mechanism Shifts via FANS: Function And Noise Separation using Flows"

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

run:

```experimenet
python -m experiments.experiment_script --model fans
python -m experiments.experiment_script --model iscan
python -m experiments.experiment_script --model linearccp
python -m experiments.gpr_experiment
python -m experiments.splitkci_experiment
```

## Analysis of Results

The results are saved at results directory.
```analysis
python analysis/analyze_fans_results.py
python analysis/analyze_iscan_results.py
python analysis/analyze_linearccp_results.py
python analysis/analyze_gpr_results.py
python analysis/analyze_splitkci_results.py
```

## Results

Our model achieves the following performance on :


>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 