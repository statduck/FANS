# Dissecting Causal Mechanism Shifts via FANS: Function And Noise Separation using Flows

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

![image](https://github.com/user-attachments/assets/e93b1074-62a4-44d9-98df-84e5ed1f48d0)

![image](https://github.com/user-attachments/assets/19cfe614-f3d4-4479-baa1-1030a79a7958)


## Contributing

If you'd like to contribute, or have any suggestions for the code, you can contact us at gd.seo@yonsei.ac.kr or open an issue on this GitHub repository.
