# An Agent-Based Financing Model for Post-Earthquake Housing Recovery

This repository contains the code developed for the paper: An Agent-Based Financing Model for Post-Earthquake Housing Recovery: Quantifying Inequitable Recovery Across Income Groups. If you find this code useful in your research, please consider citing.

![FinancingModel.png](FinancingModel.png)

## Running Code

1. Change parameters in `configs/basecase.py`
    - A sample input file is available in `data/building_damage_owner_pkl`
2. Open `sanjose_model.py` and change line 12 according to the name from step 1. Example: if the name is: `basecase.py`, then line 12 becomes: `from configs.basecase import *`
3. Run the code: `python sanjose_model.py`
4. Results will be saved according to the name defined in `configs/basecase.py`

## Visualizing Results

- `visualization_utils.py`: Functions for visualizing results
- `Visualize Results.ipynb`: Jupyter Notebook for visualizing
