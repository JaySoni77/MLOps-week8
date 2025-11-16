# MLflow Assignment – Data Poisoning on Iris Dataset

This project demonstrates data poisoning on the Iris dataset and analyzes its impact on model performance using MLflow. The assignment includes generating poisoned datasets at multiple levels, training a Decision Tree model, and comparing validation outcomes across poisoning intensities.

## Objectives

- Apply label poisoning at 5%, 10%, and 50% levels using random label flipping.
- Train a Decision Tree classifier on both clean and poisoned datasets.
- Log all training runs to MLflow, including metrics, parameters, tags, and artifacts.
- Evaluate how poisoning affects validation accuracy.
- Provide thoughts on mitigation strategies and how data quantity requirements change when quality drops.

## Poisoning Script

`poisoning.py` accepts:

- `--data_path` for source dataset
- `--percent` for poisoning level

It randomly corrupts label values and outputs a new poisoned CSV. This simulates a real-world data poisoning attack.

## Training Script

`train.py` accepts:

- `--data_path` for clean or poisoned dataset

It trains a Decision Tree model and logs:

- Accuracy score
- Parameters
- Model artifact
- Tags identifying poisoning level
- Named MLflow runs for easy comparison

## Validation Outcomes

Increasing poisoning significantly reduces accuracy:

- **5%** → slight degradation
- **10%** → moderate drop
- **50%** → severe collapse

MLflow visualizes this progression clearly.

## Mitigation & Data Quantity Insights

### Mitigation Strategies

To mitigate poisoning attacks:

- Apply data validation, anomaly checks, and drift detection
- Use more robust algorithms
- Track data provenance and secure pipelines
- Maintain human-in-the-loop verification

### Data Quantity Requirements

As data quality drops, quantity requirements rise:

- **Mild noise** needs ~20% more data
- **Moderate noise** needs ~40% more data
- **Heavy noise** may require 3–5× more clean samples to stabilize model performance
