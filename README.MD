MLOps Assignment — Data Poisoning on Iris Dataset (MLflow Integration)

This project demonstrates data poisoning, model training, and MLflow experiment tracking using the Iris dataset.
The objective is to analyze how model performance degrades as label noise increases and to explore mitigation strategies.

🚀 1. Data Poisoning Script (poisoning.py)

This script simulates label poisoning attacks on the Iris dataset.

Usage
python poisoning.py --data_path data/iris.csv --percent 10


This will:

Randomly select 10% of samples

Randomly flip their labels

Save output as data/iris_poisoned_10.csv

Poisoning levels tested: 5%, 10%, 50%

🤖 2. Model Training Script (train.py)

This script trains a Decision Tree Classifier on any dataset passed as input and logs results to MLflow.

Usage
python train.py --data_path data/iris_poisoned_10.csv


MLflow logs include:

Accuracy score

Model parameters

Dataset used

Poison percentage

Run name (e.g., poisoned_10_percent)

Model artifact + signature

📊 3. Observations (Validation Outcomes)

Model accuracy decreases as poisoning increases:

0% (clean) → highest accuracy (~95–97%)

5% → small drop

10% → moderate drop

50% → severe degradation (model becomes unreliable)

All comparisons visible in MLflow UI.

🛡️ 4. Mitigation Strategies

Data validation & drift detection

Outlier and anomaly detection

Robust algorithms (e.g., ensemble methods)

Human-in-the-loop label review

Dataset versioning & provenance tracking

Continuous model monitoring in production

📈 5. Data Quantity vs Data Quality

When data gets noisier, the amount of clean data needed increases:

5% noise → need ~20% more data

10% noise → need ~30–40% more data

50% noise → may require 3–5× more clean samples

This ensures the model has enough signal to overcome the noise.

✅ Conclusion

This project successfully demonstrates:

Data poisoning at multiple levels

Retraining & logging with MLflow

Performance comparison

Practical insights into robustness and data quality requirements