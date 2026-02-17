# generate_dataset.py
import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Number of rows
n = 1000

# Make sure the data folder exists
if not os.path.exists('data'):
    os.makedirs('data')

# Generate synthetic dataset
df = pd.DataFrame({
    'loan_amount': np.random.randint(1000, 50000, n),
    'term': np.random.choice([24, 36, 48, 60], n),
    'interest_rate': np.round(np.random.uniform(5, 20, n), 1),
    'income': np.random.randint(20000, 120000, n),
    'credit_score': np.random.randint(300, 850, n),
    'num_of_loans': np.random.randint(1, 6, n),
    'total_debt': np.random.randint(0, 50000, n),
    'total_credit_limit': np.random.randint(5000, 60000, n),
    'defaulted': np.random.choice([0, 1], n, p=[0.85, 0.15])
})

# Save to CSV inside the data folder
df.to_csv('data/loan_data.csv', index=False)

print("Synthetic dataset created with 1000 rows at data/loan_data.csv")
