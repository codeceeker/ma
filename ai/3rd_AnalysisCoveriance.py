import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = iris.data
feature_names = iris.feature_names
target = iris.target

# Create a DataFrame from the dataset
df = pd.DataFrame(data, columns=feature_names)
df['species'] = pd.Categorical.from_codes(target, iris.target_names)

# Add a hypothetical continuous covariate with random values
np.random.seed(42)  # Set a seed for reproducibility
covariate_values = np.random.rand(150) * 10  # Random values scaled by 10
df['covariate'] = covariate_values

# Replace spaces and parentheses in column names
df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Fit ANCOVA model
formula = 'covariate ~ species + ' + ' + '.join(df.columns[:-2])  # Use all columns except 'species' and 'covariate'
model = sm.OLS.from_formula(formula, data=df)
result = model.fit()

# Print ANCOVA summary
print(result.summary())
