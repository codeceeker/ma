import pandas as pd

# Sample data (replace this with your own dataset)
data = {
    'Variable1': [1, 2, 3, 4, 5],
    'Variable2': [5, 4, 3, 2, 1],
    'Variable3': [2, 3, 1, 5, 4],
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Print the correlation matrix
print("Correlation Matrix:-")
print(correlation_matrix)
