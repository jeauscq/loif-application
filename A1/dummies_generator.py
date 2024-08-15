import pandas as pd
import numpy as np

"""
This code creates a vast amount of num_samples with random generated numbers
that are bounded by the limitations of the original problem to solve.
"""


# Define the number of samples
num_samples = 1000  # You can adjust the number of samples as needed

# Generate random samples for each column within the specified ranges
LLL = np.random.uniform(10, 30, num_samples)
WWW = np.random.uniform(1, 5, num_samples)
TTT = np.random.uniform(0.1, 1, num_samples)
alpha = np.random.uniform(0, 15, num_samples)

# Create a DataFrame with the generated data
data = {
    'LLL': LLL,
    'WWW': WWW,
    'TTT': TTT,
    'alpha': alpha
}
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_filename = 'random_dataset.xlsx'
df.to_excel(output_filename, index=False)
print(f"Random dataset saved to {output_filename}")
