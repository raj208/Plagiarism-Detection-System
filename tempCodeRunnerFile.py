import pandas as pd

# Load the dataset
df = pd.read_csv(r'archive\train_snli.txt', sep='\t', header=None)


# Name the columns
df.columns = ['source_text', 'student_text', 'label']

# Check the data
# print(df.head())



# Check for nulls or issues
print(df.isnull().sum())

# Remove any rows with missing values (just in case)
df.dropna(inplace=True)

# Convert labels to integers (if they’re not already)
df['label'] = df['label'].astype(int)