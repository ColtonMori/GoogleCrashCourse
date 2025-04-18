import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv("california_housing_train.csv")

print(training_df.describe())