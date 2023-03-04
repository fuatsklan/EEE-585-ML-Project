import numpy as np

def arange_data(df, n):  # df = data, n = sample size.

    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data frame.
    Y = df['Fire Alarm'].to_numpy()[0:n]  # Split the labels.
    # Drop the unwanted columns:
    df.drop(['CNT'], inplace=True, axis=1)
    df.drop(['Unnamed: 0'], inplace=True, axis=1)
    df.drop(['Fire Alarm'], inplace=True, axis=1)

    df.drop(['UTC'],axis=1,inplace=True)
    # Convert the data to array and apply min-max scaling.
    design = df.to_numpy()
    maximum_values = []
    minimum_values = []
    for i in range(len(df.columns.values.tolist())):
        minimum = np.min(design[:, i])
        maximum = np.max(design[:, i])
        maximum_values.append(maximum)
        minimum_values.append(minimum)
    for i in range(len(df.columns.values.tolist())):
        design[:, i] = (design[:, i] - minimum_values[i]) / (maximum_values[i] - minimum_values[i])

    design_1 = np.c_[np.ones((np.shape(design)[0], 1)), design]  # Add the DC terms.

    X = design_1[0:n].reshape((n, 13))  # Declare the design matrix.

    Beta = np.random.normal(0, 1, size=(13, 1)).reshape((13, 1))  # Initialize the parameters for GD.

    # Beta = np.zeros((14,1))

    return X, Y, Beta  # Return X set, Y set, and parameters.