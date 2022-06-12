# StockXtrapolate

StockXtra object can take 5 parameters:
1. train - tuple containing the location of the training file and the column for the opening prices (0-indexed). (required)
2. test - tuple containing the location of the testing file and the column for the opening prices (0-indexed). (required)
3. load - True if the model is being loaded from a location "loc", default=False.
4. save - True if the model needs to be saved in a location "loc", default=True.
5. loc - the location for either load or save, or both, default=location of the py file.

The locations for the training and testing files need to be in a tuple.


# Explanations for each function in the class

1. __init__: initializes the object with the given parameters.
2. updateTrainingData: creates and returnes the training numpy arrays
3. createModel: Takes in the training arrays returned in "updateTrainingData". If the class object is initialized with load=True, loads that model. Else, creates a Sequential object with the given parameters. If the class is initialized with save=True, saves the model. It then returns the model created/loaded.
4. testData: Uses the testing file that the class was initializd with and the model returned in the "createModel" function to test the model. Calls the "plot" method to plot the original prices from the test file and the predicted prices that were obtained from using the model.
5. plot: Explained above.
