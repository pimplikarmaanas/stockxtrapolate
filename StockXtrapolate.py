# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import load_model

class StockXtra:
    scaler = MinMaxScaler(feature_range=(0,1))

    def __init__(self, train, test, load=False, save=True, loc=os.getcwd()):
        self.train = train
        self.test = test
        self.load = load
        self.save = save
        self.loc = loc

    def updateTrainingData(self):
        x_train, y_train = [], []
        training_data = pd.read_csv(self.train[0])
        training_set = training_data.iloc[:, self.train[1]:self.train[1]+1].values
        scaled_tSet = self.scaler.fit_transform(training_set)

        for i in range(60, len(training_set)):
            x_train.append(scaled_tSet[i-60:i, 0])
            y_train.append(scaled_tSet[i, 0])
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return x_train, y_train

    def createModel(self, numEpochs, numBatches, x_train, y_train):
        if self.load:
            model = load_model(self.loc)
        else:
            model = Sequential()
            
            #input layer
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            # Hidden layer 1
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            #Hidden layer 2
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            #Hidden Layer 3
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            #output layer
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=numEpochs, batch_size=numBatches)
            
        if self.save:
            model.save(self.loc)

        return model

    def testData(self, model):
        training_data = pd.read_csv(self.train[0])
        testing_data = pd.read_csv(self.test[0])
        real_prices = testing_data.iloc[:, self.test[1]:self.test[1]+1].values
        
        total_dataset = pd.concat((training_data['Open'], testing_data['Open']), axis=0)
        
        inputs = total_dataset[len(total_dataset) - len(testing_data) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)

        x_test = []
        for i in range(60, len(inputs)):
            x_test.append(inputs[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices = model.predict(x_test)
        predicted_prices = self.scaler.inverse_transform(predicted_prices)
        
        self.plot(real_prices, predicted_prices)
    
    def plot(self, real_prices, predicted_prices):
        plt.plot(real_prices, color = 'black', label = 'Real Stock Price')
        plt.plot(predicted_prices, color = 'green', label = 'Prediced Stock Price')
        plt.title('Stock Price prediction')
        plt.xlabel('Time (days)')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    train_location = ('AAPLTraining.csv', 3)
    test_location = ('AAPL.csv', 1)

    obj = StockXtra(train_location, test_location)
    x_train, y_train = obj.updateTrainingData()
    model = obj.createModel(100, 32, x_train, y_train)
    obj.testData(model)