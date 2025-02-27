import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

#Load dataset
dataset = pd.read_csv('cancer.csv')

#Split features and target
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Define model architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Train model
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_test, y_test))

#Evaluation
model.evaluate(x_test, y_test)
