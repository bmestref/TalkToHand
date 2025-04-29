from tensorflow.keras import layers, models
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def HandTranslate(num_classes=3):
    model = models.Sequential()
    model.add(layers.Input(shape=(42,)))  # 21 keypoints * 2 coordinates
    model.add(layers.Reshape((21, 2)))    # (time steps, features)

    # Reduce filter size for speed
    model.add(layers.Conv1D(4, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(8, kernel_size=3, padding='same', activation='relu'))

    # Use global pooling instead of flatten (much faster)
    model.add(layers.GlobalMaxPooling1D())

    # Smaller dense layer
    model.add(layers.Dense(32, activation='relu'))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

model = HandTranslate(num_classes = 7)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()

# for i, file in enumerate(os.listdir(r'./data')):
#     if i == 0:
#         df = pd.read_csv(file)
#         df.drop(columns = ['Unnamed: 0'], inplace = True)
#     else:
#         new_df = pd.read_csv(file)
#         new_df.drop(columns = ['Unnamed: 0'], inplace = True)

#         df = pd.concat([df, new_df])

df = pd.read_csv('ABCDEFG_medium_hand.csv')
df.drop(columns = ['Unnamed: 0'], inplace = True)
df['Letter'] = df['Letter'].map({'a': 0, 'b': 1, 'c': 2, 'd':3, 'e':4, 'f':5, 'g':6})
x_train, x_test, y_train, y_test = train_test_split(df.drop(columns = ['Letter']), df['Letter'], 
                                                    shuffle = True,
                                                    stratify = df['Letter'], test_size = 0.2, random_state = 68)


model.fit(x_train, y_train, epochs = 20, validation_split = 0.2, batch_size = 32)

model.save('RightHandModel.keras')

# Test the accuracy of the model
y_pred = np.argmax(model.predict(x_test), axis=1)  # get class index with highest probability
print(y_pred)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()







