import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

categories_input = [
    'A1 - Огромный',
    'A2 - Средний',
    'A3 - Маленький',
    'B1 - Деловое',
    'B2 - Вечернее',
    'B3 - Высокая мода',
    'B4 - Повседневное',
    'B5 - Молодежное',
    'C1 - Старший',
    'C2 - Средний',
    'C3 - Подростковый',
    'D1 - Люкс',
    'D2 - Масс-маркет',
    'D3 - Премиум'
]

categories_output = [
    'R1 - LVMH',
    'R2 - Melon Fashion Group',
    'R3 - Inditex'
]

input_data = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
])

output_data = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
])

model = keras.Sequential([
    Flatten(input_shape=(14, 1)),
    Dense(14, activation='relu'),
    Dense(3, activation='softmax'),
])
print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(input_data, output_data, epochs=99, validation_split=0.1)

x = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
input_understandable = [categories_input[i] for i, x in enumerate(x) if x == 1]

x = np.expand_dims(x, axis=0)
res = model.predict(x)
print(res)
print()
print(input_understandable)
print(categories_output[np.argmax(res)])
print()

print(np.argmax(res))

input_string = input("Введите строку: ")
output_string = ",".join(input_string)
print(output_string)

input_test = input_data
output_test = output_data

# Predicting the test dataset
predictions = model.predict(input_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(output_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(true_classes, predicted_classes))
