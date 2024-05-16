import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

x = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]

categories_input = [
    'А1 - Эмоциональный',
    'А2 - Описательный',
    'А3 - Философский',
    'А4 - Простой и понятный',
    'B1 - Веселое',
    'B2 - Трогательное',
    'B3 - Таинственное',
    'B4 - Успокаивающее',
    'C1 - Легкий',
    'C2 - Средний',
    'C3 - Сложный',
    'D1 - Короткие статьи',
    'D2 - Средние рассказы',
    'D3 - Полные книги',
    'E1 - История',
    'E2 - Наука',
    'E3 - Искусство',
    'E4 - Спорт'
]

categories_output = [
    'R1 - Фантастика',
    'R2 - Детектив',
    'R3 - Роман',
    'R4 - Научно-популярная литература',
    'R5 - Другое'
]

input_data = np.array([
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
])

output_data = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0]
])

model = keras.Sequential([
    Flatten(input_shape=(18, 1)),
    Dense(18, activation='relu'),
    Dense(5, activation='softmax'),
])
print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(input_data, output_data, epochs=99, validation_split=0.1)

input_understandable = [categories_input[i] for i, x in enumerate(x) if x == 1]

x = np.expand_dims(x, axis=0)
res = model.predict(x)
print(res)
print()
print(input_understandable)
print(categories_output[np.argmax(res)])
print()

print(np.argmax(res))
