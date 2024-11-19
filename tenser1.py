import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Cargar el dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos (escala de 0 a 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Construir el modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Aplana las imágenes 28x28 a un vector 1D
    layers.Dense(128, activation='relu'),  # Capa densa con 128 neuronas y activación ReLU
    layers.Dropout(0.2),  # Capa de Dropout para evitar sobreajuste
    layers.Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (dígitos del 0 al 9)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluar el modelo en los datos de prueba
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nPrecisión en los datos de prueba: {test_acc}')

# Mostrar la curva de precisión durante el entrenamiento
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.show()
