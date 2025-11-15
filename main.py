import keras
from keras.utils import to_categorical
import numpy as np
from keras import models,layers
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
'''
1. La gráfica de líneas que muestren la evolución de la pérdida y la tasa de acierto para el conjunto de 
entrenamiento y para el conjunto de validación durante el entrenamiento de la red (en Keras, esta 
información la devuelve la función fit, que es la que realiza el entrenamiento).
2. La gráfica de barras con el tiempo de entrenamiento y la tasa de acierto finales con el conjunto de 
test (en Keras, esto lo conseguimos con la función evaluate), para comparar resultados de varios 
modelos.
3. La matriz de confusión para los resultados con el conjunto de test (investiga cómo puedes mostrar 
esa información de manera rápida y elegante, quizás tengas que instalar alguna otra librería de 
Python).
'''

def cargar_y_preprocesar_cifar10_mlp():
    # Carga datos
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train = X_train.reshape((X_train.shape[0], -1)).astype("float32") / 255.0 #-1 hace que numpy lo calcule por si mismo
    X_test  = X_test.reshape((X_test.shape[0], -1)).astype("float32")/255.0

    # One-hot de las clases (0..9)
    num_clases = 10
    y_train_cat = to_categorical(y_train, num_clases) #todo a arrays de 10 posiciones, uno para cada imagen 1 donde pertnece 0 en el resto
    y_test_cat  = to_categorical(y_test, num_clases)

    return X_train, y_train_cat, X_test, y_test_cat

def compilar_mlp(input_dim,num_clases=10):

    model = models.Sequential()
    model.add(layers.Input(shape = (input_dim,))) #el numero de caracteristicas que tiene cada muestra en mi caso 3072, 32*32*3
    model.add(layers.Dense(48, activation="sigmoid"))
    model.add(layers.Dense(num_clases, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
     )
    print(model.summary())
    return model

def probar_MLP():
    X_train, y_train, X_test, y_test = cargar_y_preprocesar_cifar10_mlp()

    input_dim = X_train.shape[1]
    num_clases = y_train.shape[1]

    model = compilar_mlp(input_dim, num_clases)
    model.summary()

    t0 = time.time()
    history = model.fit( #entrenamos con los parametros indicados pro la práctica
        X_train, y_train,
        validation_split=0.1,
        batch_size=32,
        epochs=10,
        verbose=2
    )
    train_time = time.time() - t0

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0) #evaluamos con el conjunto de test

    # ░░░░░░░░░░░░░░░░░░░░░░░
    # 1) Gráficas de entrenamiento
    # ░░░░░░░░░░░░░░░░░░░░░░░
    epochs = range(1, len(history.history["loss"]) + 1)

    # Pérdida
    plt.figure()
    plt.plot(epochs, history.history["loss"], label="train loss")
    plt.plot(epochs, history.history["val_loss"], label="val loss")
    plt.title("Evolución de la pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mlp_loss.png")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history.history["accuracy"], label="train acc")
    plt.plot(epochs, history.history["val_accuracy"], label="val acc")
    plt.title("Evolución del accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mlp_accuracy.png")
    plt.close()

    # ░░░░░░░░░░░░░░░░░░░░░░░
    # 2) Gráfica resumen (tiempo + accuracy)
    # ░░░░░░░░░░░░░░░░░░░░░░░
    plt.figure()
    plt.bar(["Train time (s)", "Test accuracy"], [train_time, test_acc])
    plt.title("Resumen del modelo MLP")
    plt.tight_layout()
    plt.savefig("mlp_resumen.png")
    plt.close()

    # ░░░░░░░░░░░░░░░░░░░░░░░
    # 3) Matriz de confusión (NECESARIA)
    # ░░░░░░░░░░░░░░░░░░░░░░░
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(8, 8))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Matriz de confusión - MLP CIFAR10")
    plt.tight_layout()
    plt.savefig("mlp_confusion_matrix.png")
    plt.close()

    return test_loss, test_acc

if __name__ == "__main__":
    probar_MLP()

