import keras
from keras.utils import to_categorical
import numpy as np
from keras import models, layers
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataclasses import dataclass
from typing import List

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

# ------------------------------
# Class de los mlps
# ------------------------------
@dataclass
class MLPConfig:
    nombre: str                     #nombre para graficas
    capas: List[int]      #vector con las neuronas por capa
    activation: str               #activacion de las capas intermedias
    epochs: int = 10              #épocas
    batch_size: int = 32          #tamaño de batch


def cargar_y_preprocesar_cifar10_mlp():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # shape es (50000, 32, 32, 3) -> (50000, 3072), normalizando píxeles a [0, 1]
    X_train = X_train.reshape((X_train.shape[0], -1)).astype("float32") / 255.0
    X_test  = X_test.reshape((X_test.shape[0], -1)).astype("float32") / 255.0

    num_clases = 10
    # One-hot encoding: cada etiqueta pasa a vector de 10 posiciones
    y_train_cat = to_categorical(y_train, num_clases)
    y_test_cat  = to_categorical(y_test, num_clases)

    return X_train, y_train_cat, X_test, y_test_cat



def compilar_mlp(config: MLPConfig, input_dim: int, num_clases: int = 10):
    model = models.Sequential(name=config.name)

    
    model.add(layers.Input(shape=(input_dim,)))

    #capas
    for units in config.capas:
        model.add(layers.Dense(units, activation=config.activation))

    #capa de salida: num_clases neuronas con softmax
    model.add(layers.Dense(num_clases, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(model.summary())
    return model


# ------------------------------
# Entrenar y evaluar un MLP según una config
# ------------------------------
def probar_MLP(config: MLPConfig):
    X_train, y_train, X_test, y_test = cargar_y_preprocesar_cifar10_mlp()

    input_dim = X_train.shape[1]
    num_clases = y_train.shape[1]

    model = compilar_mlp(config, input_dim, num_clases)

    t0 = time.time()
    history = model.fit(  # entrenamos con los parámetros indicados
        X_train, y_train,
        validation_split=0.1,
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=2
    )
    train_time = time.time() - t0

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  # evaluación en test

    # 1) Gráficas de entrenamiento
    epochs_range = range(1, len(history.history["loss"]) + 1)

    #pérdida
    plt.figure()
    plt.plot(epochs_range, history.history["loss"], label="train loss")
    for x,y in zip(epochs_range,history.history["loss"]):
        plt.annotate(
            f"{y:.3f}",
            (x,y),
            textcoords="offset points",
            xytext=(0,8), #8 arriba del punto
            ha="center" #alineamiento horizontal
        )
    plt.plot(epochs_range, history.history["val_loss"], label="val loss")
    plt.title(f"Evolución de la pérdida - {config.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{config.name}_loss.png")
    plt.close()

    #Accuracy
    plt.figure()
    plt.plot(epochs_range, history.history["accuracy"], label="train acc")
    plt.plot(epochs_range, history.history["val_accuracy"], label="val acc")
    plt.title(f"Evolución del accuracy - {config.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{config.name}_accuracy.png")
    plt.close()

    # 2) Gráfica resumen (tiempo + accuracy)
    plt.figure()
    plt.bar(["Train time (s)", "Test accuracy"], [train_time, test_acc])
    plt.title(f"Resumen del modelo {config.name}")
    plt.tight_layout()
    plt.savefig(f"{config.name}_resumen.png")
    plt.close()

    #3) matriz de confusión (usando stickit learn)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(8, 8))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Matriz de confusión - {config.name}")
    plt.tight_layout()
    plt.savefig(f"{config.name}_confusion_matrix.png")
    plt.close()

    return {
        "name": config.name,
        "train_time": train_time,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


# ------------------------------
#varios modelos distintos
# ------------------------------
if __name__ == "__main__":
    # Aquí defines las "características" de cada MLP:
    configs = [
        MLPConfig(
            nombre="mlp_48_sigmoid",
            hidden_layers=[48],        # 1 capa oculta de 48 neuronas
            activation="sigmoid",
            epochs=10,
            batch_size=32,
        ),
        MLPConfig(
            nombre="mlp_48_sigmoid_100",
            capas=[48],        # 1 capa oculta de 48 neuronas
            activation="sigmoid",
            epochs=100,
            batch_size=32,
        ),
    ]

    resultados = []
    for cfg in configs:
        print(f"\n=== Entrenando {cfg.name} ===")
        res = probar_MLP(cfg)
        resultados.append(res)

    #de momento no usar
    
    '''
    nombres = [r["name"] for r in resultados]
    accs = [r["test_acc"] for r in resultados]
    tiempos = [r["train_time"] for r in resultados]
    plt.figure()
    plt.bar(nombres, accs)
    plt.title("Comparación de accuracy en test entre modelos")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("comparativa_accuracy_modelos.png")
    plt.close()

    plt.figure()
    plt.bar(nombres, tiempos)
    plt.title("Comparación de tiempo de entrenamiento entre modelos")
    plt.ylabel("Tiempo (s)")
    plt.tight_layout()
    plt.savefig("comparativa_tiempo_modelos.png")
    plt.close()
    '''
