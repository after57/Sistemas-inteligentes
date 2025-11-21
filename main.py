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

@dataclass #dataclass que representa los mlps
class MLPConfig:
    nombre: str                     #nombre para graficas
    capas: List[int]      #vector con las neuronas por capa
    activation: str               #activacion de las capas intermedias
    epochs: int = 10              #épocas
    batch_size: int = 32          #tamaño de batch


def cargar_datos():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    #shape es (50000, 32, 32, 3) -> (50000, 3072), normalizando píxeles a [0, 1]
    X_train = X_train.reshape((X_train.shape[0], -1)).astype("float32") / 255.0
    X_test  = X_test.reshape((X_test.shape[0], -1)).astype("float32") / 255.0

    num_clases = 10
    #cada etiqueta pasa a vector de 10 posiciones (one-hot encoding)
    y_train_cat = to_categorical(y_train, num_clases)
    y_test_cat  = to_categorical(y_test, num_clases)

    return X_train, y_train_cat, X_test, y_test_cat, y_test

def compilar_mlp(config: MLPConfig, input_dim: int, num_clases: int = 10):
    model = models.Sequential(name=config.nombre)
    
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
    return model


def ejecutar_MLP(config: MLPConfig):
    X_train, y_train, X_test , y_test, y_test_labels = cargar_datos()

    input_dim = X_train.shape[1]
    num_clases = y_train.shape[1]

    model = compilar_mlp(config,input_dim,num_clases)
    t0 = time.time()
    my_callbacks = [
        keras.callbacks.EarlyStopping(patience=2)
    ]
    history = model.fit(
        X_train,y_train,
        validation_split = 0.1, 
        batch_size = config.batch_size,
        epochs = config.epochs,
        verbose = 0,
        callbacks = my_callbacks
    )
    train_time = time.time() - t0
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1) #nos quedamos con el mayor valor para cada imagen la que el mlp piensa que es mas esa

    return {
        "train_time": train_time,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history.history,
        "y_pred": y_pred,
        "y_test": y_test_labels
    }

def entrenar_varias_veces(config: MLPConfig, repeticiones=5):
    historial = []
    for i in range(repeticiones):
        print(f"Entrenamiento {i+1}/{repeticiones}")
        resultado = ejecutar_MLP(config)
        historial.append(resultado)
    return historial

def calcular_media_historial(historial):
    """
    Calcula la media del histórico cuando hay EarlyStopping
    Rellena (pad) con el último valor los arrays más cortos
    """
    keys = list(historial[0]["history"].keys())
    medias = {}
    
    # Encontrar la longitud máxima
    max_length = max(len(h["history"][keys[0]]) for h in historial) #obtenemos el maximo numero de epocas
    
    # Obtener número de épocas en cada entrenamiento
    num_epocas_por_entrenamiento = [len(h["history"][keys[0]]) for h in historial] #obtenemos el numero de epocas de cada repeticion
    
    for k in keys:
        todas = []
        for h in historial:
            arr = np.array(h["history"][k])
            # Si el array es más corto que max_length, rellenar con el último valor
            if len(arr) < max_length:
                arr = np.pad(arr, (0, max_length - len(arr)), mode='edge') #rellena hasta max-length con el ultimo valor, para graficar
            todas.append(arr)
        medias[k] = np.mean(np.array(todas), axis=0)
    
    valores_test = ['train_time', 'test_loss', 'test_acc'] #esto son valroes escalares no es necesario nada
    for v in valores_test:
        valores = np.array([h[v] for h in historial])
        medias[v] = np.mean(valores)
    
    return medias, max_length, num_epocas_por_entrenamiento

def plottear_graficas(x, y_list, labels, title, filename):
    plt.figure(figsize=(10, 6))
    for y, label in zip(y_list, labels):
        plt.plot(x, y, label=label, marker='o')
    
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfica guardada: {filename}")

def matriz_confusion(y_test, y_pred, nombre_modelo, filename):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues') #pintamos la matriz en el area de dinbujo
    plt.title(f"Matriz de Confusión - {nombre_modelo}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Matriz de confusión guardada: {filename}")

def probar_mlp(config: MLPConfig, repeticiones):
    historial = entrenar_varias_veces(config, repeticiones)
    media, max_epochs, num_epocas = calcular_media_historial(historial)
    
    # Mostrar en qué época paró cada entrenamiento
    print(f"\n=== Información de EarlyStopping ===")
    print(f"Configuración: {config.nombre}")
    for i, ep in enumerate(num_epocas):
        print(f"  Entrenamiento {i+1}: Paró en época {ep}")
    print(f"Máximo de épocas alcanzado: {max_epochs}")
    print(f"Promedio de épocas: {np.mean(num_epocas):.1f}")
    
    epochs_range = np.arange(1, max_epochs + 1)

    plottear_graficas(
        epochs_range,
        [media["accuracy"],
         media["loss"],
         media["val_accuracy"],
         media["val_loss"]],
         ["Train accuracy", "Train loss", "Val accuracy", "Val loss"],
         "Evolución entrenamiento",
         f"{config.nombre}_evolucion_entrenamiento.png"
    )

    ultimo_resultado = historial[-1]
    matriz_confusion(
        ultimo_resultado["y_test"],
        ultimo_resultado["y_pred"],
        config.nombre,
        f"{config.nombre}_matriz_confusion.png"
    )

    print(f"\nResultados finales de {config.nombre}")
    print(f"Tiempo de entrenamiento (media): {media['train_time']:.2f}s")
    print(f"Test Accuracy (media): {media['test_acc']:.4f}")
    print(f"Test Loss (media): {media['test_loss']:.4f}")
    print(f"Val Accuracy final (media): {media['val_accuracy'][-1]:.4f}") #el ultimo de ellos
    print(f"Val Loss final (media): {media['val_loss'][-1]:.4f}") #el ultimo de ellos


if __name__ == "__main__":
    
    configs = [
        MLPConfig( #mlp 1
            nombre="mlp1",
            capas=[48],        # 1 capa oculta de 48 neuronas
            activation="sigmoid",
            epochs=10,
            batch_size=32,
        ),
        MLPConfig( #mlp 2, el callback para de media ahi
            nombre="mlp2",
            capas=[48],        # 1 capa oculta de 48 neuronas
            activation="sigmoid",
            epochs=150,
            batch_size=32,
        )
    ]

    probar_mlp(configs[1],5)


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
