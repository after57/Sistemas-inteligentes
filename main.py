import keras
from keras.utils import to_categorical
import numpy as np
from keras import models, layers
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataclasses import dataclass
from typing import List
import tensorflow as tf

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
    verbose: int = 0

@dataclass #dataclass para los distintos earlyStops
class EarlyStoppingConfig:
    monitor: str = 'val_loss'
    patience: int = 10
    restore_weights: bool = True
    min_delta: float = 0.001
    verbose: int = 1

_datos_cacheados = None

def cargar_datos():
    global _datos_cacheados
    
    if _datos_cacheados is not None:
        return _datos_cacheados
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    #shape es (50000, 32, 32, 3) -> (50000, 3072), normalizando píxeles a [0, 1]
    X_train = X_train.reshape((X_train.shape[0], -1)).astype("float32") / 255.0
    X_test  = X_test.reshape((X_test.shape[0], -1)).astype("float32") / 255.0

    num_clases = 10
    #cada etiqueta pasa a vector de 10 posiciones (one-hot encoding)
    y_train_cat = to_categorical(y_train, num_clases)
    y_test_cat  = to_categorical(y_test, num_clases)

    _datos_cacheados = (X_train, y_train_cat, X_test, y_test_cat, y_test)

    return _datos_cacheados

def calcular_media_historial(historial):

    keys = list(historial[0]["history"].keys()) #sacamos todas las métricas,
    print(f"Keys del histro {keys}")
    #recordar que historial es una lista de diccionarios y history es otro diccionario dentro de ese con train y val
    medias = {}
    
    #encontrar la longitud máxima
    max_length = max(len(h["history"][keys[0]]) for h in historial) #obtenemos el maximo numero de epocas, el len de una metrica de history
    
    #numero de épocas por cada repitcion del entrenamiento, varia por el factor de aleatorio de cada entrenamiento
    num_epocas_por_entrenamiento = [len(h["history"][keys[0]]) for h in historial] #obtenemos el numero de epocas de cada repeticion
    
    for k in keys: 
        todas = []
        for h in historial:
            arr = np.array(h["history"][k]) #en history estaba acc,loss,val_loss y val_acc

            if len(arr) < max_length: #si faltan valores para calcular la media
                arr = np.pad(arr, (0, max_length - len(arr)), mode='edge') #rellena hasta max-length con el ultimo valor, para graficar y la media
            todas.append(arr)
        medias[k] = np.mean(np.array(todas), axis=0) #hacemos la media aritmetica entre columnas es decir epoca a epoca con los 5 entrenamientos

    valores_test = ['train_time', 'test_loss', 'test_acc'] #esto son valroes escalares no es necesario nada
    for v in valores_test:
        valores = np.array([h[v] for h in historial])
        medias[v] = np.mean(valores)
    
    return medias, max_length, num_epocas_por_entrenamiento

def comparativa_modelos(resultados, filename):
 
    nombres = list(resultados.keys()) #recordemos resultados es un diccionario de diccionarios, a cada nombre esta el train,history y test_acc
    accs = [resultados[n]["test_acc"] for n in nombres]
    tiempos = [resultados[n]["train_time"] for n in nombres]
    
    fig, ax1 = plt.subplots(figsize=(14, 7)) #tamaño de la imagen
    
    x = np.arange(len(nombres)) #creamos un array para plotear los distintos modelos
    width = 0.35 #el ancho de cada barra, una dede x-width/2 hasta x+width/2
    
    #eje izquierdo: Accuracy
    bars1 = ax1.bar(x - width/2, accs, width, alpha=0.8, color='steelblue', label='Test Accuracy')
    ax1.set_xlabel("Modelo", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Test Accuracy", fontsize=11, fontweight='bold', color='steelblue')
    ax1.set_ylim([0, 1]) #de 0 a 1
    #ax1.tick_params(axis='y', labelcolor='steelblue')
    #ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars1, accs): #añadimos los valores encima de cada barra
        height = bar.get_height() #a que altura termina la barra
        ax1.text(bar.get_x() + bar.get_width()/2., height, #corrdenadas del texto
                f'{acc:.4f}', ha='center', va='bottom', fontsize=9, color='steelblue', fontweight='bold')
    
    #eje derecho: Tiempo
    ax2 = ax1.twinx() #segundo eje y con el mismo eje x
    bars2 = ax2.bar(x + width/2, tiempos, width, alpha=0.8, color='orange', label='Tiempo (s)')
    ax2.set_ylabel("Tiempo (segundos)", fontsize=11, fontweight='bold', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    for bar, t in zip(bars2, tiempos): #añadimos los valores encima de cada barra
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.2f}s', ha='center', va='bottom', fontsize=9, color='orange', fontweight='bold')
    
    # Configurar eje X
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres, rotation=45, ha='right', fontsize=9) #
    
    plt.title("Comparación de Accuracy y Tiempo de Entrenamiento", fontsize=13, fontweight='bold', pad=20)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)  #las lineas son los colores
    
    plt.tight_layout() #ajustar margenes
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfica comparativa guardada: {filename}")

def plottear_graficas(x, y_list, labels, title, filename):
    plt.figure(figsize=(10, 6)) #definimos fig_size

    for y, label in zip(y_list, labels): #cada valor y va asociado a la etiqueta de que estamos plotteando, x siempre son las épocas
        plt.plot(x, y, label=label, marker='o')
    
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout() #para que se ajuste el padding y no se colapsen las imagenes
    plt.savefig(filename)
    plt.close()
    print(f"Gráfica guardada: {filename}")

def matriz_confusion(y_test, y_pred, nombre_modelo, filename):
    cm = confusion_matrix(y_test, y_pred) #valor real y prediccion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues') #pintamos la matriz en el area de dinbujo, colormap azul podría poner otro
    plt.title(f"Matriz de Confusión - {nombre_modelo}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Matriz de confusión guardada: {filename}")

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

def entrenar_MLP(config: MLPConfig, ea: EarlyStoppingConfig, usar_ea: bool = True):

    X_train, y_train, X_test , y_test, y_test_labels = cargar_datos()

    input_dim = X_train.shape[1]
    num_clases = y_train.shape[1]

   
    my_callbacks = None
    if usar_ea:
        my_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor= ea.monitor,
                patience = ea.patience,
                restore_best_weights = ea.restore_weights,
                min_delta = ea.min_delta,
                verbose = ea.verbose
            )
        ]
    tf.keras.backend.clear_session()
    model = compilar_mlp(config,input_dim,num_clases)
    
    t0 = time.time()
    history = model.fit(
        X_train,y_train,
        validation_split = 0.1, 
        batch_size = config.batch_size,
        epochs = config.epochs,
        verbose = config.verbose,
        callbacks =  my_callbacks if my_callbacks else []
    )
    train_time = time.time() - t0
    test_loss, test_acc = model.evaluate(X_test, y_test,verbose=config.verbose)
    #esto es para la matriz de confusion donde usaremos esto y el etiquetado del test es decir y_test (sin codificacion one-hot)
    y_pred_probs = model.predict(X_test, verbose=0) #las probabilidades de cada clase para cada imagen de test
    y_pred = np.argmax(y_pred_probs, axis=1) #nos quedamos con el mayor valor para cada imagen la que el mlp piensa que es mas esa
    return{
        "train_time": train_time,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history.history, #train y validation esta aqui, accuracy,loss,val_acc,val_loss
        "y_pred": y_pred, #esto son los valores para la matriz de confusion
        "y_test": y_test_labels #el resultado que es sin codificacion one hot
    }
    
def probar_mlp(config: MLPConfig, ea: EarlyStoppingConfig, repeticiones: int = 5, usar_ea: bool = True):
    historial = []
    for i in range(repeticiones):
        resultado = entrenar_MLP(config,ea,usar_ea)
        historial.append(resultado)

    media, max_epochs, num_epocas = calcular_media_historial(historial)
    #mostrar en qué época paró cada entrenamiento
    print(f"Información de EarlyStopping")
    print(f"Configuración: {config.nombre}")
    for i,ep in enumerate(num_epocas):
        print(f"Entrenamiento {i+1} paro en la epoca: {ep}")
    print(f"Máximo de épocas alcanzado: {max_epochs}")
    print(f"Promedio de épocas: {np.mean(num_epocas):.1f}")
    
    epochs_range = np.arange(1, max_epochs + 1)

    plottear_graficas(
        epochs_range,
        [media["accuracy"], #y_list
         media["loss"],
         media["val_accuracy"],
         media["val_loss"]],
         ["Train accuracy", "Train loss", "Val accuracy", "Val loss"], #y_labels
         "Evolución entrenamiento",
         f"{config.nombre}_evolucion_entrenamiento.png" #nombre para el archivo
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
    print(f"Val Loss final (media): {media['val_loss'][-1]:.4f}",'\n') #el ultimo de ellos

    return {
        "test_acc": media['test_acc'],
        "test_loss": media['test_loss'],
        "train_time": media['train_time'],
        "media": media  #el array con las medias de los valores del entrenamiento
    }

def comparar_earlystoppings(config: MLPConfig, ea_configs: List[EarlyStoppingConfig], repeticiones=5): #para el mlp2, probar diferentes stops
    resultados = {} #diccionario para guardar los resultado de cada earlyStopping con el modelo
    
    print("Compracion earlystoppings")
    
    for ea in ea_configs:
        config_nombre = f"pat_{ea.patience}_delta_{ea.min_delta}"
        print(f"Probando earlyStopping {config_nombre}")
        config_con_nombre = MLPConfig( #mismo modelo pero el nombre acompañado del early_stopping
            nombre=f"{config.nombre}_{config_nombre}",
            capas=config.capas,
            activation=config.activation,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0
        )
        
        resultado = probar_mlp(config_con_nombre, ea, repeticiones)
        resultados[config_nombre] = resultado #creamos un diccionario de diccionario donde es { nombre:{"train_time": valor}}
    
    # Gráficas comparativas
    comparativa_modelos(
        resultados,
        "comparativa_earlystopping_accuracy.png"
    )

def probar_batch_size(config: MLPConfig, ea: EarlyStoppingConfig, batch_sizes: list[int]):
    resultados = {}
    print("Comparando batch_sizes")

    for bs in batch_sizes:
        config_nombre = f"batch_{bs}"
        print(f"Porbando el batch_size {bs}")
        config_con_parametros = MLPConfig(
            nombre=f"{config.nombre}_{config_nombre}",
            capas = config.capas,
            activation = config.activation,
            epochs = config.epochs,
            batch_size = bs,
            verbose = 0
        )
        resultado = probar_mlp(config_con_parametros,ea,5)
        resultados[config_nombre] = resultado

    comparativa_modelos(
        resultados,
        "comprativa_batch_sizes.png"
    )
    


if __name__ == "__main__":
    
    batch_sizes = [16,32,64,128,256,512,1024,2048] #batch_sizes a probar
    #mejor batch_size 256
    configs = [
        MLPConfig( #mlp 1
            nombre="mlp1",
            capas=[48],        # 1 capa oculta de 48 neuronas
            activation="sigmoid",
            epochs=10,
            batch_size=32,
            verbose=1
        ),
        MLPConfig( #mlp 2, el callback para de media ahi
            nombre="mlp2",
            capas=[48],        
            activation="sigmoid",
            epochs=200,
            batch_size=32,
            verbose=0
        ),
        MLPConfig( #mlp 3, usamos el callback 2 mejor callback obetenido hasta ahora, muchas epocas
            nombre="mlp3",
            capas=[48],        
            activation="sigmoid",
            epochs=200,
            batch_size=256, #el mejor segun las pruebas
            verbose=0
        )
    ]
    early_stopping_configs = [ #earlystoppings a probar, grafica ya generada
        EarlyStoppingConfig(monitor='val_loss', patience=2, min_delta=0.005, verbose=0),
        EarlyStoppingConfig(monitor='val_loss', patience=3, min_delta=0.002, verbose=0),
        EarlyStoppingConfig(monitor='val_loss', patience=5, min_delta=0.001, verbose=0), #para mi este es el mejor
        EarlyStoppingConfig(monitor='val_loss', patience=7, min_delta=0.0005, verbose=0),
        EarlyStoppingConfig(monitor='val_loss', patience=10, min_delta=0.0001, verbose=0)
    ]

    #probar_mlp(configs[1],early_stopping_configs[0],5,False)
    #comparar_earlystoppings(configs[1],early_stopping_configs,5)
    probar_batch_size(configs[2],early_stopping_configs[2],batch_sizes)







