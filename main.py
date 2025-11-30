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
    initializer: str = "glorot_uniform" #es el que viene por defecto sino lo especificas

@dataclass #dataclass para los distintos earlyStops
class EarlyStoppingConfig:
    monitor: str = 'val_loss'
    paciencia: int = 10
    restore_weights: bool = True
    min_delta: float = 0.001
    verbose: int = 1

class MejorasConfig:        
    dropout: float = 0.0       # 0 = sin Dropout
    use_batchnorm: bool = False     # BatchNormalization

_datos_cacheados = None

def cargar_datos(mejora: bool = False):
    global _datos_cacheados
    
    if _datos_cacheados is not None and not mejora:
        return _datos_cacheados
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
   
    #shape es (50000, 32, 32, 3) -> (50000, 3072), normalizando píxeles a [0, 1]
    if not mejora:
        X_train = X_train.reshape((X_train.shape[0], -1)).astype("float32") / 255.0
        X_test  = X_test.reshape((X_test.shape[0], -1)).astype("float32") / 255.0
    else:
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0
    
    num_clases = 10
    #cada etiqueta pasa a vector de 10 posiciones (one-hot encoding)
    y_train_cat = to_categorical(y_train, num_clases)
    y_test_cat  = to_categorical(y_test, num_clases)

    if mejora:
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomBrightness(factor=0.2)
        ])

        X_train_augmented = []
        for img in X_train:
            aug_img = data_augmentation(img[np.newaxis, ...], training=True)[0]
            X_train_augmented.append(aug_img)
        X_train_augmented = np.array(X_train_augmented)
        X_train_flat = X_train_augmented.reshape((X_train_augmented.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        return X_train_flat, y_train_cat, X_test_flat, y_test_cat, y_test

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
    accuracys = [resultados[n]["test_acc"] for n in nombres]
    tiempos = [resultados[n]["train_time"] for n in nombres]
    
    fig, ax1 = plt.subplots(figsize=(14, 7)) #tamaño de la imagen
    
    x = np.arange(len(nombres)) #creamos un array para plotear los distintos modelos
    width = 0.35 #el ancho de cada barra, una dede x-width/2 hasta x+width/2
    
    #eje izquierdo: Accuracy
    bars1 = ax1.bar(x - width/2, accuracys, width, alpha=0.8, color='steelblue', label='Test Accuracy')
    ax1.set_xlabel("Modelo", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Test Accuracy", fontsize=11, fontweight='bold', color='steelblue')

    acc_min, acc_max = min(accuracys), max(accuracys)
    ax1.set_ylim([acc_min - 0.05 * (acc_max-acc_min), acc_max + 0.05 * (acc_max-acc_min)])
    #ax1.tick_params(axis='y', labelcolor='steelblue')
    #ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars1, accuracys): #añadimos los valores encima de cada barra
        height = bar.get_height() #a que altura termina la barra
        ax1.text(bar.get_x() + bar.get_width()/2., height, #corrdenadas del texto
                f'{acc:.4f}', ha='center', va='bottom', fontsize=9, color='steelblue', fontweight='bold')
    
    #eje derecho: Tiempo
    ax2 = ax1.twinx() #segundo eje y con el mismo eje x
    bars2 = ax2.bar(x + width/2, tiempos, width, alpha=0.8, color='orange', label='Tiempo (s)')
    ax2.set_ylabel("Tiempo (segundos)", fontsize=11, fontweight='bold', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    min_tiempo, max_tiempo = min(tiempos) , max(tiempos)
    ax2.set_ylim([min_tiempo - 0.05 * (max_tiempo-min_tiempo), max_tiempo + 0.005 * (max_tiempo - min_tiempo)])
    
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

    plt.figure(figsize=(10, 6))  # definimos fig_size
    
    #primer eje (izquierda
    ax1 = plt.gca()
    ax1.plot(x, y_list[0], label=labels[0], marker='o', color='b')
    ax1.plot(x, y_list[2], label=labels[2], marker='o', color='g')
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Accuracy", color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    acc_values = np.concatenate((y_list[0],y_list[2]))
    acc_min, acc_max = min(acc_values), max(acc_values)
    ax1.set_ylim([acc_min - 0.05 * (acc_max-acc_min), acc_max + 0.005 * (acc_max-acc_min)])
    
    #segundo eje de dibujo
    ax2 = ax1.twinx()
    ax2.plot(x, y_list[1], label=labels[1], marker='o', color='r')
    ax2.plot(x, y_list[3], label=labels[3], marker='o', color='orange')
    ax2.set_ylabel("Loss", color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    loss_values = np.concatenate((y_list[1], y_list[3]))
    loss_min, loss_max = min(loss_values), max(loss_values)
    ax2.set_ylim([loss_min - 0.05 * (loss_max-loss_min), loss_max + 0.05 * (loss_max-loss_min)])
    
    #título
    plt.title(title)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfica guardada: {filename}")


def matriz_confusion(y_test, y_pred, nombre_modelo, filename):
    cm = confusion_matrix(y_test, y_pred) #valor real y prediccion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Greys') #pintamos la matriz en el area de dibujo
    plt.title(f"Matriz de Confusión - {nombre_modelo}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Matriz de confusión guardada: {filename}")

def compilar_mlp(config: MLPConfig, input_dim: int, num_clases: int = 10, mejora: MejorasConfig = None):

    model = models.Sequential(name=config.nombre)
    model.add(layers.Input(shape=(input_dim,)))

    # Si no hay mejoras, usar configuración básica
    if mejoras is None:
        mejoras = MejorasConfig()
    

    for i, neuronas in enumerate(config.capas):

        model.add(layers.Dense(
            neuronas, 
            kernel_initializer=config.initializer
        ))
        
        if mejoras.use_batchnorm:
            model.add(layers.BatchNormalization())
        
        #leakyRelu tiene que añadir su propia capa
        if config.activation == "leaky_relu":
            model.add(layers.LeakyReLU(negative_slope=0.1))
        else:
            model.add(layers.Activation(config.activation))
        
        #dropout, si esta especificado
        if mejoras.dropout > 0 and i < len(config.capas) - 1: #la ultima capa no se pueden apagar neuronas porque no habría resultado
            model.add(layers.Dropout(mejoras.dropout_rate))

    model.add(layers.Dense(num_clases, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def entrenar_MLP(config: MLPConfig, ea: EarlyStoppingConfig, usar_ea: bool = True):

    X_train, y_train, X_test , y_test, y_test_labels = cargar_datos()
    #cargamos los datos y las codificaciones one_hot, mas las etiquetas para la matriz de confusión luego

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
    
def ejecutar_mlp(config: MLPConfig, ea: EarlyStoppingConfig, repeticiones: int = 5, usar_ea: bool = True):
    mejor_entrenamiento = None
    mejor_test_acc = 0
    historial = []
    for i in range(repeticiones):
        resultado = entrenar_MLP(config,ea,usar_ea)
        if resultado['test_acc'] > mejor_test_acc:
            mejor_entrenamiento = i
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

    mejor_resultado = historial[mejor_entrenamiento]
    matriz_confusion(
        mejor_resultado["y_test"],
        mejor_resultado["y_pred"],
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
        config_nombre = f"{ea.monitor}_pat_{ea.patience}_delta_{ea.min_delta}"
        print(f"Probando earlyStopping {config_nombre}")
        config_con_nombre = MLPConfig( #mismo modelo pero el nombre acompañado del early_stopping
            nombre=f"{config.nombre}_{config_nombre}",
            capas=config.capas,
            activation=config.activation,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0
        )
        
        resultado = ejecutar_mlp(config_con_nombre, ea, repeticiones)
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
        resultado = ejecutar_mlp(config_con_parametros,ea,5)
        resultados[config_nombre] = resultado

    comparativa_modelos(
        resultados,
        "comprativa_batch_sizes.png"
    )

def probar_activaciones_inicializaciones(config: MLPConfig, ea: EarlyStoppingConfig, activaciones_inicializaciones: List[tuple], repeticiones: int = 5):
  
    resultados = {}
    print("Comparando activaciones e inicializaciones")

    for act, init in activaciones_inicializaciones: #recorremos la tupla
        config_nombre = f"{act}_{init}"
        print(f"Probando activación {act} con inicialización {init}")
        
        config_con_parametros = MLPConfig(
            nombre=f"{config.nombre}_{config_nombre}",
            capas=config.capas,
            activation=act,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            initializer=init
        )
        resultado = ejecutar_mlp(config_con_parametros, ea, repeticiones)
        resultados[config_nombre] = resultado


    comparativa_modelos(resultados,"comparativa_activaciones_inicializaciones.png")

def probar_neuronas(config: MLPConfig, ea: EarlyStoppingConfig, neuronas: List[int], repeticiones: int = 5):
    resultados = {}
    print("Comparando neuronas")
    for n in neuronas:
        config_nombre = f"neuronas_{n}"
        print(f"Probando {n} neuronas")
        config_con_parametros = MLPConfig(
            nombre=f"{config.nombre}_{config_nombre}",
            capas=[n],
            activation=config.activation,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            initializer=config.initializer
        )
        resultado = ejecutar_mlp(config_con_parametros,ea,repeticiones)
        resultados[config_nombre] = resultado
    
    comparativa_modelos(resultados,"comprativa_neuronas.png")

def probar_capas(config: MLPConfig, ea:EarlyStoppingConfig, neuronas: List[List[int]], repeticiones: int = 5):
    resultados = {}
    for capas in neuronas:
        nombre = str(capas)
        nombre = nombre.replace('[','').replace(']','').replace(',','_').replace(' ','')
        config_nombre = f"capas_{nombre}"
        print(f"Probando capas {capas}")
        config_con_parametros = MLPConfig(
            nombre = f"{config.nombre}_{config_nombre}",
            capas = [n for n in capas],
            activation = config.activation,
            epochs = config.epochs,
            batch_size = config.batch_size,
            verbose = 0,
            initializer = config.initializer
        )
        resultado = ejecutar_mlp(config_con_parametros,ea,repeticiones)
        resultados[config_nombre] = resultado

    comparativa_modelos(resultados,"comprativa_capas.png")


if __name__ == "__main__":
    
    batch_sizes = [16,32,64,100,128,200,256,300,512,1024,2048] #batch_sizes a probar
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
        MLPConfig( #mlp 3, usamos el callback 9 mejor callback obetenido hasta ahora, muchas epocas
            nombre="mlp3",
            capas=[48],        
            activation="sigmoid",
            epochs=200,
            batch_size=200, #el mejor segun las pruebas
            verbose=0
        ),
        MLPConfig( #mlp 4, usamos el callback 9 mejor callback obetenido hasta ahora, muchas epocas
            nombre="mlp4",
            capas=[48],        
            activation="leaky_relu", #mejor resultado leaky_relu con he_normal
            epochs=200,
            batch_size=200, #el mejor segun las pruebas
            verbose=0,
            initializer= "he_normal"
        ),
        MLPConfig( #mlp 5, usamos el callback 9 mejor callback obetenido hasta ahora, muchas epocas
            nombre="mlp5",
            capas=[80],   #mejor relación resultado-tiempo     
            activation="leaky_relu",
            epochs=200,
            batch_size=200, #el mejor segun las pruebas
            verbose=0,
            initializer= "he_normal"
        ),
        MLPConfig( #mlp 6, usamos el callback 9 mejor callback obetenido hasta ahora, muchas epocas
            nombre="mlp6",
            capas=[80], #mejor configuracion las 80 neuronas en una capa, al menos por ahora        
            activation="leaky_relu",
            epochs=200,
            batch_size=200, #el mejor segun las pruebas
            verbose=0,
            initializer= "he_normal"
        )
    ]
    early_stopping_configs = [ #earlystoppings a probar, grafica ya generada
        EarlyStoppingConfig(monitor='val_loss', paciencia=2, min_delta=0.005, verbose=0),
        EarlyStoppingConfig(monitor='val_loss', paciencia=3, min_delta=0.002, verbose=0),
        EarlyStoppingConfig(monitor='val_loss', paciencia=5, min_delta=0.001, verbose=0), 
        EarlyStoppingConfig(monitor='val_loss', paciencia=7, min_delta=0.0005, verbose=0),
        EarlyStoppingConfig(monitor='val_loss', paciencia=10, min_delta=0.0001, verbose=0),
        EarlyStoppingConfig(monitor='val_loss', paciencia=15, min_delta=0.00009, verbose=0),
        EarlyStoppingConfig(monitor='val_accuracy', paciencia=2, min_delta=0.005, verbose=0),
        EarlyStoppingConfig(monitor='val_accuracy', paciencia=3, min_delta=0.002, verbose=0),
        EarlyStoppingConfig(monitor='val_accuracy', paciencia=5, min_delta=0.001, verbose=0), 
        EarlyStoppingConfig(monitor='val_accuracy', paciencia=7, min_delta=0.0005, verbose=0),#para mi este es el mejor
        EarlyStoppingConfig(monitor='val_accuracy', paciencia=10, min_delta=0.0001, verbose=0),
        EarlyStoppingConfig(monitor='val_accuracy', paciencia=15, min_delta=0.00009, verbose=0)
    ]

    activaciones_inicializaciones = [
        ("relu", "he_normal"),
        ("sigmoid", "glorot_uniform"), #mejor resultado arquitectura simple
        ("tanh", "glorot_uniform"),
        ("relu", "he_uniform"),
        ("leaky_relu", "he_normal"),  # para LeakyReLU habría que añadir la capa manualmente
        ("leaky_relu","he_uniform")
    ]
    neuronas = [12,24,48,60,80,96,100,120,150,170,200] #entre 60-80 neuronas lo mejor

    capas = [
    #uniformes
    [80],
    [40, 40],
    [27, 27, 26],
    [20, 20, 20, 20],
    [16, 16, 16, 16, 16],
    
    #decrecientes 
    [50, 30],
    [60, 20],
    [40, 30, 10],
    [50, 20, 10],
    [30, 25, 15, 10],
    
    #crecientes 
    [10, 70],
    [10, 30, 40],
    [20, 30, 30],
    [10, 20, 25, 25],
    
    #cuello de botella 
    [40, 20, 40],
    [30, 15, 35],
    [35, 10, 35],
    [30, 20, 15, 15],
]
    #ejecutar_mlp(configs[0],early_stopping_configs[9],5,False)
    #ejecutar_mlp(configs[1],early_stopping_configs[0],5,False)
    #comparar_earlystoppings(configs[1],early_stopping_configs,5)
    #probar_batch_size(configs[2],early_stopping_configs[9],batch_sizes)
    #probar_activaciones_inicializaciones(configs[3],early_stopping_configs[9],activaciones_inicializaciones,5)
    #probar_neuronas(configs[4],early_stopping_configs[9],neuronas,5)
    #probar_capas(configs[5],early_stopping_configs[9],capas,5)
    









