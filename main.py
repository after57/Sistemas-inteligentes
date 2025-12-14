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

@dataclass
class MejorasConfig:        
    dropout: float = 0.0       # 0 = sin Dropout
    use_batchnorm: bool = False     #batchNormalization
    usar_augmented_data: bool = False
    descripcion: str = None

@dataclass
class CNNConfig:
    nombre: str
    bloques_conv: List[tuple]  # Lista de tuplas (num_filtros, kernel_size)
    pool_size: tuple = (2, 2)
    dense_units: int = 100
    activation: str = "relu"
    epochs: int = 100
    batch_size: int = 32
    verbose: int = 0
    initializer: str = "he_normal"

_datos_cacheados = None

data_augmentation = keras.Sequential([ #nueva forma de usar el imageDataGenerator, secuencial que aplica los filtros
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05), #puede rotar entre 0 y 0.05
    layers.RandomZoom(0.05),
])

def cargar_datos_augmented(batch_size=32):
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0 #normalizamos
    X_test  = X_test.astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10) #codificacion one-hot
    y_test_cat  = to_categorical(y_test, 10)

    #generamos datasets, las transfomraciones se hacen despues on fly, porque es una dataset perezoso, revisar el articulo si tengo dudas
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat)) #data-set base, par imagen(32x32x3)-etiqueta(one-hot)
    train_ds = train_ds.shuffle(5000) #mezclamos para romper el orden
    #funcion para aplicar las modificaciones, se modifica y se aplana, además se forma el par con la etiqueta
    #da warining esta linea puesto que no puede hacer parse a grafo de la función inline
    train_ds = train_ds.map(lambda x, y: (tf.reshape(data_augmentation(x, training=True), [-1]), y),
                            num_parallel_calls=tf.data.AUTOTUNE) #paralelizamos para que vaya mas rapido
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE) #agrupamos en el tamaño del batch
    #prefetch prepara el siguiente dataset

    test_ds = tf.data.Dataset.from_tensor_slices((X_test.reshape((X_test.shape[0], -1)), y_test_cat))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds, y_test

def cargar_datos():
    global _datos_cacheados

    if _datos_cacheados is not None:
        return _datos_cacheados

    # Cargar y normalizar
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test, 10)

    # Aplanar directamente (para MLP)
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat  = X_test.reshape((X_test.shape[0], -1))

    _datos_cacheados = (X_train_flat, y_train_cat, X_test_flat, y_test_cat, y_test)
    return _datos_cacheados

def calcular_media_historial(historial):

    keys = list(historial[0]["history"].keys()) #sacamos todas las métricas
    print(f"Keys del histro {keys}")
    #recordar que historial es una lista de diccionarios y history es otro diccionario dentro de ese con train y val
    medias = {}
    
    #encontrar la longitud máxima
    max_length = max(len(h["history"][keys[0]]) for h in historial) #obtenemos el maximo numero de epocas, el len de una metrica de history
    
    #numero de épocas por cada repitcion del entrenamiento, varia por el factor aleatorio de cada entrenamiento
    num_epocas_por_entrenamiento = [len(h["history"][keys[0]]) for h in historial] #obtenemos el numero de epocas de cada entrenamiento
    
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
    
    for bar, acc in zip(bars1, accuracys): #añadimos los valores encima de cada barra
        height = bar.get_height() #a que altura termina la barra
        ax1.text(bar.get_x() + bar.get_width()/2., height, #corrdenadas del texto
                f'{acc:.4f}', ha='center', va='bottom', fontsize=9, color='steelblue', fontweight='bold')
    
    #eje derecho: Tiempo
    ax2 = ax1.twinx() #segundo eje
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
    #dajamos unos margenes
    
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
    matriz = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    matriz.plot(ax=ax, cmap='Greys') #pintamos la matriz en el area de dibujo
    plt.title(f"Matriz de Confusión - {nombre_modelo}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Matriz de confusión guardada: {filename}")

def compilar_mlp(config: MLPConfig, input_dim: int, num_clases: int = 10, mejoras: MejorasConfig = None):

    model = models.Sequential(name=config.nombre)
    model.add(layers.Input(shape=(input_dim,)))

    if mejoras is None: #para poder hacer las comprobaciones despues, la base no lleva nada, es como no usar
        mejoras = MejorasConfig()
    
    for i, neuronas in enumerate(config.capas):

        model.add(layers.Dense(
            neuronas, 
            kernel_initializer=config.initializer
        ))
        
        if mejoras.use_batchnorm: #normalizacion del batch
            model.add(layers.BatchNormalization())
        
        #leakyRelu hay que especificarlo asi
        if config.activation == "leaky_relu":
            model.add(layers.LeakyReLU(negative_slope=0.1))
        else:
            model.add(layers.Activation(config.activation))
        
        #dropout, si esta especificado
        if mejoras.dropout > 0 and i < len(config.capas) - 1: #la ultima capa no se pueden apagar neuronas porque no habría resultado
            model.add(layers.Dropout(mejoras.dropout))

    model.add(layers.Dense(num_clases, activation="softmax")) #capa de salida

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def entrenar_MLP(config: MLPConfig, ea: EarlyStoppingConfig,
                 usar_ea: bool = True,
                 mejoras: MejorasConfig = None):

    #añadimos el early_stopping
    my_callbacks = []
    if usar_ea:
        my_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=ea.monitor,
                patience=ea.paciencia,
                restore_best_weights=ea.restore_weights,
                min_delta=ea.min_delta,
                verbose=ea.verbose
            )
        )

    tf.keras.backend.clear_session() #esto es para limpiar el historial

    if mejoras.usar_augmented_data:

        #usar pipeline dinámico
        train_ds, test_ds, y_test_labels = cargar_datos_augmented(config.batch_size)
        print("Datos cargados")
        input_dim = 32 * 32 * 3
        num_clases = 10
        model = compilar_mlp(config, input_dim, num_clases, mejoras)

        t0 = time.time()
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=config.epochs,
            verbose=config.verbose,
            callbacks=my_callbacks
        )
        train_time = time.time() - t0

        test_loss, test_acc = model.evaluate(test_ds, verbose=config.verbose)
        y_pred_probs = model.predict(test_ds, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        return {
            "train_time": train_time,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history.history,
            "y_pred": y_pred,
            "y_test": y_test_labels.flatten()
        }

    else:
        
        X_train, y_train, X_test, y_test, y_test_labels = cargar_datos()
        input_dim = X_train.shape[1]
        num_clases = y_train.shape[1]

        model = compilar_mlp(config, input_dim, num_clases, mejoras)

        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            batch_size=config.batch_size,
            epochs=config.epochs,
            verbose=config.verbose,
            callbacks=my_callbacks
        )
        train_time = time.time() - t0

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=config.verbose)
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        return {
            "train_time": train_time,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history.history,
            "y_pred": y_pred,
            "y_test": y_test_labels
        }

def ejecutar_mlp(
        config: MLPConfig, ea: EarlyStoppingConfig, repeticiones: int = 5, usar_ea: bool = True,
        mejora: MejorasConfig = None
        ):
    
    mejor_entrenamiento = None
    mejor_test_acc = 0
    historial = []
    for i in range(repeticiones):
        resultado = entrenar_MLP(config,ea,usar_ea,mejora)
        if resultado['test_acc'] > mejor_test_acc:
            mejor_entrenamiento = i
        historial.append(resultado)

    media, max_epochs, num_epocas = calcular_media_historial(historial)
    #mostrar en qué época paró cada entrenamiento (esta en num_epocas)
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
        config_nombre = f"{ea.monitor}_pat_{ea.paciencia}_delta_{ea.min_delta}"
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

def probar_mlp7_mejoras(
        config_base: MLPConfig, ea: EarlyStoppingConfig, mejoras_configs: List[MejorasConfig], 
        capas_a_probar: List[List[int]], activaciones_inicializaciones: List[tuple], repeticiones: int = 5
    ):
    resultados_globales = {}
    
    print(f"Probando mlp7")

    for capas in capas_a_probar:
        
        nombre_capas = str(capas).replace('[', '').replace(']', '').replace(',', '_').replace(' ', '')
        resultados_por_capa = {}  #para no imrpimir todo sjuntos

        for act, init in activaciones_inicializaciones:
            for mejora in mejoras_configs:
                mejora_nombre = mejora.descripcion.replace(' ', '_').replace('+', '')
                config_nombre = f"mlp7_{nombre_capas}_{act}_{init}_{mejora_nombre}"
                
                print(f"Probando: {capas} + {act}/{init} + {mejora.descripcion}")
                
                config = MLPConfig(
                    nombre=config_nombre,
                    capas=capas,
                    activation=act,
                    epochs=config_base.epochs,
                    batch_size=config_base.batch_size,
                    verbose=1,
                    initializer=init
                )
                
                resultado = ejecutar_mlp(config, ea, repeticiones, usar_ea=True, mejora=mejora)
                clave = f"{nombre_capas}_{act}_{mejora_nombre}"
                
                #guardamos en el global y en el de la capa
                resultados_globales[clave] = resultado
                resultados_por_capa[clave] = resultado
        # generar comparativa solo para esta arquitectura
        comparativa_modelos(resultados_por_capa, f"mlp7_comparativa_{nombre_capas}.png")
    
    return resultados_globales

def cargar_datos_cnn():
    """Carga datos para CNN manteniendo estructura espacial (sin aplanar)"""
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return X_train,y_train_cat,X_test,y_test_cat,y_test

def compilar_cnn(config: CNNConfig, input_shape: tuple = (32, 32, 3), num_clases: int = 10): #input shape en cifar 10 es 32x32x3
    """Compila una CNN según la configuración especificada"""
    
    model = models.Sequential(name=config.nombre)
    model.add(layers.Input(shape=input_shape))
    
    # Bloques convolucionales
    for num_filtros, kernel_size in config.bloques_conv:
        # Capa convolucional
        model.add(layers.Conv2D(
            num_filtros,
            kernel_size=kernel_size,
            activation=config.activation,
            kernel_initializer=config.initializer,
            padding='same'
        ))
        
        # MaxPooling
        model.add(layers.MaxPooling2D(pool_size=config.pool_size))
    
    # Aplanar para capas densas
    model.add(layers.Flatten())
    
    # Capa densa oculta
    model.add(layers.Dense(config.dense_units, activation=config.activation,
                          kernel_initializer=config.initializer))
    
    # Capa de salida
    model.add(layers.Dense(num_clases, activation="softmax"))
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def entrenar_CNN(config: CNNConfig, ea: EarlyStoppingConfig, usar_ea: bool = True):
    """Entrena una CNN y devuelve métricas"""
    
    my_callbacks = []
    if usar_ea:
        my_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=ea.monitor,
                patience=ea.paciencia,
                restore_best_weights=ea.restore_weights,
                min_delta=ea.min_delta,
                verbose=ea.verbose
            )
        )
    
    tf.keras.backend.clear_session()
    
    # Cargar datos sin aplanar
    X_train, y_train, X_test, y_test, y_test_labels = cargar_datos_cnn()
    
    model = compilar_cnn(config)
    
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=config.verbose,
        callbacks=my_callbacks
    )
    train_time = time.time() - t0
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=config.verbose)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    return {
        "train_time": train_time,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history.history,
        "y_pred": y_pred,
        "y_test": y_test_labels.flatten()
    }

def ejecutar_cnn(config: CNNConfig, ea: EarlyStoppingConfig, repeticiones: int = 5, 
                 usar_ea: bool = True):
    """Ejecuta múltiples entrenamientos de una CNN y calcula promedios"""
    
    mejor_entrenamiento = None
    mejor_test_acc = 0
    historial = []
    
    for i in range(repeticiones):
        print(f"\nEntrenamiento {i+1}/{repeticiones}")
        resultado = entrenar_CNN(config, ea, usar_ea)
        if resultado['test_acc'] > mejor_test_acc:
            mejor_test_acc = resultado['test_acc']
            mejor_entrenamiento = i
        historial.append(resultado)
    
    media, max_epochs, num_epocas = calcular_media_historial(historial)
    
    # Información de EarlyStopping
    print(f"\nInformación de EarlyStopping")
    print(f"Configuración: {config.nombre}")
    for i, ep in enumerate(num_epocas):
        print(f"Entrenamiento {i+1} paró en la época: {ep}")
    print(f"Máximo de épocas alcanzado: {max_epochs}")
    print(f"Promedio de épocas: {np.mean(num_epocas):.1f}")
    
    epochs_range = np.arange(1, max_epochs + 1)
    
    # Gráfica de evolución
    plottear_graficas(
        epochs_range,
        [media["accuracy"], media["loss"], media["val_accuracy"], media["val_loss"]],
        ["Train accuracy", "Train loss", "Val accuracy", "Val loss"],
        "Evolución entrenamiento CNN",
        f"{config.nombre}_evolucion_entrenamiento.png"
    )
    
    # Matriz de confusión del mejor entrenamiento
    mejor_resultado = historial[mejor_entrenamiento]
    matriz_confusion(
        mejor_resultado["y_test"],
        mejor_resultado["y_pred"],
        config.nombre,
        f"{config.nombre}_matriz_confusion.png"
    )
    
    # Resultados finales
    print(f"\nResultados finales de {config.nombre}")
    print(f"Tiempo de entrenamiento (media): {media['train_time']:.2f}s")
    print(f"Test Accuracy (media): {media['test_acc']:.4f}")
    print(f"Test Loss (media): {media['test_loss']:.4f}")
    print(f"Val Accuracy final (media): {media['val_accuracy'][-1]:.4f}")
    print(f"Val Loss final (media): {media['val_loss'][-1]:.4f}", '\n')
    
    return {
        "test_acc": media['test_acc'],
        "test_loss": media['test_loss'],
        "train_time": media['train_time'],
        "media": media
    }

def comparar_earlystoppings_cnn(config: CNNConfig, ea_configs: List[EarlyStoppingConfig], repeticiones=5):
    """Prueba diferentes configuraciones de EarlyStopping para CNNs"""
    resultados = {}
    
    print("Comparación EarlyStoppings para CNN")
    
    for ea in ea_configs:
        config_nombre = f"{ea.monitor}_pat_{ea.paciencia}_delta_{ea.min_delta}"
        print(f"\nProbando EarlyStopping: {config_nombre}")
        
        config_con_nombre = CNNConfig(
            nombre=f"{config.nombre}_{config_nombre}",
            bloques_conv=config.bloques_conv,
            pool_size=config.pool_size,
            dense_units=config.dense_units,
            activation=config.activation,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            initializer=config.initializer
        )
        
        resultado = ejecutar_cnn(config_con_nombre, ea, repeticiones)
        resultados[config_nombre] = resultado
    
    # Gráfica comparativa
    comparativa_modelos(
        resultados,
        f"{config.nombre}_comparativa_earlystopping.png"
    )
    
    return resultados

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
        ),
        MLPConfig( #mlp 7, usamos el callback 9 mejor callback obetenido hasta ahora, muchas epocas
            nombre="mlp7",
            capas=[80], #mejor configuracion las 80 neuronas en una capa, al menos por ahora        
            activation="leaky_relu",
            epochs=200,
            batch_size=200, #el mejor segun las pruebas
            verbose=1,
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

    activaciones_inicializaciones_mlp7 = [ #las que voy a probar en el mlp7
        #("leaky_relu", "he_normal"), #la mejor hasta ahora
        ("relu", "he_normal"), #dio mal resultado pero por volver a probar
        #sigmoid no la pruebo porque todos los artículos no la recomiendan para mlps sino para las recurrentes
    ]

    mejores_capas = [
        [300,200,100,50,50],
        #[80],
        #[500],
        #[60, 20],
        #[50, 30],
        #[40,40],
        #[100], #añadidas nuevas con 100 neuronas por probar, aunque mis mejores resultados han sido los de arriba
        #[60,40],
        #[40,30,20,10],
        #[30,30,10,10,10,10]
    ]

    mejoras_mlp7 = [
        #sin aumento de data
        #MejorasConfig(dropout=0.0, use_batchnorm=False, usar_augmented_data=False, descripcion="Baseline"),
        #MejorasConfig(dropout=0.1, use_batchnorm=False, usar_augmented_data=False, descripcion="Dropout_0.1_noAug"),
        #MejorasConfig(dropout=0.2, use_batchnorm=False, usar_augmented_data=False, descripcion="Dropout_0.2_noAug"),
        #MejorasConfig(dropout=0.0, use_batchnorm=True, usar_augmented_data=False, descripcion="BatchNorm_noAug"),
        #MejorasConfig(dropout=0.1, use_batchnorm=True, usar_augmented_data=False, descripcion="Dropout_0.1+BatchNorm_noAug"),

        #con aumento de dat
        #MejorasConfig(dropout=0.0, use_batchnorm=False, usar_augmented_data=True, descripcion="Baseline_Aug"),
        #MejorasConfig(dropout=0.1, use_batchnorm=False, usar_augmented_data=True, descripcion="Dropout_0.1_Aug"),
        #MejorasConfig(dropout=0.2, use_batchnorm=False, usar_augmented_data=True, descripcion="Dropout_0.2_Aug"),
        #MejorasConfig(dropout=0.0, use_batchnorm=True, usar_augmented_data=True, descripcion="BatchNorm_Aug"),
        MejorasConfig(dropout=0.05, use_batchnorm=True, usar_augmented_data=True, descripcion="Dropout_0.05+BatchNorm_Aug"),
    ]
    #el augmentation on the fly añade mucho tiempo, realentiza todo mirar de probar otra vez cacheandolo


    #ejecutar_mlp(configs[0],early_stopping_configs[9],5,False)
    #ejecutar_mlp(configs[1],early_stopping_configs[0],5,False)
    #comparar_earlystoppings(configs[1],early_stopping_configs,5)
    #probar_batch_size(configs[2],early_stopping_configs[9],batch_sizes)
    #probar_activaciones_inicializaciones(configs[3],early_stopping_configs[9],activaciones_inicializaciones,5)
    #probar_neuronas(configs[4],early_stopping_configs[9],neuronas,5)
    #probar_capas(configs[5],early_stopping_configs[9],capas,5)
    #probar_mlp7_mejoras(configs[6], early_stopping_configs[9], mejoras_mlp7, mejores_capas, activaciones_inicializaciones_mlp7,1)
    cnn_configs = [

        CNNConfig(
            nombre="cnn1",
            bloques_conv=[(16, 3), (32, 3)],  # (num_filtros, kernel_size)
            pool_size=(2, 2),
            dense_units=100,
            activation="relu",
            epochs=200,
            batch_size=32,
            verbose=1,
            initializer="he_normal"
        )
    ]
    comparar_earlystoppings_cnn(cnn_configs[0],early_stopping_configs,5)
    
    
