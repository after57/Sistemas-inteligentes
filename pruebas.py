import keras
import matplotlib.pyplot as plt
from random import sample

def show_image(imagen, titulo, filename):
    plt.figure()
    plt.suptitle(titulo)
    plt.imshow(imagen, cmap = "Greys")   # CIFAR10 es RGB
    plt.show()
    

(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

for i in sample(list(range(len(X_train))), 3):
    titulo = f"Mostrando imagen X_train[{i}] -- Y_train[{i}] = {Y_train[i]}"
    filename = f"cifar_{i}.png"
    show_image(X_train[i], titulo, filename)
    print("Guardada:", filename)


def plot_curva(Y, titulo, xscale = "linear", yscale = "linear"):
 plt.title(titulo)
 plt.plot(Y)
 plt.xscale(xscale)
 plt.yscale(yscale)
 plt.show()

plot_curva(Y_test[:20], "Etiquetas de los primeros 20 valores")
