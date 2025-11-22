Para el del batch size, a menor numero de batch size mas actualizaciones del backpropagation se hacen,
es decir es 50000 (el numero de imagenes) entre el batch size * el numero de epocas, a igualdad de epocas el efecto del batch size se nota mas.

Mejor epoca mlp2 es 10 puesto que el callback de media para ahi en 5 ejecuciones

Para mi mlp2 lo que mejor me ha ido ha sido con early-stopping val loss y 5 epocas de paciencia,
cuando deja de mejorar restaura los mejores pesos de las 5 epocas que ese monitorean