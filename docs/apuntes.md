Para el del batch size, a menor numero de batch size mas actualizaciones del backpropagation se hacen,
es decir es 50000 (el numero de imagenes) entre el batch size * el numero de epocas, a igualdad de epocas el efecto del batch size se nota mas.

Mejor epoca mlp2 es 10 puesto que el callback de media para ahi en 5 ejecuciones

Para mi mlp2 lo que mejor me ha ido ha sido con early-stopping val loss y 5 epocas de paciencia,
cuando deja de mejorar restaura los mejores pesos de las 5 epocas que ese monitorean


Referencia: https://www.google.com/search?q=best+activation+layers+for+hidden+layers+mlps&oq=best+activation+layers+for+hidden+layers+mlps&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIHCAEQABjvBTIKCAIQABiABBiiBDIHCAMQABjvBTIHCAQQABjvBTIHCAUQABjvBdIBCDg1NzhqMGoxqAIAsAIA&sourceid=chrome&ie=UTF-8

https://medium.datadriveninvestor.com/deep-learning-best-practices-activation-functions-weight-initialization-methods-part-1-c235ff976ed
https://blog.paperspace.com/the-absolute-guide-to-keras/
https://medium.com/data-science/data-augmentation-and-handling-huge-datasets-with-keras-a-simple-way-240481069376
https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/