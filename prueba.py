import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())  # en TF 2.16 quizá deprecated, pero muchas veces sigue funcionando
