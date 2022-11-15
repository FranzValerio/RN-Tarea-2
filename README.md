# Transfer Learning: Reconocimiento facial usando el CelebA dataset
## Francisco Valentín Valerio López
## Redes Neuronales Artificiales - Dr. Jorge Velázquez Castro

### Introducción

El **Transfer Learning** (o Transferencia del aprendizaje) es una metodología del Deep Learning en la que un modelo que se ha entrenado para una tarea en específico se utiliza como punto de partida para un modelo que realiza otra tarea similar. Suele ser mucho más fácil y rápido actualizar y volver a entrenar una red con Transfer Learning que entrenarla desde cero. Esta metodología se utiliza con frecuencia en aplicaciones de detección de objetos, reconocimiento de imágenes, reconocimiento de voz, entre otras.

Esta técnica se utiliza habitualmente ya que:

- Permite entrenar modelos con pocos datos de entrenamiento.
- Ayuda a reducir el tiempo de entrenamiento y los recursos informáticos. Los pesos no se aprenden desde cero, dado que el modelo previamente entrenado ya ha aprendido los pesos a partir de entrenamientos previos.
- Permite aprovechar arquitecturas de modelos desarrolladas por la comunidad de investigadores de Deep Learning, incluidas arquitecturas de uso habitual como GoogLeNet y ResNet.

### Flujo de trabajo del Transfer Learning

A pesar de la existencia de una gran variedad de arquitecturas y aplicaciones de Transfer Learning, la mayoría de los flujos de trabajo siguen una serie de pasos en común.

1. Se toman las capas de un modelo previamente entrenado.
2. Se "congelan" las capas del modelo anterior, para evitar destruir la información que contienen durante las futuras rondas de entrenamiento.
3. Se añaden nuevas capas entrenables sobre las capas congeladas. Estas van a aprender a convertir las características del modelo pre entrenado en predicciones sobre un nuevo conjunto de datos.
4. Se entrenan las nuevas capas en el nuevo conjunto de datos.

Podemos resumir el flujo en el siguiente esquema.

![TL-Workflow](https://la.mathworks.com/discovery/transfer-learning/_jcr_content/mainParsys/image_226673988_copy.adapt.full.medium.jpg/1634621566490.jpg)

La principal ventaja de este método es que podemos lograr mayor precisión en menos tiempo.

![Improvements-TL](https://la.mathworks.com/discovery/transfer-learning/_jcr_content/mainParsys/image_226673988_copy_200198420.adapt.full.medium.jpg/1634621566519.jpg)

### Problema específico: Reconocimiento facial

#### Estrategia

Para esta tarea, el objetivo es entrenar una red neuronal artificial capaz de reconocer un rostro en particular. Específicamente, mi rostro. 😮.
¿Cómo se abordó? La idea general fue utilizar la técnica de Transfer Learning, primero entrenando una red neuronal convolucional que aprenda a reconocer atributos de imágenes de rostros de un gran conjunto de datos. Ese modelo servirá como nuestra base.

Posteriormente, se debe construir un modelo usando las capas convolucionales del modelo entrenado pero quitando el clasificador para determinar sólo si es el rostro o no. Para ello es necesario congelar los pesos del modelo base para después entrenar el nuevo modelo.

#### Datos

Específicamente. Se utilizó el conjunto de imágenes [CelebFaces Attributes (CelebA)](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) que consta de `202,599` imágenes de rostros de celebridades, acompañadas de `40` atributos binarios (`1`o `-1`). Este conjunto ha sido pre-procesado de tal forma que las imágenes de los rostros están recortadas y alineadas. Además, se incluye un archivo en formato `csv`que contiene las etiquetas de los atributos para cada imagen. Se trata de un archivo con 202,600 filas y 41 columnas, donde los `1` representan un atributo *positivo* (que sí tiene el atributo) y los `-1` representan un atributo *negativo*.


#### Modelo
Como se mencionó anteriormente, se utilizó una red neuronal convolucional, debido a su gran desempeño en tareas de identificación de patrones en imágenes. La arquitectura propuesta se enfocó en lograr reducir el número de parámetros entrenables, debido a su alto costo computacional.

Se utilizó un modelo de capas secuencial, utilizando `tf.keras.Sequential()`. La arquitectura de red seleccionada consiste en una red neuronal con:

- 1 Capa de entrada de `shape = ((192,192,3))`.
- 6 Capas ocultas intermedias convolucionales Conv2D de 32 filtros, con tamaño de kernel igual a 3 y con función de activación `ReLU`.
- 6 Capas ocultas de MaxPooling2D asociadas a sus capas convolucionales.
- 1 Capa Flatten (aplanamiento) después de las Conv2D y sus MaxPooling.
- 1 Capa Dense de 128 neuronas y función de activación `Sigmoid`.
- 1 Capa de salida Dense con 40 neuronas de salida (una para cada etiqueta).

El resumen de la arquitectura propuesta se puede apreciar con detalle en la siguiente imagen:

![arch_cnn](https://user-images.githubusercontent.com/67620297/201844565-2ced5e39-f6e8-4ddb-af0e-208566041fdd.png)

#### Configuraciones de compilación

En cuanto a la compilación del modelo, se probó con varias configuraciones para la opción del optimizador, la función de costo y la métrica utilizada para evaluar el desempeño.

1. `Modelo 1` *Optimizador*: **ADAM**, *Loss*: **Binary Cross Entropy**, *Métrica*: **Binary Accuracy**.
2. `Modelo 2` *Optimizador*: **SGD**, *Loss*: **Mean Squared Error (MSE)**, *Métrica*: **Binary Accuracy**.
3. `Modelo 3` *Optimizador*: **ADAM**, *Loss*: **Cross Entropy**, *Métrica*: **Binary Accuracy**.


#### Entrenamiento

Posteriormente, se entrenó el modelo definido y se registraron los resultados obtenidos usando `Tensorboard`. Cada una de las propuestas en la compilación fueron entrenadas durante `30 epochs`, teniendo un tiempo promedio de época de `288 s`, haciendo un tiempo de entrenamiento de cada modelo de aproximadamente 3 horas. 

Los resultados obtenidos para el valor de la función de costo y las distintas métricas se presentan a continuación:

##### `Modelo 1`

Loss

![imagen](https://user-images.githubusercontent.com/67620297/201852567-9629fe9e-4553-4ab1-a329-b82cd29a9a69.png)


Binary Accuracy

![imagen](https://user-images.githubusercontent.com/67620297/201852526-24b38eac-9a8a-4b44-8d97-15da7f5fccce.png)


##### `Modelo 2`

Loss

![imagen](https://user-images.githubusercontent.com/67620297/201853208-26392833-cbbf-4056-86d4-e2479df08ce8.png)


Binary Accuracy

![m2_acc](https://user-images.githubusercontent.com/67620297/201852299-52a2fb25-7b04-4bef-85e8-ae06591e4ad3.png)


##### `Modelo 3`

Loss
![imagen](https://user-images.githubusercontent.com/67620297/201852935-ba2000d4-c7a0-4fb5-97db-73e228a73328.png)


Binary Accuracy
![imagen](https://user-images.githubusercontent.com/67620297/201852978-91d95fff-0a85-40f4-ab3d-78ede0093e6c.png)

Podemos observar que el modelo que tuvo mejor desempeño fue el segundo, ya que se obtuvo un valor aproximado de `loss = 0.1358` y `binary_accuracy = 0.8057`.

#### Transfer Learning

Una vez alcanzado el mejor resultado de esos 3 modelos, se guardó el segundo en formato `H5` y se utilizó para entrenar la red de reconocimiento facial.

Dado que no soy muy fotogénico, el nuevo conjunto de entrenamiento dispone de muy pocas imágenes (menos de 200), por lo que la técnica de Transfer Learning es adecuada. 

Para poder comenzar con esta parte, tuve que elaborar una carpeta, `fotos` con subdirectorios, llamados `train`, `validation` y `test`, en los que manualmente coloqué a su vez dos carpetas con las dos categorías posibles `paco` y `no paco`. Cade destacar que las fotos que no correspondían a mi persona fueron una muestra proporcional extraída aleatoriamente del CelebA. 

Así mismo, utilicé la técnica de `Data Augmentation` para incrementar el número de imágenes de entrenamiento de forma sintética, a través de transformaciones aleatorias que fueron implementadas a través de la API `keras.preprocessing.image.ImageDataGenerator()`. Aprovechando esa funcionalidad, tambié reescalé las imágenes y les asigné las mismas dimensiones que las del CelebA.

El modelo utilizado para entrenar estas imágenes se construyó encima del `Modelo 2` previamente entrenado. Para ello, lo cargué a mi código y comencé a construir una nueva red secuencial, con la misma arquitectura que el anterior, excepto por las últimas dos capas de clasificación, a saber añadí:

- 1 Capa Densa de 128 neuronas con función de activación `sigmoid`.
- 1 Capa Densa de salida de 1 neurona con función de activación `linear`.

Uno de los pasos fundamentales en el flujo del Transfer Learning es, como ya comentamos, mantener la información aprendida por el primer modelo. Esto se consigue fácilmente al pedirle a TensorFlow que no entrene ciertas capas a voluntad. En mi caso, solicité que mantuviera la información heredada por las capas convolucionales del modelo, y que sólo entrenara las dos últimas capas de clasificación.

El resumen de este modelo se presenta a continuación.

![imagen](https://user-images.githubusercontent.com/67620297/201856455-458bcaab-67d7-46b0-9148-4627c4e6874f.png)

Notemos que a pesar de tener la misma arquitectura, el número de parámetros entrenables es muchísimo menor (`4,353`) comparados al número de parámetros entrenables de la primera red (`56,520`). 

Para la compilación del modelo, se optó por utilizar la configuración del `Modelo 2`, modificando un poco el `learning rate` del optimizador, poniendolo como `learning_rate = 0.00001`, para lograr que la función de costo descendiera más rápidamente. 

Finalmente, el modelo fue entrenado durante `150 epochs` con los datos de entrenamiento y de validación que corresponden a mi rostro y a las muestras del CelebA. 

Después del entrenamiento se obtuvieron los siguientes resultados.

Loss

![imagen](https://user-images.githubusercontent.com/67620297/201857526-fdf1b216-9ee7-4088-9bc7-47c17115b7cf.png)

Binary Accuracy

![imagen](https://user-images.githubusercontent.com/67620297/201857608-c7ec3679-5da0-465d-8a53-b46ee57c9bac.png)

Notamos que en cuanto a la función de costo, se tiene un buen resultado ya que a la época 150 se obtiene el valor `loss = 0.2469`. Sin embargo, la exactitud sufrió considerablemente, teniendo un comportamiento irregular un poco antes de la época 40, para después estabilizarse y estancarse en `binary_accuracy = 0.5457`. Un valor demasiado bajo. 😞

#### Conclusiones

Desafortunadamente, no se cumplieron las expectativas de mi modelo. El entrenamiento del clasificador final, no dio resultados aceptables para la exactitud de la clasificación de mi rostro. Esto puedo atribuirselo a la poca cantidad de datos de entrenamiento así como de un posible mal enfoque en la resolución de esta tarea.

No obstante, este proyecto me sirvió para comprender el potencial que tienen los métodos de aprendizaje profundo, en específico esta metodología del Transfer Learning. Durante el proceso de búsqueda de información, pude comprobar que dentro de la comunidad científica de las redes neuronales y el Machine Learning, existen arquitecturas específicas y modelos pre-entrenados sobre conjuntos de imágenes muchísimo más grandes que el conjunto que se utilizó para estre proyecto (ImageNet $\approx$ 1.2 mil millones de imágenes y 1000 características). Por lo que es posible aplicar este mismo método en otras tareas que se lleguen a presentar en el futuro, haciendo uso de los buenos resultados que la comunidad ha ido obteniendo con el paso del tiempo.

Finalmente puedo decir, que para un primer acercamiento a las Redes Neuronales Convolucionales, este proyecto fue retador, así como divertido y un poco desesperante en ocasiones. Me llevo el conocimiento obtenido durante cada tropiezo, cada error de compilación, y cada re instalación de TensorFlow que tuve que hacer, debido a mi inexperiencia y a la fe ciega de soluciones de StackOverflow. 😃
