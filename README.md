# Algoritmo de clasificación de cancer de lesiones en exámenes mamográficos.

Este proyecto persigue el objetivo de crear una herramienta que permita clasificar las lesiones 
presentes en imágenes mamográficas como benignas o malignas.

Para la realización de esta aplicación se han utilizado 4 arquitecturas que componen el estado 
del arte en tareas de clasificación de imagen, como son: _VGG16_, _ResNet50_, _DenseNet121_ y _InceptionV3_.
Las predicciones realizadas por cada arquitectura son combinadas en un _Random Forest_ para obtener la
predicción final de cada instancia (probabilidad de que un cáncer sea maligno).

Para la realización de la aplicación, se han utilizado 3 bases de datos distintas:
- CBIS-DDDSM: Disponible en https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
- INBreast: Disponible en https://www.kaggle.com/martholi/inbreast?select=inbreast.tgz
- MIAS: Disponible en https://www.kaggle.com/kmader/mias-mammography/version/3?select=Info.txt

Este repositorio pretende servir de base para que otras personas puedan aprovechar el trabajo realizado
y unirse en la causa para la lucha contra el cáncer de seno. De este modo, a continuación se detallará la estructura
del repositorio así como los objetivos de cada módulo o paquete.

- `bin`: Contiene los archivos necesarios para crear una interfaz gráfica a partir de la librería `Pyqt5`. Entre estos
destaca la carpeta `hooks` con las dependencias necesarias a instalar en la aplicación.

- `notebooks`: Contiene algunos análisis _adhoc_ para la realización de la herramienta (procesado de imágenes o creación
de la combinación secuencial de clasificadores).

- `src`: Contiene los paquetes y los _scripts_ principales de ejecución de código. Este paquete se divide en:
    - `algoriths`: Módulo utilizado para crear las redes neuronales de clasificación y de segmentación (_on going_).
    **En este módulo se deberían de añadir todas aquellas arquitecturas de red nuevas a incluir**. Por otra parte,
    también existen scripts para la creación secuencial de clasificadores a partir de un _Random Forest_. La generación
    de nuevos algorítmos podría introducirse en este script. 
    - `breast_cancer_dataset`: Módulo que contiene los scripts utilizados para realizar el procesado de datos de 
    cada set de datos individual (CBIS, MIAS e INBreast). Estos scripts se encuentran en el paquete `databases` del 
    módulo, de modo que **para cualquier base de datos que se desee añadir, será necesario introducir su procesado en este
    paquete**. Por otra parte, el script _database_generator.py_ crea el set de datos utilizado por los algorítmos de 
    _deep learning_ utilizados uniendo cada base de datos individual contenida en el paquete `databases`. 
    Asimismo, se aplican técnicas de _data augmentation_ y se realiza el split de datos en entrenamiento y validación.
     - `data_viz`: módulo utilizado para generar visualizaciones de los resultados obtenidos por las redes.
     - `preprocessing`: módulo que contiene las funciones de preprocesado genéricas aplicadas a todos los conjuntos de 
     datos. Además, contiene las funcionalidades necesarias para estandarizar las imágenes a formato _png_ o _jpg_.
     **Cualquier procesado nuevo a añadir, deberá hacerse en este módulo**.
     - `static`: módulo que contiene los archivos estáticos utilizados para la creación de la interfaz gráfica del 
     programa como ficheros _.css_, _.html_ e imágenes.
     - `user_interace`:  módulo utilizado para crear la aplicación `Pyqt5` de clasificación de imágenes de seno.
     - `utils`: módulo genérico en el cual configurar las rutas de las bases de datos dentro del entorno local desde 
     dónde se esté ejecutando el aplicativo, así como la configuración de los hiperparámetros de las redes neuronales. 
     - `main_train.py`: script utilizado para realizar generar el pipeline de entrenamiento, desde la obtención de datos
     hasta la creación y el entrenamiento de cada modelo.
     - `main.py`: script utilizado para lanzar la aplicación final realizada.
     
Juntamente con los módulos contenidos en esta descripción, se crearán un conjunto de carpetas adicionales. Estas carpetas
no están contenidas en el repositorio por motivos de capacidad de almacenaje. A continuación se detallan los módulos y 
sus objetivos:

- `logging`: Carpeta que contendrá los logs de ejecuciones del programa, como por ejemplo los errores producidos durante
el procesado de las imagenes.
- `models`: Carpeta que contendrá los modelos almacenados juntamente con las predicciones realizadas durante el entrenamiento. 
- `data`: Carpeta que contendrá las imagenes de cada set de datos convertidas (sub-directorio _01_CONVERTED_) y 
procesadas (sub-directorio _02_PROCESED_). Esta carpeta tiene el objetivo de reiterar el proceso de procesado de imagenes
una vez realizado.