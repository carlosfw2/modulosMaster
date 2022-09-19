from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import *
import pandas as pd

# File location and type
file_location = "/FileStore/tables/PilotosV5.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

#The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

#Vemos si hay valores nulos o NaN en cada una de las columnas
from pyspark.sql.functions import col,sum,expr, when, isnan, count
display(df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]))

#Tipo de cada una de las columnas
df.printSchema()

#Para operar sobre ellas, realizamos un cambio de String a Integer/Double en función de lo que necesitemos
df2 = df.withColumn("NGps", df["NGps"].cast(IntegerType())).withColumn("TemporadasCompletadas", df["TemporadasCompletadas"].cast(IntegerType())) \
.withColumn("MediaParrilla", df["MediaParrilla"].cast(DoubleType())).withColumn("MediaCarrera", df["MediaCarrera"].cast(DoubleType())).withColumn("Diferencia", df["Diferencia"].cast(DoubleType())) \
.withColumn("%Victorias", df["%Victorias"].cast(DoubleType())).withColumn("%Poles", df["%Poles"].cast(DoubleType())).withColumn("%Podiums", df["%Podiums"].cast(DoubleType())) \
.withColumn("%VR", df["%VR"].cast(DoubleType())).withColumn("%Abandonos", df["%Abandonos"].cast(DoubleType())).withColumn("Nvictorias", df["Nvictorias"].cast(IntegerType())) \
.withColumn("Npoles", df["Npoles"].cast(IntegerType())).withColumn("Npodios", df["Npodios"].cast(IntegerType())).withColumn("NVR", df["NVR"].cast(IntegerType())) \
.withColumn("Nabandonos", df["Nabandonos"].cast(IntegerType())).withColumn("NVueltasDadas", df["NVueltasDadas"].cast(IntegerType())).withColumn("VueltasLiderada", df["VueltasLiderada"].cast(IntegerType())) \
.withColumn("Debut", df["Debut"].cast(IntegerType())).withColumn("Retiro", df["Retiro"].cast(IntegerType())).withColumn("Campeon", df["Campeon"].cast(IntegerType())) \
.withColumn("MejorCoche", df["MejorCoche"].cast(IntegerType())).withColumn("PorcentajeMejorCoche*100", df["PorcentajeMejorCoche*100"].cast(DoubleType())) \
.withColumn("Puntos/NGps", df["Puntos/NGps"].cast(DoubleType()))

df2.printSchema()

#Correlación de cada una de las variables con la variable objetivo.
#Queremos hallar que pilotos desde el año 2000 deberían ser campeones del mundo por las estadísticas que tiene actualmente
import six
for i in df2.columns:
    if not( isinstance(df2.select(i).take(1)[0][0], six.string_types)):
        print( i,"Su correlación con Campeon es:", df2.stat.corr('Campeon',i))

#Dibujo de la matriz
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14,13))
sns.heatmap(df2.select([c for c in df2]).toPandas().corr(),linewidths=0.2,vmax=1.0, square=True, linecolor="white", annot=True)
display(plt.show())
plt.gcf().clear()
plt.clear()

#Variables que vamos a utilizar para predecir
#Preferimos las variables en % en función de las carreras disputadas, por eso no elegimos las 4 variables posteriores(NVictorias, Npoles...)
#No escogemos las 2 primeras variables para que no influya si ha tenido o no el mejor coche.
features = ['%Victorias','%Poles','%Podiums','%VR','Puntos/NGps']

# Preparar los datos para podérselos pasar a los distintos modelos
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

# Definimos un array donde vas a ir añadiendo todos los datos que le vamos a pasar a nuestra Tuberia
stages = [] 
# A continuación definimos otro array con las variables de tipo cadena que tenemos
ColumnasCategoricas = ['piloto']

for Col in ColumnasCategoricas:
# las indexamos con el método StringIndexer
    stringIndexer = StringIndexer(inputCol=Col, outputCol=Col + "Index")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[Col + "classVec"])
# Lo añadimos para que a continuación se ejecute en la tuberia.
    stages += [stringIndexer, encoder]

# Convertir la etiqueta en índices de etiquetas usando el StringIndexer y se añaden al array
labelstring = StringIndexer(inputCol="Campeon", outputCol="label")
stages += [labelstring]

# Pasamos en un vector los datos numéricos que finalmente le vamos a pasar al modelo y los añadimos a las etapas de la tuberia 
vectorAssembler = VectorAssembler (inputCols = features, outputCol = 'features')  
stages += [vectorAssembler]

# Tenemos ya nuestra variable objetivo renombrada como label y todas las variables predictoras que son de tipo numérico en un vector
# y la variable nombre en un StringIndexer

#Definimos los stages por los que va a pasar
from pyspark.ml import Pipeline
pipeline = Pipeline().setStages(stages)
pipelineModel = pipeline.fit(df2)
dfFinal = pipelineModel.transform(df2)

#DATAFRAME CON PILOTOS QUE QUEREMOS TESTEAR
df_filtered= dfFinal.filter((dfFinal.Debut == "2010")|(dfFinal.Debut == "2011")|(dfFinal.Debut == "2012")|(dfFinal.Debut == "2013")|(dfFinal.Debut == "2014")
                            |(dfFinal.Debut == "2015")|(dfFinal.Debut == "2016")|(dfFinal.Debut == "2017")|(dfFinal.Debut == "2018")|(dfFinal.Debut == "2019")|(dfFinal.Debut == "2020")
                           |(dfFinal.Debut == "2009")|(dfFinal.Debut == "2008")|(dfFinal.Debut == "2007")|(dfFinal.Debut == "2006")|(dfFinal.Debut == "2005")|(dfFinal.Debut == "2004"))

display(df_filtered)

#DATAFRAME CON PILOTOS QUE QUEREMOS ENTRENAR
dfFinal = dfFinal.filter("not (Debut = '2004')")
dfFinal = dfFinal.filter("not (Debut = '2005')")
dfFinal = dfFinal.filter("not (Debut = '2006')")
dfFinal = dfFinal.filter("not (Debut = '2007')")
dfFinal = dfFinal.filter("not (Debut = '2008')")
dfFinal = dfFinal.filter("not (Debut = '2009')")
dfFinal = dfFinal.filter("not (Debut = '2010')")
dfFinal = dfFinal.filter("not (Debut = '2011')")
dfFinal = dfFinal.filter("not (Debut = '2012')")
dfFinal = dfFinal.filter("not (Debut = '2013')")
dfFinal = dfFinal.filter("not (Debut = '2014')")
dfFinal = dfFinal.filter("not (Debut = '2015')")
dfFinal = dfFinal.filter("not (Debut = '2016')")
dfFinal = dfFinal.filter("not (Debut = '2017')")
dfFinal = dfFinal.filter("not (Debut = '2018')")
dfFinal = dfFinal.filter("not (Debut = '2019')")
dfFinal = dfFinal.filter("not (Debut = '2020')")

#Escogemos el dataframe completo para entrenar
(train, test) = dfFinal.randomSplit ([1.0, 0.0],  1234) 
print ("Recuento de conjunto de datos de entrenamiento:" + str (train.count())) 
print ("Recuento de conjunto de datos de prueba: "+ str (test.count()))

#3) Implementación y análisis de algún modelo de clasificación desarrollados en la API de Spark API de Spark:

#RANDOM FOREST
#Construimos el modelo de regresión logística sobre el conjunto de datos de entrenamiento (train)
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees = 20, maxDepth=2)

#entrenar el modelo rf sobre el subconjunto train
rfModel = rf.fit(train)

#Predecimos ,usando el modelo entrenado con el dataframe train, el dataframe con los pilotos del 2000 hacia delante
predict = rfModel.transform (df_filtered)
display(predict.select("piloto","label","prediction"))

#Tras la ejecución, vemos que Max Verstappen y Valteri Bottas nos predice que pueden ser campeones del mundo aunque aún no lo han sido.
#Valteri Bottas lleva varios años con el mejor coche de la parrilla lo que provoca sus altos números
#Max Verstappen no tiene un coche tan bueno(2-3º mejor coche) pero es el futuro de la F1 y uno de los mejores pilotos actuales

#Evaluamos la efectividad de la predicción realizada.
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
score = predict.select("prediction", "label")
accuracyscore = MulticlassClassificationEvaluator(predictionCol="prediction",metricName="accuracy")
rate = accuracyscore.evaluate(score)*100
print("accuracy: {}%" .format(round(rate,2)))

#Area ROC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predict, {evaluator.metricName: "areaUnderROC"})))

# Realizamos una validación cruzada para garantizar que los resultados obtenidos son independientes de
# la partición entre datos de entrenamiento y prueba.
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# Ejecución de la validación cruzada
cv = CrossValidator(estimator=rf, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, numFolds=3)

cvModel = cv.fit(train)

# Calculamos la exactitud del modelo tras la validación cruzada 
predicts = cvModel.transform(df_filtered)
score = predicts.select("prediction", "label")
accuracyscore = MulticlassClassificationEvaluator(predictionCol="prediction",metricName="accuracy")
rate = accuracyscore.evaluate(score)*100
print("accuracy: {}%" .format(round(rate,2)))
accuracy = accuracyscore.evaluate(predicts)
print("Test Error = %g" % (1.0 - accuracy))

#4) Implementación y análisis de algún modelo de regresión desarrollados en la API de Spark API de Spark:

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression

#Elegimos las variables con las que vamos a estudiar el modelo, en nuestro caso Campeon
assembler = VectorAssembler(
    inputCols=["Campeon"],
    outputCol="features")

pipeline = Pipeline(stages=[assembler])
dfej4 = pipeline.fit(df2).transform(df2)

glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3,
                                  featuresCol="features", labelCol="Campeon")

# Entrenamos el modelo
model = glr.fit(dfej4)

# Imprimimos los coeficientes y el término independiente (intercept) para el modelo de regresión lineal
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Resumen del modelo ejecutado sobre el conjunto de entrenamiento
summary = model.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()

#5)Implementación y análisis de algún modelo de agrupación desarrollados en la API de Spark API de Spark:

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler


# Transforma todas las características en un vector usando VectorAssembler
# Se han descartado algunas características por no encajar en el objetivo del algoritmo de k-means que se busca
numericCols = ['PorcentajeMejorCoche*100','Puntos/NGps','MediaParrilla','MediaCarrera','Diferencia','%Victorias','%Poles','%Podiums','%VR','%Abandonos']

assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
dfej5 = assembler.transform(df2)

# Ajustar el modelo k-means
kmeans = (KMeans()
          .setK(8)
          .setSeed(123)
          .setFeaturesCol("features")
          .setInitMode("random")
          .setMaxIter(10))
model = kmeans.fit(dfej5)

# Realizar predicciones
predictions = model.transform(dfej5)

# Evaluación de los clusters evaluando la silueta obtenida
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silueta = " + str(silhouette))

# Mostrar centroides.
ctr = []
centers = model.clusterCenters()
print("Centroides: ")
for center in centers:
    ctr.append(center)
    print(center)

#Convertimos el modelo predictions en un dataframe de pandas y vemos en la columna Prediction el cluster al que pertenece cada piloto
pandasDF=predictions.toPandas()
pandasDF2 = pandasDF[['piloto','Campeon','%Podiums','PorcentajeMejorCoche*100','prediction']]
pandasDF2.sort_values(by=['prediction'], inplace=True)
display(pandasDF2)

#Sorprendente Stirling Moss como único representante en el cluster 7 que no es campeon. 
#James Hunt en el cluster 0, único piloto que no ha tenido el mejor coche en ningún año.

#6 Uso de  y análisis de algún algoritmo de optimización de parámetros de alguno de los tres apartados anteriores, este conjunto de algoritmos se deben encontrar desarrollados en la API de Spark API de Spark
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# Preparamos el train y test a partir del dataframe del ejercicio 3.
train, test = df_filtered.randomSplit([0.9, 0.1], seed=12345)

lr = LinearRegression(maxIter=10)

# TrainValidationSplit prueba todas las combinaciones de valores posibles y determina  el mejor modelo
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

# Se ha escogido la linea de regresion
# Introducimos el Estimator y evaluator
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% de los datos se van a entrenar y 20% validación
                           trainRatio=0.8)

# Elegimos los mejores parametros
model = tvs.fit(train)

# Se hace la predicion en la parte test
model.transform(test)\
    .select("features", "label", "prediction")\
    .show()


