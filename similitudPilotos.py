from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# File location and type
file_location = "/FileStore/tables/PilotosV5-4.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

#Para operar sobre ellas, realizamos un cambio de String a Integer/Double en función de lo que necesitemos
df2 = df.withColumn("NGps", df["NGps"].cast(IntegerType())).withColumn("TemporadasCompletadas", df["TemporadasCompletadas"].cast(IntegerType())) \
.withColumn("MediaParrilla", df["MediaParrilla"].cast(DoubleType())).withColumn("MediaCarrera", df["MediaCarrera"].cast(DoubleType())).withColumn("Diferencia", df["Diferencia"].cast(DoubleType())) \
.withColumn("%Victorias", (df["%Victorias"]/100).cast(DoubleType())).withColumn("%Poles", (df["%Poles"]/100).cast(DoubleType())).withColumn("%Podiums", (df["%Podiums"]/100).cast(DoubleType())) \
.withColumn("%VR", (df["%VR"]/100).cast(DoubleType())).withColumn("%Abandonos", (df["%Abandonos"]/100).cast(DoubleType())).withColumn("Nvictorias", df["Nvictorias"].cast(IntegerType())) \
.withColumn("Npoles", df["Npoles"].cast(IntegerType())).withColumn("Npodios", df["Npodios"].cast(IntegerType())).withColumn("NVR", df["NVR"].cast(IntegerType())) \
.withColumn("Nabandonos", df["Nabandonos"].cast(IntegerType())).withColumn("NVueltasDadas", df["NVueltasDadas"].cast(IntegerType())).withColumn("VueltasLiderada", df["VueltasLiderada"].cast(IntegerType())) \
.withColumn("Debut", df["Debut"].cast(IntegerType())).withColumn("Retiro", df["Retiro"].cast(IntegerType())).withColumn("Campeon", df["Campeon"].cast(IntegerType())) \
.withColumn("MejorCoche", df["MejorCoche"].cast(IntegerType())).withColumn("PorcentajeMejorCoche*100", (df["PorcentajeMejorCoche*100"]/100).cast(DoubleType())) \
.withColumn("Puntos/NGps", (df["Puntos/NGps"]/25).cast(DoubleType())).withColumn("GanaCompanero", df["GanaCompanero"].cast(IntegerType())) \
.withColumn("PierdevsCompanero", df["PierdevsCompanero"].cast(IntegerType())).withColumn("%Companero", (df["%Companero"]/100).cast(DoubleType())) \
.withColumn("Ncampeonatos", df["Ncampeonatos"].cast(IntegerType())).withColumn("Puntos", df["Puntos"].cast(IntegerType()))
df2 = df2.withColumn("PorcentajeVueltasLider",(df2["VueltasLiderada"]/df2["NVueltasDadas"]))
df2.printSchema()

df3 = df2[('piloto','PorcentajeMejorCoche*100','%Companero','Puntos/NGps','MediaParrilla','MediaCarrera','%Victorias','%Poles','%Abandonos','%Podiums','%VR','PorcentajeVueltasLider',"NCampeonatos")]
dfPandas = df3.select("*").toPandas()

#Tablas necesarias X e Y con los valores y los pilotos
X, y = dfPandas.iloc[:,1:13].values, dfPandas.iloc[:,0].values

#Escalamos las variables para el PCA
sc = StandardScaler()
X_std = sc.fit_transform(X)
pca = PCA(n_components = 12)
pca.fit(X_std)
X_pca = pca.transform(X_std)

comprobacion = pd.DataFrame(data=X_std, index=y, columns = ('%Victorias','%Poles','%Podiums','%VR','%Abandonos','%Compañero','PorcentajeVueltasLider','Puntos/NGps',
                                                 'PorcentajeMejorCoche*100','MediaParrilla','MediaCarrera','NCampeonatos'))
comprobacion.describe()

df_pca_resultado = pd.DataFrame(data=X_pca[:,0:13], columns = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12"], index=y)
df_pca_resultado.sort_values(by=['PC1'])

df11 = df_pca_resultado.T.corr(method='pearson')
df11

df11 = df11.filter(like='HAMILTON Lewis', axis=0)
df11 =df11.T
df11 = df11*100

df11.sort_values(by=['HAMILTON Lewis'],ascending=False).head(11).iloc[1:]


