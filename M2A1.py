
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col

#Carga de datos del archivo SP1.csv en un dataframe con nombre dffutbol
dffutbol = spark.read.option("header","true").option("inferSchema","true").csv('FileStore/tables/SP1.csv')

#Caso1: En este caso calculamos que porcentaje de los tiros realizados por un equipo van a puerta tanto en casa como fuera. 
#Creación de un dataframe dfcaso1 donde seleccionamos las siguientes columnas
dfcaso1 = dffutbol.select("HomeTeam","AwayTeam","HS","AS","HST","AST")

#Para obtener la solución, realizamos la agrupación por el nombre del equipo y con el método agg realizamos una división de las sumas de tiros a puerta y tiros cuyo resultado aparece en una columna nueva con el nombre del método alias. Por último, utilizamos F.round para redondear a 2 decimales y sort para ordenar por la columna que elijamos de manera descendente.
tirosCasa = dfcaso1.groupby((dfcaso1.HomeTeam).alias("Equipo")).agg(F.round((F.sum(dfcaso1.HST)/F.sum(dfcaso1.HS)*100),2).alias("% CASA")).sort(col("% CASA").desc())
print(tirosCasa)

tirosFuera = dfcaso1.groupby((dfcaso1.AwayTeam).alias("Equipo")).agg(F.round((F.sum(dfcaso1.AST)/F.sum(dfcaso1.AS)*100),2).alias("% FUERA")).sort(col("% FUERA").desc())
print(tirosFuera)

#Caso 2: Numero de faltas que un equipo tiene que realizar para que le saquen tarjeta amarilla.
#En primer lugar, seleccionamos un dataframe con los datos que necesitamos y sustituimos los posibles valores nulos por 0.
df22 = dffutbol.select("HomeTeam","AwayTeam","HY","AY","HF","AF")
df22 = df22.fillna(0)

#Sumamos las faltas totales en casa y fuera de cada equipo
faltasCasa = df22.groupby((df22.HomeTeam).alias("Equipo")).agg(F.sum(df22.HF).alias("FC"))
faltasFuera =df22.groupby((df22.AwayTeam).alias("Equipo")).agg(F.sum(df22.AF).alias("FF"))

#Creamos un nuevo dataframe con el nombre de cada equipo y el número de faltas totales mediante el uso de join.
df33 = faltasCasa.join(faltasFuera,"Equipo",how='left_outer')
df33 = df33.withColumn("Faltas", df33.FC + df33.FF)

#Hacemos exactamente el mismo procedimiento con las tarjetas amarillas y unimos ambos dataframes creados.
tarjetasCasa = df22.groupby((df22.HomeTeam).alias("Equipo")).agg(F.sum(df22.HY).alias("TC"))
tarjetasFuera =df22.groupby((df22.AwayTeam).alias("Equipo")).agg(F.sum(df22.AY).alias("TF"))
df44 = tarjetasCasa.join(tarjetasFuera,"Equipo",how='left_outer')
df44 = df44.withColumn("Tarjetas", df44.TC + df44.TF)
dffinal = df44.join(df33,"Equipo",how='left_outer')

#Para finalizar, realizamos el calculo haciendo redondeo a 2 decimales y lo añadimos a una columna llamada Faltas/tarjeta.
dffinal = dffinal.withColumn("Faltas/Tarjeta", F.round((dffinal.Faltas/dffinal.Tarjetas),2))
dffinal = dffinal.select("Equipo","Faltas/Tarjeta").sort(col("Faltas/Tarjeta").desc())
print(dffinal)

#Caso 3: Nº veces que cada equipo ha dejado su portería a 0
#Creamos 2 dataframes que filtramos por goles encajados = 0 tanto en casa como fuera.
df3 = dffutbol.select("HomeTeam","AwayTeam","FTHG","FTAG")
df3 = df3.filter(df3["FTAG"] < 1)

#A la columna count() resultante la renombramos con el método withColumnRenamed.
vecesCasa = df3.groupBy((df3.HomeTeam).alias("Equipo")).count().withColumnRenamed('count','PIC')

#Repetimos el proceso para partidos fuera de casa.
df4 = dffutbol.select("HomeTeam","AwayTeam","FTHG","FTAG")
df4 = df4.filter(df4["FTHG"]<1)
vecesFuera = df4.groupBy((df4.AwayTeam).alias("Equipo")).count().withColumnRenamed('count','PIF')

#Realizamos un join para unir ambas columnas resultantes y creamos una nueva "Porterias imbatidas" con la suma de ambas.
df5 = vecesCasa.join(vecesFuera,"Equipo",how='left_outer')
df5 = df5.withColumn('Porterias imbatidas',vecesCasa.PIC+vecesFuera.PIF)
df5 = df5.select("Equipo","Porterias imbatidas").sort(col("Porterias imbatidas").desc())
print(df5)

#Punto 5: Calculo de 4 primeros clasificados si tenemos en cuenta resultados hasta el descanso.
#Creamos dataframe dfcaso4 con solo 2 columnas: Equipo local y resultado.
dfcaso4 = dffutbol.select("HomeTeam","HTR")

#Realizamos un replace de cada posible valor texto por valor numérico
dfcaso4 = dfcaso4.replace('H','3')
dfcaso4 = dfcaso4.replace('A','0')
dfcaso4 = dfcaso4.replace('D','1')

#Sumamos los puntos de cada equipo local agrupados en una columna PL.
dfcaso4F = dfcaso4.groupBy((dfcaso4.HomeTeam).alias("Equipo")).agg(F.sum(dfcaso4.HTR)).withColumnRenamed('sum(HTR)','PL')

#Repetimos proceso para equipos visitante.
dfcaso41 = dffutbol.select("AwayTeam","HTR")
dfcaso41 = dfcaso41.replace('A','3')
dfcaso41 = dfcaso41.replace('H','0')
dfcaso41 = dfcaso41.replace('D','1')
dfcaso41F = dfcaso41.groupBy((dfcaso41.AwayTeam).alias("Equipo")).agg(F.sum(dfcaso41.HTR)).withColumnRenamed('sum(HTR)','PF')

#Nuevo dataframe dffinal5 donde unimos las columnas de ambos dataframes.
dffinal5 = dfcaso4F.join(dfcaso41F,"Equipo",how='left_outer')

#Creamos una nueva columna "PuntosTotales" donde sumamos el valor de ambas columnas anteriores y la ordenamos en orden descendente para obtener los 4 primeros clasificados
dffinal5 = dffinal5.withColumn("PuntosTotales", dffinal5.PF + dffinal5.PL).select("Equipo","PuntosTotales").sort(col("PuntosTotales").desc()).show(4)


