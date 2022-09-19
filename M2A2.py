
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.types import *

#Cargamos los 4 dataframes separando cada campo por el delimitador ;
#Dataframe 1
ruta_fichero = "/FileStore/tables/TAL_ALD-e8f12.csv"
tipo_fichero = "csv"

#Opciones de lectura
inferir_esquema = "true"
primer_fila_cabecera = "true"
delimitador = ";"

df1 = (spark.read.format(tipo_fichero)
      .option("inferSchema", inferir_esquema)
      .option("header", primer_fila_cabecera)
      .option("sep", delimitador)
      .load(ruta_fichero))

#Dataframe 2
ruta_fichero = "/FileStore/tables/TAL_COL-db584.csv"
tipo_fichero = "csv"

#Opciones de lectura
inferir_esquema = "true"
primer_fila_cabecera = "true"
delimitador = ";"

df2 = (spark.read.format(tipo_fichero)
      .option("inferSchema", inferir_esquema)
      .option("header", primer_fila_cabecera)
      .option("sep", delimitador)
      .load(ruta_fichero))

#Dataframe 3

ruta_fichero = "/FileStore/tables/TAL_CBA-45daa.csv"
tipo_fichero = "csv"

#Opciones de lectura
inferir_esquema = "true"
primer_fila_cabecera = "true"
delimitador = ";"

df3 = (spark.read.format(tipo_fichero)
      .option("inferSchema", inferir_esquema)
      .option("header", primer_fila_cabecera)
      .option("sep", delimitador)
      .load(ruta_fichero))

#Dataframe 4

ruta_fichero = "/FileStore/tables/TAL_HUR-d2ee8.csv"
tipo_fichero = "csv"

#Opciones de lectura
inferir_esquema = "true"
primer_fila_cabecera = "true"
delimitador = ";"

df4 = (spark.read.format(tipo_fichero)
      .option("inferSchema", inferir_esquema)
      .option("header", primer_fila_cabecera)
      .option("sep", delimitador)
      .load(ruta_fichero))

#Selección de las columnas en las que estamos interesados de cada dataframe

df1 = df1.select("Equipo","Nombre jugador","Tiempo en juego","Pases","% efectividad","Pelotas recuperadas ")
df2 = df2.select("_c0","_c3","Tiempo en juego","Pases","% efectividad","Pelotas recuperadas ")
df3 = df3.select("Equipo","Nombre jugador","Tiempo en juego","Pases","% efectividad","Pelotas recuperadas ")
df4 = df4.select("_c0","_c3","Tiempo en juego","Pases","% efectividad","Pelotas recuperadas ")

#En df2 y df4 cambiamos el nombre de la columna _c3
df2 = df2.withColumnRenamed('_c3','Nombre jugador')
df2 = df2.withColumnRenamed('_c0','Equipo')
df4 = df4.withColumnRenamed('_c3','Nombre jugador')
df4 = df4.withColumnRenamed('_c0','Equipo')

#Realizamos el mismo procedimiento para cada dataframe anterior asi que solo vamos a explicar el df1.
#Creamos una nueva columna donde asignamos una id incremental a cada fila del dataframe. De esta forma, filtramos el dataframe con los jugadores de Talleres y borramos la columna "Equipo" al no necesitarla más.
df1 = df1.withColumn("index",monotonically_increasing_id())
df11 = df1.filter(df1.index.between(14,27)).drop("Equipo","index")
#Renombramos las 4 columnas a utilizar. Esto lo hacemos para poder hacer más adelante el join correctamente y el método no confunda columnas con nombres iguales.
df11 = df11.withColumnRenamed('Tiempo en juego','Min1')
df11 = df11.withColumnRenamed('Pases','Pas1')
df11 = df11.withColumnRenamed('% efectividad','Ef1')
df11 = df11.withColumnRenamed('Pelotas recuperadas ','Rec1')

#Para realizar los cálculos posteriores, dividimos los valores de "% efectividad" y "Tiempo en juego"
split_col = F.split(df11['Min1'],',')
split_col2 = F.split(df11['Ef1'],'%')
#Creamos 2 nuevas columnas con la parte del valor que nos interesa de los 2 campos anteriores.Con esto, tenemos la parte entera de los "Minutos jugados" y el valor de "Efectividad" sin %
df11 = df11.withColumn('Min11',split_col.getItem(0)).withColumn('Ef11',split_col2.getItem(0))
df11 = df11.select("Nombre jugador","Min11","Pas1","Ef11","Rec1")
#Con estos valores, creamos 2 columnas para castear los valores a Integer al ser de tipo String.
df11 = df11.withColumn("Min11",df11["Min11"].cast(IntegerType()))
df11 = df11.withColumn("Ef11",df11["Ef11"].cast(IntegerType()))
#Por último, añadimos un campo que tiene valor 0 o 1 en función de si ha participado en el partido con al menos 1 minuto jugado.
df11 = df11.withColumn("Pj1", F.when(df11.Min11 > 0, 1).otherwise(0))

df2 = df2.withColumn("index",monotonically_increasing_id())
df22 = df2.filter(df2.index.between(14,27)).drop("Equipo","index")
df22 = df22.withColumnRenamed('Tiempo en juego','Min2')
df22 = df22.withColumnRenamed('Pases','Pas2')
df22 = df22.withColumnRenamed('% efectividad','Ef2')
df22 = df22.withColumnRenamed('Pelotas recuperadas ','Rec2')

split_col = F.split(df22['Min2'],',')
split_col2 = F.split(df22['Ef2'],'%')
df22 = df22.withColumn('Min22',split_col.getItem(0)).withColumn('Ef22',split_col2.getItem(0))
df22 = df22.select("Nombre jugador","Min22","Pas2","Ef22","Rec2")
df22 = df22.withColumn("Min22",df22["Min22"].cast(IntegerType()))
df22 = df22.withColumn("Ef22",df22["Ef22"].cast(IntegerType()))
df22 = df22.withColumn("Pj2", F.when(df22.Min22 > 0, 1).otherwise(0))

df3 = df3.withColumn("index",monotonically_increasing_id())
df33 = df3.filter(df3.index.between(14,27)).drop("Equipo","index")
df33 = df33.withColumnRenamed('Tiempo en juego','Min3')
df33 = df33.withColumnRenamed('Pases','Pas3')
df33 = df33.withColumnRenamed('% efectividad','Ef3')
df33 = df33.withColumnRenamed('Pelotas recuperadas ','Rec3')

split_col = F.split(df33['Min3'],',')
split_col2 = F.split(df33['Ef3'],'%')
df33 = df33.withColumn('Min33',split_col.getItem(0)).withColumn('Ef33',split_col2.getItem(0))
df33 = df33.select("Nombre jugador","Min33","Pas3","Ef33","Rec3")
df33 = df33.withColumn("Min33",df33["Min33"].cast(IntegerType()))
df33 = df33.withColumn("Ef33",df33["Ef33"].cast(IntegerType()))
df33 = df33.withColumn("Pj3", F.when(df33.Min33 > 0, 1).otherwise(0))

df4 = df4.withColumn("index",monotonically_increasing_id())
df44 = df4.filter(df4.index.between(14,27)).drop("Equipo","index")
df44 = df44.withColumnRenamed('Tiempo en juego','Min4')
df44 = df44.withColumnRenamed('Pases','Pas4')
df44 = df44.withColumnRenamed('% efectividad','Ef4')
df44 = df44.withColumnRenamed('Pelotas recuperadas ','Rec4')

split_col = F.split(df44['Min4'],',')
split_col2 = F.split(df44['Ef4'],'%')
df44 = df44.withColumn('Min44',split_col.getItem(0)).withColumn('Ef44',split_col2.getItem(0))
df44 = df44.select("Nombre jugador","Min44","Pas4","Ef44","Rec4")
df44 = df44.withColumn("Min44",df44["Min44"].cast(IntegerType()))
df44 = df44.withColumn("Ef44",df44["Ef44"].cast(IntegerType()))
df44 = df44.withColumn("Pj4", F.when(df44.Min44 > 0, 1).otherwise(0))


#Hacemos join de los 4 dataframes y con las operaciones anteriores, ya podemos sustituir todos los valores nulos por 0.
dffinal = df11.join(df22,"Nombre jugador",how='full_outer')
dffinal = dffinal.join(df33,"Nombre jugador",how='full_outer')
dffinal = dffinal.join(df44,"Nombre jugador",how='full_outer')
dffinal = dffinal.na.fill(0)
#Sumamos los partidos jugados de cada jugador.
dffinal = dffinal.withColumn('PartidosJugados', dffinal.Pj1 + dffinal.Pj2 + dffinal.Pj3 + dffinal.Pj4)

#Creamos las 4 columnas que van a servirnos para calcular el indice de ponderación. 
#Excepto MinutosFinales, el resto se ha calculado dividiendo por el número de partidos jugados para obtener la media por partido.
dffinal = dffinal.withColumn('MinutosFinales', dffinal.Min11+dffinal.Min22+dffinal.Min33+dffinal.Min44)
dffinal = dffinal.withColumn("Pases",F.round((dffinal.Pas1+dffinal.Pas2+dffinal.Pas3+dffinal.Pas4)/dffinal.PartidosJugados,0))
dffinal = dffinal.withColumn("Efectividad",F.round((dffinal.Ef11+dffinal.Ef22+dffinal.Ef33+dffinal.Ef44)/dffinal.PartidosJugados,1))
dffinal = dffinal.withColumn("Recuperaciones",F.round((dffinal.Rec1+dffinal.Rec2+dffinal.Rec3+dffinal.Rec4)/dffinal.PartidosJugados,0))

#Indice de influencia: Damos un peso a cada campo y lo redondeamos a 1 decimal como hemos hecho antes
dffinal = dffinal.withColumn("Indice de influencia",F.round(((dffinal.MinutosFinales*0.15) + (dffinal.Pases*0.45) + (dffinal.Efectividad*0.20) + (dffinal.Recuperaciones)*0.20),1))
#Seleccionamos las columnas que vamos a mostrar y ordenamos el dataframe por el indice de influencia en orden descendente
dffinal = dffinal.select("Nombre jugador","MinutosFinales","Pases","Efectividad","Recuperaciones","Indice de influencia").sort("Indice de influencia",ascending=False)
dffinal.show(30)


#Dar una conclusión de cada jugador TOP de porque su influencia
'''
Facundo Medina: Es el jugador que mas minutos ha jugado, realiza 15 pases mas que el 2º en este aspecto, 85,3% de efectividad y 3º en recuperaciones.
Andres Cubas: 3º jugador con mas minutos, 2º en pases realizados por partido, 83,3% de efectividad y el jugador con mas recuperaciones.
Nahuel Tenaglia: Jugador con mas minutos. Cumple en el resto de aspectos, pero sin destacar en ninguno de ellos.
'''


#Una opinión subjetiva para contrarrestrarlo
'''
Facundo Medina: Marcaje individual o presión alta en su zona para que no pueda recibir de manera cómoda.
Andres Cubas: Centrar el juego por una zona de influencia a la que le cueste llegar. Aumentar velocidad de juego para el mismo objetivo.
Nahuel Tenaglia: A pesar de su influencia, no sería objetivo prioritario.
'''


#Explicar porque el valor de cada ponderación
'''
He tenido bastantes problemas para realizar la ponderación porque no lo tenía nada claro en cuanto a la importancia de minutos jugados. Finalmente, he decidido por darle un peso bastante bajo
porque era la única manera de potenciar jugadores que realizan un muy buen trabajo y no conocemos el motivo de la ausencia de minutos(Jugadores como Rafael Perez, Jose Mauri,Juan Cruz Komar o Javier Gandolfi).

De todos modos, no lo tengo nada claro porque habría que tener en cuenta otros muchos aspectos para hacer un buen indice de influencia. Probablemente los jugadores que se han visto perjudicado por la ponderación, sean muy importantes pero con lo que hemos analizado no identificamos esa importancia.
'''


