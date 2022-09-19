# Importa aquí las librerías que vayas a utilizar
import numpy as np
import pandas as pd
import traceback
import re
import matplotlib.pyplot as plt

#Crea una variable llamada raw_data de tipo DataFrame que tenga la información del fichero my_players_info.csv

path = "my_players_info.csv"
raw_data = pd.read_csv(path, sep=';')

print(raw_data.info())

#Ejercicio 1: Función para extraer la altura

def extraer_altura(df):
    df['Altura'].fillna('-1', inplace=True)
    df['Altura'] = df['Altura'].apply(lambda x: x.replace(",",".").replace("'",".")) 
    df['Altura'] = df['Altura'].apply(lambda x: x.strip()) 
    df['Altura'] = df['Altura'].apply(lambda x: float(x[0:4]))
    df.loc[df['Altura']==-1, 'Altura'] = np.nan
        
    assert(isinstance(df, pd.DataFrame))
    return df

#Ejercicio 2: Función para extraer el peso
def extraer_peso(df):
    df['Peso'].fillna('-1', inplace=True)
    df['Peso'] = df['Peso'].apply(lambda x: x.strip()) 
    df['Peso'] = df['Peso'].apply(lambda x: str(x).split()[0])
    df['Peso'] = df['Peso'].apply(lambda x: float(x))
    df.loc[df['Peso']==-1, 'Peso'] = np.nan
        
    assert(isinstance(df, pd.DataFrame))
    return df

#Ejercicio 3: Calcular la categoría de la OMS usando el Indice de masa Corporal (IMC)
raw_data = pd.read_csv(path, sep=";" )
df_4 = extraer_altura(raw_data)
df_4 = extraer_peso(df_4)
def clasificacion_oms(df):
    df['IMC'] = (df_4['Peso'])/(df_4['Altura']*df_4['Altura'])
    
    df['clasificacion_oms']= np.where(df['IMC'] < 18.5,"delgadez",np.where((df['IMC'] >=18.5 )& (df['IMC'] <25.0),"normal",
                    np.where((df['IMC'] >=25.0 )& (df['IMC'] <30.0),"sobrepeso",np.where(df['IMC'] >30.0,"obeso",None))))
    
    assert(isinstance(df, pd.DataFrame))
    return df 

clasificacion_oms(raw_data)

#Ejercicio 4: Función para extraer el club en el que debutó y el año de debut
raw_data[['Debut','Debut deportivo']].tail(3)

def extraer_datos_debut(df):
    año = df.Debut.map(lambda x: str(x).split()[-1])
    añodd = df['Debut deportivo'].map(str)
    añodd = añodd.apply(lambda x: x.split("(")[0]).apply(lambda x: x.split()[-1] if len(x)>4 else x) 
    
    df['fecha_debut']= np.where((año == añodd) & (año != str(np.nan)) & (añodd != "nan") ,año,
                            np.where((año != añodd) & (año != "nan") & (añodd != "nan") ,añodd,
                    np.where((año != añodd ) & (añodd == "nan"),año,np.where((año=="NaN") & (añodd=="nan"),np.nan,añodd))))
    
    
    df['club_debut'] = df['Debut deportivo'].apply(lambda x: str(x).split("(")[-1].split(")")[0])
    df.drop('Debut deportivo',inplace=True,axis=1)
    df.drop('Debut',inplace=True,axis=1)
    assert(isinstance(df, pd.DataFrame))
    return df

extraer_datos_debut(raw_data)

#Ejercicio 5: Función para los goles en clubes
raw_data['Golesen clubes'] = pd.to_numeric(raw_data['Goles en clubes'],errors='coerce')

#Ejercicio 6:
jugador_mas_delgado = raw_data[raw_data.IMC == raw_data.IMC.min()]
jugador_mas_delgado = jugador_mas_delgado[['nombre','Posición','IMC','Club']]
print(jugador_mas_delgado)

jugador_mas_obeso = raw_data[raw_data.IMC == raw_data.IMC.max()]
jugador_mas_obeso = jugador_mas_obeso[['nombre','Posición','IMC','Club']]
print(jugador_mas_obeso)

#Ejercicio 7:Realizar una visualizacion que permita ver la relacion entre IMC y la posición del jugador en el equipo

raw_data = raw_data.replace(['Guardameta','Arquero'],['Portero','Portero'])

raw_data = raw_data.replace(['Defensa','Defensa central','Defensa central Mediocentro defensivo','Defensor','Central','Defensor central',
   'Defensa centralLateral','Defensa central/Lateral izquierdo','defensa','Defensor Central'],
    ['Defensa central','Defensa central','Defensa central', 'Defensa central', 'Defensa central','Defensa central',
     'Defensa central','Defensa central','Defensa central','Defensa central'])

raw_data = raw_data.replace(['Lateral derechoDefensa central derecho','Lateral','Lateral izquierdo','Lateral derecho , Extremo derecho , Mediapunta',
    'Lateral derecho','Lateral derecho  Interior derecho', 'Lateral derechoMediocentro defensivo','Lateral Izquierdo',
    'Defensa lateral derecho','Lateral izquierdo - Centrocampista','carilero','Lateral derechoExtremo derecho',
    'lateral derecho','Defensa lateral izquierdo','Lateral derecho.','Lateral izquierdo/Interior izquierdo',
    'Defensa (lateral izquierdo o central)'],
    ['Laterales','Laterales','Laterales','Laterales','Laterales','Laterales','Laterales','Laterales','Laterales',
     'Laterales','Laterales','Laterales','Laterales','Laterales','Laterales','Laterales','Laterales'])

raw_data = raw_data.replace(['Interior derecho Interior izquierdo','MediapuntaInterior izquierdo','Interior derecho',
'Interior/Extremo Izquierdo','MediapuntaInterior derecho','Interior o Lateral derecho','Interior izquierdo'],
['Interiores','Interiores','Interiores','Interiores','Interiores','Interiores','Interiores'])

raw_data = raw_data.replace(['Extremo','Extremo derecho','Extremo derechoSegundo delantero','Delantero extremo izquierdo',
'Extremo izquierdo','extremo derecho','Extremo Izquierdo|Mediapunta','ExtremoMediapunta','Delantero Extremo'],
['Extremo','Extremo','Extremo','Extremo','Extremo','Extremo','Extremo','Extremo','Extremo'])

raw_data = raw_data.replace(['Delantero','delantero centro','Centrodelantero','delantero', 'Delantero / Mediapunta',
'Delantero centro','Delantero, centrocampista','Delantero y centrocampista'],
['Delantero','Delantero','Delantero','Delantero','Delantero','Delantero','Delantero','Delantero'])

raw_data = raw_data.replace(['Mediocentro','Mediocentro defensivo','Mediapunta','Centrocampista defensivo',
    'Volante','Centrocampista/Lateral derecho','Centrocampista ofensivo','Mediapunta, Delantero','DefensaCentrocampista',
 'Volante de marca','Medio centro, Medio centro defensivo','Centrocampista / Extremo','Pivote','Mediocampista',
    'mediocentro','Medio','Mediocentro o defensa','Mediocampista ofensivo','medio centro ofensivo/interior derecho'],
  ['Centrocampista','Centrocampista','Centrocampista','Centrocampista','Centrocampista','Centrocampista','Centrocampista',
'Centrocampista','Centrocampista','Centrocampista','Centrocampista','Centrocampista','Centrocampista','Centrocampista',
  'Centrocampista','Centrocampista','Centrocampista','Centrocampista','Centrocampista'])

raw_data['Posición'].unique().tolist()

sumatorios = raw_data.groupby('Posición')['IMC'].mean().round(2)
sumatorios = sumatorios.reindex(['Delantero','Extremo','Interiores','Centrocampista','Laterales','Defensa central','Portero'])
print(sumatorios)

sumatorios.plot(kind='barh', color=['Orange','Orange','Green','Green','Red','Red','Blue'])
for index, value in enumerate(sumatorios):
    plt.text(value, index, str(value))
plt.title('IMC/Posición')
plt.xlabel('IMC')
plt.ylabel('Posición')



