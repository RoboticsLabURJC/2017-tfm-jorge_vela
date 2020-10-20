# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:41:03 2020

@author: User
"""
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

import numpy as np
from sklearn.metrics import precision_recall_curve


precission_tot = 0
recall_tot = 0
used = 0

umbral = 20
'''
PARA UTILIZAR ESTE PROGRAMA, EN LA MISMA CARPETA TIENE QUE ESTAR 
results, CON LOS FICHEROS TXT OBTENIDOS EN BADACOST, Y training/label_2
con los ficheros txt que queremos comparar
'''
for file in range(747):
    t_p = 0
    f_p = 0
    f_n = 0
    file_name = file + 6733; #PARA OBTENER DEL 006733.txt en adelante
    
    f1 = "results/00" + str(file_name) + ".txt" #6733.txt"
    f2 = "training/label_2/00"+ str(file_name) + ".txt" 
    
    file1 = open(f1, 'r')        #ABRIR EL FICHERO DE DETECCIONES BADACOST
    Lines_get = file1.readlines()
        
    for detection in Lines_get: #LEER LINEA A LINEA
        positive = 0
        x = str(detection)
        values = x.split(" ")
        if(values[0] == "Car"): #SI ES UN COCHE, COMPARAR CON EL OTRO FICHERO
            coords = values[4:8]
            file_compare =  open(f2, 'r') 
            Lines_compare = file_compare.readlines()
            for detection2 in Lines_compare:#LEER LINEA A LINEA PARA COMPARAR SI ES COCHE
                y = str(detection2)
                coords2 = y.split(" ")
                if(coords2[0] == "Car"):
                    coords2 = coords2[4:8] #SE COMPARAN LAS COORDENADAS, QUE SEAN MENOS VALOR A UN UMBRAL DADO
                    if (abs(float(coords[0]) - float(coords2[0])) < umbral and abs(float(coords[1]) - float(coords2[1])) < umbral):
                        if(abs(float(coords[2]) - float(coords2[2])) < umbral and abs(float(coords[3]) - float(coords2[3])) < umbral ):
                            positive = 1
                            break
            t_p = t_p + positive
            f_p = f_p + (positive-1)*-1
    
    file1.close()        
    file_compare.close()
    
    file3 =  open(f2, 'r') 
    Lines3 = file3.readlines()    #SE HACE EL MISMO PROCESO PERO COMPARANDO EL OTRO FICHERO CON BADACOST
    for detection3 in Lines3:     #ASI SE PUEDE VER QUE POSITIVOS NO HAN SIDO DETECTADOS POR NUESTRO DETECTOR
        negative = 0
        x = str(detection3)
        values_compare = x.split(" ")
        if(values_compare[0] == "Car"):
            coords3 = values_compare[4:8]
            file4 =  open(f1, 'r') 
            Lines_get = file4.readlines()
            for detection4 in Lines_get:
                values_get = str(detection4)
                coords4 = values_get.split(" ")
                if(coords4[0] == "Car"):
                    coords4 = coords4[4:8]
                    if (abs(float(coords3[0]) - float(coords4[0])) < umbral and abs(float(coords3[1]) - float(coords4[1])) < umbral):
                        if(abs(float(coords3[2]) - float(coords4[2])) < umbral and abs(float(coords3[3]) - float(coords4[3])) < umbral ):
                            negative = negative + 1
                            break    
    
       
            f_n = f_n + (negative-1)*-1
        
    file3.close()
    file4.close()
        
    if(t_p == 0 and f_p == 0 and f_n == 0):
        #NO HAY COCHE
        continue
    if(t_p == 0 and f_p == 0):
        #?
        #print(file, t_p, f_p, f_n)
        continue
    elif(t_p == 0 and f_n == 0):
        #?
        continue
    else:
        precision = t_p / (t_p + f_p)
        recall = t_p / (t_p + f_n)
        precission_tot = precission_tot + precision
        recall_tot = recall_tot + recall
        used = used + 1
    
    #print("Precision:",precision , "recall",recall )

print(precission_tot / used, recall_tot/used , used)







