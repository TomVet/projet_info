from math import *
import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
import time

def recuperer_donnee_csv(fichier, separateur=","):
    """
    Créée une liste de liste contenant les données de fichier.
    Parameters
    ----------
    fichier : string
        chemin du fichier csv a lire
        ce fichier ne doit contenir que des float.
    separateur : string, optional
        string contenant le séparateur utilisé dans fichier.
        The default is ",".
    Returns
    -------
    data : np.array
        array de dimension 2.
    """
    with open(fichier, newline="", encoding="utf-8") as data_file:
        data = []
        data_reader = csv.reader(data_file, delimiter=separateur)
        for ligne in data_reader:
            data.append(ligne)
        data = np.array(data)
        data = data.astype(np.float64)
        return data

# naive bayes sklearn

def liste_classe(data):
    classes=[]
    for point in data:
        classes.append(point[-1])
    return classes

def liste_donnes(points):
    donnees=[]
    for point in points:
        liste=[point[i] for i in range(len(points[0])-1)]
        donnees.append(liste)
        liste=[]
    return donnees

def proba_naives_sklearn(point,points):
    start = time.time()
    X = liste_donnes(points)
    Y = liste_classe(points)
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X, Y)
    GaussianNB()
    clf_pf = GaussianNB()
    clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    end = time.time()
    temps = (end - start)
    return (clf_pf.predict_proba(point)),(int(clf.predict(point))),temps

# naive bayes programmé

def calcul_ecart(points):
    classes = set(point[-1] for point in points)
    return int(min(classes))

def calcul_esp(points):
# renvoie lesperance de chaque catégorie en fonction de sa classe
    dim_point = len(points[0]) - 1
    esperance = []
    esperance_par_classe = []
    classes = set(point[-1] for point in points)
    somme = 0
    compteur = 0
    for classe in classes:
        for rang in range(dim_point):
            for point in points:
                if point[-1] == classe:
                    compteur += 1
                    somme += point[rang]
            esperance.append(somme/compteur)
            somme = 0
            compteur = 0
        esperance_par_classe.append([esperance,classe])
        esperance = []
    return esperance_par_classe


def calcul_var(points,ecart):
# renvoie la variance de chaque catégorie en fonction de sa classe
    dim_point = len(points[0]) - 1
    variance = []
    variance_par_classe = []
    classes = set(point[-1] for point in points)
    somme = 0
    compteur = 0
    for classe in classes:
        for rang in range(dim_point):
            for point in points:
                if point[-1] == classe:
                    compteur += 1
                    #print(compteur)
                    somme = somme + (point[rang] - calcul_esp(points)[int(classe-ecart)][0][int(rang)])**2
                    #print(somme)
            variance.append((somme/(compteur-1)))
            somme = 0
            compteur = 0
        variance_par_classe.append([variance,classe])
        variance = []
    return variance_par_classe

C=recuperer_donnee_csv("C:/Users/jbcor/Downloads/dataset.csv")
D=[[0.9978,3.51,5],[0.9968,3.2,5],[0.997,3.26,5]]
E=[[182,81.6,30,1],[180,74.8,25,1],[152,45.4,15,0],[168,68,20,0],[165,59,18,0],[175,68,23,0]]
F=[[180,86.2,28,1],[170,77.1,30,1]]
A=[[182,81.6,30,1],[180,86.2,28,1],[170,77.1,30,1],[180,74.8,25,1],[152,45.4,15,0],[168,68,20,0],[165,59,18,0],[175,68,23,0]]
B=[[190,70,28]]

def calcul_proba_classe(points):
# ex male ou femelle
    liste_classe = set([point[-1] for point in points])
    liste_proba = []
    for classe in liste_classe:
        compteur = 0
        for point in points:
            if point[-1] == classe:
                compteur += 1
        liste_proba.append([compteur/len(points),classe])
    return liste_proba


def calcul_proba_categorie_sachant_classe(point,points,categorie,classe,ecart,esperance,variance):
    esp = esperance[int(classe-ecart)][0][categorie]
    var = variance[int(classe-ecart)][0][categorie]
    proba = exp((-((point[0][categorie]-esp)**2)/(2*var)))/(sqrt(2*float(pi)*var))
    return proba


def calcul_constante(points,ecart,point,esp,var):
    classes = set(point[-1] for point in points)
    proba=1
    constante=0
    liste_proba=[]
    for classe in classes:
        for i in range(len(points[0])-1):
            proba=proba*calcul_proba_categorie_sachant_classe(point,points,i,classe,ecart,esp,var)
        proba=proba*float(calcul_proba_classe(points)[int(classe-ecart)][0])
        liste_proba.append(proba)
        proba=1
    for i in range(len(liste_proba)):
        constante+=liste_proba[i]
    return constante

def calcul_proba_bayes(point,points):
    start = time.time()
    ecart=calcul_ecart(points)
    proba_classe = calcul_proba_classe(points)
    classes = set(point[-1] for point in points)
    liste_proba = []
    esp = calcul_esp(points)
    var = calcul_var(points,ecart)
    for classe in classes:
        proba_categorie_sachant_classe = 1
        for i in range(len(points[0])-1):
            proba_categorie_sachant_classe = proba_categorie_sachant_classe*calcul_proba_categorie_sachant_classe(point,points,i,classe,ecart,esp,var)
        liste_proba.append([(proba_classe[int(classe-ecart)][0]*proba_categorie_sachant_classe/calcul_constante(points,ecart,point,esp,var)),classe])
    end = time.time()
    temps = (end - start)
    return liste_proba,max(liste_proba)[1],temps

def comparateur(liste_test,dataset):
    classes=liste_classe(liste_test)
    points=liste_donnes(liste_test)
    temps_1=0
    temps_2=0
    succes_1=0
    succes_2=0
    taille=len(liste_test)
    for i in range(len(liste_test)):
        Liste=proba_naives_sklearn([points[i]],dataset)
        liste=calcul_proba_bayes([points[i]],dataset)
        temps_1+=Liste[2]
        temps_2+=liste[2]
        if Liste[1]==classes[i]:
            succes_1+=1
        if liste[1]==classes[i]:
            succes_2+=1
        liste_1=0
        liste_2=0
    temps_1=temps_1/taille
    temps_2=temps_2/taille
    succes_1=succes_1/taille
    succes_2=succes_2/taille
    return (succes_1,temps_1),(succes_2,temps_2)

