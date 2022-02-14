from math import *
from sympy import *


def calcul_esp(points): # renvoie lesperance de chaque catégorie en fonction de sa classe
    dim_point = len(points[0]) - 1
    esperances = []
    esperance_par_classe=[]
    classes=set(point[-1] for point in points)
    somme=0
    compteur=0
    for classe in classes:
        for rang in range(dim_point):
            for point in points:
                if point[-1]==classe:
                    compteur+=1
                    somme += point[rang]
            esperances.append(somme/compteur)
            somme=0
            compteur=0
        esperance_par_classe.append([esperances,classe])
        esperances=[]
    return esperance_par_classe


def calcul_var(points):
    dim_point = len(points[0]) - 1
    variances = []
    variance_par_classe=[]
    classes=set(point[-1] for point in points)
    somme=0
    compteur=0
    for classe in classes:
        for rang in range(dim_point):
            for point in points:
                if point[-1]==classe:
                    compteur+=1
                    somme = somme + (point[rang]-calcul_esperance(points)[0][1])**2
            variances.append((somme/(compteur-1)))
            somme=0
            compteur=0
        variance_par_classe.append([variances,classe])
        variances=[]
    return variance_par_classe


A=[[1,25,33,64,3,16,1,0],[2,54,65,25,26,3,7,1],[21,5,2,5,5,6,6,1],[52,3,6,94,6,4,4,0]]

def calcul_proba_classe(points): #ex male ou femelle
    liste_classe=set([point[-1]for point in points])
    liste_proba=[]
    for classe in liste_classe:
        compteur=0
        for point in points:
            if point[-1]==classe:
                compteur += 1
        liste_proba.append([compteur/len(points),classe])
    return liste_proba

    #le 2eme [] permet d'ingnorer la classe rendu par les fonctions calcul_variance et esperance
#calcul des P(x1,x2,....) par intervalle sur loi normale

def calcul_proba_categorie_sachant_classe(point,points,categorie,classe):
    esp=calcul_esp(points)[classe][0][categorie]
    var=calcul_var(points)[classe][0][categorie]
    proba=exp((-((point[categorie]-esp)**2)/(2*var**2)))/sqrt(2*float(pi)*var**2)
    return proba

def calcul_proba_categorie(points,borne_inferieur,borne_superieur,categorie,classe):
    x=Symbol('x')
    esp=calcul_esp(points)[classe][0][categorie]
    var=calcul_var(points)[classe][0][categorie]
    proba=integrate(exp((-((x-esp)**2)/(2*var**2)))/sqrt(2*float(pi)*var**2),(x,borne_inferieur,borne_superieur))
    return proba

def calcul_proba_bayes(point,points,borne_inf,borne_sup,classe):#borne_inf et borne_sup sont des listes contenant les bornes inferieur ou superieur des integrations de la loi normale lié à chaque catégories
    proba_classe=calcul_proba_classe(points)[classe][0]
    proba_categorie=1
    proba_categorie_sachant_classe=1
    for i in range(len(points[0])-1):
        proba_categorie=proba_categorie*calcul_proba_categorie(points,borne_inf[i],borne_sup[i],classe)
        proba_categorie_sachant_classe=proba_categorie_sachant_classe*calcul_proba_categorie_sachant_classe(point,points,i,classe)
    return proba_classe*proba_categorie_sachant*classe/proba_categorie


