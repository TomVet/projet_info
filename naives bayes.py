from math import *
from sympy import *

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


def calcul_var(points):
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
                    somme = somme + (point[rang]-calcul_esp(points)[0][1])**2
            variance.append((somme/(compteur-1)))
            somme = 0
            compteur = 0
        variance_par_classe.append([variance,classe])
        variance = []
    return variance_par_classe

def calcul_esp_categorie(points,categorie):
# renvoie l'esperance d'une categorie indépendamment de sa classe
    esp = 0
    for point in points:
        esp = esp+point[categorie]
    return esp/len(points)

def calcul_var_categorie(points,categorie):
# renvoie la variance d'une categorie indépendamment de sa classe
    var = 0
    for point in points:
        var = var + (point[categorie]-calcul_esp_categorie(points,categorie))**2
    return var/(len(points)-1)

A=[[1,25,33,64,3,16,1,0],[2,54,65,25,26,3,7,1],[21,5,2,5,5,6,6,1],[52,3,6,94,6,4,4,0]]
B=[2,5,7,6,8,4,4]

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

def calcul_proba_categorie_sachant_classe(point,points,categorie,classe):
    esp = calcul_esp(points)[classe][0][categorie]
    var = calcul_var(points)[classe][0][categorie]
    proba = exp((-((point[categorie]-esp)**2)/(2*var**2)))/sqrt(2*float(pi)*var**2)
    return proba

def creer_intervalle(points,categorie):
#categorie est un chiffre (la xème catégorie étudiée)
#renvoie une liste represenant un intervalle divisé en 10 segments, chaque point etant la bordure d'un segment
    points_categorie = []
    for point in points:
        points_categorie.append(point[categorie])
    borne_inf = min(points_categorie)
    borne_sup = max(points_categorie)
    intervalle = [borne_inf+k*(borne_sup-borne_inf)/10 for k in range(11)]
    return intervalle

def calcul_proba_categorie(points,point,categorie):
    x = Symbol('x')
    intervalle = creer_intervalle(points,categorie)
    borne=[]
    for i in range(11):
        if intervalle[i] <= point[categorie] <= intervalle[i+1]:
            borne.append(intervalle[i])
            borne.append(intervalle[i+1])
# si la valeur du nouveau point appartient à l'intervalle étudié, alors les bornes de l'intervalle seront pris comme borne d'intégration
    esp = calcul_esp_categorie(points,categorie)
    var = calcul_var_categorie(points,categorie)
    proba = integrate(exp((-((x-esp)**2)/(2*var**2)))/sqrt(2*float(pi)*var**2),(x,borne[0],borne[1]))
    return proba

def calcul_proba_bayes(point,points):
# on calcule indépendamment chaque membre de la formule de bayes puis on les rassemble
# renvoie une liste donnant la probabilité d'appartenance du nouveau point à chaque classe
    classes = set(point[-1] for point in points)
    liste_proba = []
    for classe in classes:
        proba_classe = calcul_proba_classe(points)[classe][0]
        proba_categorie = 1
        proba_categorie_sachant_classe = 1
        for i in range(len(points[0])-1):
            proba_categorie = proba_categorie*calcul_proba_categorie(points,point,i)
            proba_categorie_sachant_classe = proba_categorie_sachant_classe*calcul_proba_categorie_sachant_classe(point,points,i,classe)
        liste_proba.append(proba_classe*proba_categorie_sachant_classe/proba_categorie)
    return liste_proba

