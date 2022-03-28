from math import *

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
D=[[0.997,3.65]]
A=[[182,81.6,30,1],[180,86.2,28,1],[170,77.1,30,1],[180,74.8,25,1],[152,45.4,15,0],[168,68,20,0],[165,59,18,0],[175,68,23,0]]
B=[[183,59,20]]
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


def calcul_proba_categorie_sachant_classe(point,points,categorie,classe,ecart):
    esp = calcul_esp(points)[int(classe-ecart)][0][categorie]
    var = calcul_var(points,ecart)[int(classe-ecart)][0][categorie]
    proba = exp((-((point[0][categorie]-esp)**2)/(2*var)))/(sqrt(2*float(pi)*var))
    return proba


def calcul_constante(points,ecart,point):
    classes = set(point[-1] for point in points)
    proba=1
    constante=0
    liste_proba=[]
    for classe in classes:
        for i in range(len(points[0])-1):
            proba=proba*calcul_proba_categorie_sachant_classe(point,points,i,classe,ecart)
        proba=proba*float(calcul_proba_classe(points)[int(classe-ecart)][0])
        liste_proba.append(proba)
        proba=1
    for i in range(len(liste_proba)):
        constante+=liste_proba[i]
    return constante

def calcul_proba_bayes(point,points):
    ecart=calcul_ecart(points)
    classes = set(point[-1] for point in points)
    liste_proba = []
    for classe in classes:
        proba_classe = calcul_proba_classe(points)[int(classe-ecart)][0]
        #print(proba_classe)
        proba_categorie_sachant_classe = 1
        for i in range(len(points[0])-1):
            proba_categorie_sachant_classe = proba_categorie_sachant_classe*calcul_proba_categorie_sachant_classe(point,points,i,classe,ecart)
        liste_proba.append([(proba_classe*proba_categorie_sachant_classe/calcul_constante(points,ecart,point)),classe])
    return liste_proba,max(liste_proba)[1]
