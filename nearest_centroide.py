# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:03:56 2021

@author: TomDION
"""

import csv
import numpy as np
#import matplotlib.pyplot as plt
#import sklearn.datasets as dataset


def calcul_coordonnees_centroide(liste_coordonne):
    """
    Calcule le centroide des points de liste_coordonne de dimension
    nb_dimension

    Parameters
    ----------
    liste_coordonne : np.array
        liste de coordonnee de point de meme dimension.

    Returns
    -------
    coordonnees : np.array
        liste des coordonnee du centroide calcule.

    """
    coordonnees = np.array([])
    # on calcule la dimension de l'espace considere pour le centroide
    nb_dimension = len(liste_coordonne[0])
    # on calcul les coordonnees du centroide dans chaque dimension
    for dimension in range(nb_dimension):
        somme = 0
        # on somme les coordonnees de chaque points
        for point in liste_coordonne:
            somme += point[dimension]
        # on ajoute la somme / par le nombre de point a coordonnees
        coordonnees = np.append(coordonnees, [somme/len(liste_coordonne)])
    return coordonnees

def calcul_distance_euclidienne(point_1, point_2):
    """
    Calcul de la distance euclidienne au carre entre les points 1 et 2

    Parameters
    ----------
    point_1 : np.array
        liste des coordonnees du point 1.
    point_2 : np.array
        liste des coordonnees du point 2.

    Returns
    -------
    distance : float
        distance euclidienne au carre entre les point 1 et 2.

    """
    # on calcule la dimension de l'espace des 2 points
    nb_dimension = len(point_1)
    distance = 0
    # on fait la somme au carre des coordonnees des points 1 et 2 dans chaque dimension
    for dimension in range(nb_dimension):
        somme = (point_1[dimension] - point_2[dimension])**2
        distance += somme
    return distance


def find_nearest_centroid(point, centroides):
    """
    permet de trouver le centroide le plus proche du point

    Parameters
    ----------
    point : np.array
        liste des coordonnee du point.
    centroides : np.array
        liste de coordonnee de centroides.

    Returns
    -------
    indice_du_min : int
        indice du centroide le plus proche de point dans la liste centroides.

    """
    distance_min = calcul_distance_euclidienne(point, centroides[0])
    indice_du_min = 0
    # on parcour la liste des centroides
    for indice, centroide in enumerate(centroides):
        # on calcul la distance entre le centroide et le point
        distance = calcul_distance_euclidienne(point, centroide)
        # si la nouvelle distance est plus petite que le minimum elle devient le minimum
        if distance_min > distance:
            distance_min = distance
            # on conserve l´indice, dans la liste de centroides, du centroide le plus proche
            indice_du_min = indice
    return indice_du_min

def recuperer_donnee_csv(fichier, separateur=','):
    """
    cree une liste de liste contenant les donnees de fichier

    Parameters
    ----------
    fichier : string
        chemin du fichier csv a lire
        ce fichier ne doit contenir que des float.
    separateur : string, optional
        string contenant le separateur utiliser dans fichier. The default is ','.

    Returns
    -------
    data : np.array
        array de dimension 2.

    """
    data = np.array([])
    with open(fichier, newline='', encoding='utf-8') as dataFile:
        dataReader = csv.reader(dataFile, delimiter=separateur)
        nb_ligne = 0
        for ligne in dataReader:
            ligne_data = np.array([])
            nb_ligne += 1
            nb_colone = 0
            for element in ligne:
                ligne_data = np.append(ligne_data, float(element))
                nb_colone += 1
            data = np.concatenate((data, ligne_data))

    data = np.resize(data, (nb_ligne, nb_colone))
    return data


"""
# on cree un dataset en 2 dimension pour tester l´algorithme et pouvoir le visualiser
# nombre d'echantillon : 100
# nombre de parametre pour definir un point : 2
# nombre de parametre informatif : 2
# nombre de parametre redondant : 0
# nombre de classe = 3
# nombre de groupe par classe = 1

data = dataset.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1)


'''on separe le dataset en fonction des classes'''
# on cherche ou sont les points de la classe 1 dans la liste data
coor_classe_1 = np.where(data[1] == 0)
# on met les coordonnees des points de la classe 1 dans une nouvelle liste
classe_1 = data[0][coor_classe_1]
x_classe_1 = [point[0] for point in classe_1]
y_classe_1 = [point[1] for point in classe_1]
# on calcule le centroide de la classe 1
centroide_1 = calcul_coordonnees_centroide(classe_1)
x_1 = centroide_1[0]
y_1 = centroide_1[1]

# on cherche ou sont les points de la classe 2 dans la liste data
coor_classe_2 = np.where(data[1] == 1)
# on met les coordonnees des points de la classe 2 dans une nouvelle liste
classe_2 =  data[0][coor_classe_2]
x_classe_2 = [point[0] for point in classe_2]
y_classe_2 = [point[1] for point in classe_2]
# on calcule le centroide de la classe 2
centroide_2 = calcul_coordonnees_centroide(classe_2)
x_2 = centroide_2[0]
y_2 = centroide_2[1]

# on cherche ou sont les points de la classe 3 dans la liste data
coor_classe_3 = np.where(data[1] == 2)
# on met les coordonnees des points de la classe 3 dans une nouvelle liste
classe_3 = data[0][coor_classe_3]
x_classe_3 = [point[0] for point in classe_3]
y_classe_3 = [point[1] for point in classe_3]
# on calcul le centroide de la classe 3
centroide_3 = calcul_coordonnees_centroide(classe_3)
x_3 = centroide_3[0]
y_3 = centroide_3[1]


plt.plot(x_1, y_1, 'b^')
plt.plot(x_classe_1, y_classe_1, 'b.')
plt.plot(x_2, y_2, 'r^')
plt.plot(x_classe_2, y_classe_2, 'r.')
plt.plot(x_3, y_3, 'g^')
plt.plot(x_classe_3, y_classe_3, 'g.')
plt.show()
"""

# lecture dataset en csv
dataset = recuperer_donnee_csv('heart.csv')

# separation en fonction de target
nb_parametre = 13
malade = np.array([])
nb_malade = 0
sain = np.array([])
nb_sain = 0
for ligne in dataset:
    if ligne[13] == 1:
        malade = np.append(malade, ligne[:13])
        nb_malade += 1
    elif ligne[13] == 0:
        sain = np.append(sain, ligne[:13])
        nb_sain += 1

malade = np.resize(malade, (nb_malade, nb_parametre))
sain = np.resize(sain, (nb_sain, nb_parametre))

centroide_malade = calcul_coordonnees_centroide(malade)
centroide_sain = calcul_coordonnees_centroide(sain)
centroides = [centroide_sain, centroide_malade]

testdata = recuperer_donnee_csv('heart_test.csv')

nb_bon = 0
nb_faux = 0

# sain correspond a 0
# malade correspond a 1

for test in testdata:
    if find_nearest_centroid(test[:13], centroides) == 1 and test[13] == 1:
        nb_bon+= 1
    elif find_nearest_centroid(test[:13], centroides) == 0 and test[13] == 0:
        nb_bon += 1

print('Précision : ', nb_bon/len(testdata) * 100, '%')


# lecture dataset en csv
dataset = recuperer_donnee_csv('water_potability.csv',';')

# separation en fonction de target
nb_parametre = 9
potable = np.array([])
nb_potable = 0
non_potable = np.array([])
nb_non_potable = 0
for ligne in dataset:
    if ligne[nb_parametre] == 1:
        potable = np.append(potable, ligne[:nb_parametre])
        nb_potable += 1
    elif ligne[nb_parametre] == 0:
        non_potable = np.append(non_potable, ligne[:nb_parametre])
        nb_non_potable += 1

potable = np.resize(potable, (nb_potable, nb_parametre))
non_potable = np.resize(non_potable, (nb_non_potable, nb_parametre))

centroide_potable = calcul_coordonnees_centroide(potable)
centroide_non_potable = calcul_coordonnees_centroide(non_potable)
centroides = [centroide_non_potable, centroide_potable]

testdata = recuperer_donnee_csv('water_potability_test_1.csv',';')

nb_bon = 0
nb_faux = 0

# non_potable correspond a 0
# potable correspond a 1

for test in testdata:
    if find_nearest_centroid(test[:nb_parametre], centroides) == 1 and test[nb_parametre] == 1:
        nb_bon+= 1
    elif find_nearest_centroid(test[:nb_parametre], centroides) == 0 and test[nb_parametre] == 0:
        nb_bon += 1

print('Précision : ', nb_bon/len(testdata) * 100, '%')
