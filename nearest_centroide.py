# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:03:56 2021

@author: TomDION
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dataset


def coordonnees_centroide(liste_coordonne):
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

def distance_euclidienne(point_1, point_2):
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


def centroide_proche(point, centroides):
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
        indice du centroide le plus proche de point dans la liste centroide.

    """
    distance_min = 0
    indice_du_min = 0
    # on parcour la liste des centroides
    for indice, centroide in enumerate(centroides):
        # on calcul la distance entre le centroide et le point
        distance = distance_euclidienne(point, centroide)
        # si la nouvelle distance est plus petite que le minimum elle devient le minimum
        if distance_min > distance:
            distance_min = distance
            # on conserve l´indice, dans la liste de centroides, du centroide le plus proche
            indice_du_min = indice
    return indice_du_min


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
centroide_1 = coordonnees_centroide(classe_1)
x_1 = centroide_1[0]
y_1 = centroide_1[1]

# on cherche ou sont les points de la classe 2 dans la liste data
coor_classe_2 = np.where(data[1] == 1)
# on met les coordonnees des points de la classe 2 dans une nouvelle liste
classe_2 =  data[0][coor_classe_2]
x_classe_2 = [point[0] for point in classe_2]
y_classe_2 = [point[1] for point in classe_2]
# on calcule le centroide de la classe 2
centroide_2 = coordonnees_centroide(classe_2)
x_2 = centroide_2[0]
y_2 = centroide_2[1]

# on cherche ou sont les points de la classe 3 dans la liste data
coor_classe_3 = np.where(data[1] == 2)
# on met les coordonnees des points de la classe 3 dans une nouvelle liste
classe_3 = data[0][coor_classe_3]
x_classe_3 = [point[0] for point in classe_3]
y_classe_3 = [point[1] for point in classe_3]
# on calcul le centroide de la classe 3
centroide_3 = coordonnees_centroide(classe_3)
x_3 = centroide_3[0]
y_3 = centroide_3[1]


plt.plot(x_1, y_1, 'b^')
plt.plot(x_classe_1, y_classe_1, 'b.')
plt.plot(x_2, y_2, 'r^')
plt.plot(x_classe_2, y_classe_2, 'r.')
plt.plot(x_3, y_3, 'g^')
plt.plot(x_classe_3, y_classe_3, 'g.')
plt.show()

with open('heart.csv', newline='', encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        print('; '.join(row))
