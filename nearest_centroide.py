# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:03:56 2021

@author: fireb
"""

import numpy as np
# import matplotlib.pyplot as plt
import sklearn.datasets as dataset

data = dataset.make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1)

test = np.array([(1,3),(3,2),(9,4)])

def coordonnees_centroide(liste_coordonne):
    """
    Calcule le centroide des points de liste_coordonne dans de dimension
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
    # on calcule la dimension de l'espace considérer pour le centroide
    nb_dimension = len(liste_coordonne[1])
    # on calcul les coordonnees du centroide dans chaque dimension
    for dimension in range(nb_dimension):
        somme = 0
        # on somme les coordonnees de chaque point
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
    for indice, centroide in enumerate(centroides):
        distance = distance_euclidienne(point, centroide)
        if distance_min < distance:
            distance_min = distance
            indice_du_min = indice
    return indice_du_min
