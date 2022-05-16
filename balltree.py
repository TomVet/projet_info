# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:31:29 2022.

@author: fireb
"""

import nearest_centroide_v2 as n
from collections import Counter
import time


def trouver_coordonnees_centroide(points):
    """
    Calcule le centroide des points de liste_points de dimension.

    Parameters
    ----------
    points : list
        liste de coordonnée de point de même dimension.

    Returns
    -------
    centroide : list
        liste des coordonnée du centroide calculé.

    """
    centroide = []
    # on calcule la dimension de l'espace considéré pour le centroide
    # print('test')
    # print(points)
    nb_dimension = len(points[0])
    # on calcule les coordonnées du centroide dans chaque dimension
    for dimension in range(nb_dimension):
        somme = 0
        # on somme les coordonnées de chaque points
        for point in points:
            somme += point[dimension]
        # on ajoute la somme / par le nombre de point a coordonnées
        centroide.append(somme/len(points))
    return centroide


def separe_liste(points):
    """


    Parameters
    ----------
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    points_1 : TYPE
        DESCRIPTION.
    points_2 : TYPE
        DESCRIPTION.

    """
    centroide = trouver_coordonnees_centroide(points)
    distances = []
    for point in points:
        distances.append(n.calcul_distance_euclidienne(point[:-1], centroide))
    distances_triee = sorted(distances)
    centre_1 = points[distances.index(distances_triee[-1])]
    centre_2 = points[distances.index(distances_triee[-2])]
    points_1 = [centre_1]
    points_2 = [centre_2]
    for point in points:
        dist_1 = n.calcul_distance_euclidienne(centre_1[:-1], point[:-1])
        dist_2 = n.calcul_distance_euclidienne(centre_2[:-1], point[:-1])
        if dist_1 < dist_2:
            points_1.append(point)
        else:
            points_2.append(point)
    return points_1, points_2


def ball_tree(dataset, profondeur):
    """


    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    profondeur : TYPE
        DESCRIPTION.

    Returns
    -------
    listes : TYPE
        DESCRIPTION.

    """
    listes = separe_liste(dataset)
    for _ in range(profondeur - 1):
        nouvelle_listes = []
        for liste in listes:
            if len(liste) == 1:
                nouvelle_listes.append(liste)
            else:
                liste_1, liste_2 = separe_liste(liste)
                nouvelle_listes.append(liste_1)
                nouvelle_listes.append(liste_2)
        listes = nouvelle_listes
    return listes


def classe_liste(points):
    """
    Reste.

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    classes = []
    for point in points:
        classes.append(point[-1])
    classe = Counter(classes).most_common(1)
    return classe[0][0]


def centroide_classe_liste(listes):
    """


    Parameters
    ----------
    listes : TYPE
        DESCRIPTION.

    Returns
    -------
    centroides : TYPE
        DESCRIPTION.
    classes : TYPE
        DESCRIPTION.

    """
    centroides = []
    classes = []
    for ind, points in enumerate(listes):
        centroides.append(trouver_coordonnees_centroide(points))
        classes.append(classe_liste(points))
    return centroides, classes

def prediction(point, centroides, classes):
    """


    Parameters
    ----------
    point : TYPE
        DESCRIPTION.
    centroides : TYPE
        DESCRIPTION.
    classes : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    dist_min = n.calcul_distance_euclidienne(centroides[0][:-1], point[:-1])
    centroide_min = centroides[0]
    for centroide in centroides:
        dist = n.calcul_distance_euclidienne(centroide[:-1], point[:-1])
        if dist < dist_min:
            dist_min = dist
            centroide_min = centroide
    return classes[centroides.index(centroide_min)]


def classification_balltree(precision, dataset, datatest, separateur=','):
    """
    Reste.

    Parameters
    ----------
    precision : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.
    datatest : TYPE
        DESCRIPTION.
    separateur : TYPE, optional
        DESCRIPTION. La valeur par défaut est ','.

    Returns
    -------
    fiabilite : TYPE
        DESCRIPTION.
    temps : TYPE
        DESCRIPTION.

    """
    start = time.time()
    listes = ball_tree(n.recuperer_donnee_csv(dataset, separateur), precision)
    centroides, classes = centroide_classe_liste(listes)
    nb_bon = 0
    test_data = n.recuperer_donnee_csv(datatest, separateur=',')
    nb_test = len(test_data)
    for test in test_data:
        if prediction(test, centroides, classes) == test[-1]:
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    end = time.time()
    temps = (end - start) * 1000
    return fiabilite, temps


def comparaison(donnee, precision, separateur=","):
    """
    Compare notre algorithme et celui de scikit-learn.

    Parameters
    ----------
    donnee : tuple
        tuple contenant : (nom du dataset, chemin dataset, chemin datatest).
    separateur : string, optional
        string contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Print
    -------
    Dataset :
    Nombre de classe :
    Notre algorithme :
        Précision : 0.00 %
        Temps d'execution : 0.000 ms
    Algorithme du module :
        Précision : 0.00 %
        Temps d'execution : 0.000 ms

    """
    nom, dataset, datatest = donnee
    fiabilite_1, temps_1 = classification_balltree(
        precision, dataset, datatest, separateur)
    fiabilite_2, temps_2 = 1, 1
    nb_classe = 2
    print(f"""Dataset : {nom}\nNombre de classe : {nb_classe :.0f}
    Notre algorithme :\n\tPrécision : {fiabilite_1 :.2f} %
    \tTemps d'execution : {temps_1 :.3f} ms\n\tAlgorithme du module :
    \tPrécision : {fiabilite_2 :.2f} %
    \tTemps d'execution : {temps_2 :.3f} ms\n""")

comparaison(n.HEART, 7)
comparaison(n.WATER_POTABILITY, 30)
comparaison(n.DIABETES, 30)
comparaison(n.IRIS, 30)
