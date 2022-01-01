# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:03:56 2021

@author: TomDION
"""

import csv
import time
import numpy as np
from sklearn.neighbors import NearestCentroid

# On crée les fonctions necessaire à la réalisation d'un algorithme de classification centroide
# le plus proche

def calcul_coordonnees_centroide(liste_coordonne):
    """
    Calcule le centroide des points de liste_coordonne de dimension
    nb_dimension

    Parameters
    ----------
    liste_coordonne : list
        liste de coordonnee de point de meme dimension.

    Returns
    -------
    coordonnees : list
        liste des coordonnee du centroide calcule.

    """
    coordonnees = []
    # on calcule la dimension de l'espace considere pour le centroide
    nb_dimension = len(liste_coordonne[0])
    # on calcul les coordonnees du centroide dans chaque dimension
    for dimension in range(nb_dimension):
        somme = 0
        # on somme les coordonnees de chaque points
        for point in liste_coordonne:
            somme += point[dimension]
        # on ajoute la somme / par le nombre de point a coordonnees
        coordonnees = np.append(coordonnees, somme/len(liste_coordonne))
    return coordonnees

def calcul_distance_euclidienne(point_1, point_2):
    """
    Calcul de la distance euclidienne au carre entre les points 1 et 2

    Parameters
    ----------
    point_1 : list
        liste des coordonnees du point 1.
    point_2 : list
        liste des coordonnees du point 2.

    Returns
    -------
    distance : float
        distance euclidienne au carre entre les points 1 et 2.

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
    point : list
        liste des coordonnee du point.
    centroides : list
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
    data : list
        list de dimension 2.

    """
    with open(fichier, newline='', encoding='utf-8') as data_file:
        data=[]
        data_reader = csv.reader(data_file, delimiter=separateur)
        for ligne in data_reader:
            data.append(ligne)
        data = np.array(data)
        data = data.astype(np.float64)
        return data

def calcul_centroides(fichier, nb_classe, separateur=','):
    """
    Calcul les centroides pour chaque classe de fichier

    Parameters
    ----------
    fichier : string
        chemin du fichier csv avec les donnees d'entrtainement
        ce fichier ne doit contenir que des float.
    nb_classe : int
        nombre de classe dans le fichier
    separateur : string, optional
        string contenant le separateur utiliser dans fichier. The default is ','.

    Returns
    -------
    centroides : np.array
        liste des coordonnees des centroides de chaque classe.
    nb_parametres : int
        nombre de parametre pour definir chaque classe.

    """
    dataset = recuperer_donnee_csv(fichier, separateur)
    nb_parametres = len(dataset[1]) - 1
    centroides = []
    for classe in range(nb_classe):
        liste_classe = []
        for point in dataset:
            if point[-1] == classe:
                liste_classe.append(point[:nb_parametres])
        centroides = np.append(centroides, calcul_coordonnees_centroide(liste_classe))
    centroides = np.resize(centroides, (nb_classe, nb_parametres))
    return centroides, nb_parametres

def tester_data(fichier, centroides, nb_parametres, separateur=','):
    """
    test la precision de l'algorithme

    Parameters
    ----------
    fichier : string
        chemin du fichier csv avec les donnees de test
        ce fichier ne doit contenir que des float.
    centroides : np.array
        liste des coordonnees des centroides de chaque classe.
    nb_parametres : int
        nombre de parametre pour definir chaque classe.
    separateur : string, optional
        string contenant le separateur utiliser dans fichier. The default is ','.

    Returns
    -------
    nb_test : int
        nombre de test effectuer.
    nb_bon : int
        nombre de test reussi.

    """
    test_data = recuperer_donnee_csv(fichier, separateur)
    nb_test = len(test_data)
    nb_classe = len(centroides)
    nb_bon = 0
    for test in test_data:
        for classe in range(nb_classe):
            if (find_nearest_centroid(test[:nb_parametres], centroides), test[-1]) == (classe, classe):
                nb_bon += 1
                break
    return (nb_test, nb_bon)

def centroide_plus_proche(dataset, datatest, nb_classe, separateur=','):
    """
    Test l'algorithme avec comme données d'apprentissage dataset et comme
    données de test datatest puis calcul la précision et la vitesse d'éxecution
    du programme

    Parameters
    ----------
    dataset : string
        chemin du fichier csv avec les donnees d'entrtainement
        ce fichier ne doit contenir que des float.
    datatest : string
        chemin du fichier csv avec les donnees de test
        ce fichier ne doit contenir que des float.
    nb_classe : int
        nombre de classe dans le fichier.
    separateur : string, optional
        string contenant le separateur utiliser dans fichier. The default is ','.

    Returns
    -------
    precision : float
        precision de l'algorithme sur cet ensemble de donnees (en pourcentage).
    temps : float
        temps d'execution de la fonction en milliseconde

    """
    start = time.time()
    centroides, nb_parametres = calcul_centroides(dataset, nb_classe, separateur)
    nb_test, nb_bon = tester_data(datatest, centroides, nb_parametres, separateur)
    precision = nb_bon / nb_test * 100
    end = time.time()
    temps = (end - start) * 1000
    return precision, temps


# on crée les fonction pour comparer l'algorithme que l'on a fait avec celui d'une bibliothèque

def apprentissage(fichier, clf, separateur=','):
    """
    Ajuste le modèle en fonction des données de fichier

    Parameters
    ----------
    fichier : string
        chemin du fichier csv avec les donnees d'entrtainement
        ce fichier ne doit contenir que des float.
    clf : fonction
        fonction de classification de la bibliothèque scikitlearn, ici NearestCentroid().
    separateur : string, optional
        string contenant le separateur utiliser dans fichier. The default is ','.

    Returns
    -------
    None.

    """
    dataset = recuperer_donnee_csv(fichier, separateur)
    echantillon = []
    cibles = []
    for point in dataset:
        echantillon.append(point[:-1])
        cibles.append(point[-1])
    echantillon = np.resize(echantillon, (len(cibles), len(dataset[1])-1))
    clf.fit(echantillon, cibles)

def test_donnees(fichier, clf, separateur=','):
    """
    test l'algorithme de classification pour les données de fichier

    Parameters
    ----------
    fichier : string
        chemin du fichier csv avec les donnees de test
        ce fichier ne doit contenir que des float.
    clf : fonction
        fonction de classification de la bibliothèque scikitlearn, ici NearestCentroid().
    separateur : string, optional
        string contenant le separateur utiliser dans fichier. The default is ','.

    Returns
    -------
    precision : float
        precision de l'algorithme sur cet ensemble de donnees (en pourcentage).

    """
    datatest = recuperer_donnee_csv(fichier, separateur)
    nb_bon = 0
    nb_test = len(datatest)
    for point in datatest:
        if clf.predict([point[:-1]]) == point[-1]:
            nb_bon += 1
    precision = nb_bon / nb_test * 100
    return precision

def centroide_plus_proche_sklearn(dataset, datatest, separateur=','):
    """
    Réalise l'apprentissage et le test de l'algorithme

    Parameters
    ----------
    dataset : np.array
        chemin du fichier csv avec les donnees d'entrtainement
        ce fichier ne doit contenir que des float.
    datatest : np.array
        chemin du fichier csv avec les donnees de test
        ce fichier ne doit contenir que des float.
    separateur : string, optional
        string contenant le separateur utiliser dans fichier. The default is ','.

    Returns
    -------
    precision : float
        precision de l'algorithme sur cet ensemble de donnees (en pourcentage).
    temps : float
        temps d'execution de la fonction en milliseconde

    """
    start = time.time()
    clf = NearestCentroid()
    apprentissage(dataset, clf, separateur)
    precision = test_donnees(datatest, clf, separateur)
    end = time.time()
    temps = (end - start) * 1000
    return precision, temps


def comparaison(dataset, datatest, nb_classe, separateur=','):
    """
    Fait tourner l'algorithme de classification que l'on àa créée et celui de
    la bibliothèque scikitlearn et imprime les précision et temps d'exection
    des deux algorithmes

    Parameters
    ----------
    dataset : np.array
        chemin du fichier csv avec les donnees d'entrtainement
        ce fichier ne doit contenir que des float.
    datatest : np.array
        chemin du fichier csv avec les donnees de test
        ce fichier ne doit contenir que des float.
    nb_classe : int
        nombre de classe dans le fichier.
    separateur : string, optional
        string contenant le separateur utiliser dans fichier.
        The default is ','.

    Returns
    -------
    None.

    """
    precision_1, temps_1 = centroide_plus_proche(dataset, datatest, nb_classe, separateur)
    precision_2, temps_2 = centroide_plus_proche_sklearn(dataset, datatest, separateur)
    print(f"Notre algorithme :\n\tPrécision : {precision_1 :.2f} %\n\tTemps d'execution : \
{temps_1 :.3f} ms\nAlgorithme du module :\n\tPrécision : {precision_2 :.2f} %\n\tTemps \
d'execution : {temps_2 :.3f} ms\n")


comparaison('dataset_formater/heart.csv', 'dataset_formater/heart_test.csv', 2)
comparaison('dataset_formater/water_potability.csv', 'dataset_formater/water_potability_test.csv', 2, ';')
comparaison('dataset_formater/diabetes.csv', 'dataset_formater/diabetes_test.csv', 2, ';')
comparaison('dataset_formater/iris.csv', 'dataset_formater/iris_test.csv', 3, ';')
