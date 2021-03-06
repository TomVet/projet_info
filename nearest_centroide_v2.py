# -*- coding: utf-8 -*-


import csv
import time
import numpy as np
from sklearn.neighbors import NearestCentroid

HEART = ("Maladie cardiaque", "dataset_formater/heart.csv",
         "dataset_formater/heart_test.csv")
WATER_POTABILITY = ("Potabilité de l'eau",
                    "dataset_formater/water_potability.csv",
                    "dataset_formater/water_potability_test.csv")
DIABETES = ("Diabète", "dataset_formater/diabetes.csv",
            "dataset_formater/diabetes_test.csv")
IRIS = ("Iris", "dataset_formater/iris.csv", "dataset_formater/iris_test.csv")


# Définition des fonctions pour faire un algorithme de classification centroide
# le plus proche .
# _____________________________________________________________________________

def calcul_coordonnees_centroide(liste_coordonne):
    """
    Calcule le centroide des points de liste_coordonne de dimension.

    Parameters
    ----------
    liste_coordonne : list
        liste de coordonnée de point de même dimension.

    Returns
    -------
    coordonnees : list
        liste des coordonnée du centroide calculé.

    """
    coordonnees = []
    # on calcule la dimension de l'espace considéré pour le centroide
    nb_dimension = len(liste_coordonne[0])
    # on calcule les coordonnées du centroide dans chaque dimension
    for dimension in range(nb_dimension):
        somme = 0
        # on somme les coordonnées de chaque points
        for point in liste_coordonne:
            somme += point[dimension]
        # on ajoute la somme / par le nombre de point a coordonnées
        coordonnees = np.append(coordonnees, somme/len(liste_coordonne))
    return coordonnees


def calcul_distance_euclidienne(point_1, point_2):
    """
    Calcule de la distance euclidienne au carré entre les points 1 et 2.

    Parameters
    ----------
    point_1 : list
        liste des coordonnées du point 1.
    point_2 : list
        liste des coordonnées du point 2.

    Returns
    -------
    distance : float
        distance euclidienne au carré entre les points 1 et 2.

    """
    # on calcule la dimension de l'espace des 2 points
    nb_dimension = len(point_1)
    distance = 0
    # on fait la somme au carré des coordonnées des points 1 et 2 dans chaque
    # dimension
    for dimension in range(nb_dimension):
        somme = (point_1[dimension] - point_2[dimension]) ** 2
        distance += somme
    return distance


def find_nearest_centroid(point, centroides):
    """
    Permet de trouver le centroide le plus proche du point.

    Parameters
    ----------
    point : list
        liste des coordonnées du point.
    centroides : list
        liste de coordonnée de centroides.

    Returns
    -------
    classe_du_min : int
        classe du centroide le plus proche de point dans la liste centroides.

    """
    distance_min = calcul_distance_euclidienne(point, centroides[0][1])
    classe_du_min = centroides[0][0]
    # on parcoure la liste des centroides
    for classe, centroide in centroides:
        # on calcule la distance entre le centroide et le point
        distance = calcul_distance_euclidienne(point, centroide)
        # si la nouvelle distance est plus petite que le minimum
        # elle devient le minimum
        if distance_min > distance:
            distance_min = distance
            # on conserve la classe du centroide le plus proche
            classe_du_min = classe
    return classe_du_min


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
        La valeur par défaut est ",".

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


def calcul_centroides(fichier, separateur=","):
    """
    Calcule les centroides pour chaque classe de fichier.

    Parameters
    ----------
    fichier : string
        chemin du fichier csv avec les données d'entrainement
        ce fichier ne doit contenir que des float.
    separateur : string, optional
        string contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Returns
    -------
    centroides : np.array
        liste des coordonnées des centroides de chaque classe.
    nb_parametres : int
        nombre de parametres pour définir chaque point.
    classes : set
        set des classes du dataset

    """
    dataset = recuperer_donnee_csv(fichier, separateur)
    nb_parametres = len(dataset[1]) - 1
    centroides = []
    classes = {point[-1] for point in dataset}
    for classe in classes:
        liste_classe = []
        for point in dataset:
            if point[-1] == classe:
                liste_classe.append(point[:nb_parametres])
        centroide = classe, calcul_coordonnees_centroide(liste_classe)
        centroides.append(centroide)
    return centroides, nb_parametres, classes


def tester_data(fichier, centroides, nb_parametres, classes, separateur=","):
    """
    Test la précision de l'algorithme.

    Parameters
    ----------
    fichier : string
        chemin du fichier csv avec les données de test
        ce fichier ne doit contenir que des float.
    centroides : np.array
        liste des coordonnées des centroides de chaque classe.
    nb_parametres : int
        nombre de paramètres pour définir chaque classe.
    classes : set
        set des classes du dataset
    separateur : string, optional
        string contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Returns
    -------
    nb_test : int
        nombre de test effectuer.
    nb_bon : int
        nombre de test réussi.

    """
    test_data = recuperer_donnee_csv(fichier, separateur)
    nb_test = len(test_data)
    nb_bon = 0
    for test in test_data:
        for classe in classes:
            if (find_nearest_centroid(test[:nb_parametres], centroides),
                    test[-1]) == (classe, classe):
                nb_bon += 1
                break
    return nb_test, nb_bon


def centroide_plus_proche(dataset, datatest, separateur=","):
    """
    Test l'algorithme et renvoie le précision et la vitesse d'éxecution.

    Utilise comme données d'apprentissage dataset et comme
    données de test datatest

    Parameters
    ----------
    dataset : string
        chemin du fichier csv avec les données d'entrainement
        ce fichier ne doit contenir que des float.
    datatest : string
        chemin du fichier csv avec les données de test
        ce fichier ne doit contenir que des float.
    separateur : string, optional
        string contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Returns
    -------
    fiabilite : float
        précision de l'algorithme sur cet ensemble de données (en pourcentage).
    temps : float
        temps pour classer un point en milliseconde.

    """
    start = time.time()
    centroides, nb_parametres, classes = calcul_centroides(dataset, separateur)
    nb_test, nb_bon = tester_data(datatest, centroides, nb_parametres,
                                  classes, separateur)
    fiabilite = nb_bon / nb_test * 100
    end = time.time()
    temps = (end - start) * 1000 / nb_test
    return fiabilite, temps, classes


# Définition des fonctions pour utilisée l'algorithme nearest centroide de
# sklear.
# _____________________________________________________________________________


def apprentissage(fichier, clf, separateur=","):
    """
    Ajuste le modèle en fonction des données de fichier.

    Parameters
    ----------
    fichier : string
        chemin du fichier csv avec les données d'entrainement
        ce fichier ne doit contenir que des float.
    clf : fonction
        fonction de classification de la bibliothèque scikitlearn,
        ici NearestCentroid().
    separateur : string, optional
        string contenant le separateur utiliser dans fichier.
        La valeur par défaut est ",".

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


def test_donnees(fichier, clf, separateur=","):
    """
    Test l'algorithme de classification pour les données de fichier.

    Parameters
    ----------
    fichier : string
        chemin du fichier csv avec les donnees de test
        ce fichier ne doit contenir que des float.
    clf : fonction
        fonction de classification de la bibliothèque scikitlearn,
        ici NearestCentroid().
    separateur : string, optional
        string contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Returns
    -------
    fiabilite : float
        précision de l'algorithme sur cet ensemble de données (en pourcentage).
    nb_test : int
        nombre de test éffectué pour calculer la fiabilité

    """
    datatest = recuperer_donnee_csv(fichier, separateur)
    nb_bon = 0
    nb_test = len(datatest)
    for point in datatest:
        if clf.predict([point[:-1]]) == point[-1]:
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    return fiabilite, nb_test


def centroide_plus_proche_sklearn(dataset, datatest, separateur=","):
    """
    Réalise l'apprentissage et le test de l'algorithme.

    Parameters
    ----------
    dataset : np.array
        chemin du fichier csv avec les données d'entrainement
        ce fichier ne doit contenir que des float.
    datatest : np.array
        chemin du fichier csv avec les données de test
        ce fichier ne doit contenir que des float.
    separateur : string, optional
        string contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Returns
    -------
    fiabilite : float
        précision de l'algorithme sur cet ensemble de données (en pourcentage).
    temps : float
        temps pour classer un point en milliseconde.

    """
    start = time.time()
    clf = NearestCentroid()
    apprentissage(dataset, clf, separateur)
    fiabilite, nb_test = test_donnees(datatest, clf, separateur)
    end = time.time()
    temps = (end - start) * 1000 / nb_test
    return fiabilite, temps


def comparaison(donnee, separateur=","):
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
    fiabilite_1, temps_1, nb_classe = centroide_plus_proche(dataset, datatest,
                                                            separateur)
    fiabilite_2, temps_2 = centroide_plus_proche_sklearn(dataset, datatest,
                                                         separateur)
    nb_classe = len(nb_classe)
    print(f"""Dataset : {nom}\nNombre de classe : {nb_classe :.0f}
    Notre algorithme :\n\tPrécision : {fiabilite_1 :.2f} %
    \tTemps d'execution : {temps_1 :.3f} ms\n\tAlgorithme du module :
    \tPrécision : {fiabilite_2 :.2f} %
    \tTemps d'execution : {temps_2 :.3f} ms\n""")


# print("Nearest centroide :\n_________________________________________________")
# comparaison(HEART)
# comparaison(WATER_POTABILITY)
# comparaison(DIABETES)
# comparaison(IRIS)
