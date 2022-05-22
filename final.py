# -*- coding: utf-8 -*-
"""
Fais .

test
"""

import csv
import time
from collections import Counter
import sys
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

HEART = ("Maladie cardiaque", "dataset_formater/heart.csv",
         "dataset_formater/heart_test.csv")
WATER_POTABILITY = ("Potabilité de l'eau",
                    "dataset_formater/water_potability.csv",
                    "dataset_formater/water_potability_test.csv")
DIABETES = ("Diabète", "dataset_formater/diabetes.csv",
            "dataset_formater/diabetes_test.csv")
IRIS = ("Iris", "dataset_formater/iris.csv", "dataset_formater/iris_test.csv")


# Définition de fonction utile pour les 3 algorithmes
# _____________________________________________________________________________


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


def calcul_coordonnees_centroide(points):
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


# Définition des fonctions pour faire un algorithme de classification centroide
# le plus proche .
# _____________________________________________________________________________


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


def comparaison_nearest_centroide(donnee, separateur=","):
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


print("Nearest centroide :\n_________________________________________________")
comparaison_nearest_centroide(HEART)
comparaison_nearest_centroide(WATER_POTABILITY)
comparaison_nearest_centroide(DIABETES)
comparaison_nearest_centroide(IRIS)


# _____________________________________________________________________________
# Définition des fonctions pour faire un algorithme de classification Balltree.
# _____________________________________________________________________________

def separe_liste(points):
    """
    Sépare la liste de point en 2 sous listes.

    Calcul le centroide de points et trouve les 2 points les plus éloigné du
    centroide. Puis sépare les points en fonction de leurs distance au 2 points
    les plus éloigné du centroide.

    Parameters
    ----------
    points : list
        liste à séparer.

    Returns
    -------
    points_1 : list
        1 er sous liste.
    points_2 : list
        2eme sous liste.

    """
    centroide = calcul_coordonnees_centroide(points)
    distances = []
    for point in points:
        distances.append(calcul_distance_euclidienne(point[:-1], centroide))
    distances_triee = sorted(distances)
    centre_1 = points[distances.index(distances_triee[-1])]
    centre_2 = points[distances.index(distances_triee[-2])]
    if (set(centre_1) == set(centre_2)):
        points.pop(distances.index(distances_triee[-1]))
        return [centre_1], points
    points_1 = []
    points_2 = []
    for point in points:
        dist_1 = calcul_distance_euclidienne(centre_1[:-1], point[:-1])
        dist_2 = calcul_distance_euclidienne(centre_2[:-1], point[:-1])
        if dist_1 < dist_2:
            points_1.append(point)
        else:
            points_2.append(point)
    return points_1, points_2


def ball_tree(dataset, profondeur):
    """
    Compartimente dataset selon l'algorithme BallTree.

    Parameters
    ----------
    dataset : list
        Liste de points à séparer.
    profondeur : int
        Nombre de fois que la liste de départ est séparé.

    Returns
    -------
    listes : list
        liste de listes créé à partir de dataset selon l'algorithme BallTree.

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
    Renvoie la classe la plus représenté dans la liste points.

    Parameters
    ----------
    points : list
        Liste de point pour laquelle il faut déterminer la classe.

    Returns
    -------
    classe : int
        classe la plus représenter dans la liste points.

    """
    classes = []
    for point in points:
        classes.append(point[-1])
    classe = Counter(classes).most_common(1)
    classe = classe[0][0]
    return classe


def centroide_classe_liste(listes):
    """
    Calcul les centroides et classes des listes de listes.

    Parameters
    ----------
    listes : listes
        liste des listes de points pour lesquelle on veut déterminer la classe
        et le centroide.

    Returns
    -------
    centroides : list
        Liste des coordonnées des centroides des listes de listes rangée dans
        le même ordre.
    classes : list
        Liste des classes des listes de listes rangée dans le même ordre.

    """
    centroides = []
    classes = []
    for points in listes:
        centroides.append(calcul_coordonnees_centroide(points))
        classes.append(classe_liste(points))
    return centroides, classes


def prediction(point, centroides, classes):
    """
    Recherche le centroides le plus proche du point et renvoie sa classe.

    Parameters
    ----------
    point : list
        Coordonnée du point pour lequelle on veux déterminé la classe.
    centroides : list
        Liste des centroides des sous .
    classes : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    dist_min = calcul_distance_euclidienne(centroides[0][:-1], point[:-1])
    centroide_min = centroides[0]
    for centroide in centroides:
        dist = calcul_distance_euclidienne(centroide[:-1], point[:-1])
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
    listes = ball_tree(recuperer_donnee_csv(dataset, separateur), precision)
    centroides, classes = centroide_classe_liste(listes)
    nb_bon = 0
    test_data = recuperer_donnee_csv(datatest, separateur=',')
    nb_test = len(test_data)
    for test in test_data:
        if prediction(test, centroides, classes) == test[-1]:
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    end = time.time()
    temps = (end - start) * 1000
    return fiabilite, temps


# Définition des fonctions pour utilisée l'algorithme ballTree de sklear.
# _____________________________________________________________________________


def apprentissage_sklearn(fichier, separateur=','):
    neigh = KNeighborsClassifier(algorithm='ball_tree')
    dataset = recuperer_donnee_csv(fichier, separateur)
    echantillon = []
    cibles = []
    for point in dataset:
        echantillon.append(point[:-1])
        cibles.append(point[-1])
    echantillon = np.resize(echantillon, (len(cibles), len(dataset[1])-1))
    neigh.fit(echantillon, cibles)
    return neigh


def test_donnees_sklearn(fichier, neigh,  separateur):
    datatest = recuperer_donnee_csv(fichier, separateur)
    nb_bon = 0
    nb_test = len(datatest)
    for point in datatest:
        if neigh.predict([point[:-1]]) == point[-1]:
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    return fiabilite


def ball_tree_sklearn(dataset, datatest, separateur=','):
    start = time.time()
    neigh = apprentissage_sklearn(dataset, separateur)
    fiabilite = test_donnees_sklearn(datatest, neigh, separateur)
    end = time.time()
    temps = (end - start) * 1000
    return fiabilite, temps


def comparaison_balltree(donnee, precision, separateur=","):
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
    fiabilite_2, temps_2 = ball_tree_sklearn(dataset, datatest)
    nb_classe = 2
    print(f"""Dataset : {nom}\nNombre de classe : {nb_classe :.0f}
    Notre algorithme :\n\tPrécision : {fiabilite_1 :.2f} %
    \tTemps d'execution : {temps_1 :.3f} ms\n\tAlgorithme du module :
    \tPrécision : {fiabilite_2 :.2f} %
    \tTemps d'execution : {temps_2 :.3f} ms\n""")



print("Balltree :\n_________________________________________________")
comparaison_balltree(HEART, 7)
comparaison_balltree(WATER_POTABILITY, 30)
comparaison_balltree(DIABETES, 30)
comparaison_balltree(IRIS, 30)
