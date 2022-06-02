# -*- coding: utf-8 -*-
"""
On cherche à comparer 3 algorithme de classification.

On s'intéresse ici à :
    Nearest centroide
    BallTree
    Naive Bayes
Nous implementerons ces 3 algorithmes et pour vérifié leurs implementation on
utilise le module sklearn dans lequelle ces algorithmes sont implémenté.

_______________________________________________________________________________

Pour tester les performances des algorithmes, ils sont testé sur 4 dataset:
    - Dataset HEART :
        https://www.kaggle.com/ronitf/heart-disease-uci
    - Dataset WATER_POTABILITY :
        https://www.kaggle.com/botrnganh/knearestneighbours
    - Dataset DIABETES :
        https://www.kaggle.com/abdallamahgoub/diabetes
    - Dataset IRIS :
        https://www.kaggle.com/shivamkumarsingh1/knn-classifier

_______________________________________________________________________________

Prérequis :
    Installer le module sklearn et tqdm
    Utiliser au minimum python 3.8
    Mettre le programe python et les datasets dans le même dossier.

Pour tester le programme il suffit d'executer le fichier.
"""

import csv
import time
from collections import Counter
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

HEART = ("Maladie cardiaque", "heart.csv", "heart_test.csv")
WATER_POTABILITY = ("Potabilité de l'eau", "water_potability.csv",
                    "water_potability_test.csv")
DIABETES = ("Diabète", "diabetes.csv", "diabetes_test.csv")
IRIS = ("Iris", "iris.csv", "iris_test.csv")


# Définition de fonction utile pour les 3 algorithmes.
# _____________________________________________________________________________


def recuperer_donnee_csv(fichier, separateur=","):
    """
    Créée une liste de liste contenant les données de `fichier`.

    Chaque ligne de `fichier` devient une sous liste de `data`.

    Paramètres
    ----------
    fichier : string
        Chemin du fichier csv a lire.
        Ce fichier ne doit contenir que des float.
    separateur : string, optional
        String contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Retours
    -------
    data : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant les points
        de `fichier`.
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
    Calcul de la distance euclidienne au carré entre `points_1` et `point_2`.

    Soient `point_1` = [x1, x2, ..., xn] et `point_2` = [y1, y2, ..., yn]
    alors distance = somme de i=1 à n de (xi - yi)**2 .

    Paramètres
    ----------
    point_1 : array_like
        Liste des coordonnées de `point_1` sans la classe.
    point_2 : array_like
        Liste des coordonnées de `point_2` sans la classe.

    Retours
    -------
    distance : float
        Distance euclidienne au carré entre `points_1` et `points_2`.

    """
    nb_parametre = len(point_1)
    distance = 0
    for parametre in range(nb_parametre):
        somme = (point_1[parametre] - point_2[parametre]) ** 2
        distance += somme
    return distance


def calcul_coordonnees_centroide(points):
    """
    Calcul le centroide des points de `points` de dimension `nb_dimension`.

    Soient (A1, A2, ..., An) n point de dimension nb_parametre
    centroide = 1/n * somme de i=1 à n de Ai .

    Paramètres
    ----------
    points : array_like
        Liste de la forme (nb_point, nb_parametre) contenant les coordonnées
        des points sans leurs classe.

    Retours
    -------
    centroide : array_like
        Liste des coordonnées du centroide calculé.

    """
    centroide = []
    nb_parametre = len(points[0])
    # on calcul les coordonnées du centroide dans chaque dimension
    for parametre in range(nb_parametre):
        somme = 0
        for point in points:
            somme += point[parametre]
        centroide.append(somme / len(points))
    return centroide


def recuperer_classe(data):
    """
    Renvoie une liste contenant la classe de chaque point.

    Paramètres
    ----------
    data : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant les points
        dont on veut extraire les classes.

    Retours
    -------
    classes : array_like
        Liste de la forme (nb_point) contenant les classes des points de
        `data` dans le même ordre que les points.

    """
    classes = []
    for point in data:
        classes.append(point[-1])
    return classes


def liste_donnes(points):
    """
    Renvoie la liste des points sans leurs classes.

    On retire donc le dernière élement de chaque point.

    Paramètres
    ----------
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant les points
        dont on veut extraire les coordonnées.
    Retours
    -------
    donnees : array_like
        Liste de la forme (nb_point, nb_parametre) contenant les points
        sans leurs classes.

    """
    donnees = [point[:-1] for point in points]
    return donnees


# _____________________________________________________________________________

# Définition des fonctions pour faire un algorithme de classification centroide
# le plus proche .
# _____________________________________________________________________________


def find_nearest_centroid(point, centroides):
    """
    Permet de trouver le centroide le plus proche du `point`.

    Calcul la distance entre le `point` et tout les centroides puis renvoie
    la classe du centroide pour lequel la distance est minimale.

    Paramètres
    ----------
    point : array_like
        Liste de la forme (nb_parametre) contenant les coordonnées du point.
    centroides : array_like
        Liste de la forme (nb_classes) contenant des tuples
        de la forme (classe_centroide, coordonnées_centroide).

    Retours
    -------
    classe_du_min : int
        Classe du centroide le plus proche de point dans la liste centroides.

    """
    distance_min = calcul_distance_euclidienne(point, centroides[0][1])
    classe_du_min = centroides[0][0]
    for classe, centroide in centroides:
        distance = calcul_distance_euclidienne(point, centroide)
        if distance_min > distance:
            distance_min = distance
            classe_du_min = classe
    return classe_du_min


def calcul_centroides(dataset):
    """
    Calcul les centroides pour chaque classe de `dataset`.

    Regroupe les points par classes puis calcul les centroides de chaque
    classes.

    Paramètres
    ----------
    dataset : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées.

    Retours
    -------
    centroides : array_like
        Liste de la forme (nb_classes, 2, nb_parametre) contenant des tuples
        de la forme (classe_centroide, coordonnées_centroide).
    nb_parametres : int
        Nombre de paramètres pour définir chaque point.
    classes : set
        Set des classes du dataset

    """
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


def tester_data(fichier, centroides, nb_parametres, separateur=","):
    """
    Test la précision de l'algorithme.

    Verifie la prédiction de l'algorithme en appliquant l'algorithme à des
    points connue dont on connait la classe que l'on compare avec la
    prédiction.

    Paramètres
    ----------
    fichier : string
        Chemin du fichier csv avec les données de test.
        Ce fichier ne doit contenir que des float.
    centroides : array_like
        Liste de la forme (nb_classes, 2, nb_parametre) contenant des tuples
        de la forme (classe_centroide, coordonnées_centroide).
    nb_parametres : int
        Nombre de paramètres pour définir chaque point.
    classes : set
        Set des classes du dataset
    separateur : string, optional
        String contenant le séparateur utilisé dans `fichier`.
        La valeur par défaut est ",".

    Retours
    -------
    fiabilite : float
        Précision de l'algorithme sur cet ensemble de données (en pourcentage).
    nb_test : int
        Nombre de test effectué.

    """
    test_data = recuperer_donnee_csv(fichier, separateur)
    nb_test = len(test_data)
    nb_bon = 0
    for test in test_data:
        if (find_nearest_centroid(test[:nb_parametres], centroides) ==
                test[-1]):
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    return fiabilite, nb_test


def centroide_plus_proche(dataset, datatest, separateur=","):
    """
    Test l'algorithme et renvoie le précision et la vitesse d'éxecution.

    Utilise comme données d'apprentissage `dataset` et comme
    données de test `datatest`.

    Paramètres
    ----------
    dataset : string
        Chemin du fichier csv avec les données d'entrainement.
        Ce fichier ne doit contenir que des float.
    datatest : string
        Chemin du fichier csv avec les données de test.
        Ce fichier ne doit contenir que des float.
    separateur : string, optional
        String contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Retours
    -------
    fiabilite : float
        précision de l'algorithme sur cet ensemble de données (en pourcentage).
    temps : float
        temps pour classer un point en milliseconde.

    """
    list_dataset = recuperer_donnee_csv(dataset, separateur)
    start = time.time()
    centroides, nb_parametres, classes = calcul_centroides(list_dataset)
    tps_app = (time.time() - start) * 1000
    start = time.time()
    fiabilite, nb_test = tester_data(datatest, centroides, nb_parametres,
                                     separateur)
    end = time.time()
    temps = (end - start) * 1000 / nb_test
    return fiabilite, temps, classes, tps_app


# Définition des fonctions pour utilisée l'algorithme nearest centroide de
# sklear.
# _____________________________________________________________________________


def apprentissage(fichier, clf, separateur=","):
    """
    Ajuste le modèle en fonction des données de fichier.

    Paramètres
    ----------
    fichier : string
        Chemin du fichier csv avec les données d'entrainement.
        Ce fichier ne doit contenir que des float.
    clf : fonction
        Fonction de classification de la bibliothèque scikitlearn,
        ici NearestCentroid().
    separateur : string, optional
        String contenant le separateur utiliser dans fichier.
        La valeur par défaut est ",".

    Retours
    -------
    None.

    """
    start = time.time()
    dataset = recuperer_donnee_csv(fichier, separateur)
    echantillon = []
    cibles = []
    for point in dataset:
        echantillon.append(point[:-1])
        cibles.append(point[-1])
    echantillon = np.resize(echantillon, (len(cibles), len(dataset[1]) - 1))
    clf.fit(echantillon, cibles)
    end = time.time()
    temps_app = (end - start) * 1000
    return temps_app


def test_donnees(fichier, clf, separateur=","):
    """
    Test l'algorithme de classification pour les données de `fichier`.

    Paramètres
    ----------
    fichier : string
        Chemin du fichier csv avec les donnees de test.
        Ce fichier ne doit contenir que des float.
    clf : fonction
        Fonction de classification de la bibliothèque scikitlearn,
        ici NearestCentroid().
    separateur : string, optional
        String contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Retours
    -------
    fiabilite : float
        Précision de l'algorithme sur cet ensemble de données (en pourcentage).
    nb_test : int
        Nombre de test éffectué pour calculer la fiabilité.

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

    Paramètres
    ----------
    dataset : string
        Chemin du fichier csv avec les données d'entrainement.
        Ce fichier ne doit contenir que des float.
    datatest : string
        Chemin du fichier csv avec les données de test.
        Ce fichier ne doit contenir que des float.
    separateur : string, optional
        String contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Retours
    -------
    fiabilite : float
        Précision de l'algorithme sur cet ensemble de données (en pourcentage).
    temps : float
        Temps pour classer un point en milliseconde.
    temps_app : float
        Temps pour réaliser l'apprentissage.

    """
    start = time.time()
    clf = NearestCentroid()
    temps_app = apprentissage(dataset, clf, separateur)
    fiabilite, nb_test = test_donnees(datatest, clf, separateur)
    end = time.time()
    temps = (end - start) * 1000 / nb_test
    return fiabilite, temps, temps_app


# _____________________________________________________________________________

# Définition des fonctions pour faire un algorithme de classification Balltree.
# _____________________________________________________________________________


def separe_liste(points):
    """
    Sépare la liste de point en 2 sous listes.

    Calcul le centroide de points et trouve les 2 points les plus éloigné du
    centroide. Puis sépare les points en fonction de leurs distance au 2
    points les plus éloigné du centroide.

    Paramètres
    ----------
    points : array_like
        Liste à séparer de la forme (nb_point, nb_parametre).

    Retours
    -------
    points_1 : array_like
        1er sous liste.
    points_2 : array_like
        2eme sous liste.

    """
    # Si la liste ne comporte que 2 points on sépare juste la liste en 2.
    if len(points) == 2:
        points_1 = [points[0]]
        points_2 = [points[1]]
    else:
        centroide = calcul_coordonnees_centroide(points)
        distances = []
        # On calcul les distance entre le centroide de `points` et les points
        # pour pouvoir trouver les 2 points les plus éloigné.
        for point in points:
            distances.append(calcul_distance_euclidienne(point[:-1],
                                                         centroide))
        distances_triee = sorted(distances)
        centre_1 = points[distances.index(distances_triee[-1])]
        centre_2 = points[distances.index(distances_triee[-2])]
        longueur = -3
        # Si `centre_1` = `centre_2` on choisi un autre point.
        while np.array_equal(centre_1, centre_2):
            centre_2 = points[distances.index(distances_triee[longueur])]
            longueur -= 1
            if longueur == - len(points):
                return points
        points_1 = []
        points_2 = []
        # On attribue chaque point de `points` à une des 2 nouvelle liste.
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

    Paramètres
    ----------
    dataset : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées.
    profondeur : int
        Nombre de fois que la liste de départ est séparé.

    Retours
    -------
    listes : array_like
        Liste de listes de points créé à partir de dataset selon l'algorithme
        BallTree.

    """
    listes = separe_liste(dataset)
    for _ in range(profondeur - 1):
        nouvelle_listes = []
        for liste in listes:
            # On ne sépare pas une liste contenant un unique point.
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
    Renvoie la classe la plus représenté dans la liste `points`.

    Paramètres
    ----------
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées avec en dernier leurs
        classes.

    Retours
    -------
    classe : int
        Classe la plus représenter dans la liste `points`.

    """
    classes = []
    for point in points:
        classes.append(point[-1])
    classe = Counter(classes).most_common(1)
    classe = classe[0][0]
    return classe


def centroide_classe_liste(listes):
    """
    Calcul les centroides et classes des listes de `listes`.

    Paramètres
    ----------
    listes : array_like
        Liste des listes de points pour lesquelle on veut déterminer les
        classes et les centroides.

    Retours
    -------
    centroides : array_like
        Liste des coordonnées des centroides des listes de `listes` rangée dans
        le même ordre.
    classes : array_like
        Liste des classes des listes de `listes` rangée dans le même ordre.

    """
    centroides = []
    classes = []
    for points in listes:
        centroides.append(calcul_coordonnees_centroide(points))
        classes.append(classe_liste(points))
    return centroides, classes


def prediction(point, centroides, classes):
    """
    Recherche le centroide le plus proche du `point` et renvoie sa classe.

    Paramètres
    ----------
    point : array_like
        Liste de la forme (nb_parametre + 1) contenant les coordonnées et la
        classe de `point`.
    centroides : array_like
        Liste de  la forme (nb_centroide, nb_parametre) contenant les
        coordonées des centroides.
    classes : array_like
        Liste de la forme (nb_centroiode) contenant les classes des centroides.

    Retours
    -------
    classe : float
        Classe prédite par le balltree pour `point`.

    """
    dist_min = calcul_distance_euclidienne(centroides[0][:-1], point[:-1])
    centroide_min = centroides[0]
    for centroide in centroides:
        dist = calcul_distance_euclidienne(centroide[:-1], point[:-1])
        if dist < dist_min:
            dist_min = dist
            centroide_min = centroide
    classe = classes[centroides.index(centroide_min)]
    return classe


def classification_balltree(profondeur, dataset, datatest, separateur=","):
    """
    Réalise l'apprentissage et le test de l'algorithme.

    Paramètres
    ----------
    profondeur : int
        Nombre de fois que la liste de départ est séparé.
    dataset : string
        Chemin du fichier csv avec les données d'entrainement.
        Ce fichier ne doit contenir que des float.
    datatest : string
        Chemin du fichier csv avec les données de test.
        Ce fichier ne doit contenir que des float.
    separateur : string, optional
        String contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Retours
    -------
    fiabilite : float
        Précision de l'algorithme sur cet ensemble de données (en pourcentage).
    temps : float
        Temps pour classer un point en milliseconde.
    temps_app : float
        Temps pour réaliser l'apprentissage.

    """
    start = time.time()
    listes = ball_tree(recuperer_donnee_csv(dataset, separateur), profondeur)
    centroides, classes = centroide_classe_liste(listes)
    temps_app = (time.time() - start) * 1000
    start = time.time()
    nb_bon = 0
    test_data = recuperer_donnee_csv(datatest, separateur=",")
    nb_test = len(test_data)
    for test in test_data:
        if prediction(test, centroides, classes) == test[-1]:
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    temps = (time.time() - start) * 1000 / nb_test
    return fiabilite, temps, temps_app


# Définition des fonctions pour utilisée l'algorithme ballTree de sklear.
# _____________________________________________________________________________


def apprentissage_sklearn(fichier, separateur=","):
    """
    Réalise l'apprentissage pour le balltree de sklearn.

    Paramètres
    ----------
    fichier : string
        Chemin du fichier csv avec les données d'entrainement.
        Ce fichier ne doit contenir que des float.
    separateur : string, optional
        String contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".
    Retours
    -------
    tree : object
        Classifier sklearn.
    """
    tree = KNeighborsClassifier(
        algorithm="ball_tree", leaf_size=10, n_neighbors=3)
    dataset = recuperer_donnee_csv(fichier, separateur)
    tree.fit(liste_donnes(dataset), recuperer_classe(dataset))
    return tree


def test_donnees_sklearn(fichier, tree, separateur=","):
    """
    Test le Balltree de sklearn.

    Paramètres
    ----------
    fichier : string
        Chemin du fichier csv avec les données d'entrainement.
        Ce fichier ne doit contenir que des float.
    tree : object
        Classifier sklearn.
    separateur : string, optional
        String contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Retours
    -------
    fiabilite : float
        Précision de l'algorithme sur cet ensemble de données (en pourcentage).
    nb_test : int
        Nombre de test éffectué pour calculer la fiabilité.

    """
    datatest = recuperer_donnee_csv(fichier, separateur)
    nb_bon = 0
    nb_test = len(datatest)
    prediction_sk = tree.predict(liste_donnes(datatest))
    classes = recuperer_classe(datatest)
    for indice, classe in enumerate(prediction_sk):
        if classe == classes[indice]:
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    return fiabilite, nb_test


def ball_tree_sklearn(dataset, datatest, separateur=","):
    """
    Réalise l'apprentissage et le test du balltree de sklearn.

    Paramètres
    ----------
    dataset : string
        Chemin du fichier csv avec les données d'entrainement.
        Ce fichier ne doit contenir que des float.
    datatest : string
        Chemin du fichier csv avec les données de test.
        Ce fichier ne doit contenir que des float.
    separateur : string, optional
        String contenant le séparateur utilisé dans fichier.
        La valeur par défaut est ",".

    Retours
    -------
    fiabilite : float
        Précision de l'algorithme sur cet ensemble de données (en pourcentage).
    temps : float
        Temps pour classer un point en milliseconde.
    temps_app : float
        Temps pour réaliser l'apprentissage.

    """
    start = time.time()
    tree = apprentissage_sklearn(dataset, separateur)
    end_1 = time.time()
    fiabilite, nb_test = test_donnees_sklearn(datatest, tree, separateur)
    end_2 = time.time()
    temps_app = (end_1 - start) * 1000
    temps = (end_2 - start) * 1000 / nb_test
    return fiabilite, temps, temps_app


# _____________________________________________________________________________

# Définition des fonctions pour faire un algorithme de classification Naive
# Bayes.
# _____________________________________________________________________________


def proba_naives_sklearn(point, points):
    """
    Reste.

    Paramètres
    ----------
    point : array_like
        Liste de la forme (nb_parametre) contenant les coordonnées du
        `point`.
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées avec leurs classes.

    Retours
    -------
    prediction_sk : float
        Classe prédite par Naive Bayes pour `point`.


    """
    points_sc = liste_donnes(points)
    classes = recuperer_classe(points)
    clf = GaussianNB()
    clf.fit(points_sc, classes)
    GaussianNB()
    clf_pf = GaussianNB()
    clf_pf.partial_fit(points_sc, classes, np.unique(classes))
    GaussianNB()
    prediction_sk = clf.predict(point)
    return prediction_sk


# naive bayes programmé


def calcul_ecart(points):
    """
    Calcul l'écart entre la première classe et 0.

    Paramètres
    ----------
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées avec leurs classes.

    Retours
    -------
    ecart : int

    """
    classes = set(point[-1] for point in points)
    ecart = int(min(classes))
    return ecart


def calcul_esp(points):
    """
    Calcul l'esperance de chaque catégorie en fonction de sa classe.

    Paramètres
    ----------
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées avec leurs classes.

    Retours
    -------
    esperance_par_classe : array_like
        Liste de la forme (nb_classe, nb_parametre) contenant les espérances
        de chaque paramètre pour chaque classe.

    """
    dim_point = len(points[0]) - 1
    esperance_par_classe = []
    classes = {point[-1] for point in points}
    for classe in classes:
        esperance = []
        for rang in range(dim_point):
            somme = 0
            compteur = 0
            for point in points:
                if point[-1] == classe:
                    compteur += 1
                    somme += point[rang]
            esperance.append(somme / compteur)
        esperance_par_classe.append([esperance, classe])
    return esperance_par_classe


def calcul_var(points, ecart):
    """
    Renvoie la variance de chaque catégorie en fonction de sa classe.

    Paramètres
    ----------
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées avec leurs classes.

    Retours
    -------
    variance_par_classe : array_like
        Liste de la forme (nb_classe, nb_parametre) contenant les variances
        de chaque paramètre pour chaque classe.

    """
    dim_point = len(points[0]) - 1
    variance_par_classe = []
    classes = set(point[-1] for point in points)
    esp = calcul_esp(points)
    for classe in classes:
        variance = []
        for rang in range(dim_point):
            somme = 0
            compteur = 0
            for point in points:
                if point[-1] == classe:
                    compteur += 1
                    somme += (
                        point[rang] - esp[int(classe - ecart)][0][rang])**2
            variance.append((somme / (compteur - 1)))
        variance_par_classe.append([variance, classe])
    return variance_par_classe


def calcul_proba_classe(points):
    """
    Calcul les probas pour les points de `points` d'être dans chaque classes.

    Paramètres
    ----------
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées avec leurs classes.

    Retours
    -------
    liste_proba : array_like
        Liste de la forme (nb_classe) contenant la probabilité d'appartenir
        à chaque classe.
    """
    liste_classe = set(point[-1] for point in points)
    liste_proba = []
    for classe in liste_classe:
        compteur = 0
        for point in points:
            if point[-1] == classe:
                compteur += 1
        liste_proba.append([compteur / len(points), classe])
    return liste_proba


def calcul_proba_categorie_sachant_classe(
    point, categorie, classe, ecart, esperance, variance
):
    """
    Calcul les proban de chaque paramètres sachant la classe de `point`.

    Paramètres
    ----------
    point : array_like
        Liste de la forme (nb_parametre) contenant les coordonnées du point.
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées.
    categorie : int
        Rang du paramètre étudié.
    classe : int
        Rrang de la classe étudié.
    ecart : int
    esperance : array_like
        Liste de la forme (nb_classe, nb_parametre) contenant les espérances
        de chaque paramètre pour chaque classe.
    variance : array_like
        Liste de la forme (nb_classe, nb_parametre) contenant les variances
        de chaque paramètre pour chaque classe.
    Retours
    -------
    proba : int
        Probabilité de la catégorie en fonction de la classe.
    """
    esp = esperance[int(classe - ecart)][0][categorie]
    var = variance[int(classe - ecart)][0][categorie]
    proba = np.exp((-((point[0][categorie] - esp) ** 2) / (2 * var))) / (
        np.sqrt(2 * np.pi * var)
    )
    return proba


def calcul_proba_bayes(point, points):
    """
    Calcul la classe à laquelles `point` à le plus de chance d'appartenir.

    Paramètres
    ----------
    point : array_like
        Liste de la forme (nb_parametre) contenant les coordonnées du
        `point`.
    points : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        représenter par les listes de leurs coordonnées avec leurs classes.

    Retours
    -------
    prediction_b : float
        Classe prédite por `point`.

    """
    ecart = calcul_ecart(points)
    proba_classe = calcul_proba_classe(points)
    classes = set(point[-1] for point in points)
    liste_proba = []
    esp = calcul_esp(points)
    var = calcul_var(points, ecart)
    for classe in classes:
        proba_categorie_sachant_classe = 1
        for i in range(len(points[0]) - 1):
            proba_categorie_sachant_classe = (
                proba_categorie_sachant_classe
                * calcul_proba_categorie_sachant_classe(
                    point, i, classe, ecart, esp, var
                )
            )
        liste_proba.append([(proba_classe[int(classe - ecart)][0] *
                             proba_categorie_sachant_classe), classe])
    prediction_nb = max(liste_proba)[1]
    return prediction_nb


def comparateur(liste_test, dataset):
    """
    Test les 2 algorithmes.

    Paramètres
    ----------
    liste_test : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        dont on connait la classe.
    dataset : array_like
        Liste de la forme (nb_point, nb_parametre + 1) contenant des points
        servant pour l'apprentissage.

    Retours
    -------
    fiabilite_sk : float
        Précision de sklearn sur cet ensemble de données (en pourcentage).
    temps_sk : float
        Temps pour classer un point avec sklearn en milliseconde.
    fiabilite_nb : float
        Précision de `calcul_naive_bayes` sur cet ensemble de données
        (en pourcentage).
    temps_nb : float
        Temps pour classer un point avec `calcul_naive_bayes` en milliseconde.

    """
    classes = recuperer_classe(liste_test)
    points = liste_donnes(liste_test)
    temps_sk = 0
    temps_nb = 0
    fiabilite_sk = 0
    fiabilite_nb = 0
    taille = len(liste_test)
    for i in range(taille):
        start = time.time()
        prediction_sk = proba_naives_sklearn([points[i]], dataset)
        temps_sk += time.time() - start
        start = time.time()
        prediciton_b = calcul_proba_bayes([points[i]], dataset)
        temps_nb += time.time() - start
        if prediction_sk == classes[i]:
            fiabilite_sk += 1
        if prediciton_b == classes[i]:
            fiabilite_nb += 1
    temps_sk = temps_sk / taille * 1000
    temps_nb = temps_nb / taille * 1000
    fiabilite_sk = fiabilite_sk / taille * 100
    fiabilite_nb = fiabilite_nb / taille * 100
    return fiabilite_sk, temps_sk, fiabilite_nb, temps_nb


def comparaison(donnee, precision, separateur=","):
    """
    Reste.

    Paramètres
    ----------
    donnee : TYPE
        DESCRIPTION.
    precision : TYPE
        DESCRIPTION.
    separateur : TYPE, optional
        DESCRIPTION. The default is ",".

    Retours
    -------
    None.

    """
    nom, dataset, datatest = donnee
    fiabilite_1, tps_point_1, classes, tps_app_1 = centroide_plus_proche(
        dataset, datatest, separateur
    )
    fiabilite_2, tps_point_2, tps_app_2 = centroide_plus_proche_sklearn(
        dataset, datatest, separateur
    )
    nb_classe = len(classes)
    fiabilite_3, tps_point_3, tps_app_3 = classification_balltree(
        precision, dataset, datatest, separateur
    )
    fiabilite_4, tps_point_4, tps_app_4 = ball_tree_sklearn(dataset, datatest)
    data = recuperer_donnee_csv(dataset)
    test = recuperer_donnee_csv(datatest)
    fiabilite_5, tps_point_5, fiabilite_6, tps_point_6 = comparateur(test,
                                                                     data)
    textes = [
        (
            fiabilite_1,
            tps_point_1,
            tps_app_1,
            fiabilite_2,
            tps_point_2,
            tps_app_2,
            "\tNearest centroide :",
        ),
        (
            fiabilite_3,
            tps_point_3,
            tps_app_3,
            fiabilite_4,
            tps_point_4,
            tps_app_4,
            "\tBalltree :",
        ),
    ]
    print(
        f"""---------------------------------------\nDataset : {nom}
Nombre de classe : {nb_classe :.0f}"""
    )
    for text in textes:
        (
            fiabilite_1,
            tps_point_1,
            tps_app_1,
            fiabilite_2,
            tps_point_2,
            tps_app_2,
            algo,
        ) = text
        print(algo + "\n\t___________________")
        print(
            f"""\t\tNotre algorithme :\n\t\t\tPrécision : {fiabilite_1 :.2f} %
        \tTemps apprentissage : {tps_app_1 :.2f} ms
        \tTemps classement d'un point : {tps_point_1 :.2f} ms\n\n\t\tSklearn :
        \tPrécision : {fiabilite_2 :.2f} %
        \tTemps apprentissage : {tps_app_2 :.2f} ms
        \tTemps classement d'un point : {tps_point_2 :.2f} ms"""
        )
    print(
        f"""\tNaives Bayes :\n\t___________________\n\t\tNotre Algorithme :
        \tPrécision : {fiabilite_6 :.2f} %
        \tTemps classement d'un point : {tps_point_6 :.2f} ms\n\n\t\tSklearn :
        \tPrécision : {fiabilite_5 :.2f} %
        \tTemps classement d'un point : {tps_point_5 :.2f} ms"""
    )
    print("---------------------------------------")


comparaison(HEART, 15)
comparaison(DIABETES, 15)
comparaison(IRIS, 15)
comparaison(WATER_POTABILITY, 15)
