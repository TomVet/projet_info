from collections import Counter
import csv
import time
import numpy as np
from sklearn.neighbors import BallTree

heart = 'C:/Users/felix/Downloads/Informatique/projet_info-main/projet_info-main/dataset_formater/heart.csv'
heart_test = 'C:/Users/felix/Downloads/Informatique/projet_info-main/projet_info-main/dataset_formater/heart_test.csv'
water_potability = 'dataset_formater/water_potability.csv'
water_potability_test = 'dataset_formater/water_potability_test.csv'
diabetes = 'dataset_formater/diabetes.csv'
diabetes_test = 'dataset_formater/diabetes_test.csv'
iris = 'dataset_formater/iris.csv'
iris_test = 'dataset_formater/iris_test.csv'


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

def trouver_coordonnees_centroide(liste_coordonne):
    """
    Calcule le centroide des points de liste_coordonne de dimension
    nb_dimension
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
    nb_dimension = len(liste_coordonne[0][:-1])
    # on calcule les coordonnées du centroide dans chaque dimension
    for dimension in range(nb_dimension):
        somme = 0
        # on somme les coordonnées de chaque points
        for point in liste_coordonne:
            somme += point[dimension]
        # on ajoute la somme / par le nombre de point a coordonnées
        coordonnees = np.append(coordonnees, somme/len(liste_coordonne))
    coordonnees = np.append(coordonnees, 0)
    return coordonnees

def D_euclidienne(point_1, point_2):
    """
    Calcule de la distance euclidienne au carré entre les points 1 et 2
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
    nb_dimension = len(point_1[:-1])
    distance = 0
    # on fait la somme au carré des coordonnées des points 1 et 2 dans chaque dimension
    for dimension in range(nb_dimension):
        somme = (point_1[dimension] - point_2[dimension])**2
        distance += somme
    return distance


def trouver_P_loin(liste):
#la fonction va determiner le point le plus eloigné du centroid de la liste
    centroid = trouver_coordonnees_centroide(liste)
    point_1 = liste[0]
    for indice in range(len(liste)): # si la distance entre le point de rang k et le centroid est inferieur à la distance entre le point de rang i et le centroid, alors k revient i

        point_2 = centroid
        point_3 = liste[indice]
        if D_euclidienne(point_1, point_2) < D_euclidienne(point_2, point_3):
            point_1 = liste[indice]
    return point_1

def creer_nouvelleliste(liste, P_loin):
    # on cree une liste qui va prendre tout les points situés dans la sphere de rayon (distance "point le plus loin" jusquà "centroid")
    nouvelleliste  =  []
    centroid = trouver_coordonnees_centroide(liste)
    for point in liste:
        if D_euclidienne(point, P_loin) <= D_euclidienne(centroid, P_loin):
            nouvelleliste.append(point)
    return (nouvelleliste)

def creer_liste_complementaire(liste, P_loin):
    # on cree une liste qui va prendre tout les autres points n'ayant pas été pris dans nouvelleliste
    liste_complementaire = []
    centroid = trouver_coordonnees_centroide(liste)
    for point in liste:
        if D_euclidienne(point,P_loin) > D_euclidienne(centroid,P_loin):
            liste_complementaire.append(point)
    return (liste_complementaire)

def balltree(precision, liste):
# la précision correspond au nombre de sous liste qui vont etre formé à partir de la liste initial (représente donc aussi le nombre d'itération du programme)
    listes_crees = [liste]
# rassemble l'ensemble des listes crées qui sont des sous ensembles de la liste initiales
    liste_a_diviser = liste
    for tour in range(precision):
        P_loin = trouver_P_loin(liste_a_diviser)
        nouvelleliste = creer_nouvelleliste(liste_a_diviser, P_loin)
        liste_complementaire = creer_liste_complementaire(liste_a_diviser, P_loin)
# on a fractionné la liste initiale en 2 listes qui regroupent ensemble tout les points
        listes_crees.pop(0) #on supprime la liste qui a été divisée
        listes_crees.append(nouvelleliste) # on ajoute les 2 listes créés qui vont devenir de futur listes à diviser
        listes_crees.append(liste_complementaire)
        liste_a_diviser = listes_crees[0]
# on recommence le programme avec une des 2 listes qui viennent d'etre crées
    return listes_crees



def creer_liste_centroid(liste_de_listes): # on definit une fonction qui va déterminer le centroid de toutes les sous listes crées par balltree (liste_de_listes est le listes_crees)
    liste_centroid = []
    for liste in liste_de_listes:
        centroid_de_laliste = trouver_coordonnees_centroide(liste)
        liste_centroid.append(centroid_de_laliste)
    return liste_centroid

def retrouver_liste(centroid_obtenu, liste_de_listes, liste_centroid):
    rang=liste_centroid.index(centroid_obtenu)
    return liste_de_listes[rang] #renvoie la sous liste associé au nouveau point donné pour la prédiction

def centroide_plus_proche(liste_centroid, nouveau_point): # on cherche le centroid le plus proche du nouveau point
    indice_centroid_proche = 0
    centroid_proche = liste_centroid[indice_centroid_proche]
    for indice in range(len(liste_centroid)):
        point_1 = liste_centroid[indice_centroid_proche]
        point_2 = liste_centroid[indice]
        if D_euclidienne(point_1,nouveau_point) > D_euclidienne(point_2,nouveau_point):
            indice_centroid_proche = indice

    return centroid_proche


def classe_liste(points):
    classes = []
    for point in points:
        classes.append(point[-1])
        classe = Counter(classes).most_common(1)
    return classe[0][0]

def prediction(point, liste_de_listes, centroides):
    centroide = centroide_plus_proche(centroides, point)
    classe = classe_liste(retrouver_liste(centroide, liste_de_listes , centroides))
    return classe

def classification_balltree(precision, dataset, datatest, separateur=','):
    start = time.time()
    listes = balltree(precision, recuperer_donnee_csv(dataset, separateur))
    centroides = creer_liste_centroid(listes)
    nb_bon = 0
    test_data = recuperer_donnee_csv(datatest, separateur=',')
    nb_test = len(test_data)
    for test in test_data:
        if prediction(test, listes, centroides) == test[-1]:
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    end = time.time()
    temps = (end - start) * 1000
    return fiabilite, temps

def ball_tree_sklearn(precison, listes):
    clf = BallTree(listes)
    indice = clf.query(listes, k=1, return_distance=False)
    centroid_proche = liste_centroid[indice]
    return centroid_proche

def prediction_sklearn(liste_de_listes, centroides):
    centroide = ball_tree_sklearn(precision, liste)
    classe = classe_liste(retrouver_liste(centroide, liste_de_listes , centroides))
    return classe

def classification_balltree_sklearn(precision, dataset, datatest, separateur=','):
    start = time.time()
    listes = ball_tree_sklearn(precision, recuperer_donnee_csv(dataset, separateur))
    centroides = creer_liste_centroid(listes)
    nb_bon = 0
    test_data = recuperer_donnee_csv(datatest, separateur=',')
    nb_test = len(test_data)
    for test in test_data:
        if prediction_sklearn(listes, centroides) == test[-1]:
            nb_bon += 1
    fiabilite = nb_bon / nb_test * 100
    end = time.time()
    temps = (end - start) * 1000
    return fiabilite, temps

from sklearn.neighbors import KNeighborsClassifier

def ball_tree_sklearn(dataset, datatest, separateur=','):
    neigh = KNeighborsClassifier(n_neighbors = len(dataset-1),algorithm = 'ball_tree') 
    neigh.fit(dataset, y)
    
    print(neigh.predict(datatest)


def comparaison(dataset, datatest, precision, separateur=','):
    fiabilite_1, temps_1 = classification_balltree(precision, dataset, datatest, separateur)
    fiabilite_2, temps_2 = classification_balltree_sklearn(precision, dataset, datatest, separateur)
    print(f"Notre algorithme :\n\tPrécision : {fiabilite_1 :.2f} %\n\tTemps d'execution : \
{temps_1 :.3f} ms\nAlgorithme du module :\n\tPrécision : {fiabilite_2 :.2f} %\n\tTemps \
d'execution : {temps_2 :.3f} ms\n")

comparaison(heart, heart_test, 5)
comparaison(water_potability, water_potability_test, 5)
comparaison(diabetes, diabetes_test, 5)
comparaison(iris, iris_test, 5)
