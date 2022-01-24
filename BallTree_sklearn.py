from sklearn.neighbors import BallTree
import numpy as np



def retrouver_liste(centroid_obtenu, liste_de_listes, liste_centroid):
    rang=liste_centroid.index(centroid_obtenu)
    return liste_de_listes[rang] #renvoie la sous liste associé au nouveau point donné pour la prédiction



def ball_tree_sklearn(dataset, datatest, separateur=','):

    clf = BallTree(data)
    indice = clf.query(data, k=1, return_distance=False)
    

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
