import numpy as np

test = np.array([(1,3),(3,2),(9,4)])

def centroide(liste_coordonne):
    """
    Calcule le centroide des points de liste_coordonne dans de dimension
    nb_dimension

    Parameters
    ----------
    liste_coordonne : np.array
        liste de coordonnee de point de meme dimension.

    Returns
    -------
    coordonnee_centroide : np.array
        liste des coordonnee du centroide calcule.

    """
    coordonnee_centroide = np.array([])
    # on calcule la dimension de l'espace considérer pour le centroide
    nb_dimension = len(liste_coordonne[1])
    # on calcul les coordonnees du centroide dans chaque dimension
    for dimension in range(nb_dimension):
        somme = 0
        # on somme les coordonnees de chaque point
        for point in liste_coordonne:
            somme += point[dimension]
        # on ajoute la somme / par le nombre de point a coordonne_centroide
        coordonnee_centroide = np.append(coordonnee_centroide, [somme/len(liste_coordonne)])
    return coordonnee_centroide

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

centroides = {''}

def centroide_proche(point, centroides):
    distance_min = 0
    rang_du_min = 0
    for centroide in centroides:
        distance = distance_euclidienne(point, centroide)
        if distance_min < distance:
            distance_min = distance
