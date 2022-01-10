import csv
import numpy as np
data  =  np.array([[ 0.50170141,  1.46027261],
       [ 0.97413385,  0.6756709 ],
       [-0.17649309, -1.44746182],
       [-0.30586378, -1.07902476],
       [ 1.20221275, -0.23151285],
       [ 0.06981749, -1.06081874],
       [-0.85017335, -0.84245093],
       [-0.83247692, -0.95133504],
       [ 0.47296343,  1.21798404],
       [ 2.56608803, -0.05580103],
       [-1.10900372, -1.14478982],
       [-0.10330393, -0.1310852 ],
       [ 1.11981807, -1.33795926],
       [-0.52838736, -0.1285451 ],
       [ 2.50072817, -0.0504333 ],
       [ 2.66461543,  0.5241306 ],
       [ 0.07789832, -1.96159762],
       [ 1.77076715,  2.009976  ],
       [-0.60754177, -2.85748894],
       [ 1.3419128 ,  1.00674139],
       [ 1.77808038, -0.61140296],
       [-1.87657177, -0.75621871],
       [-0.45251957, -1.22483606],
       [-0.81237637, -0.99422874],
       [ 1.33908386, -1.51000216],
       [ 0.22526135, -0.63649467],
       [-0.39763935, -0.97229169],
       [-0.06975968, -1.38742715],
       [ 2.2011879 ,  0.89394132],
       [ 1.52405996, -0.13247843],
       [-0.1725551 , -1.07792929],
       [-1.28838949, -1.03368359],
       [ 0.04777995, -0.61307349],
       [ 0.04643614, -2.33815556],
       [ 1.78669557, -1.98416597],
       [ 1.38352719, -1.00212029],
       [ 0.73665638,  0.87218699],
       [ 0.13502378,  0.46108463],
       [ 1.61493839,  1.7259827 ],
       [ 3.16233102,  0.17102427],
       [-0.7937284 , -1.05385348],
       [ 1.42707999,  0.10295644],
       [-0.80863498, -1.23140216],
       [-1.21302857, -1.17533513],
       [-1.21431731, -1.02483268],
       [ 1.55117664,  1.19544266],
       [ 1.35588487,  0.30365803],
       [-0.90538176, -0.82585213],
       [ 1.32384277,  0.99830437],
       [-0.72827283, -0.90257227],
       [ 0.69317043, -0.38964404],
       [ 2.16056177,  2.16351984],
       [ 1.25863047,  1.65018581],
       [-0.92167555, -0.81992795],
       [ 0.23044462,  1.67154087],
       [ 0.85490783,  0.62708547],
       [ 2.07962214,  0.00376494],
       [-0.37038456, -0.86121597],
       [ 0.14592578, -0.05129392],
       [-0.95360918, -0.87906511],
       [-1.52416947, -0.90292055],
       [ 0.86535561,  2.13001973],
       [-0.88913375, -0.9663168 ],
       [-0.967569  , -1.08887845],
       [ 1.00771383,  2.01773258],
       [ 3.34761268, -0.29272765],
       [-0.64403857, -0.97925761],
       [-1.05359672, -0.80166895],
       [ 0.66231173, -1.10871655],
       [ 2.44274413, -1.17471458],
       [-0.9431884 , -1.23816601],
       [ 1.35888299,  1.68099039],
       [-1.14701437, -1.06246221],
       [-0.94889076, -1.34966394],
       [ 2.12201037,  1.53293206],
       [-0.81517599, -0.93959529],
       [ 0.7543151 , -1.54799905],
       [-1.60516443, -1.01022526],
       [-0.50906522, -1.16430558],
       [-1.09403278, -1.00283777],
       [ 0.81822478, -1.01614577],
       [ 1.90640335,  0.98524983],
       [ 0.11620641, -0.67173328],
       [ 0.61885922,  0.79241477],
       [ 0.75119567,  1.32295809],
       [-1.64603422, -0.8516942 ],
       [-0.73940729, -1.10391545],
       [-1.00117526, -1.15152382],
       [-0.11289362,  0.87458094],
       [ 0.38816204, -1.38469197],
       [ 0.38337377,  1.05182534],
       [-0.6668804 , -1.01869223],
       [ 1.28981576, -0.49026195],
       [ 1.00901316, -0.81117648],
       [ 1.15138896, -0.85184834],
       [ 1.11109866,  1.4723829 ],
       [ 1.38973445,  0.9336832 ],
       [ 2.31657724, -0.78652379],
       [ 1.41037629,  1.63239969],
       [ 0.27188321,  0.6682191 ]])

data2  =  np.array([[ 0.50170141,  1.46027261],
       [ 0.97413385,  0.6756709 ],
       [-0.17649309, -1.44746182],
       [-0.30586378, -1.07902476],
       [ 1.20221275, -0.23151285],
       [ 0.06981749, -1.06081874],
       [-0.85017335, -0.84245093],
       [-0.83247692, -0.95133504],
       [ 0.47296343,  1.21798404],
       [ 2.56608803, -0.05580103],
       [-1.10900372, -1.14478982],
       [-0.10330393, -0.1310852 ],
       [ 1.11981807, -1.33795926],
       [-0.52838736, -0.1285451 ],
       [ 2.50072817, -0.0504333 ],
       [ 2.66461543,  0.5241306 ],
       [ 0.07789832, -1.96159762],
       [ 1.77076715,  2.009976  ],
       [-0.60754177, -2.85748894],
       [ 1.3419128 ,  1.00674139],
       [ 1.77808038, -0.61140296],
       [-1.87657177, -0.75621871],
       [-0.45251957, -1.22483606],
       [-0.81237637, -0.99422874],
       [ 1.33908386, -1.51000216],
       [ 0.22526135, -0.63649467],
       [-0.39763935, -0.97229169],
       [-0.06975968, -1.38742715],
       [ 2.2011879 ,  0.89394132],
       [ 1.52405996, -0.13247843],
       [-0.1725551 , -1.07792929],
       [-1.28838949, -1.03368359],
       [ 0.04777995, -0.61307349],
       [ 0.04643614, -2.33815556],
       [ 1.78669557, -1.98416597],
       [ 1.38352719, -1.00212029],
       [ 0.73665638,  0.87218699],
       [ 0.13502378,  0.46108463],
       [ 1.61493839,  1.7259827 ],
       [ 3.16233102,  0.17102427],
       [-0.7937284 , -1.05385348],
       [ 1.42707999,  0.10295644],
       [-0.80863498, -1.23140216],
       [-1.21302857, -1.17533513],
       [-1.21431731, -1.02483268],
       [ 1.55117664,  1.19544266],
       [ 1.35588487,  0.30365803],
       [-0.90538176, -0.82585213],
       [ 1.32384277,  0.99830437],
       [-0.72827283, -0.90257227],
       [ 0.69317043, -0.38964404],
       [ 2.16056177,  2.16351984],
       [ 1.25863047,  1.65018581],
       [-0.92167555, -0.81992795],
       [ 0.23044462,  1.67154087],
       [ 0.85490783,  0.62708547],
       [ 2.07962214,  0.00376494],
       [-0.37038456, -0.86121597],
       [ 0.14592578, -0.05129392],
       [-0.95360918, -0.87906511],
       [-1.52416947, -0.90292055],
       [ 0.86535561,  2.13001973],
       [-0.88913375, -0.9663168 ],
       [-0.967569  , -1.08887845],
       [ 1.00771383,  2.01773258],
       [ 3.34761268, -0.29272765],
       [-0.64403857, -0.97925761],
       [-1.05359672, -0.80166895],
       [ 1.00901316, -0.81117648],
       [ 1.15138896, -0.85184834],
       [ 1.11109866,  1.4723829 ],
       [ 1.38973445,  0.9336832 ],
       [ 2.31657724, -0.78652379],
       [ 1.41037629,  1.63239969],
       [ 0.27188321,  0.6682191 ]])

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
    coordonnees  =  np.array([])
    nb_dimension  =  len(liste_coordonne[0])
    for i in range(nb_dimension):
        somme  =  0
        for point in liste_coordonne:
            somme  =  somme+point[i]
        coordonnees  =  np.append(coordonnees, [somme/len(liste_coordonne)])
    return coordonnees

def D_euclidienne(point_1, point_2):
    nb_dimension  =  len(point_1)
    distance  =  0
    for dimension in range(nb_dimension):

        somme  =  (point_1[dimension] - point_2[dimension])**2
        distance +=  somme
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

def creer_nouvelleliste(liste,P_loin):
    # on cree une liste qui va prendre tout les points situés dans la sphere de rayon (distance "point le plus loin" jusquà "centroid")
    nouvelleliste  =  []
    centroid = trouver_coordonnees_centroide(liste)
    for point in liste:
        if D_euclidienne(point,P_loin) <= D_euclidienne(centroid,P_loin):
            nouvelleliste.append(point)
    return (nouvelleliste)

def creer_liste_complementaire(liste,P_loin):
    # on cree une liste qui va prendre tout les autres points n'ayant pas été pris dans nouvelleliste
    liste_complementaire = []
    centroid = trouver_coordonnees_centroide(liste)
    for point in liste:
        if D_euclidienne(point,P_loin) > D_euclidienne(centroid,P_loin):
            liste_complementaire.append(point)
    return (liste_complementaire)

def balltree(precision,liste):
# la précision correspond au nombre de sous liste qui vont etre formé à partir de la liste initial (représente donc aussi le nombre d'itération du programme)
    listes_crees = [liste]
# rassemble l'ensemble des listes crées qui sont des sous ensembles de la liste initiales
    liste_a_diviser = liste
    for tour in range(precision):
        P_loin = trouver_P_loin(liste_a_diviser)
        nouvelleliste = creer_nouvelleliste(liste_a_diviser,P_loin)
        liste_complementaire = creer_liste_complementaire(liste_a_diviser,P_loin)
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

def retrouver_liste(centroid_obtenu,liste_de_listes,liste_centroid):
    rang=liste_centroid.index(centroid_obtenu)
    return liste_de_listes[rang] #renvoie la sous liste associé au nouveau point donné pour la prédiction

def prediction(liste_centroid,nouveau_point): # on cherche le centroid le plus proche du nouveau point
    indice_centroid_proche = 0
    centroid_proche = liste_centroid[indice_centroid_proche]
    for indice in range(len(liste_centroid)):
        point_1 = liste_centroid[indice_centroid_proche]
        point_2 = liste_centroid[indice]
        if D_euclidienne(point_1,nouveau_point) > D_euclidienne(point_2,nouveau_point):
            indice_centroid_proche = indice

    return centroid_proche
