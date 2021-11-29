def coordonnees_centroide(liste_coordonne):
    coordonnees = np.array([])

    nb_dimension = len(liste_coordonne)

    for dimension in range(nb_dimension):
        somme = 0

        for point in liste_coordonne:
            somme += point[dimension]

        coordonnees = np.append(coordonnees, [somme/len(liste_coordonne)])
    return coordonnees
def distance_euclidienne(point_1, point_2):
    nb_dimension = len(point_1)
    distance = 0

    for dimension in range(nb_dimension):
        somme = (point_1[dimension] - point_2[dimension])**2
        distance += somme
    return distance

#debut code


def trouvercentroid(nouvelleliste): # doit trouver le centroid de nouvelleliste
    centroid=nouvelleliste[0]
    s=0
   for point in nouvelle liste : # chercher le point avec la plus petite distance euclidienne par rapport aux autres points
       for i in range(len(nouvelleliste)):
           point_2=nouvelleliste[i]
           s=s+distance_euclidienne(point,point_2)



def centroid(liste_coordonne,centroid0): #determine le point le plus eloigné de centroid0, ce point sera le centre du prochain cercle
    centroid=liste_coordonne[0]
    for i in range(len(liste_coordonne)-1):
        point_1=liste_coordonne[i]
        point_2=centroid0
        point_3=liste_coordonne[i+1]
        if distance_euclidienne(point_1, point_2) < distance_euclidienne(point_2, point_3):
            centroid=liste_coordonne[i+1]
    return(centroid)

def listecentroid(liste_coordonne,centroid,centroid0): # son centroid est donné par centroid(,)
    listecentroid=np.array([centroid])
    for point in liste_coordonne :
        if distance_euclidienne(point,centroid) < distance_euclidienne(centroid,centroid0): # la distance entre les 2 centroids represente le rayon du cercle
        listecentroid.append(point)
    return listecentroid

def nouvelleliste(liste_coordonne,centroid,centroid0):
    nouvelleliste=np.array([])
    for point in liste_coordonne :
        if distance_euclidienne(point,centroid) > distance_euclidienne(centroid,centroid0):
            nouvelleliste.append(point)
    return nouvelleliste

# avec ces 3 fonctions on peut fractionner une liste en 2 nouvelles listes contenants des points differents et chaque liste est associé à son centroid qui sera stocké dans une liste, la prédiction sera une comparaison des différentes distance euclidienne entre le nouveau point mystère et les centroids obtenus (seuls les centroids finaux sont stockés)

def balltree(precision,liste_coordonne):
    listeensemble=np.array([])
    centroid0=coordonnees_centroid(liste_coordonne)
    nouvelleliste=nouvelleliste(liste_coordonne,centroid,centroid0)
    listecentroid=listecentroid(liste_coordonne,centroid,centroid0)
    listeensemble.append([nouvelleliste,trouvercentroid(nouvelleliste)])
    listeensemble.append([listecentroid,centroid(liste_coordonne,centroid0)])
    for liste in listeensemble: # sans fin, besoin condition de fin
        centroid0=liste[1]
        centroid=centroid(liste[0],centroid0)
        nouvelleliste=nouvelleliste(liste[0],centroid,centroid0)
        listecentroid=listecentroid(liste[0],centroid,centroid0)
        listeensemble.append([nouvelleliste,trouvercentroid(nouvelleliste)])
        listeensemble.append([listecentroid,centroid(liste_coordonne,centroid0)])
        listeensemble.remove(liste) #element à la position 0, qui vient detre utilisé dans la boucle
            if len(listeensemble)==precision:
                return listeensemble # ou un tour de plus pour fractionner la derniere liste puis reunir les liste[1] pour faire la prédiction
def listedescentroids(listeensemble):

def prediction(listedescentroids,pointmystere):
    # chercher le centroid avec la plus petite distance entre lui et le point






