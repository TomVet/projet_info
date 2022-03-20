import csv


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
        The default is ",".

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

#print(recuperer_donnee_csv("C:/Users/jbcor/Downloads/dataset.csv"))

import numpy as np

C=recuperer_donnee_csv("C:/Users/jbcor/Downloads/dataset.csv")

def liste_classe(data):
    classes=[]
    for point in data:
        classes.append(point[2])
    return classes

def liste_donnes(data):
    donnes=[]
    for point in data:
        donnes.append([point[0],point[1]])
    return donnes

def proba_naives_sklearn(points,point):
    X = liste_donnes(points)
    Y = liste_classe(points)
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X, Y)
    GaussianNB()
    clf_pf = GaussianNB()
    clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    return clf_pf.predict_proba([point]),int(clf.predict([point]))