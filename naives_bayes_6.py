import csv
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB


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
        data = np.arraye(data)
        data = data.astype(np.float64)
        return data

# naive bayes sklearn


def liste_classe(data):
    """
    renvoie une liste contenant la classe de chaque point

    Parameters
    ----------
    data : list
        liste de points

    Returns
    -------
    classes : list
        liste d'entiers

    """
    classes = []
    for point in data:
        classes.append(point[-1])
    return classes


def liste_donnes(points):
    """
    Renvoie une liste contenant les points issu de cette liste sans leur
    coordonnée

    Parameters
    ----------
    points : list
        liste de points

    Returns
    -------
    donnees : list
        liste de points

    """
    donnees = []
    for point in points:
        liste = [point[i] for i in range(len(points[0])-1)]
        donnees.append(liste)
        liste = []
    return donnees


def proba_naives_sklearn(point, points):
    start = time.time()
    X = liste_donnes(points)
    Y = liste_classe(points)
    clf = GaussianNB()
    clf.fit(X, Y)
    GaussianNB()
    clf_pf = GaussianNB()
    clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    end = time.time()
    temps = (end - start)
    return (clf_pf.predict_proba(point)), (int(clf.predict(point))), temps

# naive bayes programmé


def calcul_ecart(points):
    """
    # calcul l'écart entre la classe de et le rang de la classe parmis
    les autres classes ex : vin à pH 6 qui est le 3eme pH testé
    (classe = 6, rang = 3 ), pour eviter un décalage lors de l'utilisation
    d'une fonction

    Parameters
    ----------
    points : list
        liste de points

    Returns
    -------
    integer

    """
    classes = set(point[-1] for point in points)
    return int(min(classes))


def calcul_esp(points):
    """
    Renvoie l'esperance de chaque catégorie en fonction de sa classe

    Parameters
    ----------
    points : list
        liste de points

    Returns
    -------
    esperance_par_classe : list
        liste d'entiers

    """
    dim_point = len(points[0]) - 1
    esperance = []
    esperance_par_classe = []
    classes = set(point[-1] for point in points)
    somme = 0
    compteur = 0
    for classe in classes:
        for rang in range(dim_point):
            for point in points:
                if point[-1] == classe:
                    compteur += 1
                    somme += point[rang]
            esperance.append(somme/compteur)
            somme = 0
            compteur = 0
        esperance_par_classe.append([esperance, classe])
        esperance = []
    return esperance_par_classe


def calcul_var(points, ecart):
    """
    Renvoie la variance de chaque catégorie en fonction de sa classe

    Parameters
    ----------
    points : list
        liste de points
    ecart : integer

    Returns
    -------
    variance_par_classe : list
        liste d'entiers

    """
    dim_point = len(points[0]) - 1
    variance = []
    variance_par_classe = []
    classes = set(point[-1] for point in points)
    somme = 0
    compteur = 0
    for classe in classes:
        for rang in range(dim_point):
            for point in points:
                if point[-1] == classe:
                    compteur += 1
                    somme += (point[rang] -
                              calcul_esp(points)[int(classe - ecart)][0][int(rang)])**2
            variance.append((somme/(compteur - 1)))
            somme = 0
            compteur = 0
        variance_par_classe.append([variance, classe])
        variance = []
    return variance_par_classe


def calcul_proba_classe(points):
    """
    Renvoie la liste de probabilité qu'un point prit dans la liste
    appartienne à chaqu'une des classe (ex homme ou femme)

    Parameters
    ----------
    points : list
        liste de points

    Returns
    -------
    liste_proba : list
        liste de probabilitées

    """
    liste_classe = set([point[-1] for point in points])
    liste_proba = []
    for classe in liste_classe:
        compteur = 0
        for point in points:
            if point[-1] == classe:
                compteur += 1
        liste_proba.append([compteur/len(points), classe])
    return liste_proba


def calcul_proba_categorie_sachant_classe(point, points, categorie, classe, ecart, esperance, variance):
    """

    Parameters
    ----------
    point : list
        coordonnées du point étudié
    points : list
        liste de points
    categorie : integer
        rang du paramètre étudié
    classe : integer
        rang de la classe étudié
    ecart : integer
    esperance : list
        liste des espérances d'une catégorie en fonction des classes
    variance : list
        liste des variances d'une catégorie en fonction des classes


    Returns
    -------
    proba : integer
        probabilité de la catégorie en fonction de la classe

    """
    esp = esperance[int(classe-ecart)][0][categorie]
    var = variance[int(classe-ecart)][0][categorie]
    proba = np.exp((-((point[0][categorie]-esp)**2) /
                   (2*var)))/(np.sqrt(2*float(np.pi)*var))
    return proba


def calcul_constante(points, ecart, point, esp, var):
    """
    Calcul le terme constant de la loi de bayes (l'évidence), la probabilité
    d'apparetance à une catégorie

    Parameters
    ----------
    points : list
        liste de points
    ecart : integer
    point : list
        coordonnées du point étudié
    esp : list
        liste des espérances de chaque catégorie en fonction de la classe
    var : list
        liste des variances de chaque catégorie en fonction de la classe

    Returns
    -------
    constante : integer

    """
    classes = set(point[-1] for point in points)
    proba = 1
    constante = 0
    liste_proba = []
    for classe in classes:
        for i in range(len(points[0])-1):
            proba = proba * \
                calcul_proba_categorie_sachant_classe(
                    point, points, i, classe, ecart, esp, var)
        proba = proba*float(calcul_proba_classe(points)[int(classe-ecart)][0])
        liste_proba.append(proba)
        proba = 1
    for i in range(len(liste_proba)):
        constante += liste_proba[i]
    return constante


def calcul_proba_bayes(point, points):
    """


    Parameters
    ----------
    point : TYPE
        DESCRIPTION.
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    liste_proba : list
        liste avec la probabilité d'appartenance du point à chaque classe
    integer
        classe ayant la plus grande probabilité d'etre celle du point étudié
    temps : integer
        temps moyenne d'une boucle d'apprentissage

    """
    start = time.time()
    ecart = calcul_ecart(points)
    proba_classe = calcul_proba_classe(points)
    classes = set(point[-1] for point in points)
    liste_proba = []
    esp = calcul_esp(points)
    var = calcul_var(points, ecart)
    for classe in classes:
        proba_categorie_sachant_classe = 1
        for i in range(len(points[0])-1):
            proba_categorie_sachant_classe = proba_categorie_sachant_classe * \
                calcul_proba_categorie_sachant_classe(
                    point, points, i, classe, ecart, esp, var)
        liste_proba.append([(proba_classe[int(classe-ecart)][0]*proba_categorie_sachant_classe /
                           calcul_constante(points, ecart, point, esp, var)), classe])
    end = time.time()
    temps = (end - start)
    return liste_proba, max(liste_proba)[1], temps


def comparateur(liste_test, dataset):
    """
    Compare les succes des 2 algorithmes avec une liste dont la  classe
    est deja connue

    Parameters
    ----------
    liste_test : list
        liste de points dont on connait la classe et qu'on a retiré de dataset
    dataset : list
        liste de points

    Returns
    -------
    integer
        temps et précision de chaque algorithme

    """
    classes = liste_classe(liste_test)
    points = liste_donnes(liste_test)
    temps_1 = 0  # temps de sk_learn
    temps_2 = 0  # temps de calcul_proba_bayes
    succes_1 = 0  # succes de sk_learn
    succes_2 = 0  # succes de calcul_proba_bayes
    taille = len(liste_test)
    for i in range(len(liste_test)):
        Liste = proba_naives_sklearn([points[i]], dataset)
        liste = calcul_proba_bayes([points[i]], dataset)
        temps_1 += Liste[2]
        temps_2 += liste[2]
        if Liste[1] == classes[i]:
            succes_1 += 1
        if liste[1] == classes[i]:
            succes_2 += 1
    temps_1 = temps_1/taille
    temps_2 = temps_2/taille
    succes_1 = succes_1/taille
    succes_2 = succes_2/taille
    return (succes_1, temps_1), (succes_2, temps_2)
