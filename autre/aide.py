# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:53:50 2021

@author: fireb
"""

import random as rd
import csv
import numpy as np


def nb_aleatoire_entre(a, b, nb):
    liste = {rd.randint(a, b)}
    while len(liste) < nb:
        liste.add(rd.randint(a, b))
    return liste

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
