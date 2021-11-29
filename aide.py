# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:53:50 2021

@author: fireb
"""

import random as rd

def nb_aleatoire_entre(a, b, nb):
    liste = {rd.randint(a, b)}
    while len(liste) < nb:
        liste.add(rd.randint(a, b))
    return liste
