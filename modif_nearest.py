# lecture dataset en csv
dataset = recuperer_donnee_csv(nom_du_fichier)

# separation en fonction de target
def etudier_donnees_csv(nom_du_fichier):    
    nb_parametre = nb_colone-1
    Etudes_verifiees = np.array([])
    nb_Etudes_verifiees = 0
    Etudes_non_verifiees = np.array([])
    nb_Etudes_non_verifiees = 0
    for ligne in dataset:
        if ligne[nb_parametre] == 1:
        Etudes_verifiees = np.append(Etudes_verifiees, ligne[nb_parametre])
        nb_Etudes_verifiees += 1
        elif ligne[nb_colone-1] == 0:
        Etudes_non_verifiees = np.append(Etudes_non_verifiees, ligne[nb_parametre])
        nb_Etudes_non_verifiees += 1
        Etudes_verifiees = np.resize(Etudes_verifiees, (nb_Etudes_verifiees, nb_parametres))
        Etudes_non_verifiees = np.resize(Etudes_non_verifiees, (nb_Etudes_non_verifiees, nb_parametres))
        
        centroide_Etudes_verifiees = calcul_coordonnees_centroide(Etudes_verifiees)
        centroide_Etudes_non_verifiees = calcul_coordonnees_centroide(Etudes_non_verifiees)
        centroides = [centroide_Etudes_non_verifiees, centroide_Etudes_verifiees]
    



def tester_donnees(nom_du_fichier_test):
    testdata = recuperer_donnee_csv(nom_du_fichier_test)
    
    nb_bon = 0
    nb_faux = 0

# Etudes_non_verifiees correspond a 0
# Etudes_verifiees correspond a 1

   for test in testdata:
       if find_nearest_centroid(test[:13], centroides) == 1 and test[13] == 1:
           nb_bon+= 1
       elif find_nearest_centroid(test[:13], centroides) == 0 and test[13] == 0:
           nb_bon += 1

   return('Précision : ', nb_bon/len(testdata) * 100, '%')
