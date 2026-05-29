import os
from collections import defaultdict
import tqdm

def trouver_doublons(repertoire_source, fichier_sortie="doublon.csv"):
    # Dictionnaire pour stocker les noms de fichiers (sans extension) et leurs chemins
    fichiers_par_nom = defaultdict(list)

    # Extensions à considérer
    extensions_valides = {'.jpg', '.png'}

    # Parcours récursif des sous-répertoires
    for racine, _, fichiers in os.walk(repertoire_source):
        for fichier in tqdm.tqdm(fichiers):
            nom, extension = os.path.splitext(fichier)
            if extension.lower() in extensions_valides:
                chemin_absolu = os.path.abspath(os.path.join(racine, fichier))
                fichiers_par_nom[nom].append(chemin_absolu)

    # Filtrer les doublons (noms avec plus d'un chemin)
    doublons = {nom: chemins for nom, chemins in fichiers_par_nom.items() if len(chemins) > 1}

    if doublons:
        with open(fichier_sortie, 'w') as f:
            for nom, chemins in doublons.items():
                ligne = f"{nom}\t" + "\t".join(chemins) + "\n"
                f.write(ligne)

        print(f"Les doublons ont été enregistrés dans '{fichier_sortie}'.")
    else:
        print("Aucun doublon trouvé.")

    return doublons

# Exemple d'utilisation
if __name__ == "__main__":
    repertoire = '/home/aobled/Downloads/Aircraft_DATASET'
    doublons = trouver_doublons(repertoire)
