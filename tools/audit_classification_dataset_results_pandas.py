import os
import pandas as pd

def print_audit_report(csv_path="audit_classification_results.csv"):
    # Résoudre le chemin du fichier par rapport à l'emplacement du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_csv_path = os.path.join(script_dir, csv_path)
    
    if not os.path.exists(full_csv_path):
        print(f"❌ Fichier non trouvé : {full_csv_path}")
        print("Veuillez d'abord exécuter audit_classification_dataset.py pour générer les résultats.")
        return
        
    print(f"📂 Chargement des résultats depuis {csv_path}...\n")
    df = pd.read_csv(full_csv_path)
    
    if len(df) == 0:
        print("❌ Le fichier CSV est vide.")
        return
        
    total = len(df)
    errors = len(df[df["status"] == "ERROR"])
    accuracy = ((total - errors) / total * 100) if total > 0 else 0
    
    print(f"======================================")
    print(f"📊 BILAN DE L'AUDIT DE CLASSIFICATION")
    print(f"======================================")
    print(f"   Total analysé : {total} images (crops)")
    print(f"   Erreurs       : {errors}")
    print(f"   Précision     : {accuracy:.2f} %")
    
    print(f"\n   Répartition de la précision par split :")
    split_acc = df.groupby('split')['status'].apply(lambda x: (x == 'CORRECT').mean() * 100)
    for s, acc in split_acc.items():
        print(f"     - {s.upper()} : {acc:.2f} %")
    
    if errors > 0:
        print("\n⚠️ TOP 10 PIRES ERREURS (Haute confiance du modèle, mais classé ERROR) :")
        worst = df[df["status"] == "ERROR"].sort_values(by="confidence", ascending=False).head(100)
        print(worst[["base_image_name", "true_class", "pred_class", "confidence", "split"]].to_string(index=False))

if __name__ == "__main__":
    print_audit_report()
