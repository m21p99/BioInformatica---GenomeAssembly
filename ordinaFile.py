import os
import re

def leggi_e_ordina_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    sezioni = content.split("NumIndividui:")
    if not sezioni[0].strip():
        sezioni = sezioni[1:]
    
    sezioni_con_distanza = []
    for sezione in sezioni:
        # Utilizza un'espressione regolare per trovare la distanza di Levenshtein
        match = re.search(r"Distanza di Levenshtein dati i markers.*?e di:\s*(\d+)", sezione)
        if match:
            distanza = int(match.group(1))
            sezioni_con_distanza.append((distanza, sezione))
    
    sezioni_ordinate = sorted(sezioni_con_distanza, key=lambda x: x[0])
    
    with open(file_path, 'w') as file_ordinato:
        for _, sezione in sezioni_ordinate:
            file_ordinato.write("NumIndividui:" + sezione)
            file_ordinato.write("\n")

# Percorso del file da ordinare
files = os.listdir('.')

# Filtra i file che terminano con ".txt" e non iniziano con "requirements"
txt_files = [file for file in files if file.endswith(".txt") and not file.startswith("requirements")]

# Stampa i file trovati
print(txt_files)
for x in txt_files:
    leggi_e_ordina_file(x)