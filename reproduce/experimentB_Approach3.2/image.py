import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

# Lettura e marcatori
lettura = "TCATATCCCTAGAGTGCAATAGCTGAGTGAGTAGCCGTAGGTTCTGCGCGATGCAGTGTCCCTGAATAATCCAAACAACCTCGCCGCGGTCGCATGCGCC "
marcatori = ['AAATT', 'TCCCT', 'GTCGC', 'TTTTC']

# Funzione per spezzare la lettura in blocchi
def spezza_lettura(lettura, marcatori):
    pattern = '|'.join(marcatori)
    blocchi = re.split(f'({pattern})', lettura)
    return [blocco for blocco in blocchi if blocco]

blocchi = spezza_lettura(lettura, marcatori)

print(blocchi)


