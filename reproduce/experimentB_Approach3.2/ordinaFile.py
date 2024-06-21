import re


def extract_levenshtein_distance(line):
    match = re.search(r'Distanza di Levenshtein dati i markers \[.*\] e di: (\d+)', line)
    if match:
        return int(match.group(1))
    return None


file = ["Test_5Marks_5Dim.txt", "Test_5Marks_6Dim.txt", "Test_5Marks_7Dim.txt", "Test_5Marks_8Dim.txt",
        "Test_6Marks_5Dim.txt", "Test_6Marks_6Dim.txt", "Test_6Marks_7Dim.txt", "Test_6Marks_8Dim.txt",
        "Test_7Marks_5Dim.txt", "Test_7Marks_6Dim.txt", "Test_7Marks_7Dim.txt", "Test_8Marks_6Dim.txt",
        "Test_8Marks_7Dim"
        ".txt",
        "Test_8Marks_8Dim.txt", "Test_9Marks_6Dim.txt", "Test_9Marks_7Dim.txt"]
for x in file:
    # Leggi il file
    with open(x, "r") as file:
        lines = file.readlines()
    # Ordina le righe in base alla distanza di Levenshtein
    sorted_lines = sorted(lines, key=extract_levenshtein_distance)

    # Scrivi le righe ordinate nel file
    with open(x, "w") as file:
        file.writelines(sorted_lines)

print("File ordinato con successo!")
