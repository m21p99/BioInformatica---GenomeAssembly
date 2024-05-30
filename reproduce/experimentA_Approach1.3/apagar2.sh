{ time { python3 main.py $1 $2 &> /dev/null; }; } &> saida.txt && grep real saida.txt | sed "s/real\s*//"
