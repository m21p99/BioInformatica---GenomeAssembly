for db in $( seq 23 ); do
    { time { python3 main.py $1 $db &> /dev/null; }; } &> saida.txt && grep real saida.txt | sed "s/real\s*//"
done;
