mkdir test_ga
for j in $(seq 1 1 20)
do
    mkdir test_ga/run_$j
    for i in $(seq 19 1 23)
    do
        date
        unbuffer python3 ga.py 1000 $i | tee test_ga/run_$j/saida_$i.txt
        date
    done
done
