for iter in $(seq 17 20); do
     { time ./sA.sh $1 $2 $iter >/dev/null; } &> tempo_$1_$2_$iter
     tar -czf resultado_$1_$2_$iter.tar.gz resultado_$1_$2_$iter.txt
     rm resultado_$1_$2_$iter.txt
done;
