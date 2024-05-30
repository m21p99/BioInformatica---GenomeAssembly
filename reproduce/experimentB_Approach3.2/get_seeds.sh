for db in $(seq 23); do
    head -1 resultado_*_${db}_*.txt | grep -o "^[0-9][0-9]*$" | sed "s/.*/${db},&/"
done
