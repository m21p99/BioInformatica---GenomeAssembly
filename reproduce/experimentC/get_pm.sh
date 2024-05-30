for db in $(seq 23); do
    grep -o test.rw..[\.0-9]* resultado_*_${db}_*txt | sed "s/^.*test.rw../${db},/"
done
