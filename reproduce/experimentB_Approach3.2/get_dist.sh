for db in $(seq 23); do
    grep dist..[0-9]* resultado_*_${db}_*txt | sed "s/^.*dist../${db},/"
done
