for db in $(seq 23); do
     cat semente_*_${db}_*.txt | sed "s/.*/${db},&/"
done
