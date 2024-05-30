for db in $(seq 23); do
    for iter in $(seq 20); do
        SEED=`cat semente_*_${db}_${iter}.txt`
        EPIS=`ls -1 semente_*_${db}_${iter}.txt | cut -d$'_' -f2`
        echo "python3 main.py "$EPIS $db $SEED
    done;
done;
