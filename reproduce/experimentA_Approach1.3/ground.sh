for db in $(seq 23); do
    python3 main.py 10 $db | grep reward
done
