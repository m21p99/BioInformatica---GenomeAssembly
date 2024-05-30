rm -f results.txt
for i in $(seq 1 23)
do
    ./batchdb.sh $i > log$i.txt &
done
