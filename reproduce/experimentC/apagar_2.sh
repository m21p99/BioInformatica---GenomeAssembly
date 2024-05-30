for iter in 1 2 3 4 5 6; do
    ./apagar.sh $1 $(($2 + $iter))
done;
