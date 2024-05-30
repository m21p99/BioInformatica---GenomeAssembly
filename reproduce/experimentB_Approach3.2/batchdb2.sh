DBID=$1
GENS=10
GENSSTEP=10
MAXRUNS=20
MAXGENS=1000
SEEDSFILE=seeds$DBID.txt
DISTSFILE=dists$DBID.txt
RESULTSFILE=results.txt
rm -f $SEEDSFILE $DISTSFILE
echo "DB:"$DBID
while [ $GENS -le $MAXGENS ]
do
    echo $GENS
    if [ $GENS -ge 100 ]
    then
        GENSSTEP=100
    fi
    COUNTER=0
    for i in $(seq 1 $MAXRUNS)
    do
        DIST=`python3 ga.py $GENS $DBID 2>> $SEEDSFILE | grep -o "dist: [0-9]*" | sed "s/dist..//"`
        echo $GENS","$DIST >> $DISTSFILE
        if [ "$DIST" -eq "0" ]
        then
            ((COUNTER=COUNTER+1))
        else
            ((GENS=GENS+GENSSTEP))
            break
        fi
    done
    if [ $COUNTER -eq $MAXRUNS ]
    then
        break
    fi
done
if [ $GENS -le $MAXGENS ]
then
    OUTMSG=$DBID","$GENS","$COUNTER
else
    OUTMSG=$DBID","$MAXGENS","$COUNTER
fi
echo $OUTMSG
echo $OUTMSG >> $RESULTSFILE
