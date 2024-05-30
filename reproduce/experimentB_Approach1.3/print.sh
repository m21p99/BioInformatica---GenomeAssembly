for db in $(seq 23); do
	for iter in $(seq 20); do
		PREFIX=resultado_*_${db}_${iter}
		FILETAR=`ls -1 $PREFIX.tar.gz 2>/dev/null`
                FILETXT=`basename $FILETAR tar.gz 2>/dev/null`txt
                if [ -z "$FILETAR" ]; then
                    continue
                fi
                if test -f "$FILETAR"; then
			tar -xzvf $FILETAR &>/dev/null
			OUT=`tail -n 1 $FILETXT | grep -o test.rw..[0-9\.]*.* | sed "s/test.rw..//" | sed "s/ .* /,/"`
                        echo $db,$iter,$OUT
			rm $FILETXT
		fi
	done;
done;	
