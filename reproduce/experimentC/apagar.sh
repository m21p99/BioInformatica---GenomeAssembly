 { time python3 ga.py $1 $2 >/dev/null; } |& grep real | sed "s/real\s*//"
