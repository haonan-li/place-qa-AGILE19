files="answer query answer_noname query_noname"

for file in $files; do
    src="data/pure_"$file".txt"
    tar="hdf5/"$file".hdf5"
    allennlp elmo $src $tar --top
    echo "transfer" $src "done"
done
