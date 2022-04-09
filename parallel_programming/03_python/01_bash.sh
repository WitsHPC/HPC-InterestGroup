for i in {1..5}; do
	echo "Running $i"
	python 01_bash.py $i &
done
wait;