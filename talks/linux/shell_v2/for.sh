#!/bin/bash
# Simple for loop
for i in 1 2 3 4 5; do
    echo $i
done

echo '========='

# inclusive
for i in {5..10}; do
    echo $i
done

echo '========='

for i in `ls *`; do
    echo $i
done
