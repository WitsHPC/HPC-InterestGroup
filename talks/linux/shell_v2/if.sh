#!/bin/bash
A=1
B=2
C=1

if [ "$A" = "$B" ]; then
    echo "$A = $B is True!"
fi

if [ "$A" = "$C" ]; then
    echo "$A = $C is True!"
fi


# use -lt and -gt for less than and greater than respectively
if [ "$A" -lt "$B" ]; then
    echo "$A < $B is True!"
elif ["$A" -gt "$B"]; then
    echo "$A > $B is True!"
else
    echo "$A = $B is True"
fi
