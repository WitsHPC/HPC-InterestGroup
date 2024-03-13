import sys
from time import sleep


def main(seed: int):
    sleep(5)
    return seed ** 2

# sys.argv is the arguments from the terminal, similar to char** argv in C/C++ or String[] args in Java.
print("Result = ", main(int(sys.argv[-1])))