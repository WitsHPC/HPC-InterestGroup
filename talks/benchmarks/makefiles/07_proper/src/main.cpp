#include <iostream>
#include "math/mymaths.h"
#include "utils/utils.h"

int main(){
    int a = 5;
    int b = 7;
    int c = add(a, b);
    std::cout << a<< " + " << b << " = " << c << "\n";
    std::cout << a<< " * " << b << " = " << multiply(a, b) << "\n";
}
