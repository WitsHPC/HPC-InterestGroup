#include "utils/utils.h"

// multiply like a person of culture.
int multiply(int a, int b){    
    int ans = 0;
    for (int i=0; i<b; ++i){
        ans = add(ans, a);
    }
    return ans;
}
