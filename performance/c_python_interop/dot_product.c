extern float dot_product(int length, float* a, float* b, float* ans){
    *ans = 0;
    for (int i=0; i < length; ++i){
        *ans += a[i] * b[i];
    }
}