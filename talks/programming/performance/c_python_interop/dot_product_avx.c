#include <immintrin.h>
extern void dot_product(int length, float* a, float* b, float* ans){
    // assume length % 8 == 0
    *ans = 0;
    const int ones = 0xFFFFFFFF;
    __m256i x = _mm256_set1_epi8(0xff);
    for (int i=0; i < length; i += 8){
        // load the data
        __m256 packed_a = _mm256_load_ps (a + i);
        __m256 packed_b = _mm256_load_ps (b + i);

        // do the dot product
        __m256 temp_ans = _mm256_dp_ps(packed_a, packed_b, 0xFF);
        // and sum the results.
        *ans += temp_ans[0] + temp_ans[7];
    }
}