#include <immintrin.h>
#include <omp.h>
#include <stdio.h>

#include <cmath>
#include <string>

#include "utils.h"
typedef double (*integral_calc_func)(double x_start, double x_end, double dx);

/**
 * @brief This calculates the integral of 4 / (1 + x^2) from start to end using dx given.
 * @return double 
 */
double calc_integral(double x_start, double x_end, double dx) {
    double ans = 0;
    double x;
    // our counter
    int i;
    // how many loop iterations should we do
    int n = ceil((x_end - x_start) / dx);
    for (i = 0; i < n; ++i) {
        // get x
        x = i * dx + x_start;
        // increment ans
        ans += dx * 4.0 / (1.0 + x * x);
    }
    // return
    return ans;
}

/**
 * @brief This calculates the integral using one core using the given function.
 * 
 * @param dx 
 * @param func 
 * @return double 
 */
double calc_integral_serial(double dx, integral_calc_func func) {
    return func(0.0, 1.0, dx);
}

/**
 * @brief This calculates the integral in chunks using openmp, and each chunk is calculated using the given function.
 * 
 * @param dx 
 * @param func 
 * @return double 
 */
double calc_integral_par(double dx, integral_calc_func func) {
    // a global answer, the total
    double global_ans = 0.0;
    // make a parallel region
#pragma omp parallel
    {
        // how many threads do we have?
        int num_threads = omp_get_num_threads();
        // what is my id
        int my_id = omp_get_thread_num();
        // how much should each thread integrate
        double per_thread = 1.0 / (double)num_threads;
        // where should I start and end
        double my_start = my_id * per_thread;
        double my_end = (1 + my_id) * per_thread;
        // get my answer
        double my_ans = func(my_start, my_end, dx);
        // now, this increment should not happen concurrently, as we might have a race condition.
        // Hence the omp atomic that ensures only one update at a time occurs.
#pragma omp atomic
        global_ans += my_ans;
    }

    return global_ans;
}

/**
 * @brief Calculates the integral using a parallel for loop
 * 
 * @param dx 
 * @return double 
 */
double calc_integral_par_for(double dx) {
    double ans = 0;
    double x;
    // our counter
    int i;
    // how many loop iterations should we do
    int n = ceil((1.0 - 0.0) / dx);
#pragma omp parallel for private(x) reduction(+ \
                                              : ans)
    for (i = 0; i < n; ++i) {
        // get x
        x = i * dx;
        // increment ans
        ans += dx * 4.0 / (1.0 + x * x);
    }
    // return
    return ans;
}

/**
 * @brief Calculates the integral using AVX 256 instructions.
 * 
 * @param x_start 
 * @param x_end 
 * @param dx 
 * @return double 
 */
double calc_integral_avx2(double x_start, double x_end, double dx) {
    // All our variables here are __m256d, i.e. 256 bit wide array of doubles.
    // intermediate answer. _mm256_setzero_pd() does the following:
    // _mm256 -> 256 buts
    // setzero -> set to 0
    // pd -> doubles
    __m256d _ans = _mm256_setzero_pd();
    // four, we use set1, to set all values in the array to the same value.
    __m256d _four = _mm256_set1_pd(4.0);
    // four multiplied by dx, set1 again
    __m256d _four_dx = _mm256_set1_pd(4.0 * dx);

    __m256d _onetwothreefour_dx = _mm256_set_pd(0.0, 1.0 * dx, 2.0 * dx, 3.0 * dx);
    // use one
    __m256d _one = _mm256_set1_pd(1.0);
    // our dx, i.e. how large are our steps
    __m256d _dx = _mm256_set1_pd(dx);
    // where do we start
    __m256d _x_start = _mm256_set1_pd(x_start);
    // _mm256_set_pd allows you to set each element, and we set that to 0, dx, 2dx, 3dx, i.e. the first four values of x.
    // We add to that x_start, which offsets us to start where we should
    __m256d _x = _mm256_add_pd(
        _mm256_set_pd(0.0, 1.0 * dx, 2.0 * dx, 3.0 * dx),  // +
        _x_start);

    // While we haven't reached the end. This has the potential to cause problems in the case where
    // _x[0] < x_end, but _x[3] isn't. For our use case here, since our dx is quite small, we won't have that issue, but it
    // is potentially better to make this check more sophisticated.
    while (_x[3] < x_end) {
        // The following is just executing: ans = ans + 4/(1 + x * x) * dx
        _ans = _mm256_add_pd(_ans,
                             _mm256_div_pd(_four_dx,
                                           _mm256_add_pd(_one, _mm256_mul_pd(_x, _x))));

        // This is basically doing x += dx, but since we use 4 numbers at once, we need to jump 4 spaces further.
        _x = _mm256_add_pd(_x, _four_dx);
    }
    // at the end, _ans contains 4 subtotals, so we total them up.
    return _ans[0] + _ans[1] + _ans[2] + _ans[3];
}

/**
 * @brief This uses avx1 instructions to calculate the integral from start to end using dx. 
 * See calc_integral_avx2 for more details
 * 
 * @param x_start 
 * @param x_end 
 * @param dx 
 * @return double 
 */
double calc_integral_avx1(double x_start, double x_end, double dx) {
    double ans = 0;
    __m128d _ans = _mm_setzero_pd();
    __m128d _two_dx = _mm_set1_pd(2.0 * dx);
    __m128d _four_dx = _mm_set1_pd(4.0 * dx);
    __m128d _one = _mm_set1_pd(1.0);
    __m128d _dx = _mm_set1_pd(dx);
    __m128d _x = _mm_add_pd(_mm_set_pd(0.0, 1.0 * dx), _mm_set1_pd(x_start));

    while (_x[1] < x_end) {
        _ans = _mm_add_pd(_ans, _mm_div_pd(_four_dx, _mm_add_pd(_one, _mm_mul_pd(_x, _x))));
        _x = _mm_add_pd(_x, _two_dx);
    }
    return _ans[0] + _ans[1];
}

/**
 * @brief This calculates the integral using the openmp simd directive.
 * 
 * @param dx 
 * @return double 
 */
double calc_integral_omp_simd(double dx) {
    // normal setup
    double ans = 0;
    double x;
    int i;
    int n = ceil(1.0 / dx);

// we use the directive as normal, with a reduction, with a reduction
#pragma omp simd reduction(+ \
                           : ans)
    // This loop must be in 'canonical form' (https://www.openmp.org/spec-html/5.0/openmpsu40.html#x63-1260002.9.1), which basically looks like a normal for loop.
    // It must be something like (x = start; x < P; x += y)
    for (i = 0; i < n; ++i) {
        // normal loop
        x = i * dx;
        ans += dx * 4.0 / (1.0 + x * x);
    }
    // answer
    return ans;
}
/**
 * @brief This calculates the integral using the parallel for as well as the simd directives.
 * 
 * @param dx 
 * @return double 
 */
double calc_integral_omp_simd_and_par_for(double dx) {
    // standard
    double ans = 0;
    double x;
    int i;
    int n = ceil(1.0 / dx);
// only change -> add in parallel for
#pragma omp parallel for simd reduction(+ \
                                        : ans)
    // loop is the same
    for (i = 0; i < n; ++i) {
        x = i * dx;
        ans += dx * 4.0 / (1.0 + x * x);
    }
    return ans;
}

void do_comparison() {
    // we need enough computation to actually get visible improvements from the above techniques.
    
    double dx = 1e-8;
    // all functions we want to test
    std::vector<single_func> vec_of_functions = {
        [&dx]() { return calc_integral_serial(dx, calc_integral); },
        [&dx]() { return calc_integral_par(dx, calc_integral); },
        [&dx]() { return calc_integral_par_for(dx); },
        [&dx]() { return calc_integral_serial(dx, calc_integral_avx1); },
        [&dx]() { return calc_integral_par(dx, calc_integral_avx1); },
        [&dx]() { return calc_integral_serial(dx, calc_integral_avx2); },
        [&dx]() { return calc_integral_par(dx, calc_integral_avx2); },
        [&dx]() { return calc_integral_omp_simd(dx); },
        [&dx]() { return calc_integral_omp_simd_and_par_for(dx); },

    };

    // The names of the functions for decent output.
    std::vector<std::string> names = {
        "Serial",
        "Parallel",
        "OMP Parallel For",
        "AVX 128 Serial",
        "AVX 128 Par",
        "AVX 256 Serial",
        "AVX 256 Par",
        "OMP SIMD",
        "OMP SIMD + FOR ",
    };
    double ans = calc_integral_serial(dx, calc_integral);
    printf("Correct answer = %lf\n", ans);
    Timer t(vec_of_functions, names, double_comparison, std::any(ans), 50);
    t.run();
}

int main() {
    do_comparison();
    return 0;
}
