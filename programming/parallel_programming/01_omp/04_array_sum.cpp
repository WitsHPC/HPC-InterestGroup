#include <omp.h>
#include <chrono>
#include <string>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <numeric>
// This just gets a vector to add.
std::vector<int> get_vector(int nums){
    std::vector<int> v;
    // make the vector
    for (int i=0; i< nums; ++i){
        v.push_back(i);
    }
    return v;
}
// A few different functions we have. They are defined down below.
int do_sum_manual(const std::vector<int> &v);
int do_sum_serial(const std::vector<int> &v);
int do_sum_par_for(const std::vector<int> &v);
int faster_for(const std::vector<int> &v);
int for_reduction(const std::vector<int> &v);

// This is the function's signature, to be able to declare a vector of functions.
typedef int (*function) (const std::vector<int>&);

int main(){
    int N = 10000000;
    int experiments = 100;
    // The functions to compare
    std::vector<function> vec_of_functions = {
        do_sum_serial,
        do_sum_manual,
        do_sum_par_for, 
        faster_for,
        for_reduction
    };
    
    // The names of the functions for decent output.
    std::vector<std::string> names = {
        "Serial",
        "Manual",
        "Parallel For",
        "Faster For",
        "For Reduction"
    };
    std::vector<double> times(vec_of_functions.size(), 0.0);
    std::vector<int> values(vec_of_functions.size(), 0);
    // Get the vector
    auto vector_to_use = get_vector(N);
    // for each function
    for (int f =0; f < vec_of_functions.size(); ++f){
        // repeat 100 times to average out noise.
        for (int i=0; i < experiments; ++i){
            // get the time now
            auto start = std::chrono::high_resolution_clock::now();
            // perform the result
            values[f] += vec_of_functions[f](vector_to_use);
            // end time
            auto end = std::chrono::high_resolution_clock::now();
            // duration
            auto duration = std::chrono::duration<double, std::milli>(end - start).count();
            times[f] += duration;
        }
    }
    // get the correct answer to validate.
    int correct_answer = std::accumulate(vector_to_use.cbegin(), vector_to_use.cend(), 0);
    // now just print the results.
    for (int f = 0; f < vec_of_functions.size(); ++f){
        bool is_correct = correct_answer * experiments == values[f];
        printf("Method %-20s. Correct Answer = %-10s. Time = %lfms\n", names[f].c_str(), is_correct ? "YES" : "NO", times[f] / experiments);
    }
}

int do_sum_serial(const std::vector<int> &v){
    // simple array sum, serially.
    int total = 0;
    for (auto &i: v){
        total += i;
    }
    return total;
}

// Somewhat manually
int do_sum_manual(const std::vector<int> &v){
    int N = v.size();
    // shared variable
    std::vector<int> totals (omp_get_max_threads(), 0);
#pragma omp parallel
    {
        // how many threads are there?
        int num_threads = omp_get_num_threads();
        // how many numbers should each thread process?
        int per_thread = ceil((float)N / num_threads);
        // what is my index?
        int my_thread_num = omp_get_thread_num();

        // Where should this thread start and end?
        int my_start = per_thread * my_thread_num;
        int my_end = my_start + per_thread;

        // go over all my numbers and add them
        for (int i=my_start; i < my_end; ++i){
            totals[my_thread_num] += v[i];
        }

    }


    // outside region, sum up the partial sums.
    int total = 0;
    for (auto i: totals) total += i;

    return total;
}

// Parallel for
int do_sum_par_for(const std::vector<int> &v){
    int N = v.size();
    int global_total = 0;
    // intermediate results
    std::vector<int> totals (omp_get_max_threads(), 0);
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        // sum into intermediate vector.
#pragma omp for
        for (int i=0; i < N; ++i){
            totals[thread_num] += v[i];
        }

    }
    int total = 0;
    for (auto i: totals) total += i;

    return total;
}

// Somewhat faster for, using a local total and a global total.
// This could be faster because of the lack of false sharing.
int faster_for(const std::vector<int> &v){
    int N = v.size();
    int global_total = 0;
#pragma omp parallel
    {
        // this is defined per threads.
        int my_total = 0;
// update my total
#pragma omp for
        for (int i=0; i < N; ++i){
            my_total += v[i];
        }
// update the global total, but do it atomically, to avoid race conditions.
#pragma omp atomic
        global_total += my_total;
    }
    return global_total;
}

// This uses a reduction
int for_reduction(const std::vector<int> &v){
    int N = v.size();
    int global_total = 0;
// the reduction syntax is something like (operation:variable), and openmp handles most things.
#pragma omp parallel for reduction(+:global_total)
        for (int i=0; i < N; ++i){
            global_total += v[i];
        }
    return global_total;
}
