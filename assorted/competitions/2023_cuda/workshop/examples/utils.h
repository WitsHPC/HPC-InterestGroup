#include <time.h>
#include <sys/time.h>
#ifndef C03FCF51_CED7_4623_9D79_0D005991335F
#define C03FCF51_CED7_4623_9D79_0D005991335F
float get_wtime(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return ((long)t.tv_sec * 1000000 + (long)t.tv_nsec / 1000)/1000.0f;

}

#endif /* C03FCF51_CED7_4623_9D79_0D005991335F */
