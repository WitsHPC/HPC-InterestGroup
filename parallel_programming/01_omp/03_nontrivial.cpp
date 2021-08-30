#include <omp.h>
#include <stdio.h>
int main(){
    // This is actually wrong, because there is a race condition.
    int my_total = 0;
	#pragma omp parallel
	{
        // go over some numbers
        for (int i=0; i<100; ++i)
            // and add up
            my_total += omp_get_thread_num();
	}
    printf("The total sum = %d\n", my_total);
	return 0;
}
