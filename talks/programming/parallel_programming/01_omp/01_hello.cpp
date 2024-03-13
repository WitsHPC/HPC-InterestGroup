#include <omp.h>
#include <stdio.h>
int main(){
	// a parallel region
	#pragma omp parallel
	{
		// this code gets executed by every thread.
		printf("Hello World\n");
	}
	return 0;
}
