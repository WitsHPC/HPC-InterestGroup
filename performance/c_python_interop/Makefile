CC=gcc
CFLAGS=-shared
all:  dot_product.so dot_product_avx.so dot_product_faster.so
dot_product.so: dot_product.c
	${CC} $^ ${CFLAGS} -o $@

dot_product_faster.so: dot_product.c
	${CC} $^ ${CFLAGS} -o $@ -O3
	
dot_product_avx.so: dot_product_avx.c
	${CC} $^ ${CFLAGS} -o $@ -O3 -mavx -mavx2

clean:
	rm *.so