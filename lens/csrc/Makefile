# lens/csrc/Makefile — builds our own kernel library. No BLAS, no external deps.
#
# -O3            full optimisation
# -march=native  use this CPU's vector instructions (AVX2 on the Ryzen 5 4650G)
# -ffast-math    allow reassociation, which is what lets the vectoriser fuse the loops.
#                Honest caveat: this makes float results slightly non-reproducible across
#                thread counts. The L0 gate checks against numpy within a stated tolerance
#                rather than demanding bit equality.
# -fopenmp       threads for every loop nest
# -funroll-loops the inner j-loops in the matmul kernels
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -ffast-math -fopenmp -funroll-loops -fPIC -Wall -Wextra
LDFLAGS ?= -shared -fopenmp -lm
OUT     := ../liblens.so

.PHONY: all clean

all: $(OUT)

$(OUT): kern.c
	$(CC) $(CFLAGS) -o $(OUT) kern.c $(LDFLAGS)
	@echo "built $(OUT)"

clean:
	rm -f $(OUT)
