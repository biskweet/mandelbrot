CC=mpicc
FLAGS=-O3 -lm -D_FILE_OFFSET_BITS=64

all: boss-worker

boss-worker: boss-worker.c
	$(CC) $^ -o $@ $(FLAGS)
