CC=mpicc
FLAGS=-O3 -lm

all: boss-worker

boss-worker: boss-worker.c
	$(CC) $^ -o $@ $(FLAGS)
