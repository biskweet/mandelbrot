/*
 * Sorbonne Université  --  PPAR
 * Computing the Mandelbrot set, sequential version
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>           /* compile with -lm */
#include <sys/time.h>
#include <arpa/inet.h>      /* htonl */

#include <mpi.h>
#include <errno.h>
// #include "mpi.h"

#include "rasterfile.h"

#define _FILE_OFFSET_BITS 64
#define _LARGEFILE64_SOURCE


char info[] = "\
Usage:\n\
      ./mandel dimx dimy xmin ymin xmax ymax depth\n\
\n\
      dimx,dimy : dimensions of the image to generate\n\
      xmin,ymin,xmax,ymax : domain to be computed, in the complex plane\n\
      depth : maximal number of iterations\n\
\n\
Some examples of execution:\n\
      ./mandel 3840 3840 0.35 0.355 0.353 0.358 200\n\
      ./mandel 3840 3840 -0.736 -0.184 -0.735 -0.183 500\n\
      ./mandel 3840 3840 -0.736 -0.184 -0.735 -0.183 300\n\
      ./mandel 3840 3840 -1.48478 0.00006 -1.48440 0.00044 100\n\
      ./mandel 3840 3840 -1.5 -0.1 -1.3 0.1 10000\n\
";

double wallclock_time()
{
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6);
}

unsigned char cos_component(int i, double freq)
{
    double iD = i;
    iD = cos(iD / 255.0 * 2 * M_PI * freq);
    iD += 1;
    iD *= 128;
    return iD;
}

/**
 *  Save the data array in rasterfile format
 */
void save_rasterfile(char * name, size_t largeur, size_t hauteur, unsigned char * p)
{
    FILE *fd = fopen(name, "w");
    if (fd == NULL) {
        perror("Error while opening output file. ");
        exit(1);
    }

    printf("writing in %s, largeur=%zu, hauteur=%zu\n", name, largeur, hauteur);

    struct rasterfile file;
    file.ras_magic = htonl(RAS_MAGIC);
    file.ras_width = htonl(largeur);            /* width of the image, in pixels */
    file.ras_height = htonl(hauteur);           /* height of the image, in pixels */
    file.ras_depth = htonl(8);              /* depth of each pixel (1, 8 or 24 ) */
    file.ras_length = htonl(largeur * hauteur);      /* size of the image in nb of bytes */
    file.ras_type = htonl(RT_STANDARD);         /* file type */
    file.ras_maptype = htonl(RMT_EQUAL_RGB);
    file.ras_maplength = htonl(256 * 3);

    fwrite(&file, sizeof(struct rasterfile), 1, fd);

    /* Color palette: red component */
    for (int i = 255; i >= 0; i--) {
        unsigned char o = cos_component(i, 13.0);
        fwrite(&o, sizeof(unsigned char), 1, fd);
    }

    /* Color palette: green component */
    for (int i = 255; i >= 0; i--) {
        unsigned char o = cos_component(i, 5.0);
        fwrite(&o, sizeof(unsigned char), 1, fd);
    }

    /* Color palette: blue component */
    for (int i = 255; i >= 0; i--) {
        unsigned char o = cos_component(i + 10, 7.0);
        fwrite(&o, sizeof(unsigned char), 1, fd);
    }

    size_t written = 0;
    size_t chunk_size = 1024 * 1024 * 128;

    // Write file chunk by chunk
    while (written < (largeur * hauteur)) {
        printf("Writing... %.2lf%%\r", 100.0 * written / (largeur * hauteur)); fflush(fd);
        size_t write_size = (largeur * hauteur - written > chunk_size) ? chunk_size : (largeur * hauteur - written);
        size_t result = fwrite(p, sizeof(unsigned char), write_size, fd);
        if (result < write_size) {
            printf("\nWrite error (%zu bytes written over %zu)\n", written, largeur * hauteur);
            perror("");
            exit(1);
        }
        written += write_size;
    }

    printf("Writing... %.2lf%%\n", 100.0 * written / (largeur * hauteur));

    fclose(fd);
}

/**
 * Given the coordinates of a point $c = a + ib$ in the complex plane,
 * the function returns the corresponding color which estimates the
 * distance from the Mandelbrot set to this point.
 * Consider the complex sequence defined by:
 * \begin{align}
 *     z_0     &= 0 \\
 *     z_{n+1} &= z_n^2 + c
 *   \end{align}
 * The number of iterations that this sequence needs before diverging
 * is the number $n$ for which $|z_n| > 2$.
 * This number is brought back to a value between 0 and 255, thus
 * corresponding to a color in the color palette.
 */
unsigned char xy2color(double a, double b, int depth)
{
    double x = 0;
    double y = 0;
    for (int i = 0; i < depth; i++) {
        /* save the former value of x (which will be erased) */
        double temp = x;
        /* new values for x and y */
        double x2 = x * x;
        double y2 = y * y;
        x = x2 - y2 + a;
        y = 2 * temp * y + b;
        if (x2 + y2 > 4.0)
            return (i % 255);   /* diverges */
    }
    return 255;   /* did not diverge in depth steps */
}


enum FLAG {
    WORK_MESSAGE,
    END_OF_WORK,
};


void copy_lines_in_image(unsigned char * image, unsigned char const * source, unsigned int offset, unsigned int length)
{
    for (unsigned int i = 0; i < length; i++)
        image[offset + i] = source[i];
}


/*
 * Main: in each point of the grid, apply xy2color
 */
int main(int argc, char **argv)
{
    if (argc == 1) {
        fprintf(stderr, "%s\n", info);
        return 1;
    }

    /* Default values for the fractal */
    double xmin = -2;     /* Domain of computation, in the complex plane */
    double ymin = -2;
    double xmax = 2;
    double ymax = 2;
    size_t w = 3840;     /* dimensions of the image (4K HTDV!) */
    size_t h = 3840;
    int depth = 10000;     /* depth of iteration */

    /* Retrieving parameters */
    if (argc > 1)
        w = atoi(argv[1]);
    if (argc > 2)
        h = atoi(argv[2]);
    if (argc > 3)
        xmin = atof(argv[3]);
    if (argc > 4)
        ymin = atof(argv[4]);
    if (argc > 5)
        xmax = atof(argv[5]);
    if (argc > 6)
        ymax = atof(argv[6]);
    if (argc > 7)
        depth = atoi(argv[7]);

    /* Computing steps for increments */
    double xinc = (xmax - xmin) / (w - 1);
    double yinc = (ymax - ymin) / (h - 1);

    /* MPI stuff */
    int rank;
    int p;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int granularity = 5; // number of lines pr
    MPI_Status status;

    int * metadata = malloc(sizeof(unsigned int) * 2);

    // Say p-1 is the root
    if (rank == (p - 1)) {
        printf("Launching with %d threads...\n", p);

        unsigned char * image = (unsigned char *) malloc(sizeof(unsigned int) * w * h);

        // Keep track of what anyone is working on
        unsigned int * offset_process_is_working_with = (unsigned int *) malloc(sizeof(unsigned int) * (p - 1));
        for (unsigned int i = 0; i < (p - 1); i++)
            offset_process_is_working_with[i] = i * granularity;

        unsigned int processed_lines_amount;

        unsigned char * computed_lines = (unsigned char *) malloc(sizeof(unsigned char) * w * granularity);

        double beginning = wallclock_time();

        // Start listening for computed data
        for (int i = (p - 1) * granularity; i < h; i += granularity) {
            MPI_Recv(computed_lines, w * granularity, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            processed_lines_amount = granularity;

            // If there's less than `granularity` lines remaining, just request the difference
            if (offset_process_is_working_with[status.MPI_SOURCE] > (h - granularity))
                processed_lines_amount = h - offset_process_is_working_with[status.MPI_SOURCE];

            copy_lines_in_image(image, computed_lines, offset_process_is_working_with[status.MPI_SOURCE] * w, processed_lines_amount * w);

            metadata[0] = i;
            metadata[1] = (h - i) >= granularity ? granularity : (h - i);

            // Sending next line
            MPI_Send(metadata, 2, MPI_INT, status.MPI_SOURCE, WORK_MESSAGE, MPI_COMM_WORLD);

            printf("%.2f%%\r", 100.0 * i / h);

            offset_process_is_working_with[status.MPI_SOURCE] = i;
        }

        printf("%.2f%%\n", 100.0);

        double end = wallclock_time();

        // End of work, notify all processes
        for (unsigned int i = 0; i < p-1; i++) {
            MPI_Recv(computed_lines, w * granularity, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            processed_lines_amount = granularity;
            if (offset_process_is_working_with[status.MPI_SOURCE] > (h - granularity))
                processed_lines_amount = h - offset_process_is_working_with[status.MPI_SOURCE];

            copy_lines_in_image(image, computed_lines,
                                offset_process_is_working_with[status.MPI_SOURCE] * w,
                                processed_lines_amount * w);

            MPI_Send(metadata, 2, MPI_INT, status.MPI_SOURCE, END_OF_WORK, MPI_COMM_WORLD);
        }

        fprintf(stderr, "-----\nTotal computing time: %g sec\n", end - beginning);

        char filename[100];
        sprintf(filename, "mandel_%zux%zu.ras", w, h);

        save_rasterfile(filename, w, h, image);

        free(offset_process_is_working_with);
        free(image);
        free(computed_lines);

    } else {

        /* WORKER POV */

        unsigned char * subimage = (unsigned char *) malloc(sizeof(unsigned char) * w * granularity);

        unsigned int startingpoint = (rank * granularity);
        unsigned int no_lines_to_process = granularity;

        double x;
        double y;

        do {
            y = ymin + yinc * startingpoint;
            x = xmin;

            for (unsigned int i = 0; i < no_lines_to_process * w; i++) {
                if (i % w == 0)
                    x = xmin;

                subimage[i] = xy2color(x, y, depth);
                x += xinc;

                if (i % w == 0)
                    y += yinc;
            }

            MPI_Send(subimage, w * granularity, MPI_CHAR, p - 1, WORK_MESSAGE, MPI_COMM_WORLD);

            MPI_Recv(metadata, 2, MPI_INT, p - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            startingpoint = metadata[0];
            no_lines_to_process = metadata[1];

        } while (status.MPI_TAG != END_OF_WORK);

        free(subimage);
    }

    free(metadata);

    MPI_Finalize();

    return 0;
}
