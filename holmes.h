#include <fftw3.h>

#define WINDOW_LENGTH 1600
#define MEL_FILTER_COUNT 20
#define MFCC_COUNT 20
// Set NON_SYMMETRIC to 1 if you want a non-symmetric window for some reason
#define NON_SYMMETRIC 0
#define CHUNK_COUNT 63
#define HOP_SIZE 800

int mfccs_from_file(const char* filename, float*);
