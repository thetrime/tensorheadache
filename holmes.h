#include <fftw3.h>
#include "ibuprofen.h"

#define WINDOW_LENGTH 1600
#define MEL_FILTER_COUNT 20
#define MFCC_COUNT 20
// Set NON_SYMMETRIC to 1 if you want a non-symmetric window for some reason
#define NON_SYMMETRIC 0
#define CHUNK_COUNT 63
#define HOP_SIZE 800

int mfccs_from_file(const char* filename, float*);
typedef struct
{
   double** mel_bank;
   fftw_complex  *data, *fft_result;
   fftw_plan plan;
   model_t* model;
   double buffer[WINDOW_LENGTH];
   int bufptr;
} context_t;

context_t* alloc_context(char* filename, int sample_rate);
void free_context(context_t* context);
int mfccs_from_circular_buffer(context_t* context, int start, int bufsize, float*, int outptr);
int mfccs_from_file(const char* filename, float*);
double process_block_double(context_t* context, double* data, int inNumPackets);
double process_block_int16(context_t* context, int16_t* data, int inNumPackets);
