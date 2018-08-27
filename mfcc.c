// There is an example of how to compute the MFCCs at https://github.com/jsawruk/libmfcc
// This assumes that I already have the DFT though

#include <fftw3.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
   char riff_header[4];    // Contains "RIFF"
   int wav_size;           // Size of the wav portion of the file, which follows the first 8 bytes. File size - 8
   char wave_header[4];    // Contains "WAVE"

   // Format Header
   char fmt_header[4];     // Contains "fmt " (includes trailing space)
   int fmt_chunk_size;     // Should be 16 for PCM
   short audio_format;     // Should be 1 for PCM. 3 for IEEE Float
   short num_channels;
   int sample_rate;
   int byte_rate;          // Number of bytes per second. sample_rate * num_channels * Bytes Per Sample
   short sample_alignment; // num_channels * Bytes Per Sample
   short bit_depth;        // Number of bits per sample
    
   // Data
   char data_header[4];    // Contains "data"
   int data_bytes;         // Number of bytes in data. Number of samples * num_channels * sample byte size
}  wav_header_t;

int mfccs(const char* filename, float* mfccs)
{
   // First, lets read in the header
   FILE *fd = fopen(filename, "rb");
   assert(fd != NULL);

   wav_header_t header;

   assert(fread(&header, 1, sizeof(wav_header_t), fd) == sizeof(wav_header_t));
   assert(header.sample_rate == 16000);
   assert(header.fmt_chunk_size == 16);

   int sample_duration = 2; // We only care about the first 2 seconds
   int sample_count = header.sample_rate * sample_duration;
   double *samples = malloc(sizeof(double) * sample_count);
   double *frequencies = malloc(sizeof(double) * sample_count);

   // Now read in each sample, dividing by the maximum possible value to get an input between 0 and 1
   for (int i = 0 ; i < sample_count; i++)
   {
      uint16_t sample;
      if (feof(fd))
         sample = 0;
      else
         assert(fread(&sample, 1, sizeof(uint16_t), fd) == sizeof(uint16_t));
      samples[i++] = (double)sample / 65535.f;
   }
   fclose(fd);

   fftw_plan plan =  fftw_plan_r2r_1d(sample_count, samples, frequencies, FFTW_R2HC, FFTW_ESTIMATE);
   fftw_execute(plan);
   fftw_destroy_plan(plan);

   // Ok. Now we have the frequencies, and must compute the MFCCs from them
   printf("frequencies[0]: %f\n", frequencies[0]);
   return 0;
}
