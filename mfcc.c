// There is an example of how to compute the MFCCs at https://github.com/jsawruk/libmfcc
// This assumes that I already have the DFT though

// There is also http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
// which is a great theoretical guide

// The features in python are extracted from librosa, which is fortunately open source. So this code here attempts to replicate the librosa
// process

// Ultimately it boils down to
//  power_to_db(melspectrogram(y=<samples>, sr=16000))
// This requires us to compute the spectrogram and dot it with a suitable Mel filter

// So first, we must obtain the output of
//   _spectrogram(y=<samples>, S=None, n_fft=2048, hop_length=512, power=2)
// This is computed as
//  np.abs(stft(<samples>, n_fft=2048, hop_length=512))**2
//


#include <fftw3.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WINDOW_LENGTH 2048

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

// Set NOT_SYMMETRIC to 1 if you want a non-symmetric window for some reason
#define NOT_SYMMETRIC 0

void make_window(int size, double* window)
{
   //double A = 0.54; // Hamming
   double A = 0.5; // Hann
   // To make a symmetric window we effectively make the window 1 unit bigger (so it is odd in length) and discard the last element
   for (int i = 0; i < WINDOW_LENGTH; i++)
      window[i] = A - ((1-A) * cos(2 * M_PI * (i / ((WINDOW_LENGTH - NOT_SYMMETRIC) * 1.0))));


}

int mfccs(const char* filename, double* mfccs)
{
   // First, lets read in the header
   FILE *fd = fopen(filename, "rb");
   assert(fd != NULL);

   wav_header_t header;

   assert(fread(&header, 1, sizeof(wav_header_t), fd) == sizeof(wav_header_t));
   assert(header.sample_rate == 16000);
   assert(header.fmt_chunk_size == 16);

   double window[WINDOW_LENGTH];
   make_window(WINDOW_LENGTH, window);

   int sample_duration = 2;
   int sample_count = sample_duration * header.sample_rate;
   double *samples = malloc(sizeof(double) * sample_count);

   // Now read in each sample, dividing by the maximum possible value to get an input between 0 and 1
   for (int i = 0 ; i < sample_count; i++)
   {
      int16_t sample;
      if (feof(fd))
         sample = 0;
      else
      {
         int b = fread(&sample, 1, sizeof(int16_t), fd);
         assert(b >= 0);
         if (b == 0)
            sample = 0;
      }
      // Normalize the data to be in the range -1..1
      samples[i] = (double)sample / 32768.0;
   }
   printf("sample1: %.20f\n", samples[1]);
   printf("window1: %.20f\n", window[1]);
   fclose(fd);

   // Ok, now we have the samples in *samples and we must do the stft
   // First set up the buffers and build the FFT plan

   fftw_complex    *data, *fft_result, *ifft_result;
   fftw_plan       plan_forward;

   data        = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   fft_result  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   ifft_result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   plan_forward = fftw_plan_dft_1d(WINDOW_LENGTH, data, fft_result, FFTW_FORWARD, FFTW_ESTIMATE );

   // Now we read in 2048 samples at a time, starting at 0, then 512, then 1024, and so on, until we get to the end

   int block_offset = 0;
   int readIndex;
   // Process each 2048-sample block of the signal
   while (block_offset < sample_count)
   {
      // Copy the chunk into our buffer. The real part is the windowed sample. The imaginary part is 0.
      for (int i = 0; i < WINDOW_LENGTH; i++)
      {
         readIndex = block_offset + i;
         // Apply the hann window
         data[i][0] = samples[readIndex] * window[i];
         data[i][1] = 0.0;
      }

      // Actually do the FFT
      fftw_execute(plan_forward);

      // At this point we have the right answer that stft would give!

      // Next, square each element and take the absolute value. This is the spectrogram

      // Then we need the dot product of this with our Mel filter. This is the 'melspectrogram'.

      // Then finally convert the data from power to db using:
      // ref=1.0, amin=1e-10, top_db=80.0

      block_offset += 512;
  
   }
 
   fftw_destroy_plan(plan_forward);
   fftw_free(data);
   fftw_free(fft_result);
   fftw_free(ifft_result);
   return 1;
}
