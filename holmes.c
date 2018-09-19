// This is a rewrite of the mfcc code to get the same answer that mycroft gets.

#include <fftw3.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "holmes.h"

#define FFT_SIZE 512
#define FFT_COEFFICIENTS ((int)((FFT_SIZE / 2) + 1))
typedef struct
{
   char riff_header[4];    // Contains "RIFF"
   int wav_size;           // Size of the wav portion of the file, which follows the first 8 bytes. File size - 8
   char wave_header[4];    // Contains "WAVE"

   // Format Header
   char fmt_header[4];     // Contains "fmt " (includes trailing space)
   int fmt_chunk_size;     // Should be 16 for PCM
   short audio_format;     // Should be 1 for PCM. 3 for IEEE float
   short num_channels;
   int sample_rate;
   int byte_rate;          // Number of bytes per second. sample_rate * num_channels * Bytes Per Sample
   short sample_alignment; // num_channels * Bytes Per Sample
   short bit_depth;        // Number of bits per sample
    
   // Data
   char data_header[4];    // Contains "data"
   int data_bytes;         // Number of bytes in data. Number of samples * num_channels * sample byte size
}  wav_header_t;

double process_block_double(context_t* context, double* data, int inNumPackets)
{
   int i = 0;
   while (i < inNumPackets)
   {
      // Compute how many bytes would bring the buffer up to a new chunk
      int thisChunkSize = 800 - (context->bufptr % 800);
      // Do not copy more packets than we have though
      if (thisChunkSize + i > inNumPackets)
         thisChunkSize = inNumPackets - i;
      // Copy the data in to the buffer
      memcpy(&context->buffer[context->bufptr], &data[i], thisChunkSize*sizeof(double));
      context->bufptr += thisChunkSize;
      context->bufptr = context->bufptr % 1600;
      if ((context->bufptr % 800) == 0)
      {
         // If bufptr is 0 then the block from 800 to 800 is ready. Otherwise the block from 0-1600 is ready
         // The thing that I missed the first time is that precise uses a progressive buffer. New mfcc vectors are added at the end
         // which means that effectively, everything gets shuffled back one index each time we add a new vector. It is not circular unless
         // we can move the 'start' point of the tensor data forward by 1 vector each time. There might be a nicer solution than this, but
         // for now, just move the array back one, and ALWAYS write to the last spot in the tensor data stucture
         memcpy(model_data(context->model), &model_data(context->model)[13], sizeof(float)*13*28);
         mfccs_from_circular_buffer(context, context->bufptr, 1600, model_data(context->model), 28*13);
      }
      i += thisChunkSize;
   }
   return run_model(context->model);
}

double process_block_int16(context_t* context, int16_t* data, int inNumPackets)
{
   int i = 0;
   while (i < inNumPackets)
   {
      // Compute how many bytes would bring the buffer up to a new chunk
      int thisChunkSize = 800 - (context->bufptr % 800);
      // Do not copy more packets than we have though
      if (thisChunkSize + i > inNumPackets)
         thisChunkSize = inNumPackets - i;
      // Copy the data in to the buffer, sample by agonizing sample, turning each one into a double
      for (int j = 0; j < thisChunkSize; j++)
         context->buffer[context->bufptr++] = (double)data[j] / 32768.0;
      context->bufptr = context->bufptr % 1600;
      if ((context->bufptr % 800) == 0)
      {
         // If bufptr is 0 then the block from 800 to 800 is ready. Otherwise the block from 0-1600 is ready
         // The thing that I missed the first time is that precise uses a progressive buffer. New mfcc vectors are added at the end
         // which means that effectively, everything gets shuffled back one index each time we add a new vector. It is not circular unless
         // we can move the 'start' point of the tensor data forward by 1 vector each time. There might be a nicer solution than this, but
         // for now, just move the array back one, and ALWAYS write to the last spot in the tensor data stucture
         memcpy(model_data(context->model), &model_data(context->model)[13], sizeof(float)*13*28);
         mfccs_from_circular_buffer(context, context->bufptr, 1600, model_data(context->model), 28*13);
      }
      i += thisChunkSize;
   }
   return run_model(context->model);
}



double hz_to_mels(double hz)
{
   // This is the O'Shaugnessy algorithm
   return 1127.0 * log(1.0 + (hz / 700.0));
}

double mels_to_hz(double mels)
{
   return 700.0 * (exp(mels / 1127.0) - 1.0);
}

void free_mel_bank(double** mels)
{
   for (int i = 0; i < MEL_FILTER_COUNT; i++)
      free(mels[i]);
   free(mels);
}

double** make_mel_bank(double sample_rate)
{
   double max_frequency_hz = sample_rate;
   double min_frequency_hz = 0;
   double max_frequency_mels = hz_to_mels(max_frequency_hz);
   double min_frequency_mels = hz_to_mels(min_frequency_hz);
   float centroids[MEL_FILTER_COUNT+2];

   double **mels = malloc(sizeof(double*) * MEL_FILTER_COUNT);
   for (int i = 0; i < MEL_FILTER_COUNT; i++)
      mels[i] = malloc(sizeof(double) * FFT_COEFFICIENTS);


   // Conceptually we now want MEL_FILTER_COUNT+2 equally spaced points
   // The first one will be min_frequency_mels, and the last one max_frequency_mels
   // However, we want these in Hz, not Mels

   double spacing = max_frequency_mels / (double)(MEL_FILTER_COUNT+1);

   for (int i = 0; i < MEL_FILTER_COUNT+2; i++)
      centroids[i] = (int)((FFT_COEFFICIENTS * mels_to_hz(i * spacing)) / sample_rate);

   // Now fill in the array. We fill in the k possible values for each filter m
   // k is an FFT bin though, and the centroids are Hz. This is important, as librosa
   // produces a slightly different result using this technique than the bank computed
   // via the practicalcryptography method. The latter claims that nothing is degraded,
   // but for the purposes of checking my implementation is correct, it helps if the numbers
   // match!

   for (int m = 0; m < MEL_FILTER_COUNT; m++)
   {
      for (int k = 0; k < FFT_COEFFICIENTS; k++)
      {
         // There are 4 cases here. A graph might help.
         //
         // filter
         //   ^
         //   |         ^
         //   |        /|\
         //   |       / | \
         //   +------/..|..\----->  frequency(hz)
         //          |  |  |
         //          A  B  C
         //
         // For filter m, B is the 'target' centroid - which is centroids[m+1].
         // This means:
         //   * if the frequency is less than A (ie centroids[m]) then the filter is 0
         //   * if the frequency is between A and B (ie centroids[m] to centroids[m+1]) then the filter is sloping upwards
         //   * if the frequency is between B and C (ie centroids[m+1] to centroids[m+2]) then the filter is sloping downwards
         //   * if the frequency is greater than C (ie centroids[m+2]) then the filter is 0

         if (k < centroids[m])
            mels[m][k] = 0;
         else if (k <= centroids[m+1])
            mels[m][k] = ((k - centroids[m]) / (centroids[m+1] - centroids[m]));
         else if (k <= centroids[m+2])
            mels[m][k] = ((centroids[m+2] - k) / (centroids[m+2] - centroids[m+1]));
         else
            mels[m][k] = 0;
      }
   }
   return mels;
}

#define MAX(a, b) (a > b ? a : b)

double power_to_db(double power)
{
   return 10.0 * log10(MAX(1e-10, power));
}

context_t* alloc_context(char* filename, int sample_rate)
{
   context_t* context = malloc(sizeof(context_t));
   context->mel_bank = make_mel_bank(sample_rate);
   context->data= (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   context->fft_result  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   for (int i = 0; i < WINDOW_LENGTH; i++)
      context->data[i][1] = 0.0;
   context->plan = fftw_plan_dft_1d(FFT_SIZE, context->data, context->fft_result, FFTW_FORWARD, FFTW_ESTIMATE );
   context->model = load_model(filename);
   float* f = model_data(context->model);
   memset(f, 0, sizeof(float) * 13 * 29);
   memset(context->buffer, 0, sizeof(double)*WINDOW_LENGTH);
   context->bufptr = 0;
   return context;
}

void free_context(context_t* context)
{
   fftw_destroy_plan(context->plan);
   fftw_free(context->data);
   fftw_free(context->fft_result);
   free_mel_bank(context->mel_bank);
}

int mfccs_from_circular_buffer(context_t* context, int start, int bufsize, float* out, int outptr)
{
   // Initially this is going to be very expensive because we do all this setup stuff each time
   // Once it works, I can concentrate on just doing the expensive stuff once
   double melspectrogram[MEL_FILTER_COUNT];
   double block_sum;
   double* samples = context->buffer;

   // Load in the data
   //printf("Input: ");
   for (int i = 0; i < WINDOW_LENGTH; i++)
   {
      context->data[i][0] = samples[(start + i) % bufsize];
      //printf("%.3f ", context->data[i][0]);
   }
   //   printf("\n");
   // Do the FFT
   fftw_execute(context->plan);

   // Compute the absolute value of the components
   block_sum = 0;
   for (int i = 0; i <= FFT_COEFFICIENTS; i++)
   {
      context->fft_result[i][0] = ((context->fft_result[i][0] * context->fft_result[i][0]) + (context->fft_result[i][1] * context->fft_result[i][1])) / FFT_SIZE;
      block_sum += context->fft_result[i][0];
   }

   // Compute the melspectrogram
   for (int i = 0; i < MEL_FILTER_COUNT; i++)
   {
      double m = 0;
      for (int j = 0; j < FFT_COEFFICIENTS; j++)
         m = m + context->mel_bank[i][j] * context->fft_result[j][0];
      // Then we take the log of MAX(2.220446049250313e-16, m), for some reason
      m = log(MAX(2.220446049250313e-16, m));
      melspectrogram[i] = m;
   }

   // Then do the DCT to get the MFCCs
   for (int k = 0; k < MFCC_COUNT; k++)
   {
      double sum = 0;
      for (int n = 0; n < MEL_FILTER_COUNT; n++)
         sum += melspectrogram[n] * cos(M_PI * k * (2*n+1) / (2 * MEL_FILTER_COUNT));
      double result;
      if (k == 0)
         result = log(MAX(2.220446049250313e-16, block_sum));
      else
         result = 2 * sum * sqrt(1.0/(2.0*MEL_FILTER_COUNT));
      //printf("%.10f ", result);
      out[outptr] = result;
      outptr = (outptr + 1) % 377;
   }
   //printf("\n");
   return 1;
}

int mfccs_from_file(const char* filename, float* mfccs)
{
   // First, lets read in the header
   FILE *fd = fopen(filename, "rb");
   assert(fd != NULL);

   wav_header_t header;
   assert(fread(&header, sizeof(wav_header_t), 1, fd) == 1);
   assert(header.sample_rate == 16000);
   assert(header.fmt_chunk_size == 16);

   int sample_duration = 2;
   int pad_size = WINDOW_LENGTH;
   int sample_count = sample_duration * header.sample_rate;
   int chunks = (sample_count + pad_size - WINDOW_LENGTH) / HOP_SIZE + 1;
   int next_output = 0;

   // We need to leave space for 1024 samples of padding at the beginning and end
   double *samples = malloc(sizeof(double) * (sample_count + 2048));

   // Now read in each sample, dividing by the maximum possible value to get an input between 0 and 1
   for (int i = 0 ; i < sample_count; i++)
   {
      int16_t sample;
      if (feof(fd))
         sample = 0;
      else
      {
         int b = fread(&sample, sizeof(int16_t), 1, fd);
         assert(b >= 0);
         if (b == 0)
         {
            printf("EOF detected at sample %d\n", i);
            sample = 0;
         }
      }
      // Normalize the data to be in the range -1..1
      samples[i] = (double)sample / 32768.0;
   }
   fclose(fd);

   double** mel_bank = make_mel_bank(header.sample_rate);
   double melspectrogram[MEL_FILTER_COUNT][chunks];
   double block_sum[chunks];

   // Ok, now we have the samples in *samples and we must do the stft
   // First set up the buffers and build the FFT plan
   fftw_complex    *data, *fft_result;
   fftw_plan       plan_forward;

   data        = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   fft_result  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   plan_forward = fftw_plan_dft_1d(FFT_SIZE, data, fft_result, FFTW_FORWARD, FFTW_ESTIMATE );

   // Now we read in 2048 samples at a time, starting at 0, then 512, then 1024, then 1536, and so on, until we get to the end
   int block_offset = 0;
   int readIndex;

   // Process each 2048-sample block of the signal, stopping when the last block would exceed the input buffer
   for (int k = 0; block_offset < sample_count; k++)
   {
      // Copy the chunk into our buffer. The real part is the windowed sample. The imaginary part is 0.
      for (int i = 0; i < WINDOW_LENGTH; i++)
      {
         readIndex = block_offset + i;
         data[i][0] = samples[readIndex];
         data[i][1] = 0.0;
      }

      // Actually do the FFT
      fftw_execute(plan_forward);

      // Next, compute the square of the absolute value of each element. Fortunately this is just data[i][0]**2 + data[i][1]**2
      // Since we do not need the raw data again, we can just put this directly back into data[i][0]
      // Also note that we only care about the first half of the output because of symmetry around the nyquist frequency
      // This means we can go from 0 to 1024 (inclusive) and skip the rest
      block_sum[k] = 0;
      for (int i = 0; i <= FFT_COEFFICIENTS; i++)
      {
         fft_result[i][0] = ((fft_result[i][0] * fft_result[i][0]) + (fft_result[i][1] * fft_result[i][1])) / FFT_SIZE;
         block_sum[k] += fft_result[i][0];
      }

      // Then we need the dot product of this with our Mel filter. This is the 'melspectrogram'.
      for (int i = 0; i < MEL_FILTER_COUNT; i++)
      {
         double m = 0;
         for (int j = 0; j < FFT_COEFFICIENTS; j++)
            m = m + mel_bank[i][j] * fft_result[j][0];
         assert(k <= chunks);
         // Then we take the log of MAX(2.220446049250313e-16, m), for some reason
         m = log(MAX(2.220446049250313e-16, m));
         melspectrogram[i][k] = m;
      }
      block_offset += HOP_SIZE;
   }

   // And finally we must do the DCT on each melspetrogram to get the mfccs:
   //           N-1
   // y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
   //           n=0
   for (int m = 0; m < chunks; m++)
   {
      for (int k = 0; k < MFCC_COUNT; k++)
      {
         double sum = 0;
         for (int n = 0; n < MEL_FILTER_COUNT; n++)
            sum += melspectrogram[n][m] * cos(M_PI * k * (2*n+1) / (2 * MEL_FILTER_COUNT));
         double result;
         if (k == 0)
            result = log(MAX(2.220446049250313e-16, block_sum[k]));
         else
            result = 2 * sum * sqrt(1.0/(2.0*MEL_FILTER_COUNT));
         printf("mfcc[%d] = %.10f\n", next_output, result);
         mfccs[next_output++] = result;
      }
   }

   fftw_destroy_plan(plan_forward);
   fftw_free(data);
   fftw_free(fft_result);
   free_mel_bank(mel_bank);
   return 1;
}
