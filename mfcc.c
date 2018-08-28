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
#define MEL_FILTER_COUNT 26


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

// Set NON_SYMMETRIC to 1 if you want a non-symmetric window for some reason
#define NON_SYMMETRIC 0

void make_window(int size, double* window)
{
   //double A = 0.54; // Hamming
   double A = 0.5; // Hann
   // To make a symmetric window we effectively make the window 1 unit bigger (so it is odd in length) and discard the last element
   for (int i = 0; i < WINDOW_LENGTH; i++)
      window[i] = A - ((1-A) * cos(2 * M_PI * (i / ((WINDOW_LENGTH - NON_SYMMETRIC) * 1.0))));
}
/* This is what http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn1 gives
   but it turns out lots of people disagree on this process
double hz_to_mels(double hz)
{
   return 1125 * log(1 + (hz/700));
}

double mels_to_hz(double mels)
{
   return (700 * (exp(mels / 1125.0) - 1));
}
*/

double hz_to_mels(double hz)
{
   // This is the Slaney algorithm. Linear for frequencies under 1kHz and logarithmic above
   if (hz < 1000.0)
      return hz / (200.0 / 3.0);
   else
      return 15 + log(hz / 1000.0) / (log(6.4)/27.0);
}

double mels_to_hz(double mels)
{
   if (mels < 15)
      return mels * 200.0 / 3.0;
   else
      return 1000 * exp((log(6.4)/27.0) * (mels - 15));
}

void free_mel_bank(double** mels)
{
   for (int i = 0; i < MEL_FILTER_COUNT; i++)
      free(mels[i]);
   free(mels);
}

double** make_mel_bank(double sample_rate)
{
   double max_frequency_hz = sample_rate / 2.0;
   double min_frequency_hz = 0;
   double max_frequency_mels = hz_to_mels(max_frequency_hz);
   double min_frequency_mels = hz_to_mels(min_frequency_hz);
   double centroids[MEL_FILTER_COUNT+2];

   double **mels = malloc(sizeof(double*) * MEL_FILTER_COUNT);
   for (int i = 0; i < MEL_FILTER_COUNT; i++)
      mels[i] = malloc(sizeof(double) * 1025);


   // Conceptually we now want MEL_FILTER_COUNT+2 equally spaced points
   // The first one will be min_frequency_mels, and the last one max_frequency_mels
   // However, we want these in Hz, not Mels

   double spacing = max_frequency_mels / (double)(MEL_FILTER_COUNT+1);
   for (int i = 0; i < MEL_FILTER_COUNT+2; i++)
      centroids[i] = mels_to_hz(i * spacing);

   // Now fill in the array. We fill in the k possible values for each filter m
   // k is an FFT bin though, and the centroids are Hz. This is important, as librosa
   // produces a slightly different result using this technique than the bank computed
   // via the practicalcryptography method. The latter claims that nothing is degraded,
   // but for the purposes of checking my implementation is correct, it helps if the numbers
   // match!

   for (int m = 0; m < MEL_FILTER_COUNT; m++)
   {
      for (int k = 0; k < WINDOW_LENGTH; k++)
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

         // Furthermore we want to normalize so that the area of the filter is constant for all filters. This means dividing by the width of the band

         double width = (centroids[m+2] - centroids[m])/2.0;
         double frequency = (sample_rate * k) / WINDOW_LENGTH;

         if (frequency < centroids[m])
            mels[m][k] = 0;
         else if (frequency <= centroids[m+1])
            mels[m][k] = ((frequency - centroids[m]) / (centroids[m+1] - centroids[m])) / width;
         else if (frequency <= centroids[m+2])
            mels[m][k] = ((centroids[m+2] - frequency) / (centroids[m+2] - centroids[m+1])) / width;
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

   double** mel_bank = make_mel_bank(header.sample_rate);

   int sample_duration = 2;
   int sample_count = sample_duration * header.sample_rate;
   double *samples = malloc(sizeof(double) * sample_count);

   double mfcc[MEL_FILTER_COUNT];
   double melspectrogram[MEL_FILTER_COUNT];

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
   fclose(fd);

   // Ok, now we have the samples in *samples and we must do the stft
   // First set up the buffers and build the FFT plan

   fftw_complex    *data, *fft_result;
   fftw_plan       plan_forward;

   data        = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   fft_result  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WINDOW_LENGTH);
   plan_forward = fftw_plan_dft_1d(WINDOW_LENGTH, data, fft_result, FFTW_FORWARD, FFTW_ESTIMATE );

   // Now we read in 2048 samples at a time, starting at 0, then 512, then 1024, then 1536, and so on, until we get to the end

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

      // At this point we have the right answer that stft would give if we have center=False
      // If we want the answer to exactly match librosa then we need to do reflection padding too


      // Next, compute the square of the absolute value of each element. Fortunately this is just data[i][0]**2 + data[i][1]**2
      // Since we do not need the raw data again, we can just put this directly back into data[i][0]
      // Also note that we only care about the first half of the output because of symmetry around the nyquist frequency
      // This means we can go from 0 to 1024 (inclusive) and skip the rest

      for (int i = 0; i <= WINDOW_LENGTH/2; i++)
         fft_result[i][0] = fabs(((fft_result[i][0] * fft_result[i][0]) + (fft_result[i][1] * fft_result[i][1])));

      // Then we need the dot product of this with our Mel filter. This is the 'melspectrogram'.
      for (int i = 0; i < MEL_FILTER_COUNT; i++)
      {
         double m = 0;
         for (int j = 0; j < WINDOW_LENGTH/2; j++)
            m = m + mel_bank[i][j] * fft_result[j][0];
         melspectrogram[i] = power_to_db(m);
      }

      // And finally we must do the DCT on this to get the result
      //           N-1
      // y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
      //           n=0

      for (int k = 0; k < MEL_FILTER_COUNT; k++)
      {
         double sum = 0;
         for (int n = 0; n < MEL_FILTER_COUNT; n++)
            sum += melspectrogram[n] * cos(M_PI * k * (2*n+1) / (2 * MEL_FILTER_COUNT));
         if (k == 0)
            mfcc[k] = 2 * sum * sqrt(1.0/(4.0*MEL_FILTER_COUNT));
         else
            mfcc[k] = 2 * sum * sqrt(1.0/(2.0*MEL_FILTER_COUNT));
         printf("mfcc[%d] = %.10e\n", k, mfcc[k]);
      }
      assert(0);
      block_offset += 512;
  
   }
 
   fftw_destroy_plan(plan_forward);
   fftw_free(data);
   fftw_free(fft_result);
   free_mel_bank(mel_bank);
   return 1;
}
