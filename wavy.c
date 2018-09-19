#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>


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


FILE* fd = NULL;

int read_audio_samples(int16_t* samples, int buffer_size)
{
   if (fd == NULL)
   {
      assert(fd = fopen("testlong-00.wav", "rb"));
      wav_header_t header;
      assert(fread(&header, sizeof(wav_header_t), 1, fd) == 1);
      assert(header.sample_rate == 16000);
      assert(header.fmt_chunk_size == 16);
   }
   int chunk = 500;
   int count = fread(samples, sizeof(int16_t), chunk, fd);
   assert(count > 0);
   return count;
}
