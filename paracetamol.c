#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "ibuprofen.h"
#include "mfcc.h"

int main()
{
   model_t* model = load_model("model.pb");
   assert(model != NULL);

   FILE *fd = fopen("testing/negative-00.wav", "rb");
   assert(fd != NULL);

   char header[44];
   assert(fread(&header, 44, 1, fd) == 1);

   double buffer[512];
   stream_context_t* context =  make_stream_context(16000);

   for (int chunk = 0 ; chunk < 65; chunk++)
   {
      for (int i = 0; i < 512; i++)
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
         buffer[i] = (double)sample / 32768.0;
      }
      add_chunk_to_context(context, buffer);
   }
   fclose(fd);
   mfccs_from_context(context, model_data(model));
   printf("Result: %f\n", run_model(model));
   free_model(model);
   return 0;
}
