#include "holmes.h"
#include "ibuprofen.h"
#include "sizes.h"
#include <stdio.h>
#include <string.h>

model_t* model;
int run_in = 29;
double buffer[1600];
int bufptr = 0;
int m = 0;
context_t* context;

#include "block.c"

int main()
{
   int sample_rate = 16000;
   model = load_model("qqq.pb");
   context = alloc_context(sample_rate);
   float* f = model_data(model);
   memset(f, 0, sizeof(float) * 13 * 29);
   memset(buffer, 0, sizeof(double)*1600);

   FILE* fd = fopen("/tmp/out.raw", "rb");
   for (int i = 0; sizes[i] != 0; i++)
   {
      double block[sizes[i]];
      int16_t raw[sizes[i]];
      fread(raw, sizeof(int16_t), sizes[i], fd);
      for (int j = 0; j < sizes[i]; j++)
         block[j] = (double)raw[j] / 32767.0;
      process_block(block, sizes[i]);
   }
}
