#include <tensorflow/c/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "ibuprofen.h"

static void free_buffer(void* data, size_t length)
{
   free(data);
}

int mfccs_from_file(const char* filename, float*);

int main()
{
   model_t* model = load_model("model.pb");
   assert(model != NULL);

   // Fill in the input tensor
   mfccs_from_file("testing/negative-00.wav", model_data(model));

   printf("Result: %f\n", run_model(model));
   printf("Result: %f\n", run_model(model));
   printf("Result: %f\n", run_model(model));

   free_model(model);
   return 0;
}
