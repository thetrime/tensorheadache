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

   // Fill in the input tensor
   mfccs_from_file("testing/negative-07.wav", model_data(model));

   printf("Result: %f\n", run_model(model));
   printf("Result: %f\n", run_model(model));
   printf("Result: %f\n", run_model(model));

   free_model(model);
   return 0;
}
