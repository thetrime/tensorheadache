#ifndef IBUPROFEN_H
#define IBUPROFEN_H

#include <tensorflow/c/c_api.h>

typedef struct
{
   TF_Graph* graph;
   TF_Tensor* input_tensor;
   TF_Tensor* output_tensor;
   TF_Output input;
   TF_Output output;
   TF_Session *session;
} model_t;

model_t* load_model(const char* filename);
float* model_data(model_t* model);
float run_model(model_t* model);
void free_model(model_t* model);
#endif
