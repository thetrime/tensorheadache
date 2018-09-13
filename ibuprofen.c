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

model_t* load_model(const char* filename)
{
   printf("Hello from TensorFlow C library version %s\n", TF_Version());

   // I have no idea what I'm doing here. But obviously, we are going to need a graph
   model_t* model = malloc(sizeof(model_t));
   model->graph = TF_NewGraph();

   TF_Status *status = TF_NewStatus();
   // Ok good. Now we need to load our graph. First step is to load the data into a TF_Buffer
   FILE *fd;
   TF_Buffer* buffer = TF_NewBuffer();

   // Open the file
   assert((fd = fopen(filename, "rb")) != NULL);

   // For now just read the whole model into a chunk of memory
   fseek(fd, 0, SEEK_END);
   size_t s = ftell(fd);
   char *data = malloc(s);

   rewind(fd);
   fread(data, 1, s, fd);
   fclose(fd);

   buffer->data = data;
   buffer->length = s;
   buffer->data_deallocator = free_buffer;

   // OK, now we can import it
   TF_ImportGraphDefOptions *graph_opts = TF_NewImportGraphDefOptions();
   TF_GraphImportGraphDef(model->graph, buffer, graph_opts, status);
   assert(TF_GetCode(status) == TF_OK);
   TF_DeleteImportGraphDefOptions(graph_opts);
   TF_DeleteBuffer(buffer);

   // Right, next setup the input and output tensors
   int64_t inputShape[] = {1, 29, 13};
   model->input_tensor = TF_AllocateTensor(TF_FLOAT, inputShape, 3, sizeof(float) * 377);

   model->input.oper = TF_GraphOperationByName(model->graph, "net_input"); // Set up the input operation
   model->input.index = 0;
   model->output.oper = TF_GraphOperationByName(model->graph, "dense_1/Sigmoid"); // Set up the output operation
   model->output.index = 0;
   assert(model->output.oper != NULL);
   assert(model->input.oper != NULL);


   TF_SessionOptions *opts = TF_NewSessionOptions();
   model->session = TF_NewSession(model->graph, opts, status);
   assert(TF_GetCode(status) == TF_OK);
   TF_DeleteSessionOptions(opts);
   TF_DeleteStatus(status);
   return model;
}

float* model_data(model_t* model)
{
   return TF_TensorData(model->input_tensor);

}
float run_model(model_t* model)
{
   TF_Status *status = TF_NewStatus();
   TF_SessionRun(model->session,
                 /* Run Options */          NULL,
                 /* Input specification */  &model->input,  &model->input_tensor,   1,
                 /* Output specification */ &model->output, &model->output_tensor, 1,
                 /* Target operations */    NULL, 0,
                 /* Run Metadata */         NULL,
                 status);
   assert(TF_GetCode(status) == TF_OK);
   TF_DeleteStatus(status);
   float* outdata = TF_TensorData(model->output_tensor);
   return *outdata;
}

void free_model(model_t* model)
{
   TF_Status *status = TF_NewStatus();
   TF_DeleteSession(model->session, status);
   TF_DeleteStatus(status);
   TF_DeleteGraph(model->graph);
   free(model);
}

