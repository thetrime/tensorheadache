#include <tensorflow/c/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static void free_buffer(void* data, size_t length)
{
   free(data);
}

int main()
{
   printf("Hello from TensorFlow C library version %s\n", TF_Version());

   // I have no idea what I'm doing here. But obviously, we are going to need a graph
   TF_Graph *graph = TF_NewGraph();
   TF_Status *status = TF_NewStatus();
   // Ok good. Now we need to load our graph. First step is to load the data into a TF_Buffer
   FILE *fd;
   TF_Buffer* buffer = TF_NewBuffer();

   // Open the file
   assert((fd = fopen("model.pb", "rb")) != NULL);

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
   TF_GraphImportGraphDef(graph, buffer, graph_opts, status);
   assert(TF_GetCode(status) == TF_OK);
   TF_DeleteImportGraphDefOptions(graph_opts);
   TF_DeleteBuffer(buffer);

   // Right, next setup the input and output tensors
   int64_t inputShape[] = {1, 756};
   TF_Tensor * input_tensor = TF_AllocateTensor(TF_FLOAT, inputShape, 2, sizeof(float) * 756);
   TF_Tensor * output_tensor;

   TF_Output input;
   TF_Output output;

   input.oper = TF_GraphOperationByName(graph, "dense_1_input"); // Set up the input operation
   input.index = 0;
   output.oper = TF_GraphOperationByName(graph, "activation_6/Sigmoid"); // Set up the output operation
   output.index = 0;
   // Now lets make a session
   TF_SessionOptions *opts = TF_NewSessionOptions();
   TF_Session *session = TF_NewSession(graph, opts, status);
   assert(TF_GetCode(status) == TF_OK);

   // Debugging
   /*
   size_t pos = 0;
   TF_Operation* oper;
   while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL)
      printf("Operation %s\n", TF_OperationName(oper));
   */


   // And run it the model?
   TF_SessionRun(session,
                 /* Run Options */          NULL,
                 /* Input specification */  &input,  &input_tensor,   1,
                 /* Output specification */ &output, &output_tensor, 1,
                 /* Target operations */    NULL, 0,
                 /* Run Metadata */         NULL,
                 status);
   assert(TF_GetCode(status) == TF_OK);
   printf("Success?\n");

   // Clean everything up
   TF_DeleteSession(session, status);
   TF_DeleteStatus(status);
   TF_DeleteSessionOptions(opts);
   TF_DeleteGraph(graph);




   return 0;
}
