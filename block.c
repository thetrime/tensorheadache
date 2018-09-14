
void process_block(double* data, int inNumPackets)
{
   int i = 0;
   while (i < inNumPackets)
   {
      // Compute how many bytes would bring the buffer up to a new chunk
      int thisChunkSize = 800 - (bufptr % 800);
      // Do not copy more packets than we have though
      if (thisChunkSize + i > inNumPackets)
         thisChunkSize = inNumPackets - i;
      // Copy the data in to the buffer
      memcpy(&buffer[bufptr], &data[i], thisChunkSize*sizeof(double));
      bufptr += thisChunkSize;
      bufptr = bufptr % 1600;
      if ((bufptr % 800) == 0)
      {
         // If bufptr is 0 then the block from 800 to 800 is ready. Otherwise the block from 0-1600 is ready
         // The thing that I missed the first time is that precise uses a progressive buffer. New mfcc vectors are added at the end
         // which means that effectively, everything gets shuffled back one index each time we add a new vector. It is not circular unless
         // we can move the 'start' point of the tensor data forward by 1 vector each time. There might be a nicer solution than this, but
         // for now, just move the array back one, and ALWAYS write to the last spot in the tensor data stucture
         memcpy(model_data(model), &model_data(model)[13], sizeof(float)*13*28);
         mfccs_from_circular_buffer(context, buffer, bufptr, 1600, model_data(model), 28*13);
      }
      i += thisChunkSize;
   }
   double score = 0;
   score = run_model(model);
   for (i = 1 ; i < 100; i++)
      printf("%s", i <= score*100?"*":"-");
   printf("\n");
}
