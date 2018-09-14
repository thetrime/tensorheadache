
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
         //printf("Loading data from %d to %d\n", bufptr, bufptr);
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
