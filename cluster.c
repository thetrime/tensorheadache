#include <CoreFoundation/CoreFoundation.h>
#include <AudioToolbox/AudioToolbox.h>
#include <stdio.h>
#include "mfcc.h"
#include "ibuprofen.h"

stream_context_t* stream;
model_t* model;
int run_in = 64;


static void audio_callback(void *inUserData,
                           AudioQueueRef inQueue,
                           AudioQueueBufferRef inBuffer,
                           const AudioTimeStamp *inStartTime,
                           unsigned int inNumPackets,
                           const AudioStreamPacketDescription *inPacketDesc)
{
   assert(inNumPackets > 512);
   int chunks = inNumPackets / 512;
   double* data = (double*)inBuffer->mAudioData;
   double score = 0;
   double best_score = 0;
   for (int i = 0; i < chunks; i++)
   {
      add_chunk_to_context(stream, &data[512 * i]);
      if (run_in > 0)
      {
         run_in--;
         if (run_in == 0)
            printf("Ready!\n");
      }
      else
      {
         mfccs_from_context(stream, model_data(model));
         score = run_model(model);
         if (score > best_score)
            best_score = score;
         /*
         if (score > 0.9)
            fprintf(stderr, "%d", (int)(score * 100) - 90);
         else
            fprintf(stderr, ".");
         */
      }
   }
   if (run_in == 0)
      printf("%.3f\n", best_score);
   assert(AudioQueueEnqueueBuffer(inQueue, inBuffer, 0, NULL) == noErr);
}


int clusterize()
{
   model = load_model("model.pb");
   stream = make_stream_context(16000);
   AudioStreamBasicDescription recordFormat;
   memset(&recordFormat, 0, sizeof(recordFormat));

   recordFormat.mFormatID = kAudioFormatLinearPCM;
   recordFormat.mSampleRate = 16000;
   recordFormat.mFormatFlags = kAudioFormatFlagsNativeFloatPacked;
   recordFormat.mFramesPerPacket  = 1;
   recordFormat.mChannelsPerFrame = 1;
   recordFormat.mBytesPerFrame    = sizeof(Float64);
   recordFormat.mBytesPerPacket   = sizeof(Float64);
   recordFormat.mBitsPerChannel   = sizeof(Float64) * 8;

   unsigned int propSize = sizeof(recordFormat);
   assert(AudioFormatGetProperty(kAudioFormatProperty_FormatInfo,
                                 0,
                                 NULL,
                                 &propSize,
                                 &recordFormat) == noErr);

   AudioQueueRef queue = {0};
   OSStatus result = AudioQueueNewInput(&recordFormat,
                                        audio_callback,
                                        NULL,
                                        NULL,
                                        NULL,
                                        0,
                                        &queue);
   assert(result == noErr);

   AudioQueueBufferRef buffer;
        
   assert(AudioQueueAllocateBuffer(queue, 32768, &buffer) == noErr);
   assert(AudioQueueEnqueueBuffer(queue, buffer, 0, NULL) == noErr);
   printf("Please wait for a while as I fill the input buffer...\n");
   assert(AudioQueueStart(queue, NULL) == noErr);
   getchar();
   printf("Halting\n");
   return 1;
}


int main()
{
   return clusterize();
}
