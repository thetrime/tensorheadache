#include <CoreFoundation/CoreFoundation.h>
#include <AudioToolbox/AudioToolbox.h>
#include <stdio.h>
#include "mfcc.h"
#include "ibuprofen.h"

stream_context_t* stream;
model_t* model;
int run_in = 20;


static void audio_callback(void *inUserData,
                           AudioQueueRef inQueue,
                           AudioQueueBufferRef inBuffer,
                           const AudioTimeStamp *inStartTime,
                           unsigned int inNumPackets,
                           const AudioStreamPacketDescription *inPacketDesc)
{
   printf("Hello from input callback. I have %d packets for you!\n", inNumPackets);
   assert(inNumPackets > 512);
   int chunks = inNumPackets / 512;
   float* data = (float*)inBuffer->mAudioData;
   /*
   for (int j = 0; j < inNumPackets; j++)
   {
      int16_t sample = (int16_t)(32767 * data[j]);
      fwrite(&sample, sizeof(int16_t), 1, fd);
   }
   fflush(fd);
   */
   printf("Chunks in block: %d\n", chunks);
   for (int i = 0; i < chunks; i++)
   {
      add_chunk_to_context(stream, &data[512 * i]);
      if (run_in > 0)
         run_in--;
      else
      {
         //mfccs_from_context(stream, model_data(model));
         //mfccs_from_file("testing/negative-00.wav", model_data(model));
         printf("Result: %f\n", run_model(model));
      }
   }
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
   recordFormat.mBytesPerFrame    = sizeof(Float32);
   recordFormat.mBytesPerPacket   = sizeof(Float32);
   recordFormat.mBitsPerChannel   = sizeof(Float32) * 8;


/*
   recordFormat.mFormatFlags = kLinearPCMFormatFlagIsBigEndian | kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
   recordFormat.mSampleRate = 16000;
   recordFormat.mFormatID = kAudioFormatLinearPCM;
   recordFormat.mBytesPerPacket = 4;
   recordFormat.mFramesPerPacket = 1;
   recordFormat.mBytesPerFrame = 4;
   recordFormat.mChannelsPerFrame = 1;
   recordFormat.mBitsPerChannel = 16;
*/
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

   assert(AudioQueueStart(queue, NULL) == noErr);
   getchar();
   return 1;
}


int main()
{
   return clusterize();
}
