#include <CoreFoundation/CoreFoundation.h>
#include <AudioToolbox/AudioToolbox.h>
#include <stdio.h>
#include "holmes.h"
#include "ibuprofen.h"

context_t* context;

static void audio_callback(void *inUserData,
                           AudioQueueRef inQueue,
                           AudioQueueBufferRef inBuffer,
                           const AudioTimeStamp *inStartTime,
                           unsigned int inNumPackets,
                           const AudioStreamPacketDescription *inPacketDesc)
{
   double score = process_block_double(context, (double*)inBuffer->mAudioData, inNumPackets);
   for (int i = 1 ; i < 100; i++)
      printf("%s", i <= score*100?"*":"-");
   printf("\n");
   assert(AudioQueueEnqueueBuffer(inQueue, inBuffer, 0, NULL) == noErr);

}


int clusterize()
{
   int sample_rate = 16000;
   context = alloc_context("qqq.pb", sample_rate);
   AudioStreamBasicDescription recordFormat;
   memset(&recordFormat, 0, sizeof(recordFormat));

   recordFormat.mFormatID = kAudioFormatLinearPCM;
   recordFormat.mSampleRate = sample_rate;
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
   assert(AudioQueueStart(queue, NULL) == noErr);
   getchar();
   printf("Halting\n");
   return 1;
}


int main()
{
   return clusterize();
}
