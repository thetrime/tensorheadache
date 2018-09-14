#include <CoreFoundation/CoreFoundation.h>
#include <AudioToolbox/AudioToolbox.h>
#include <stdio.h>
#include "holmes.h"
#include "ibuprofen.h"

model_t* model;
int run_in = 29;

// I need to buffer 1600 samples. Each time we get data, fill up to 800 bytes of the buffer with the data, then every 800 bytes
// compute the MFCCs for the buffer. It should produce just a single vector, basically. Repeat with the next 800 bytes, wrapping at the end.
// Finally, run the simulation.

double buffer[1600];
int bufptr = 0;
int m = 0;
context_t* context;
FILE *fd1 = NULL;
FILE *fd2 = NULL;

#include "block.c"


static void audio_callback(void *inUserData,
                           AudioQueueRef inQueue,
                           AudioQueueBufferRef inBuffer,
                           const AudioTimeStamp *inStartTime,
                           unsigned int inNumPackets,
                           const AudioStreamPacketDescription *inPacketDesc)
{
   double* data = (double*)inBuffer->mAudioData;
   int16_t intdata[inNumPackets];
   for (int q = 0; q < inNumPackets; q++)
      intdata[q] = data[q] * 32767;
   fwrite(intdata, sizeof(int16_t), inNumPackets, fd1);
   fprintf(fd2, "%d, ", inNumPackets);
   fflush(fd1);
   fflush(fd2);
   process_block(data, inNumPackets);

   assert(AudioQueueEnqueueBuffer(inQueue, inBuffer, 0, NULL) == noErr);

}


int clusterize()
{
   int sample_rate = 16000;
   model = load_model("qqq.pb");
   context = alloc_context(sample_rate);
   float* f = model_data(model);
   memset(f, 0, sizeof(float) * 13 * 29);
   memset(buffer, 0, sizeof(double)*1600);
   AudioStreamBasicDescription recordFormat;
   memset(&recordFormat, 0, sizeof(recordFormat));
   fd1 = fopen("/tmp/out.raw", "wb");
   fd2 = fopen("/tmp/sizes.h", "wb");
   fprintf(fd2, "int sizes[] = {");
   fflush(fd2),

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
