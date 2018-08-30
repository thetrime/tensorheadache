#include <CoreFoundation/CoreFoundation.h>
#include <AudioToolbox/AudioToolbox.h>
#include <stdio.h>

static void audio_callback(void *inUserData,
                           AudioQueueRef inQueue,
                           AudioQueueBufferRef inBuffer,
                           const AudioTimeStamp *inStartTime,
                           unsigned int inNumPackets,
                           const AudioStreamPacketDescription *inPacketDesc)
{
   printf("Hello from input callback. I have %d packets for you!\n", inNumPackets);
   assert(AudioQueueEnqueueBuffer(inQueue, inBuffer, 0, NULL) == noErr);
}


int clusterize()
{
    AudioStreamBasicDescription recordFormat;
    memset(&recordFormat, 0, sizeof(recordFormat));
    recordFormat.mFormatFlags = kLinearPCMFormatFlagIsBigEndian | kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    recordFormat.mSampleRate = 16000;
    recordFormat.mFormatID = kAudioFormatLinearPCM;
    recordFormat.mBytesPerPacket = 4;
    recordFormat.mFramesPerPacket = 1;
    recordFormat.mBytesPerFrame = 4;
    recordFormat.mChannelsPerFrame = 1;
    recordFormat.mBitsPerChannel = 16;
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
