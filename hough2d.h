#include <iostream>
#include <math.h>

#define GRAYLEVELS 256

// CUDA functions
extern "C" void Hough2D_CUDA(float* img, int width, int height, int radMin, int radMax, int* posX, int* posY, int* maxVal, int* resRad);
extern "C" void imadjustCUDA(unsigned char *inImg, unsigned char *outImg, int width, int height, float lowPerc, float highPerc);
extern "C" void adjustGammaCUDA(unsigned char *inImg, unsigned char *outImg, int width, int height, float gamma);