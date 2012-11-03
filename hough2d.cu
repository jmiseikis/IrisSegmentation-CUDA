#include "hough2d.h"
#include "main.h"
#include <sm_11_atomic_functions.h>

#define BLOCK_SIZE 512
#define BLOCK_SIZE_HOUGH 360
#define STEP_SIZE 5
#define NUMBER_OF_STEPS 360/STEP_SIZE

// Circ mask kernel storage
__constant__ int maskKernelX[NUMBER_OF_STEPS];
__constant__ int maskKernelY[NUMBER_OF_STEPS];

// Function to set precalculated relative coordinates for circle boundary coordinates
extern "C" void setMaskKernel(int *maskX, int *maskY)
{
	cudaMemcpyToSymbol(maskKernelX, maskX, NUMBER_OF_STEPS*sizeof(int));
	cudaMemcpyToSymbol(maskKernelY, maskY, NUMBER_OF_STEPS*sizeof(int));
}

// Kernel to set all pixel values to specified value
__global__ void setAllValuesKernel(int* houghSpace, int height, int width, float value)
{
	int const index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if (index < height*width) {
		houghSpace[index] = value;
	}
	__syncthreads();
}

extern "C" void setAllValuesToCUDA(int* houghSpace, int height, int width, float value)
{
	//cout << "Setting all values to " << value << "..." << endl;
	dim3 dimGrid = (ceil((float)width*height/(float)BLOCK_SIZE));
	dim3 dimBlock = (BLOCK_SIZE);
	setAllValuesKernel<<<dimGrid, dimBlock>>>(houghSpace, height, width, value);
	cudaThreadSynchronize();
}

// Kernel to perform circular Hough transform
__global__ void houghTransformKernel(int* cudaHough, float* img, int height, int width, int radius)
{
	if (threadIdx.x < BLOCK_SIZE_HOUGH) {
		// Arrays to hold coordinates for circle pixels
		__shared__ float circVals[NUMBER_OF_STEPS];

		// There are 10 hough pixels calculated in each block
		int whichPixel = (int)threadIdx.x / NUMBER_OF_STEPS;
		// Calculate position for pixel in hough space
		int cpixIDy = (int)( ((float)(blockIdx.x*STEP_SIZE+whichPixel)) / (float)(width-(radius*2))) + radius;
		int cpixIDx = (blockIdx.x*10+whichPixel) % (width-(radius*2)) + radius;
		
		// Load image pixel from circle edge
		int xVal = cpixIDx + maskKernelX[threadIdx.x % NUMBER_OF_STEPS];
		int yVal = cpixIDy + maskKernelY[threadIdx.x % NUMBER_OF_STEPS];
		// Get the pixel value from the image
		float pixVal = img[yVal*width + xVal];
		//float pixVal = img[cpixIDy*width + cpixIDx]; // ## TO DELETE, INCORRECT

		//int houghVal = cudaHough[cpixIDy*width + cpixIDx];
		__syncthreads();

		if (pixVal > 0) {
			atomicAdd(cudaHough + cpixIDy*width + cpixIDx, 1);

		}
		__syncthreads();
	}

}

// Calls the Hough transform kernel
extern "C" void performHoughTransformCUDA(int* cudaHough, float* img, int height, int width, int radius)
{
	// Define grid and block dimensions
	dim3 dimGrid = ( ceil((float)(width-(2*radius)) * (height-(2*radius)) / (float)STEP_SIZE) );
	dim3 dimBlock = (BLOCK_SIZE_HOUGH);

	// Perform Hough transform and sync threads to get the final result
	houghTransformKernel<<<dimGrid, dimBlock>>>(cudaHough, img, height, width, radius);
	cudaThreadSynchronize();
} 

// Analyse the defined image area for circles using Hough Transform
extern "C" void Hough2D_CUDA(float* img, int width, int height, int radMin, int radMax, int* posX, int* posY, int* maxVal, int* resRad)
{
	int* houghSpace;
	houghSpace = (int*)malloc(width*height*sizeof(int));

	// Arrays for results
	int *posxArray, *posyArray, *maxValArray, *radArray;
	// Allocate correct memory for arrays
	posxArray = (int*)malloc((radMax-radMin)*sizeof(int));
	posyArray = (int*)malloc((radMax-radMin)*sizeof(int));
	maxValArray = (int*)malloc((radMax-radMin)*sizeof(int));
	radArray = (int*)malloc((radMax-radMin)*sizeof(int));

	// Allocate memory for CUDA images and matrices
	float *cudaImg;
	int *cudaHough;

	cudaMalloc((void **)&cudaImg, width*height*sizeof(float));
	cudaMalloc((void **)&cudaHough, width*height*sizeof(int));
	// Copy image from host to device
	cudaMemcpy(cudaImg, img, width*height*sizeof(float), cudaMemcpyHostToDevice);

	int ctrArr = 0, radius;
	for (int i=radMin; i < radMax; i++) {
		// Set all elements to zero
		setAllValuesToCUDA(cudaHough, height, width, 0);

		// Precalculate relX and relY
		radius = i;
		int ctr = 0;
		int* relX, *relY;
		relX = (int*)malloc(NUMBER_OF_STEPS*sizeof(int));
		relY = (int*)malloc(NUMBER_OF_STEPS*sizeof(int));
		for (int theta=0; theta < 360; theta+=STEP_SIZE) {
			// Calculate x and y coordinates
			float angle = (theta*PI) / 180;
			relX[ctr] = (int)(-radius*cos(angle));
			relY[ctr] = (int)(-radius*sin(angle));
			ctr++;
		}

		// Set mask coordinates for circle
		setMaskKernel(relX, relY);

		// Free memory
		free(relX);
		free(relY);

		//performHoughTransformCUDA(cudaHough, cudaImg, height, width, radius, relX, relY, angleNum);
		performHoughTransformCUDA(cudaHough, cudaImg, height, width, radius);

		// Copy matrix from device to host
		cudaMemcpy(houghSpace, cudaHough, width*height*sizeof(float), cudaMemcpyDeviceToHost);

		// Find max value in the houghSpace
		*maxVal = 0;
		int index;
		int tempPosX, tempPosY, tempMaxVal = 0;

		for (int y=0; y < height; y++) {
			for (int x=0; x < width; x++) {
				//index = radius*width*height + y*width + x;
				index = y*width + x;
				if (tempMaxVal < houghSpace[index]) {
					tempMaxVal = houghSpace[index];
					tempPosX = x;
					tempPosY = y;
				}
			}
		}

		// Write results to arrays
		posxArray[ctrArr] = tempPosX;
		posyArray[ctrArr] = tempPosY;
		maxValArray[ctrArr] = tempMaxVal;
		radArray[ctrArr] = i;

		cout << "Current (radius: " << i << ") MaxVal: " << maxValArray[ctrArr] << " ctr: " << ctrArr << endl;

		ctrArr++;
	} // end for

	// Find the maximum value from arrays
	*maxVal = 0;
	for (int j=0; j < (radMax-radMin); j++) {
		cout << "MaxValArray: " << maxValArray[j] << " " << posxArray[j] << " " << posyArray[j] << " " << radArray[j] << endl;
		if (*maxVal < maxValArray[j]) {
			*maxVal = maxValArray[j];
			*posX = posxArray[j];
			*posY = posyArray[j];
			*resRad = radArray[j];
		}
	}

	// Free cuda memory
	cudaFree(cudaImg);
	cudaFree(cudaHough);

	// Free array memory
	free(posxArray);
	free(posyArray);
	free(maxValArray);
	free(radArray);
}


// ##################
// #### IMADJUST ####
// ##################


__global__ void AdjustImageIntensityKernel(float *imgOut, float *imgIn, int width, int height, float lowin, float lowout, float scale)
{
    __shared__ float bufData[BLOCK_SIZE];

	// Get the index of pixel
	const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	// Load data to shared variable
	bufData[threadIdx.x] = imgIn[index];

	// Check that it's not out of bounds
	if (index < (height*width)) {
		
		// Find the according multiplier
		float tempLevel = ( bufData[threadIdx.x] - lowin)*scale + lowout;
		
		// Check that it's within required range
		if (tempLevel < 0) {
			bufData[threadIdx.x] = 0;
		}
		else if (tempLevel > 1) {
			bufData[threadIdx.x] = 1;
		}
		else {
			bufData[threadIdx.x] = tempLevel;
		}

		// Write data back
		imgOut[index] = bufData[threadIdx.x];
	}
	
	// Synchronise threads to have the whole image fully processed for output
	__syncthreads();
}

// Resize the image
__global__ void ImageScalingKernel(float *imgOut, float *imgIn, int width, int height)
{
	__shared__ float inData[BLOCK_SIZE];
	// Get the index of pixel
	const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	// Load data to shared variable
	inData[threadIdx.x] = imgIn[index];

	if ( index < (width*height) ) {
		imgOut[index] = inData[threadIdx.x] / (float)255; 
	}

	__syncthreads();
}

// the CUDA sample implementaiton can be used
void ImageHistogramCUDA(float *pSrc, int width, int height, int * imghist)
{
	const int GrayThres = 256;
	for (int i=0; i< GrayThres; i++) imghist[i] = 0; 
	for (int i=0; i< width*height; i++) {
		int level = (int) (pSrc[i]*255); 
		imghist[level]+=1; 
	}

}

// Strech limit
void ImageStretchLimitCUDA(float *pSrc, int width, int height,
					   float tol_low, float tol_high, float *low, float *high )
{
	const int GrayThres = 256;
	int imghist[256]; 

	double cdf[GrayThres], sum;
	int i;
	bool bLowFound=false, bHighFound=false;

	//histogram
	ImageHistogramCUDA(pSrc,width,height,imghist);	
	
	// the below segment can be implemented on CPU only; 
	//*************************************************
	// cdf
	cdf[0]=imghist[0];
	for (i=1;i<GrayThres;i++)
		cdf[i] = cdf[i-1] + imghist[i];		
	sum = cdf[GrayThres-1];
	for (i=0;i<GrayThres;i++)
		cdf[i] /= sum;

	// find low and high
	for (i=0;i<GrayThres;i++)
	{
		if (cdf[i]>=tol_low && (bLowFound == false))
		{
			*low = (float)(i);
			bLowFound = true;
		}

		if (cdf[i]>=tol_high && (bHighFound == false))
		{
			*high = (float)(i);
			bHighFound = true;
		}
	}
	// convert to range [0 1]
	*low /= (GrayThres-1);
	*high /= (GrayThres-1);

}

// Adjusts image intensity depending on the current gray levels of the image (histogram stretching)
extern "C" void imadjustCUDA(unsigned char *inImg, unsigned char *outImg, int width, int height, float lowPerc, float highPerc)
{
	const int grayLevels = 256;
    float lowin, highin;
	float *tempBuffer = new float[width*height];
	float *imgInput, *imgBuffer, *imgOutput;
	clock_t init, final_gpu;

	// Convert input image to float
	for (int i=0; i < (width*height); i++) {
		tempBuffer[i] = (float) inImg[i];
	}

	// ### ALLOCATE CUDA ARRAYS ###
	cudaMalloc((void **)&imgInput, width * height * sizeof(float));
	cudaMalloc((void **)&imgBuffer, width * height * sizeof(float));
	cudaMalloc((void **)&imgOutput, width * height * sizeof(float));

	// ### COPY TO CUDA MEMORY ###
	cudaMemcpy(imgInput, tempBuffer,  width * height * sizeof(float), cudaMemcpyHostToDevice);

	// Get number of blocks
	int gridSize = ceil( (float)(height*width) / (float)BLOCK_SIZE );

	// Assign sizes
	dim3 blocks( gridSize );
	dim3 threads( BLOCK_SIZE );

	// Image scaling Kernel
	ImageScalingKernel<<<blocks, threads>>>(imgBuffer, imgInput, width, height);
	cudaThreadSynchronize();

	// Copy image buffer back to host memory (for ImageStretchLimit function)
	cudaMemcpy(tempBuffer, imgBuffer, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	// find out the 1% pixel intensity value and set it to "low"
	// find out the 99% pixel intensiy valeu and set it to "high"
	//ImageStretchLimitCUDA(tempBuffer, width, height, 0.01f,0.99f,&lowin,&highin);
	ImageStretchLimitCUDA(tempBuffer, width, height, lowPerc,highPerc,&lowin,&highin);

	// Adjust image intensity
	float lowout = 0, highout = 1;
	float range = highin-lowin; 
	float rangeout = highout-lowout;  
	float scale = rangeout/range;

	printf("Adjusting image intensities on GPU (CUDA)...\n");
	// Start timer
	init = clock();
	// Call the adjust image intensity kernel
    AdjustImageIntensityKernel<<<blocks, threads>>>(imgOutput, imgBuffer, width, height, lowin, lowout, scale);
	cudaThreadSynchronize();

	// Copy the result back
	cudaMemcpy(tempBuffer, imgOutput, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	// Take time
	final_gpu=clock()-init;
	printf("Time taken for imadjust on GPU (CUDA): %f sec\n", (double)final_gpu / ((double)CLOCKS_PER_SEC));

	// convert it back to unsigned char
	for (int i =0; i< width*height; i++) {
		outImg[i] = (unsigned char) (tempBuffer[i]*255);  
	}

	// Free memory
	cudaFree(imgInput);
	cudaFree(imgBuffer);
	cudaFree(imgOutput);
}


// #####################
// #### ADJUSTGAMMA ####
// #####################


__global__ void AdjustGammaKernel(float *imgOut, float *imgIn, int width, int height, float gamma, float minVal, float maxVal)
{
    __shared__ float bufData[BLOCK_SIZE];

	// Get the index of pixel
	const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	// Load data to shared variable
	bufData[threadIdx.x] = imgIn[index];

	// Check that it's not out of bounds
	if (index < (height*width)) {
		
		// Find the according multiplier
		float tempLevel = ( bufData[threadIdx.x] - minVal) / maxVal;
		
		tempLevel = powf(tempLevel, (double)1/gamma);
		
		// Check that it's within required range
		if (tempLevel < 0) {
			bufData[threadIdx.x] = 0;
		}
		else if (tempLevel > 1) {
			bufData[threadIdx.x] = 1;
		}
		else {
			bufData[threadIdx.x] = tempLevel;
		}

		// Write data back
		imgOut[index] = bufData[threadIdx.x];
	}
	
	// Synchronise threads to have the whole image fully processed for output
	__syncthreads();
}

extern "C" void adjustGammaCUDA(unsigned char *inImg, unsigned char *outImg, int width, int height, float gamma)
{
	const int grayLevels = 256;
    float lowin, highin;
	float *tempBuffer = new float[width*height];
	float *imgInput, *imgOutput;
	clock_t init, final_gpu;
	
	float minVal = 1000, maxVal = 0;

	// Convert input image to float
	for (int i=0; i < (width*height); i++) {
		tempBuffer[i] = (float) inImg[i];
		
		// Calculate min and max values in the image ## CAN BE ADDED TO DO ON CUDA LATER ON ##
		if (minVal > tempBuffer[i]) {
			minVal = tempBuffer[i];
		}
		if (maxVal < tempBuffer[i]) {
			maxVal = tempBuffer[i];
		}
	}
	

	// ### ALLOCATE CUDA ARRAYS ###
	cudaMalloc((void **)&imgInput, width * height * sizeof(float));
	cudaMalloc((void **)&imgOutput, width * height * sizeof(float));

	// ### COPY TO CUDA MEMORY ###
	cudaMemcpy(imgInput, tempBuffer,  width * height * sizeof(float), cudaMemcpyHostToDevice);

	// Get number of blocks
	int gridSize = ceil( (float)(height*width) / (float)BLOCK_SIZE );

	// Assign sizes
	dim3 blocks( gridSize );
	dim3 threads( BLOCK_SIZE );

	printf("Adjusting gamma on GPU (CUDA)...\n");
	// Start timer
	init = clock();
	// Image scaling Kernel
	AdjustGammaKernel<<<blocks, threads>>>(imgOutput, imgInput, width, height, gamma, minVal, maxVal);
	cudaThreadSynchronize();

	// Copy the result back
	cudaMemcpy(tempBuffer, imgOutput, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	// Take time
	final_gpu=clock()-init;
	printf("Time taken for gamma adjustment on GPU (CUDA): %f sec\n", (double)final_gpu / ((double)CLOCKS_PER_SEC));

	// convert it back to unsigned char
	for (int i =0; i< width*height; i++) {
		outImg[i] = (unsigned char) (tempBuffer[i]*255);  
	}

	// Free memory
	cudaFree(imgInput);
	cudaFree(imgOutput);
}
