#include "main.h"
#include "hough2d.h"
//#include "imadjustCuda.h"

using namespace cv;

// Define scale, best accuracy/speed ratio achieved using 0.6 for test images
#define SCALE 0.6

// Function prototypes CPU
int Hough2D_CPU(int* img, int width, int height, int radius, int* posX, int* posY, int* maxVal);
int findCircleHoughCPU(IplImage *edgeImg, int radMin, int radMax, int* resX, int *resY, int *resRad, int *resVal);

// Function prototypes CUDA
int findCircleHoughCUDA(IplImage *edgeImg, int radMin, int radMax, int* resX, int *resY, int *resRad, int *resVal);
int imadjustInput(IplImage *cvImg, float lowPerc, float highPerc);

int main()
{
	// For timer
	clock_t init, final;

	// Open the file.
	IplImage *img = cvLoadImage("./images/test_3.bmp",CV_LOAD_IMAGE_GRAYSCALE);
    if (!img) {
        cout << "Error: Couldn't open the image file." << endl;
        return 1;
    }

	//IplImage *imgResult = cvLoadImage("eye_edge2.png");
	IplImage *imgResult = cvLoadImage("./images/test_3.bmp");
    if (!img) {
        cout << "Error: Couldn't open the image file." << endl;
        return 1;
    }

	// Blur the image
	cout << "Gaussian blur on CPU..." << endl;
	// Start timer
	init=clock();
	cvSmooth( img, img, CV_GAUSSIAN, 3, 3 );
	// Stop timer
	final=clock()-init;
	cout << "Time taken to perform Gaussian blurring CPU:" << (double)final / ((double)CLOCKS_PER_SEC) << "s" << endl << endl;

	// Resize the image to SCALE % of the size
	IplImage *img_small = cvCreateImage( cvSize((int)(img->width*SCALE) , (int)(img->height*SCALE) ),img->depth, img->nChannels );
	cvResize(img, img_small);

	// Adjust the intensity of the image
	imadjustInput(img_small, 0.01f, 0.99f);

	// Create new image for storing edge image
	IplImage *edgeImg = cvCreateImage( cvGetSize(img_small), IPL_DEPTH_8U, 1 );

	// Canny edge detection
	cout << "Canny edge detection on CPU..." << endl;
	// Start timer
	init=clock();
	cvCanny( img_small, edgeImg, 230, 250, 3);
	// Stop timer
	final=clock()-init;
	cout << "Time taken to perform Canny on CPU:" << (double)final / ((double)CLOCKS_PER_SEC) << "s" << endl << endl;

	// Dilate image to thicken edges
	//cvDilate( edgeImg, edgeImg, 0, 1);

	// Display edge image
	cvNamedWindow("Edge Image:", CV_WINDOW_AUTOSIZE);
    cvShowImage("Edge Image:", edgeImg);

	// Wait for the user to press a key in the GUI window.
    //cvWaitKey(0);

	int posX, posY, maxVal, radius;

	// Define minimum and maximum radius
	int radMin = (int)30 * SCALE;
	int radMax = (int)50 * SCALE;

	cout << "Finding pupil in the image using Hough transform..." << endl;
	// Start timer
	init=clock();
	// Find circles
	//findCircleHoughCPU(edgeImg, radMin, radMax, &posX, &posY, &radius, &maxVal);
	// Stop timer
	final=clock()-init;
	cout << "Time taken to perform Hough on CPU:" << (double)final / ((double)CLOCKS_PER_SEC) << "s" << endl << endl;
	// Print out results
	//cout << "CPU results: MaxVal(" << maxVal << ") was found a position (" << posX << "," << posY << ") with radius: " << radius << endl;

	// Start timer
	init=clock();
	cout << "Finding pupil in the image using Hough transform on CUDA..." << endl;

	findCircleHoughCUDA(edgeImg, radMin, radMax, &posX, &posY, &radius, &maxVal);
	// Stop timer
	final=clock()-init;
	cout << "Time taken to perform Hough on CUDA:" << (double)final / ((double)CLOCKS_PER_SEC) << "s" << endl << endl;

	// Rescale values to match full size image
	posX = (int)posX/SCALE;
	posY = (int)posY/SCALE;
	radius = (int)radius/SCALE;

	// Print out results
	cout << "CUDA results on pupil: MaxVal(" << maxVal << ") was found a position (" << posX << "," << posY << ") with radius: " << radius << endl;

	// Draw the circle
	cvCircle(imgResult, cvPoint(posX, posY), radius, CV_RGB(0,0,255), 4);
	cvShowImage("Orig Image:", imgResult);

	cvWaitKey(0);

	// ####################
	// ### FINDING IRIS ###
	// ####################
	
	// Calculate area where iris is located
	int tl_x = posX - 4*radius;
	int tl_y = posY - 4*radius;
	int dim = 8*radius;

	// Redefine radMin and radMax
	radMin = (int)radius*2*SCALE;
	radMax = (int)radius*3.5*SCALE;

	int posX_i, posY_i, maxVal_i, radius_i;

	// Crop the image
	cvSetImageROI(img, cvRect(tl_x, tl_y, dim, dim));
	IplImage *img_cropped = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	cvCopy(img, img_cropped, NULL);
	// Reset ROI
	cvResetImageROI( img );
	IplImage *img_cropped_s = cvCreateImage( cvSize((int)(img_cropped->width*SCALE) , (int)(img_cropped->height*SCALE) ),img_cropped->depth, img_cropped->nChannels );
	cvResize(img_cropped, img_cropped_s);

	// Adjust the intensity of the image
	imadjustInput(img_cropped_s, 0.3f, 0.5f);

	// Create new image for storing edge image
	IplImage *edgeImgCropped = cvCreateImage( cvGetSize(img_cropped_s), IPL_DEPTH_8U, 1 );

	// Canny edge detection
	cout << "Canny edge detection on CPU..." << endl;
	// Start timer
	init=clock();
	cvCanny( img_cropped_s, edgeImgCropped, 210, 250, 3);
	// Stop timer
	final=clock()-init;
	cout << "Time taken to perform Canny on CPU:" << (double)final / ((double)CLOCKS_PER_SEC) << "s" << endl << endl;

	// Dilate image to thicken edges
	cvDilate( edgeImgCropped, edgeImgCropped, 0, 1);

	// Display edge image
	cvNamedWindow("Edge Image:", CV_WINDOW_AUTOSIZE);
    cvShowImage("Edge Image:", edgeImgCropped);

	// Start timer
	init=clock();
	cout << "Finding iris in the image using Hough transform on CUDA..." << endl;

	findCircleHoughCUDA(edgeImgCropped, radMin, radMax, &posX_i, &posY_i, &radius_i, &maxVal_i);
	// Stop timer
	final=clock()-init;
	cout << "Time taken to perform Hough on CUDA:" << (double)final / ((double)CLOCKS_PER_SEC) << "s" << endl << endl;

	// Rescale values to match full size image
	posX_i = (int)posX_i/SCALE;
	posY_i = (int)posY_i/SCALE;
	radius_i = (int)radius_i/SCALE;

	// Print out results
	cout << "CUDA results on iris: MaxVal(" << maxVal_i << ") was found a position (" << posX_i << "," << posY_i << ") with radius: " << radius_i << endl;

	// ### TEMP ### Display the image
	cvCircle(img_cropped, cvPoint(posX_i, posY_i), radius_i, CV_RGB(0,0,255), 4);
	cvNamedWindow("ROI Image:", CV_WINDOW_AUTOSIZE);
    cvShowImage("ROI Image:", img_cropped);

	cvWaitKey(0);

    // Free the resources.
    cvDestroyWindow("Orig Image:");
    //cvReleaseImage(&img);

	cvDestroyWindow("Edge Image:");
    cvReleaseImage(&edgeImg);
    
    return 0;
}

int findCircleHoughCPU(IplImage *edgeImg, int radMin, int radMax, int* resX, int *resY, int *resRad, int *resVal)
{
	// Get dimensions of the image
	int width = edgeImg->width;
	int height = edgeImg->height;

	cout << "Width: " << width << ", Height: " << height << endl;
	cout << "Rad min: " << radMin << ", Rad max: " << radMax << endl;

	// Convert IplImage to an array
	uchar *data;
	cvGetRawData(edgeImg, (uchar**)&data);
	int* imgData;
	imgData = (int*)malloc(width*height*sizeof(int));
	for (int i=0; i < width*height; i++) {
		if (data[i] > 0) {
			imgData[i] = 1;
		}
		else {
			imgData[i] = 0;
		}
	}

	// Image array for CUDA
	float* imgDataCUDA;
	imgDataCUDA = (float*)malloc(width*height*sizeof(float));
	for (int i=0; i < width*height; i++) {
		if (data[i] > 0) {
			imgDataCUDA[i] = 1;
		}
		else {
			imgDataCUDA[i] = 0;
		}
	}

	// Saving the matrix to file
	/*ofstream fout;
	fout.open("matrix.txt");

	for (int y=0; y < height; y++) {
		for (int x=0; x < width; x++) {
			fout << imgData[y*width + x] << " ";
		}
		fout << endl;
	}
	fout.close();
	*/
	// File closed

	// Variables for Hough
	int radius;
	int posX, posY, maxVal;
	// Arrays for results
	int *posxArray, *posyArray, *maxValArray, *radArray;
	// Allocate correct memory for arrays
	posxArray = (int*)malloc((radMax-radMin)*sizeof(int));
	posyArray = (int*)malloc((radMax-radMin)*sizeof(int));
	maxValArray = (int*)malloc((radMax-radMin)*sizeof(int));
	radArray = (int*)malloc((radMax-radMin)*sizeof(int));

	// Perform task for all the radius
	int index = 0;
	for (int r=radMin; r <= radMax; r++) {
		// Perform Hough transform
		radius = r;
		Hough2D_CPU(imgData, width, height, radius, &posX, &posY, &maxVal);
		// Write results to arrays
		posxArray[index] = posX;
		posyArray[index] = posY;
		maxValArray[index] = maxVal;
		radArray[index] = radius;

		index++;
	}

	// Find the maximum value from arrays
	maxVal = 0;
	for (int i=0; i < (radMax-radMin); i++) {
		if (maxVal < maxValArray[i]) {
			maxVal = maxValArray[i];
			*resX = posxArray[i];
			*resY = posyArray[i];
			*resRad = radArray[i];
			*resVal = maxVal;
		}
	}
	
	cout << "maxVal: " << maxVal << endl;

	return 0;
}

int Hough2D_CPU(int* img, int width, int height, int radius, int* posX, int* posY, int* maxVal)
{
	int* houghSpace;
	houghSpace = (int*)malloc(width*height*sizeof(int));
	// Set all elements to zero
	for (int i=0; i < width*height; i++) {
		houghSpace[i] = 0;
	}

	// Perform hough transform
	for (int x=radius; x < width-radius; x++) {
		for (int y=radius; y < height-radius; y++) {
			// 'Draw' a circle
			for (int theta=0; theta < 360; theta+=5) {
				// Calculate x and y coordinates
				float angle = (theta*PI) / 180;
				int tempX = (int)(x - radius*cos(angle));
				int tempY = (int)(y - radius*sin(angle));
				// If edge is detected, add vote
				if (img[tempY*width+tempX] == 1) {
					houghSpace[y*width+x] = houghSpace[y*width+x] + 1;
				}
			}
		}
	}

	// Find max value in the houghSpace
	*maxVal = 0;
	int index;
	
	for (int y=0; y < height; y++) {
		for (int x=0; x < width; x++) {
			//index = radius*width*height + y*width + x;
			index = y*width + x;
			if (*maxVal < houghSpace[index]) {
				*maxVal = houghSpace[index];
				*posX = x;
				*posY = y;
			}
		}
	}

	return 0;
}

int findCircleHoughCUDA(IplImage *edgeImg, int radMin, int radMax, int* resX, int *resY, int *resRad, int *resVal)
{
	// Get dimensions of the image
	int width = edgeImg->width;
	int height = edgeImg->height;

	// ### Convert IplImage to an array ###
	unsigned char *inImg, *temp2;
	inImg = new unsigned char[edgeImg->width*edgeImg->height*edgeImg->nChannels];
	temp2 = inImg;

	// pointer to imageData
	unsigned char *temp1 = (unsigned char*) edgeImg->imageData;
	// copy imagedata to buffer row by row
	for(int i=0;i<edgeImg->height;i++)
	{
		// memory copy
		memcpy(temp2, temp1, edgeImg->width*edgeImg->nChannels);
		// imageData jump to next line
		temp1 = temp1 + edgeImg->widthStep;
		// buffer jump to next line
		temp2 = temp2+ edgeImg->width*edgeImg->nChannels;
	}

	// Saving the matrix to file
	ofstream fout;
	fout.open("houghinput.dat");

	for (int y=0; y < height; y++) {
		for (int x=0; x < width; x++) {
			fout << (int)inImg[y*width + x] << " ";
		}
		fout << endl;
	}
	fout.close();
	
	// File closed

	// Image array for CUDA
	float* imgDataCUDA;
	imgDataCUDA = (float*)malloc(width*height*sizeof(float));
	for (int i=0; i < width*height; i++) {
		if (inImg[i] > 0) {
			imgDataCUDA[i] = 1;
		}
		else {
			imgDataCUDA[i] = 0;
		}
	}

	// Variables for Hough
	int radius;
	// Arrays for results
	int *posxArray, *posyArray, *maxValArray, *radArray;
	// Allocate correct memory for arrays
	posxArray = (int*)malloc((radMax-radMin)*sizeof(int));
	posyArray = (int*)malloc((radMax-radMin)*sizeof(int));
	maxValArray = (int*)malloc((radMax-radMin)*sizeof(int));
	radArray = (int*)malloc((radMax-radMin)*sizeof(int));

	// Perform task for all the radius
	Hough2D_CUDA(imgDataCUDA, width, height, radMin, radMax, resX, resY, resVal, resRad);

	return 0;
}

int imadjustInput(IplImage *cvImg, float lowPerc, float highPerc)
{
	// Get dimensions of the image
	int width = cvImg->width;
	int height = cvImg->height;

	// ### Convert IplImage to an array ###
	unsigned char *inImg, *temp2, *outImg;
	inImg = new unsigned char[cvImg->width*cvImg->height*cvImg->nChannels];
	outImg = new unsigned char[cvImg->width*cvImg->height*cvImg->nChannels];
	temp2 = inImg;

	// pointer to imageData
	unsigned char *temp1 = (unsigned char*) cvImg->imageData;
	// copy imagedata to buffer row by row
	for(int i=0;i<cvImg->height;i++)
	{
		// memory copy
		memcpy(temp2, temp1, cvImg->width*cvImg->nChannels);
		// imageData jump to next line
		temp1 = temp1 + cvImg->widthStep;
		// buffer jump to next line
		temp2 = temp2+ cvImg->width*cvImg->nChannels;
	}

	// Adjust image intensity
	imadjustCUDA(inImg, outImg, width, height, lowPerc, highPerc);

	// Convert the image back to openCV format
	cvSetData(cvImg, outImg, width);

	return 0;
}