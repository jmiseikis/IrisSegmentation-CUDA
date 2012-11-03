GPU (CUDA) vs CPU Iris Segmentation Performance Comparison
=====================

Coursework for a class during my master studies. The goal was to compare 
the performance of a Hough transform based iris segmentation algorithm 
implemented on CPU and GPU using nVidia CUDA framework.

Dependencies:
* CUDA (and CUDA-enabled nVidia graphics card)
* OpenCV

Example input images are provided in the images folder.

The algorithm uses Hough transform to find adjust an image and find a circle 
of eye's pupil (defined maximum and minimum diameter, relative to the size of an 
input image) in the image and search for a larger one with approximately the same 
center to get the edge of eye's iris. Then iris could be segmented and used for 
biometric or any other purposes.

Results:

![Results](https://raw.github.com/jmiseikis/IrisSegmentation-CUDA/master/images/results.png)

Runtime comparison CPU vs GPU:

![Runtime](https://raw.github.com/jmiseikis/IrisSegmentation-CUDA/master/images/timing.png)

Very low-end GPU was used for testing (nVidia GeForce G210 with 512 MB), the acceleration 
would be significantly higher with medium-high end GPU. Both CPU and GPU implementations 
were __not__ highly optimised, rather used as a proof of concept, and optimisation 
would improve the runtime in both cases.

Slideshow of the presentation: http://www.slideshare.net/jmiseikis/cuda-based-iris-detection-based-on-hough-transform