# Histogram Equalization and Optimal Quantization - Computuer Vision and Image Processing 

## Overview 

So what we got in this repo:
- Loading grayscale and RGB image representations. 
- Displaying figures and images.
- Transforming RGB color images back and forth from the YIQ color space.
- Performing intensity transformations: **histogram equalization**.
- Performing **optimal quantization**

## Histogram Equalization
**The Target**: make the distribution of the intensities (the colors- 0 to 255)
 more balanced. 
 
**The Process** : First we create a histogram of the intensities, then compute
the CDF of the distribution. Now we are trying to straighten the CDF of the given picture by mapping a given intensity **_i_** to another intensity **_j_**
such CDF(j)= j/255. 

**The Result**:
![alt text](https://user-images.githubusercontent.com/60176709/114431930-bf4a1580-9bc8-11eb-9e03-8fd329c07189.png)

## Optimal Quantization 
**The Target**: As we know the standards levels of colors is 255. Suppose
we want to represent a picture with only **_k_** levels of colors, how we can do that 
and which colors to take. 

**The Process**: In order to represent all the colors in the picture with only 
**_k_** levels of colors, we perform a kind of **_k-means_**.
Given the intensities histogram, We divide the histogram into **_k_** cells and for
each cell we compute the mean.\
Now, the boundaries, do not represent the cells very well, they can be too close to their mean.
So we need to adjust the boundaries.
We’ll move each boundary to be in the middle of two means.\
Since this is not a closed solution, it forces us to do the above multiple times.
Ether a fix amount of times, or until the error (MSE) stops changing.

**The Result**:\
the original image:
![alt text](https://user-images.githubusercontent.com/60176709/114439565-c75a8300-9bd1-11eb-8355-9853ef08aaac.png)
the image with only 3 levels of GrayScale :
![alt text](https://user-images.githubusercontent.com/60176709/114439869-21f3df00-9bd2-11eb-8683-e330872cd423.png)
the error over the iterations:\
![alt text](https://user-images.githubusercontent.com/60176709/114440251-93cc2880-9bd2-11eb-9b88-9af9082929db.png)

## Gamma Correction 
Gamma correction or gamma is a nonlinear operation used to encode and decode luminance or tristimulus values in video or still image systems.

In this repo I wrote a function that performs gamma correction on an image with a given **_γ_**, using the OpenCV funtions createTrackbar
to create a slider (with gamma values from 0 to 200, but its actually from 0 to 2 since OpenCV allow only Integers values) 
and display the image with the correction.

![alt text](https://user-images.githubusercontent.com/60176709/114442795-9ed48800-9bd5-11eb-997f-05605f684436.png)