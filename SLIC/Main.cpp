
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <crtdbg.h>
#include <string>
#include "SLIC.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
#include <fstream>

#include "SLIC.h"
#include "SLICOD.h"

using namespace std;

int main(int argc, char* argv[]) { 
	
	if (argc != 6) {
		printf("Arguments is not correct: <m_spcount> <ColorSize> <JpgImage> <DepthImage> <LabelImage>");
		getchar();
		return -1;
	}
	
	/* Load the image and convert to Lab colour space. */
	IplImage *image = cvLoadImage(argv[3], 1);
	IplImage *depImg = cvLoadImage(argv[4], CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *labImg = cvLoadImage(argv[5], CV_LOAD_IMAGE_GRAYSCALE);
	
	
	unsigned int* img = NULL;
	int width(0) ;
	width = image->width;
	int height(0) ;
	height = image->height;
	int sz = width*height;
	int* labels = new int[sz];
	int numlabels(0);
	int block = atoi(argv[2]);

	int m_spcount = atoi(argv[1]);
	
	double m_compactness = 20;
	if (m_spcount > sz) printf("Number of superpixels exceeds number of pixels in the image");

	img = new unsigned int[sz];


	cv::Mat pSrc = image;
	cv::Mat dSrc = depImg;
	cv::Mat lSrc = labImg;
	
	unsigned char r, g, b, d, l;
	int x = 0;
	int y = 0;
	//int errpix = 0;
	int adrs = 0;
	for (int i = 0; i < sz; i++)
	{
		
		g = pSrc.data[3*i];
		b = pSrc.data[3*i+1];
		r = pSrc.data[3*i+2];
		d = dSrc.data[i];
		l = lSrc.data[i];
		

		// to get rid of label image noises
		//printf("(%d , %d) (%d %d %d %d) %d \n", x, y, g, b, r, d, l);
		if (l >= 40) {
			for (int j=0;;j++) {
				adrs = i - j;
				if (adrs < 0) adrs = (-1)*adrs;
				if (lSrc.data[adrs] < 40) {
					/*g = pSrc.data[3 * adrs];
					b = pSrc.data[3 * adrs + 1];
					r = pSrc.data[3 * adrs + 2];*/
					d = dSrc.data[adrs];
					l = lSrc.data[adrs];
					break;
				}
			}
		}

		x++;
		if (x >= width) {
			x = 0;
			y++;
		}
		img[i] = (d << 24) + (g << 16) + (b << 8) + r;
	}
	
	//printf("error pixels %d", errpix);

	SLIC slic;
	//slic.PerformSLICO_ForGivenK(img, width, height, labels, numlabels, m_spcount, m_compactness);//for a given number K of superpixels
	
	SLICOD slicod;
	slicod.PerformSLICOD_ForGivenK(img, width, height, labels, numlabels, m_spcount, m_compactness);//for a given number K of superpixels

	int rgbdSize =  (256 / block) * 4;
	int rgbdlSize = rgbdSize + 1;

	
	vector<vector<float>> data (numlabels, vector<float>(rgbdlSize, 0));
	vector<vector<unsigned char>> label(numlabels, vector<unsigned char>(40, 0));


	string arg1 = argv[1];
	string arg2 = argv[2];
	string arg3 = argv[3];
	ofstream ofs(arg3 + "_m" + arg1+"c"+ arg2 + ".csv" );
	ofstream ofsLabel(arg3 + "_m" + arg1 + "spmap.csv");
	x = 0;
	y = 0;
	for (int i = 0; i < sz; i++)
	{
		//printf("(%d , %d) %d \n", x, y, labels[i]);
		g = (unsigned char)pSrc.data[3 * i] /block ;
		b = (unsigned char)pSrc.data[3 * i +1] / block;
		r = (unsigned char)pSrc.data[3 * i +2] / block;
		d = (unsigned char)dSrc.data[i] / block;
		l = lSrc.data[i];

		// to get rid of label image noises
		if (l >= 40) {
			for (int j = 0;; j++) {
				adrs = i - j;
				if (adrs < 0) adrs = (-1)*adrs;
				if (lSrc.data[adrs] < 40) {
					/*g = (unsigned char)pSrc.data[3 * i] / block;
					b = (unsigned char)pSrc.data[3 * i + 1] / block;
					r = (unsigned char)pSrc.data[3 * i + 2] / block;*/
					d = (unsigned char)dSrc.data[adrs] / block;
					l = lSrc.data[adrs];
					break;
				}
			}
		}
		printf("(%d , %d) (%d %d %d %d) %d \n", x, y, g, b ,r, d ,l);
		data[labels[i]][g] += 1;
		data[labels[i]][b + 256 / block] += 1;
		data[labels[i]][r + 2 * 256 / block] += 1;
		data[labels[i]][d + 3 * 256 / block] += 1;

		label[labels[i]][l] += 1;

		ofsLabel << to_string(labels[i]);
		x++;
		if (x >= width) {
			ofsLabel <<endl;
			x = 0;
			y++;
		}
		else {
			ofsLabel << ",";
		}

	}
	
	vector<double> N(numlabels,0);

	for (int i = 0; i < numlabels; i++) {
		for (int j = 0; j < rgbdSize; j++) {
			//N[i] +=  data[i][j] * data[i][j];
			N[i] += data[i][j];
		}
	}
	int max=0;
	int maxLabel = 0;
	for (int i = 0; i < numlabels; i++) {
		max = 0;
		maxLabel = 0;
		for (int j = 0; j < 40; j++) {
			if (max < label[i][j]) {
				max = label[i][j];
				maxLabel = j;
			}
		}
	
		data[i][rgbdSize] = maxLabel;
	}

	for (int i = 0; i < numlabels; i++) {
		for (int j = 0; j < rgbdSize; j++) {
			//ofs << to_string(data[i][j]/ (sqrt(N[i]))) + ",";
			ofs << to_string(data[i][j] / (N[i])) + ",";

		}
		ofs << to_string(data[i][rgbdSize]) <<endl;
	}


	

	if (labels) delete[] labels;
	if (img) delete[] img;

	return 0;
}