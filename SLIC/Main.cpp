
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

void convert_label_to_color(string FILE_PATH, vector<vector<float>> &data, int * labels);

int main(int argc, char* argv[]) {

	if (argc != 7) {
		printf("Arguments is not correct: <m_spcount> <ColorSize> <JpgImage> <DepthImage> <LabelImage> <weight>");
		getchar();
		return -1;
	}

	/* Load the image and convert to Lab colour space. */

	//cv::Mat src = cv::imread(argv[3], 1);
	cv::Mat pSrc = cv::imread(argv[3], 1);
	//cv::cvtColer(src, pSrc, CV_BGR2HSV)
	cv::Mat dSrc = cv::imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat lSrc = cv::imread(argv[5], CV_LOAD_IMAGE_GRAYSCALE);


	
	int numlabels(0);
	int block = atoi(argv[2]);
	int m_spcount = atoi(argv[1]);
	double weight = atof(argv[6]);;
	int height = pSrc.rows;
	int width = pSrc.cols;

	int sz = width*height;
	int* labels = new int[sz];
	unsigned int* img = new unsigned int[sz];
	double m_compactness = 20;
	if (m_spcount > sz) printf("Number of superpixels exceeds number of pixels in the image");




	unsigned char r, g, b, d, l;
	int x = 0;
	int y = 0;
	int errpix = 0;
	int adrs = 0;
	for (int i = 0; i < sz; i++)
	{

		b = pSrc.data[3 * i];
		g = pSrc.data[3 * i + 1];
		r = pSrc.data[3 * i + 2];
		d = dSrc.data[i];
		l = lSrc.data[i];


		// to get rid of label image noises
		//printf("(%d , %d) (%d %d %d %d) %d \n", x, y, g, b, r, d, l);
		if (l >= 40) {
			return 0;
			for (int j = 0;; j++) {
				adrs = i - j;
				if (adrs < 0) adrs = (-1)*adrs;
				if (lSrc.data[adrs] < 40) {
					/*g = pSrc.data[3 * adrs];
					b = pSrc.data[3 * adrs + 1];
					r = pSrc.data[3 * adrs + 2];*/
					d = dSrc.data[adrs];
					l = lSrc.data[adrs];

					//errpix++;
					break;
				}
			}
		}
		//printf("%d\n", d);
		x++;
		if (x >= width) {
			x = 0;
			y++;
		}
		img[i] = (d << 24) + (b << 16) + (g << 8) + r;
	}

	//printf("%d\n", errpix);
	SLIC slic;
	//slic.PerformSLICO_ForGivenK(img, width, height, labels, numlabels, m_spcount, m_compactness);//for a given number K of superpixels

	SLICOD slicod;
	slicod.PerformSLICOD_ForGivenK(img, width, height, labels, numlabels, m_spcount, m_compactness, weight);//for a given number K of superpixels

	int rgbdSize = (256 / block) * 4;
	int rgbdlSize = rgbdSize + 1;


	vector<vector<float>> data(numlabels, vector<float>(rgbdlSize, 0));
	vector<vector<unsigned char>> label(numlabels, vector<unsigned char>(40, 0));


	string arg1 = argv[1];
	string arg2 = argv[2];
	string arg3 = argv[3];
	ofstream ofs(arg3 + "_m" + arg1 + "c" + arg2 + ".csv");
	ofstream ofsLabel(arg3 + "_m" + arg1 + "spmap.csv");
	x = 0;
	y = 0;
	for (int i = 0; i < sz; i++)
	{
		//printf("(%d , %d) %d \n", x, y, labels[i]);
		b = (unsigned char)pSrc.data[3 * i] / block;
		g = (unsigned char)pSrc.data[3 * i + 1] / block;
		r = (unsigned char)pSrc.data[3 * i + 2] / block;
		d = (unsigned char)dSrc.data[i] / block;
		l = (unsigned char)lSrc.data[i];


		//printf("(%d , %d) (%d %d %d %d) %d \n", x, y, g, b ,r, d ,l);
		data[labels[i]][b] += 1;
		data[labels[i]][g + 256 / block] += 1;
		data[labels[i]][r + 2 * 256 / block] += 1;
		data[labels[i]][d + 3 * 256 / block] += 1;

		// ‰ö‚µ‚¢‚Æ‚±‚»‚Ìˆê
		label[labels[i]][l] += 1;

		//ofsLabel << "("+to_string(labels[i])+",L" + to_string(l) + ")";
		ofsLabel << to_string(labels[i]);
		//printf("%d, %d, ", labels[i], label[labels[i]][l]);
		x++;
		if (x >= width) {
			ofsLabel << endl;
			x = 0;
			y++;
			//printf("\n");
		}
		else {
			ofsLabel << ",";
		}

	}


	vector<double> N(numlabels, 0);

	for (int i = 0; i < numlabels; i++) {
		for (int j = 0; j < rgbdSize; j++) {
			//N[i] +=  data[i][j] * data[i][j];
			N[i] += data[i][j];
		}
	}
	int max = 0;
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
	

	//for (int i = 0; i < numlabels; i++) {
	//	//if (label[i][0] < 50) continue;
	//	printf("%d [ ", i);
	//	for (int j = 0; j < 40; j++) {
	//		printf("%d, ", label[i][j]);
	//	}
	//	printf(" ] %f \n", data[i][rgbdSize]);
	//}
	//getchar();


	for (int i = 0; i < numlabels; i++) {
		for (int j = 0; j < rgbdSize; j++) {
			//ofs << to_string(data[i][j]/ (sqrt(N[i]))) + ",";
			ofs << to_string(data[i][j] / (N[i])) + ",";
		}
		ofs << to_string(data[i][rgbdSize]) << endl;
	}



	convert_label_to_color(argv[5], data, labels);
	//getchar();

	if (labels) delete[] labels;
	if (img) delete[] img;

	return 0;
}


void to_color(int label, int*b, int *g, int* r) {
	int red = 0;
	int green = 0;
	int blue = 0;
	label = label % 37;
	if (label <= 6) {
		red = 255 * label / 6;
		blue = 0;
		green = 0;
	}
	else if (6 < label || label <= 12) {
		red = 0;
		blue = 255 * (label - 6) / 6;
		green = 0;
	}
	else if (12 < label || label <= 18) {
		red = 0;
		blue = 0;
		green = 255 * (label - 12) / 6;
	}
	else if (18 < label || label <= 24) {
		red = 0;
		blue = 255 * (label - 18) / 6;
		green = 255 * (label - 18) / 6;
	}
	else if (24 < label || label <= 30) {
		red = 255 * (label - 24) / 6;
		blue = 0;
		green = 255 * (label - 24) / 6;
	}
	else {
		red = 255 * (label - 30) / 6;
		blue = 255 * (label - 30) / 6;
		green = 0;
	}

	*b = blue;
	*g = green;
	*r = red;


}

void convert_label_to_color(string FILE_PATH, vector<vector<float>> &data, int * labels) {
	cv::Mat src = cv::imread(FILE_PATH, 1);
	cv::Mat dst = src.clone();
	int height = src.rows;
	int width = src.cols;
	int size = height * width;
	int rgbd = data[0].size()-1;
	int b = 0; int g = 0; int r = 0; int l = 0;
	for (int i = 0; i < size; i++) {
		l = data[labels[i]][rgbd];
		//l = labels[i];
		to_color(l, &b, &g, &r);
		dst.data[3 * i] = b;
		dst.data[3 * i + 1] = g;
		dst.data[3 * i + 2] = r;
	}



	vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite(FILE_PATH + "_outputcolor.png", dst, compression_params);
}