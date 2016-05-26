
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

	if (argc != 6) {
		printf("Arguments is not correct: <m_spcount> <JpgImage> <DepthImage> <LabelImage> <weight>");
		getchar();
		return -1;
	}

	/* Load the image and convert to Lab colour space. */

	cv::Mat src = cv::imread(argv[2], 1);
	cv::Mat pSrc;
	cv::cvtColor(src, pSrc, CV_BGR2HSV);
	
	cv::Mat dSrc = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat lSrc = cv::imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE);


	
	int numlabels(0);
	int m_spcount = atoi(argv[1]);
	double weight = atof(argv[5]);
	int height = pSrc.rows;
	int width = pSrc.cols;

	int sz = width*height;
	int* labels = new int[sz];
	unsigned int* img = new unsigned int[sz];
	double m_compactness = 20;
	if (m_spcount > sz) printf("Number of superpixels exceeds number of pixels in the image");




	unsigned char b, g, r;
	unsigned char h, s, v, d, l;
	float sum_h = 0; float sum_s = 0; float sum_v = 0;
	int x = 0; float sum_x = 0;
	int y = 0; float sum_y = 0;
	int errpix = 0;
	int adrs = 0;
	for (int i = 0; i < sz; i++)
	{
		b = src.data[3 * i];
		g = src.data[3 * i + 1];
		r = src.data[3 * i + 2];
		d = dSrc.data[i];
		l = lSrc.data[i];


		// to get rid of label image noises

		//printf("(%d , %d) bgr(%d %d %d) \n", x, y, b, g, r);
		//printf("(%d , %d) hsv(%d %d %d %d) %d \n", x, y, h, s, v, d, l);
		
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

	//使うデータ長について
	//H(0~180) を 9 刻みで分割 → 20
	//s(0~255) を 16 刻みで分割 → 16
	//v(0~255) を 16 刻みで分割 → 16
	//Depth(0~255) を 16 刻みで分割 → 16
	//HSVそれぞれの平均で合計3
	// x, y で → 2
	// 対応ラベル → 1
	// 各スーパーピクセルに含まれるピクセル数  → 1
	// 合計75

	vector<vector<float>> data(numlabels, vector<float>(75 , 0));
	vector<vector<unsigned char>> label(numlabels, vector<unsigned char>(38, 0));

	int dataSize = 73;
	int labelCol = dataSize;
	int pixelSumCol = labelCol + 1;
	string jpgPath = argv[2];
	string s_m = argv[1];
	ofstream ofs(jpgPath + "_m" + s_m + "HSV.csv");
	ofstream ofs_spmap(jpgPath + "_m" + s_m + "spmap.csv");
	x = 0;
	y = 0;
	for (int i = 0; i < sz; i++)
	{

		h = pSrc.data[3 * i];
		s = pSrc.data[3 * i + 1];
		v = pSrc.data[3 * i + 2];
		data[labels[i]][68] += h;
		data[labels[i]][69] += s;
		data[labels[i]][70] += v;
		data[labels[i]][71] += x;
		data[labels[i]][72] += y;

		//printf("(%d , %d) %d \n", x, y, labels[i]);
		h = (unsigned char)h / 20;
		if (h >= 20) h = 19;
		s = (unsigned char)s / 16;
		v = (unsigned char)v / 16;

		d = (unsigned char)dSrc.data[i] / 16;
		l = (unsigned char)lSrc.data[i];


		//printf("(%d , %d) (%d %d %d %d) %d \n", x, y, h, s ,v, d ,l);
		data[labels[i]][h] += 1;
		data[labels[i]][s + 20] += 1;
		data[labels[i]][v + 36] += 1;
		data[labels[i]][d + 52] += 1;
		data[labels[i]][labelCol] += 1;
		data[labels[i]][pixelSumCol] += 1;

		// 怪しいとこその一
		label[labels[i]][l] += 1;

		//ofsLabel << "("+to_string(labels[i])+",L" + to_string(l) + ")";
		ofs_spmap << to_string(labels[i]);
		//printf("%d, %d, ", labels[i], label[labels[i]][l]);
		x++;
		if (x >= width) {
			ofs_spmap << endl;
			x = 0;
			y++;
			//printf("\n");
		}
		else {
			ofs_spmap << ",";
		}

	}

	int max = 0;
	int maxLabel = 0;
	for (int i = 0; i < numlabels; i++) {
		max = 0;
		maxLabel = 0;
		for (int j = 0; j < 38; j++) {
			if (max < label[i][j]) {
				max = label[i][j];
				maxLabel = j;
			}
		}
		data[i][labelCol] = maxLabel;
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
		printf("%d [ ", i);
		for (int j = 0; j < dataSize; j++) {
			//ofs << to_string(data[i][j]/ (sqrt(N[i]))) + ",";
			printf("%f, ", data[i][j] / (data[i][pixelSumCol]) );
			ofs << to_string(data[i][j] / (data[i][pixelSumCol])) + ",";
		}
		printf(" ] %f \n", data[i][labelCol]);
		ofs << to_string(data[i][labelCol]) << endl;
	}


	//convert_label_to_color(argv[5], data, labels);
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