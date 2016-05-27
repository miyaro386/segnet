
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

//使うデータ長について
//H(0~180) を 9 刻みで分割 → 20
//s(0~255) を 16 刻みで分割 → 16
//v(0~255) を 16 刻みで分割 → 16
//Depth(0~255) を 32 刻みで分割 → 8
//HSVそれぞれの平均で合計3
// x, y で → 2
// ここまで最終的に平均を取るdataSize = 73
// 対応ラベル → 1
// 各スーパーピクセルに含まれるピクセル数  → 1
// 合計75
enum Index {
	H = 0,			
	S = H + 20,			
	V = S + 16,			
	DEPTH = V + 16,		
	H_AVE = DEPTH + 8,
	S_AVE,
	V_AVE,
	D_AVE,
	X_AVE,	
	Y_AVE,						//ここまでのデータは平均を取る
	//HOG,		
	//GRAD_DIRECT = HOG  + 9,
	//SP_DENSE = GRAD_DIRECT + 18,
	SP_DENSE,
	PLANE_NORMAL_X,
	PLANE_NORMAL_Y,
	PLANE_NORMAL_Z,
	LABEL,						//データ出力はここまで
	CLOSE1,
	CLOSE2,
	CLOSE3,
	CLOSE4,
	PIXEL_SUM,					//ラスト 配列確保に使うので固定
};

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
	vector<float> gradStren(sz, 0);
	vector<float> gradDirec(sz, 0);
	unsigned int* img = new unsigned int[sz];
	double m_compactness = 20;
	if (m_spcount > sz) printf("Number of superpixels exceeds number of pixels in the image");


	unsigned char b, g, r;
	unsigned char h, s, v, d, l;
	int gradx = 0; int grady = 0;
	int x = 0; 
	int y = 0; 
	for (int i = 0; i < sz; i++)
	{
		b = src.data[3 * i];
		g = src.data[3 * i + 1];
		r = src.data[3 * i + 2];
		d = dSrc.data[i];
		l = lSrc.data[i];

		//calculate grad of each pixel
		if (x == 0)				gradx = pSrc.data[3 * i	   ] - pSrc.data[3 * i + 1];
		else if (x == width -1) gradx = pSrc.data[3 * i - 1] - pSrc.data[3 * i	  ];
		else					gradx = pSrc.data[3 * i - 1] - pSrc.data[3 * i + 1];

		if (y == 0)				  grady = pSrc.data[3 * i		 ] - pSrc.data[3 * i + width];
		else if (y == height - 1) grady = pSrc.data[3 * i - width] - pSrc.data[3 * i		];
		else					  grady = pSrc.data[3 * i - width] - pSrc.data[3 * i + width];

		gradStren[i] = abs(gradx) + abs(grady);
		gradDirec[i] = (float)atan2(grady, gradx); // note: this incledes minus
		//printf("%f, ", gradDirec[i]);
		//printf("(%d , %d) bgr(%d %d %d) \n", x, y, b, g, r);
		//printf("(%d , %d) hsv(%d %d %d %d) %d \n", x, y, h, s, v, d, l);

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


	vector<vector<float>> data(numlabels, vector<float>(PIXEL_SUM + 1, 0));
	vector<vector<unsigned char>> label(numlabels, vector<unsigned char>(38, 0));

	int dataSize = Y_AVE + 1;

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
		d = dSrc.data[i];
		data[labels[i]][H_AVE] += h;
		data[labels[i]][S_AVE] += s;
		data[labels[i]][V_AVE] += v;
		data[labels[i]][D_AVE] += d;
		data[labels[i]][X_AVE] += x;
		data[labels[i]][Y_AVE] += y;

		//printf("(%d , %d) %d \n", x, y, labels[i]);
		h = (unsigned char)h / 20;
		if (h >= 20) h = 19;
		s = (unsigned char)s / 16;
		v = (unsigned char)v / 16;
		d = (unsigned char)d / 32;
		l = (unsigned char)lSrc.data[i];


		//printf("(%d , %d) (%d %d %d %d) %d \n", x, y, h, s ,v, d ,l);
		data[labels[i]][h + H] += 1;
		data[labels[i]][s + S] += 1;
		data[labels[i]][v + V] += 1;
		data[labels[i]][d + DEPTH] += 1;
		data[labels[i]][LABEL] += 1;
		data[labels[i]][PIXEL_SUM] += 1;

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





	//正解ラベルを取得
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
		data[i][LABEL] = maxLabel;
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
	int div_num = int(sqrt(m_spcount/4)/10) * 10;
	int xMax = int(width / div_num);
	int yMax = int(height / div_num);;
	//printf("%d\n", div_num);
	
	vector< vector< vector<int> > > XYsec( yMax + 1, vector< vector<int> >(xMax + 1, vector<int>(0)));

	//平均を取るべきところまで計算
	for (int i = 0; i < numlabels; i++) {
		for (int j = 0; j < Y_AVE+1; j++) {
			data[i][j] = data[i][j] / (data[i][PIXEL_SUM]);
		}
		data[i][X_AVE] = (int)data[i][X_AVE];
		data[i][Y_AVE] = (int)data[i][Y_AVE];		
		XYsec[(int)data[i][Y_AVE]/ div_num][(int)data[i][X_AVE]/ div_num].push_back(i);
	}




	//もっとも近い4つのスーパーピクセルを取得
	int dist = 0;
	x = 0; y = 0;
	//vector<vector<int>> closeLabel(numlablels, vector<int>(4,-1));
	vector<float> min(4, 10000);
	int X = 0; int Y = 0;
	for (int i = 0; i < numlabels; i++) {
		vector<int> closeLabel;
		vector<float> min(4, 10000);
		vector<float> tempLabel(4, 0);
		X = (int)data[i][X_AVE] / div_num; Y = (int)data[i][Y_AVE] / div_num;
		for (int j = 0; j < XYsec[Y][X].size(); j++)closeLabel.push_back(XYsec[Y][X][j]);
		if (X > 0)
			for (int j = 0; j < XYsec[Y][X - 1].size(); j++)closeLabel.push_back(XYsec[Y][X - 1][j]);
		if (X < xMax)
			for (int j = 0; j < XYsec[Y][X + 1].size(); j++)closeLabel.push_back(XYsec[Y][X + 1][j]);
		if (Y > 0)
			for (int j = 0; j < XYsec[Y - 1][X].size(); j++)closeLabel.push_back(XYsec[Y - 1][X][j]);
		if (Y < yMax)
			for (int j = 0; j < XYsec[Y + 1][X].size(); j++)closeLabel.push_back(XYsec[Y + 1][X][j]);
		//左上
		if (X > 0 && Y > 0)
			for (int j = 0; j < XYsec[Y - 1][X - 1].size(); j++)closeLabel.push_back(XYsec[Y - 1][X - 1][j]);
		//右上
		if (X < xMax && Y > 0)
			for (int j = 0; j < XYsec[Y - 1][X + 1].size(); j++)closeLabel.push_back(XYsec[Y - 1][X + 1][j]);
		if (X > 0 && Y < yMax)
			for (int j = 0; j < XYsec[Y + 1][X - 1].size(); j++)closeLabel.push_back(XYsec[Y + 1][X - 1][j]);
		if (X < xMax && Y < yMax)
			for (int j = 0; j < XYsec[Y + 1][X + 1].size(); j++)closeLabel.push_back(XYsec[Y + 1][X + 1][j]);

		data[i][SP_DENSE] = closeLabel.size();

		//例外処理
		while (closeLabel.size() < 4) {
			for (int j = 0; j < XYsec[Y][X].size(); j++)closeLabel.push_back(XYsec[Y][X][j]);
			if (X > 0)
				for (int j = 0; j < XYsec[Y][X - 1].size(); j++)closeLabel.push_back(XYsec[Y][X - 1][j]);
			if (X < int(width / div_num))
				for (int j = 0; j < XYsec[Y][X + 1].size(); j++)closeLabel.push_back(XYsec[Y][X + 1][j]);
			if (Y > 0)
				for (int j = 0; j < XYsec[Y - 1][X].size(); j++)closeLabel.push_back(XYsec[Y - 1][X][j]);
			if (Y < int(height / div_num))
				for (int j = 0; j < XYsec[Y + 1][X].size(); j++)closeLabel.push_back(XYsec[Y + 1][X][j]);
		}

		//printf("%d contains ", i);
		for (int j = 0; j < closeLabel.size(); j++) {
			if (i == closeLabel[j])continue;
			dist = abs(data[i][X_AVE] - data[closeLabel[j]][X_AVE]) + abs(data[i][Y_AVE] - data[closeLabel[j]][Y_AVE]);
			for (int k = 0; k < 4; k++) {
				//min[0]が常に最も小さい
				//min[k] より小さいと分かればk以降のminは後ろにプッシュされる
				if (min[k] >= dist) {
					for (int l = 3; l > k; l--) {
						min[l] = min[l - 1];
						tempLabel[l] = tempLabel[l - 1];
					}
					min[k] = dist;
					tempLabel[k] = closeLabel[j];
					break;
				}
			}

			//printf("%d, ", closeLabel[j]);
		}//最も近いものの計算は終了

		//printf("\n");

		data[i][CLOSE1] = tempLabel[0];
		data[i][CLOSE2] = tempLabel[1];
		data[i][CLOSE3] = tempLabel[2];
		data[i][CLOSE4] = tempLabel[3];
		//printf("%f, %f, %f, %f \n", data[i][CLOSE1], data[i][CLOSE2], data[i][CLOSE3], data[i][CLOSE4]);

		//getchar();
	}


	//PLANET_NORM算出
	for (int i = 0; i < numlabels; i++) {
		vector<vector<float>> vec(4, vector<float>(3, 0));
		vector<vector<float>> faceVec(4, vector<float>(3, 0));
		vec[0][0] = data[data[i][CLOSE1]][X_AVE] - data[i][X_AVE];
		vec[0][1] = data[data[i][CLOSE1]][Y_AVE] - data[i][Y_AVE];
		vec[0][2] = data[data[i][CLOSE1]][D_AVE] - data[i][D_AVE];
		vec[1][0] = data[data[i][CLOSE2]][X_AVE] - data[i][X_AVE];
		vec[1][1] = data[data[i][CLOSE2]][Y_AVE] - data[i][Y_AVE];
		vec[1][2] = data[data[i][CLOSE2]][D_AVE] - data[i][D_AVE];
		vec[2][0] = data[data[i][CLOSE3]][X_AVE] - data[i][X_AVE];
		vec[2][1] = data[data[i][CLOSE3]][Y_AVE] - data[i][Y_AVE];
		vec[2][2] = data[data[i][CLOSE3]][D_AVE] - data[i][D_AVE];
		vec[3][0] = data[data[i][CLOSE4]][X_AVE] - data[i][X_AVE];
		vec[3][1] = data[data[i][CLOSE4]][Y_AVE] - data[i][Y_AVE];
		vec[3][2] = data[data[i][CLOSE4]][D_AVE] - data[i][D_AVE];

		faceVec[0][0] = vec[0][1] * vec[1][2] - vec[0][2] * vec[1][1];
		faceVec[0][1] = vec[0][2] * vec[1][0] - vec[0][0] * vec[1][2];
		faceVec[0][2] = vec[0][0] * vec[1][1] - vec[0][1] * vec[1][0];

		faceVec[1][0] = vec[1][1] * vec[2][2] - vec[1][2] * vec[2][1];
		faceVec[1][1] = vec[1][2] * vec[2][0] - vec[1][0] * vec[2][2];
		faceVec[1][2] = vec[1][0] * vec[2][1] - vec[1][1] * vec[2][0];

		faceVec[2][0] = vec[2][1] * vec[3][2] - vec[2][2] * vec[3][1];
		faceVec[2][1] = vec[2][2] * vec[3][0] - vec[2][0] * vec[3][2];
		faceVec[2][2] = vec[2][0] * vec[3][1] - vec[2][1] * vec[3][0];

		faceVec[3][0] = vec[3][1] * vec[0][2] - vec[3][2] * vec[0][1];
		faceVec[3][1] = vec[3][2] * vec[0][0] - vec[3][0] * vec[0][2];
		faceVec[3][2] = vec[3][0] * vec[0][1] - vec[3][1] * vec[0][0];

		for (int m = 0; m < 4; m++) {
			data[i][PLANE_NORMAL_X] += faceVec[m][0];
			data[i][PLANE_NORMAL_Y] += faceVec[m][1];
			data[i][PLANE_NORMAL_Z] += faceVec[m][2];

		}
		float norm = sqrt(data[i][PLANE_NORMAL_X] * data[i][PLANE_NORMAL_X] + data[i][PLANE_NORMAL_Y] * data[i][PLANE_NORMAL_Y] + data[i][PLANE_NORMAL_Z] * data[i][PLANE_NORMAL_Z]);
		data[i][PLANE_NORMAL_X] = data[i][PLANE_NORMAL_X] / norm;
		data[i][PLANE_NORMAL_Y] = data[i][PLANE_NORMAL_Y] / norm;
		data[i][PLANE_NORMAL_Z] = data[i][PLANE_NORMAL_Z] / norm;

	}




	//データを出力する
	for (int i = 0; i < numlabels; i++) {
		for (int j = 0; j < LABEL; j++) {
			ofs << to_string(data[i][j] ) + ",";
		}
		ofs << to_string(data[i][LABEL]) << endl;
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
	int rgbd = data[0].size() - 1;
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