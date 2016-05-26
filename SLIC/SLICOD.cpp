#define _CRT_SECURE_NO_WARNINGS
// SLICODSuperpixels.cpp : Defines the class behaviors for the application.
//

#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "SLICOD.h"

// For superpixels
const int dx4[4] = { -1,  0,  1,  0 };
const int dy4[4] = { 0, -1,  0,  1 };
//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// For supervoxels
const int dx10[10] = { -1,  0,  1,  0, -1,  1,  1, -1,  0, 0 };
const int dy10[10] = { 0, -1,  0,  1, -1, -1,  1,  1,  0, 0 };
const int dz10[10] = { 0,  0,  0,  0,  0,  0,  0,  0, -1, 1 };


SLICOD::SLICOD()
{
	m_lvec = NULL;
	m_avec = NULL;
	m_bvec = NULL;
	m_dvec = NULL;

	m_lvecvec = NULL;
	m_avecvec = NULL;
	m_bvecvec = NULL;
	m_dvecvec = NULL;
}

SLICOD::~SLICOD()
{
	if (m_lvec) delete[] m_lvec;
	if (m_avec) delete[] m_avec;
	if (m_bvec) delete[] m_bvec;
	if (m_bvec) delete[] m_dvec;


	if (m_lvecvec)
	{
		for (int d = 0; d < m_depth; d++) delete[] m_lvecvec[d];
		delete[] m_lvecvec;
	}
	if (m_avecvec)
	{
		for (int d = 0; d < m_depth; d++) delete[] m_avecvec[d];
		delete[] m_avecvec;
	}
	if (m_bvecvec)
	{
		for (int d = 0; d < m_depth; d++) delete[] m_bvecvec[d];
		delete[] m_bvecvec;
	}
	if (m_dvecvec)
	{
		for (int d = 0; d < m_depth; d++) delete[] m_dvecvec[d];
		delete[] m_dvecvec;
	}
}


//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void SLICOD::RGB2XYZ(
	const int&		sR,
	const int&		sG,
	const int&		sB,
	const int&		sD,
	double&			X,
	double&			Y,
	double&			Z,
	double&			D)
{
	double R = sR / 255.0;
	double G = sG / 255.0;
	double B = sB / 255.0;
	//double Dep = sD / 255.0;

	double r, g, b, d;

	if (R <= 0.04045)	r = R / 12.92;
	else				r = pow((R + 0.055) / 1.055, 2.4);
	if (G <= 0.04045)	g = G / 12.92;
	else				g = pow((G + 0.055) / 1.055, 2.4);
	if (B <= 0.04045)	b = B / 12.92;
	else				b = pow((B + 0.055) / 1.055, 2.4);
	//if (Dep <= 0.04045)	d = Dep / 12.92;
	//else				d = pow((Dep + 0.055) / 1.055, 2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
	//D = d
	D = sD;
	//printf("s(%d,%d,%d,%d) to X(%f,%f,%f,%f) \n", sR, sG, sB, sD, X, Y, Z, D);
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void SLICOD::RGB2LAB(const int& sR, const int& sG, const int& sB, const int& sD, double& lval, double& aval, double& bval, double& dval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z, D;
	RGB2XYZ(sR, sG, sB, sD, X, Y, Z, D);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white
	double Dr = 1.0;

	double xr = X / Xr;
	double yr = Y / Yr;
	double zr = Z / Zr;
	//double dr = D / Dr;

	double fx, fy, fz, fd;
	if (xr > epsilon)	fx = pow(xr, 1.0 / 3.0);
	else				fx = (kappa*xr + 16.0) / 116.0;
	if (yr > epsilon)	fy = pow(yr, 1.0 / 3.0);
	else				fy = (kappa*yr + 16.0) / 116.0;
	if (zr > epsilon)	fz = pow(zr, 1.0 / 3.0);
	else				fz = (kappa*zr + 16.0) / 116.0;
	//if (dr > epsilon)	fd = pow(dr, 1.0 / 3.0);
	//else				fd = (kappa*dr + 16.0) / 116.0;

	lval = 116.0*fy - 16.0;
	aval = 500.0*(fx - fy);
	bval = 200.0*(fy - fz);
	dval = D;
	//printf("X(%f,%f,%f,%f) to val(%f,%f,%f,%f) \n", X, Y, Z, D, lval, aval, bval, dval);
	//getchar();

}

//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void SLICOD::DoRGBtoLABConversion(
	const unsigned int*&		ubuff,
	double*&					lvec,
	double*&					avec,
	double*&					bvec,
	double*&					dvec)
{

	int sz = m_width*m_height;
	lvec = new double[sz];
	avec = new double[sz];
	bvec = new double[sz];
	dvec = new double[sz];

	for (int j = 0; j < sz; j++)
	{
		
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >> 8) & 0xFF;
		int b = (ubuff[j]) & 0xFF;
		int d = (ubuff[j] >> 24) & 0xFF;

		RGB2LAB(r, g, b, d, lvec[j], avec[j], bvec[j], dvec[j]);
	}
}

//===========================================================================
///	DoRGBtoLABConversion
///
/// For whole volume
//===========================================================================
void SLICOD::DoRGBtoLABConversion(
	const unsigned int**&		ubuff,
	double**&					lvec,
	double**&					avec,
	double**&					bvec,
	double**&					dvec)
{
	int sz = m_width*m_height;
	for (int d = 0; d < m_depth; d++)
	{
		for (int j = 0; j < sz; j++)
		{
			int r = (ubuff[d][j] >> 16) & 0xFF;
			int g = (ubuff[d][j] >> 8) & 0xFF;
			int b = (ubuff[d][j]) & 0xFF;
			int dep = (ubuff[d][j] >> 24) & 0xFF;

			RGB2LAB(r, g, b, dep, lvec[d][j], avec[d][j], bvec[d][j], dvec[d][j]);
		}
	}
}

//=================================================================================
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void SLICOD::DrawContoursAroundSegments(
	unsigned int*			ubuff,
	const int*				labels,
	const int&				width,
	const int&				height,
	const unsigned int&				color)
{
	const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	int sz = width*height;

	vector<bool> istaken(sz, false);

	int mainindex(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);
			for (int i = 0; i < 8; i++)
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;

					if (false == istaken[index])//comment this to obtain internal contours
					{
						if (labels[mainindex] != labels[index]) np++;
					}
				}
			}
			if (np > 1)//change to 2 or 3 for thinner lines
			{
				ubuff[mainindex] = color;
				istaken[mainindex] = true;
			}
			mainindex++;
		}
	}
}

//=================================================================================
/// DrawContoursAroundSegmentsTwoColors
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void SLICOD::DrawContoursAroundSegmentsTwoColors(
	unsigned int*			img,
	const int*				labels,
	const int&				width,
	const int&				height)
{
	const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	int sz = width*height;

	vector<bool> istaken(sz, false);

	vector<int> contourx(sz);
	vector<int> contoury(sz);
	int mainindex(0);
	int cind(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);
			for (int i = 0; i < 8; i++)
			{
				int x = k + dx[i];
				int y = j + dy[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;

					//if( false == istaken[index] )//comment this to obtain internal contours
					{
						if (labels[mainindex] != labels[index]) np++;
					}
				}
			}
			if (np > 1)
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				//img[mainindex] = color;
				cind++;
			}
			mainindex++;
		}
	}

	int numboundpix = cind;//int(contourx.size());

	for (int j = 0; j < numboundpix; j++)
	{
		int ii = contoury[j] * width + contourx[j];
		img[ii] = 0xffffff;
		//----------------------------------
		// Uncomment this for thicker lines
		//----------------------------------
		for (int n = 0; n < 8; n++)
		{
			int x = contourx[j] + dx[n];
			int y = contoury[j] + dy[n];
			if ((x >= 0 && x < width) && (y >= 0 && y < height))
			{
				int ind = y*width + x;
				if (!istaken[ind]) img[ind] = 0;
			}
		}
	}
}


//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLICOD::DetectLabEdges(
	const double*				lvec,
	const double*				avec,
	const double*				bvec,
	const double*				dvec,
	const int&					width,
	const int&					height,
	vector<double>&				edges)
{
	int sz = width*height;
	double weight = m_weight;

	edges.resize(sz, 0);
	for (int j = 1; j < height - 1; j++)
	{
		for (int k = 1; k < width - 1; k++)
		{
			int i = j*width + k;

			double xdd = 0;
			double ydd = 0;
			double tempx = dvec[i - 1] - dvec[i + 1];
			//printf("%f \n",tempx);
			double tempy = dvec[i - width] - dvec[i + width];
			if (tempx*tempx*tempx*tempx > 700)
				xdd = tempx*tempx + 7000;
			else xdd = tempx*tempx*tempx*tempx * 10;
			if (tempy*tempy*tempy*tempy > 700)
				ydd = tempy*tempy + 7000;
			else ydd = tempy*tempy*tempy*tempy * 10;
				
			double dx = (1 - weight) * ((lvec[i - 1] - lvec[i + 1])*(lvec[i - 1] - lvec[i + 1]) +
										(avec[i - 1] - avec[i + 1])*(avec[i - 1] - avec[i + 1]) +
										(bvec[i - 1] - bvec[i + 1])*(bvec[i - 1] - bvec[i + 1])) +
							weight  *	xdd;

			double dy = (1 - weight) * ((lvec[i - width] - lvec[i + width])*(lvec[i - width] - lvec[i + width]) +
										(avec[i - width] - avec[i + width])*(avec[i - width] - avec[i + width]) +
										(bvec[i - width] - bvec[i + width])*(bvec[i - width] - bvec[i + width])) +
							weight  *	ydd;
			//edges[i] = (sqrt(dx) + sqrt(dy));
			edges[i] = (dx + dy);
			//printf("(%f, %f, %f, %f)\n", (lvec[i - 1] - lvec[i + 1])*(lvec[i - 1] - lvec[i + 1]), (avec[i - 1] - avec[i + 1])*(avec[i - 1] - avec[i + 1]), (bvec[i - 1] - bvec[i + 1])*(bvec[i - 1] - bvec[i + 1]), (dvec[i - 1] - dvec[i + 1])*(dvec[i - 1] - dvec[i + 1]));
			//printf("(%f, %f, %f, %f)\n",(lvec[i - width] - lvec[i + width])*(lvec[i - width] - lvec[i + width]),(avec[i - width] - avec[i + width])*(avec[i - width] - avec[i + width]),(bvec[i - width] - bvec[i + width])*(bvec[i - width] - bvec[i + width]),(dvec[i - width] - dvec[i + width])*(dvec[i - width] - dvec[i + width]));
			//printf("(%f, %f)", dx, dy);
			//getchar();
		}
	}
}

//===========================================================================
///	PerturbSeeds
//===========================================================================
void SLICOD::PerturbSeeds(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsd,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const vector<double>&		edges)
{
	const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	int numseeds = kseedsl.size();

	for (int n = 0; n < numseeds; n++)
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for (int i = 0; i < 8; i++)
		{
			int nx = ox + dx8[i];//new x
			int ny = oy + dy8[i];//new y

			if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if (edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if (storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind / m_width;
			kseedsl[n] = m_lvec[storeind];
			kseedsa[n] = m_avec[storeind];
			kseedsb[n] = m_bvec[storeind];
			kseedsd[n] = m_dvec[storeind];
		}
	}
}


//===========================================================================
///	GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLICOD::GetLABXYSeeds_ForGivenStepSize(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsd,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					STEP,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int numseeds(0);
	int n(0);

	//int xstrips = m_width/STEP;
	//int ystrips = m_height/STEP;
	int xstrips = (0.5 + double(m_width) / double(STEP));
	int ystrips = (0.5 + double(m_height) / double(STEP));

	int xerr = m_width - STEP*xstrips;
	int yerr = m_height - STEP*ystrips;

	double xerrperstrip = double(xerr) / double(xstrips);
	double yerrperstrip = double(yerr) / double(ystrips);

	int xoff = STEP / 2;
	int yoff = STEP / 2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------
	kseedsl.resize(numseeds);
	kseedsa.resize(numseeds);
	kseedsb.resize(numseeds);
	kseedsd.resize(numseeds);
	kseedsx.resize(numseeds);
	kseedsy.resize(numseeds);

	for (int y = 0; y < ystrips; y++)
	{
		int ye = y*yerrperstrip;
		for (int x = 0; x < xstrips; x++)
		{
			int xe = x*xerrperstrip;
			int i = (y*STEP + yoff + ye)*m_width + (x*STEP + xoff + xe);

			kseedsl[n] = m_lvec[i];
			kseedsa[n] = m_avec[i];
			kseedsb[n] = m_bvec[i];
			kseedsd[n] = m_dvec[i];
			kseedsx[n] = (x*STEP + xoff + xe);
			kseedsy[n] = (y*STEP + yoff + ye);
			n++;
		}
	}


	if (perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsd, kseedsx, kseedsy, edgemag);
	}
}

//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLICOD::GetLABXYSeeds_ForGivenK(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsd,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					K,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int sz = m_width*m_height;
	double step = sqrt(double(sz) / double(K));
	int T = step;
	int xoff = step / 2;
	int yoff = step / 2;

	int n(0); int r(0);
	for (int y = 0; y < m_height; y++)
	{
		int Y = y*step + yoff;
		if (Y > m_height - 1) break;

		for (int x = 0; x < m_width; x++)
		{
			//int X = x*step + xoff;//square grid
			int X = x*step + (xoff << (r & 0x1));//hex grid
			if (X > m_width - 1) break;

			int i = Y*m_width + X;

			//_ASSERT(n < K);

			//kseedsl[n] = m_lvec[i];
			//kseedsa[n] = m_avec[i];
			//kseedsb[n] = m_bvec[i];
			//kseedsx[n] = X;
			//kseedsy[n] = Y;
			kseedsl.push_back(m_lvec[i]);
			kseedsa.push_back(m_avec[i]);
			kseedsb.push_back(m_bvec[i]);
			kseedsd.push_back(m_dvec[i]);
			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			n++;
		}
		r++;
	}

	if (perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsd ,kseedsx, kseedsy, edgemag);
	}
}


//===========================================================================
///	PerformSuperpixelSegmentation_VariableSandM
///
///	Magic SLICOD - no parameters
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
/// This function picks the maximum value of color distance as compact factor
/// M and maximum pixel distance as grid step size S from each cluster (13 April 2011).
/// So no need to input a constant value of M and S. There are two clear
/// advantages:
///
/// [1] The algorithm now better handles both textured and non-textured regions
/// [2] There is not need to set any parameters!!!
///
/// SLICODO (or SLICOD Zero) dynamically varies only the compactness factor S,
/// not the step size S.
//===========================================================================
void SLICOD::PerformSuperpixelSegmentation_VariableSandM(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsd,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	int*						klabels,
	const int&					STEP,
	const int&					NUMITR)
{


	int sz = m_width*m_height;
	double weight = m_weight;
	const int numk = kseedsl.size();
	//double cumerr(99999.9);
	int numitr(0);

	//----------------
	int offset = STEP;
	if (STEP < 10) offset = STEP*1.5;
	//----------------

	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmad(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<int> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	vector<double> distxy(sz, DBL_MAX);
	vector<double> distlab(sz, DBL_MAX);
	vector<double> distvec(sz, DBL_MAX);
	vector<double> maxlab(numk, 10 * 10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	vector<double> maxxy(numk, STEP*STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10

	double invxywt = 1.0 / (STEP*STEP);//NOTE: this is different from how usual SLICOD/LKM works

	while (numitr < NUMITR)
	{
		//------
		//cumerr = 0;
		numitr++;
		//------

		distvec.assign(sz, DBL_MAX);
		for (int n = 0; n < numk; n++)
		{
			/*int y1 = max(0, kseedsy[n] - offset);
			int y2 = min(m_height, kseedsy[n] + offset);
			int x1 = max(0, kseedsx[n] - offset);
			int x2 = min(m_width, kseedsx[n] + offset);*/
			int y1 = 0;
			int y2 = m_height;
			int x1 = 0;
			int x2 = m_width;

			if (0 < kseedsy[n] - offset) y1 = kseedsy[n] - offset;
			if (m_height > kseedsy[n] + offset) y2 = kseedsy[n] + offset;
			if (0 < kseedsx[n] - offset) x1 = kseedsx[n] - offset;
			if (m_width > kseedsx[n] + offset) x2 = kseedsx[n] + offset;


			for (int y = y1; y < y2; y++)
			{
				for (int x = x1; x < x2; x++)
				{
					int i = y*m_width + x;
					_ASSERT(y < m_height && x < m_width && y >= 0 && x >= 0);

					double l = m_lvec[i];
					double a = m_avec[i];
					double b = m_bvec[i];
					double d = m_dvec[i];

					double xdd = 0;
					double ydd = 0;
					double tempx = d - kseedsd[n];
					//printf("%f \n",tempx);
					if (tempx*tempx*tempx*tempx > 300)
						xdd = tempx*tempx + 3000;
					else xdd = tempx*tempx*tempx*tempx * 10;

				
					distlab[i] = (1-weight)	*(  (l - kseedsl[n])*(l - kseedsl[n]) +
												(a - kseedsa[n])*(a - kseedsa[n]) +
												(b - kseedsb[n])*(b - kseedsb[n]) ) +
								   weight   *   (d - kseedsd[n])*(d - kseedsd[n]);
					distxy[i] = (x - kseedsx[n])*(x - kseedsx[n]) +
						(y - kseedsy[n])*(y - kseedsy[n]);

					//------------------------------------------------------------------------
					double dist = distlab[i] / maxlab[n] + distxy[i] * invxywt;//only varying m, prettier superpixels
																			   //double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
																			   //------------------------------------------------------------------------
					//printf("(%f, %f, %f, %f)\n", l, a, b, d);
					//printf("(%f, %f, %f, %f)\n", (l - kseedsl[n])*(l - kseedsl[n]), (a - kseedsa[n])*(a - kseedsa[n]), (b - kseedsb[n])*(b - kseedsb[n]), (d - kseedsd[n])*(d - kseedsd[n]));
					//printf("(%f, %f )\n", distlab[i], distxy[i]);
					//getchar();
					
					if (dist < distvec[i])
					{
						//printf("(%f, %f, %f, %f)\n", l, a, b, d);
						//printf("(%f, %f, %f, %f)\n", (l - kseedsl[n])*(l - kseedsl[n]), (a - kseedsa[n])*(a - kseedsa[n]), (b - kseedsb[n])*(b - kseedsb[n]), (d - kseedsd[n])*(d - kseedsd[n]));
						//printf("(%f, %f) %f %d \n", distlab[i], distxy[i], dist, n);
						//printf("(%f, %f) %f %d \n", d, kseedsd[n], dist, n);
						//getchar();

						distvec[i] = dist;
						klabels[i] = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Assign the max color distance for a cluster
		//-----------------------------------------------------------------
		if (0 == numitr)
		{
			maxlab.assign(numk, 1);
			maxxy.assign(numk, 1);
		}
		{for (int i = 0; i < sz; i++)
		{
			if (maxlab[klabels[i]] < distlab[i]) maxlab[klabels[i]] = distlab[i];
			if (maxxy[klabels[i]] < distxy[i]) maxxy[klabels[i]] = distxy[i];
		}}

		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		clustersize.assign(numk, 0);
		for (int j = 0; j < sz; j++)
		{
			int temp = klabels[j];
			_ASSERT(klabels[j] >= 0);
			sigmal[klabels[j]] += m_lvec[j];
			sigmaa[klabels[j]] += m_avec[j];
			sigmab[klabels[j]] += m_bvec[j];
			sigmad[klabels[j]] += m_dvec[j];
			sigmax[klabels[j]] += (j%m_width);
			sigmay[klabels[j]] += (j / m_width);

			clustersize[klabels[j]]++;
		}

		{for (int k = 0; k < numk; k++)
		{
			//_ASSERT(clustersize[k] > 0);
			if (clustersize[k] <= 0) clustersize[k] = 1;
			inv[k] = 1.0 / double(clustersize[k]);//computing inverse now to multiply, than divide later
		}}

		{for (int k = 0; k < numk; k++)
		{
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
			kseedsd[k] = sigmad[k] * inv[k];
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
		}}
	}
}

//===========================================================================
///	SaveSuperpixelLabels
///
///	Save labels in raster scan order.
//===========================================================================
void SLICOD::SaveSuperpixelLabels(
	const int*					labels,
	const int&					width,
	const int&					height,
	const string&				filename,
	const string&				path)
{
	int sz = width*height;

	char fname[_MAX_FNAME];
	char extn[_MAX_FNAME];
	_splitpath(filename.c_str(), NULL, NULL, fname, extn);
	string temp = fname;

	ofstream outfile;
	string finalpath = path + temp + string(".dat");
	outfile.open(finalpath.c_str(), ios::binary);
	for (int i = 0; i < sz; i++)
	{
		outfile.write((const char*)&labels[i], sizeof(int));
	}
	outfile.close();
}

//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
//===========================================================================
void SLICOD::EnforceLabelConnectivity(
	const int*					labels,//input labels that need to be corrected to remove stray labels
	const int&					width,
	const int&					height,
	int*						nlabels,//new labels
	int&						numlabels,//the number of labels changes in the end if segments are removed
	const int&					K) //the number of superpixels desired by the user
{
	//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = { -1,  0,  1,  0 };
	const int dy4[4] = { 0, -1,  0,  1 };

	const int sz = width*height;
	const int SUPSZ = sz / K;
	//nlabels.resize(sz, -1);
	for (int i = 0; i < sz; i++) nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			if (0 > nlabels[oindex])
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for (int n = 0; n < 4; n++)
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if ((x >= 0 && x < width) && (y >= 0 && y < height))
					{
						int nindex = y*width + x;
						if (nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}

				int count(1);
				for (int c = 0; c < count; c++)
				{
					for (int n = 0; n < 4; n++)
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if ((x >= 0 && x < width) && (y >= 0 && y < height))
						{
							int nindex = y*width + x;

							if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if (count <= SUPSZ >> 2)
				{
					for (int c = 0; c < count; c++)
					{
						int ind = yvec[c] * width + xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if (xvec) delete[] xvec;
	if (yvec) delete[] yvec;
}

//===========================================================================
///	PerformSLICOD_ForGivenStepSize
///
/// There is option to save the labels if needed.
//===========================================================================
void SLICOD::PerformSLICOD_ForGivenStepSize(
	const unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					STEP,
	const double&				m)
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsd(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);

	//--------------------------------------------------
	m_width = width;
	m_height = height;
	int sz = m_width*m_height;
	//klabels.resize( sz, -1 );
	//--------------------------------------------------
	//klabels = new int[sz];
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	//--------------------------------------------------
	DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec, m_dvec);
	//--------------------------------------------------

	bool perturbseeds(true);
	vector<double> edgemag(0);
	if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_dvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenStepSize(kseedsl, kseedsa, kseedsb, kseedsd, kseedsx, kseedsy, STEP, perturbseeds, edgemag);

	PerformSuperpixelSegmentation_VariableSandM(kseedsl, kseedsa, kseedsb, kseedsd, kseedsx, kseedsy, klabels, STEP, 10);
	numlabels = kseedsl.size();

	int* nlabels = new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, double(sz) / double(STEP*STEP));
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }
	if (nlabels) delete[] nlabels;
}

//===========================================================================
///	PerformSLICO_ForGivenK
///
/// Zero parameter SLIC algorithm for a given number K of superpixels.
//===========================================================================
void SLICOD::PerformSLICOD_ForGivenK(
	const unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m,
	const double				weight)//weight given to spatial distance
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsd(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);

	//--------------------------------------------------
	m_width = width;
	m_height = height;
	m_weight = weight;
	int sz = m_width*m_height;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	//--------------------------------------------------
	if (1)//LAB
	{
		DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec, m_dvec);
	}
	else//RGB
	{
		m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz]; m_dvec = new double[sz];
		for (int i = 0; i < sz; i++)
		{
			m_lvec[i] = ubuff[i] >> 16 & 0xff;
			m_avec[i] = ubuff[i] >> 8  & 0xff;
			m_bvec[i] = ubuff[i]       & 0xff;
			m_dvec[i] = ubuff[i] >> 24 & 0xff;
		}
	}
	//--------------------------------------------------
	bool perturbseeds(true);
	vector<double> edgemag(0);

	if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_dvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsd, kseedsx, kseedsy, K, perturbseeds, edgemag);
	int STEP = sqrt(double(sz) / double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
	PerformSuperpixelSegmentation_VariableSandM(kseedsl, kseedsa, kseedsb, kseedsd, kseedsx, kseedsy, klabels, STEP, 10);
	numlabels = kseedsl.size();
	int* nlabels = new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, K);
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }
	if (nlabels) delete[] nlabels;
}

