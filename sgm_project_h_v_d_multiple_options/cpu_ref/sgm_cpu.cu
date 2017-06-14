#include "CTensor.h"
#include "timer.h"

#include <cmath>
#include <sstream>
#include <iostream>
#include <iomanip>


// Convert integer to std::string
template <typename T>
  std::string NumberToString ( T Number )
  {
    std::ostringstream ss;
    ss << Number;
    return ss.str();
  }

/*-------------------------------------------------------------------------
 *  32Bit RGBA color
 *-------------------------------------------------------------------------*/ 
typedef uchar4 Color;

/*-------------------------------------------------------------------------
 *  Regularization weight
 *-------------------------------------------------------------------------*/
static float const LAMBDA = 100.0f;

/*-------------------------------------------------------------------------
 *  Maximum disparity (number of labels in the message passing algorithm)
 *-------------------------------------------------------------------------*/ 
static int const MAX_DISPARITY = 50;

/*======================================================================*/
/*! 
 *   Convert CTensor to CMatrix of Colors.
 *
 *   \param out The output CMatrix
 *   \param in  The input CTensor
 */
/*======================================================================*/
void CTensorToColorCMatrix(
    CMatrix<Color>& out, const CTensor<unsigned char>& in)
{
  out.setSize(in.xSize(), in.ySize());
  for( int y = 0; y < out.ySize(); ++y )
  for( int x = 0; x < out.xSize(); ++x )
  {
    out(x,y).x = in(x,y,0); // R
    out(x,y).y = in(x,y,1); // G
    out(x,y).z = in(x,y,2); // B
  }
}


/*======================================================================*/
/*! 
 *   Compute squared distance of given pixels.
 *
 *   \param a The first pixel
 *   \param b The second pixel
 *
 *   \return L2-distance squared of a and b
 */
/*======================================================================*/
inline float unaryL2Squared(Color const &a, Color const &b)
{
  return (static_cast<float>(a.x) - static_cast<float>(b.x)) *
         (static_cast<float>(a.x) - static_cast<float>(b.x)) +
         (static_cast<float>(a.y) - static_cast<float>(b.y)) *
         (static_cast<float>(a.y) - static_cast<float>(b.y)) +
         (static_cast<float>(a.z) - static_cast<float>(b.z)) *
         (static_cast<float>(a.z) - static_cast<float>(b.z));
}


/*======================================================================*/
/*! 
 *   Compute absolute difference of given pixels.
 *
 *   \param a The first pixel
 *   \param b The second pixel
 *
 *   \return L1-distance of a and b
 */
/*======================================================================*/
inline float unaryL1(Color const &a, Color const &b)
{
  return abs(static_cast<float>(a.x) - static_cast<float>(b.x)) +
         abs(static_cast<float>(a.y) - static_cast<float>(b.y)) +
         abs(static_cast<float>(a.z) - static_cast<float>(b.z));
}


/*======================================================================*/
/*! 
 *   Compute euclidean L2 distance of given pixels.
 *
 *   \param a The first pixel
 *   \param b The second pixel
 *
 *   \return L2-distance of a and b
 */
/*======================================================================*/
inline float unaryEuclidean(Color const &a, Color const &b)
{
 return std::sqrt(unaryL2Squared(a, b));
}


/*======================================================================*/
/*! 
 *   Compute euclidean Lx-norm for pixel neighborhood 
 *
 *   \param leftImg   Left image
 *   \param rightImg  Right image
 *   \param xl        x position of left image pixel
 *   \param yl        y position of left image pixel
 *   \param xr        x position of right image pixel
 *   \param yr        y position of right image pixel
 *   \param costFunction {L1; L2} 
 *   \param N         NxN neighborhood
 *   \param option    1: L1-norm; 2: L2-norm squared
 *
 *   \return Lx-norm for pixel neighborhood
 */
/*======================================================================*/
inline float unaryLxNeighbor(CMatrix<Color> const &leftImg,
                             CMatrix<Color> const &rightImg,
                             int xl, int yl, int xr, int yr,
                             int costFunction,
                             int N)
{
  float theta = 0.0f;
  int lim = static_cast<int>(N/2);
  for (int j = -lim; j < lim; ++j) {
    for (int i = -lim; i < lim; ++i) {
      if (xl+i > 0 && xl+i < leftImg.xSize() &&
          yl+j > 0 && yl+j < leftImg.ySize() &&
          xr+i > 0 && xr+i < rightImg.xSize() &&
          yr+j > 0 && yr+j < rightImg.ySize()) {
        switch(costFunction) {
          case 1:
            theta += unaryL1(leftImg(xl+i, yl+j),
                             rightImg(xr+i, yr+j));
            break;
          case 2:
            theta += unaryL2Squared(leftImg(xl+i, yl+j),
                                    rightImg(xr+i, yr+j));
            break;
        }
      }
    }
  }

  return theta;
}


/*======================================================================*/
/*! 
 *   Compute Average pixel of NxN neighborhood 
 *
 *   \param Img       Image
 *   \param x         x position of image pixel
 *   \param y         y position of image pixel
 *   \param N         NxN neighborhood
 *
 *   \return average pixel Color for the neighborhood
 */
/*======================================================================*/
inline Color averagePixel(CMatrix<Color> const &Img,
                          int x, int y, int N)
{
  Color averagePixel;
  float averagePixelX = 0.0f;
  float averagePixelY = 0.0f;
  float averagePixelZ = 0.0f;

  int lim = N/2;
  for (int j = -lim; j < lim; ++j) {
    for (int i = -lim; i < lim; ++i) {
      if (x+i > 0 && x+i < Img.xSize() &&
          y+j > 0 && y+j < Img.ySize()) {
        averagePixelX += static_cast<float>(Img(x+i, y+j).x);
        averagePixelY += static_cast<float>(Img(x+i, y+j).y);
        averagePixelZ += static_cast<float>(Img(x+i, y+j).z);
      }
    }
  }

  averagePixel.x = averagePixelX/(N*N);
  averagePixel.y = averagePixelY/(N*N);
  averagePixel.z = averagePixelZ/(N*N);
  return averagePixel;
}


/*======================================================================*/
/*! 
 *   Compute difference of given pixels.
 *
 *   \param a The first pixel
 *   \param b The second pixel
 *
 *   \return difference in Colors of a and b
 */
/*======================================================================*/
inline float4 pixelDifference(Color const &a, Color const &b)
{
  float4 pixelDifference;
  pixelDifference.x = static_cast<float>(a.x) - static_cast<float>(b.x);
  pixelDifference.y = static_cast<float>(a.y) - static_cast<float>(b.y);
  pixelDifference.z = static_cast<float>(a.z) - static_cast<float>(b.z);
  return pixelDifference;
}


/*======================================================================*/
/*! 
 *   Compute dot product of given pixels.
 *
 *   \param a The first pixel
 *   \param b The second pixel
 *
 *   \return dot product of a and b
 */
/*======================================================================*/
inline float pixelDotProd(float4 const &a, float4 const &b)
{
  return a.x * b.x +
         a.y * b.y +
         a.z * b.z;
}


/*======================================================================*/
/*! 
 *   Compute Normalized cross-correlation (NCC) 
 *
 *   \param leftImg   Left image
 *   \param rightImg  Right image
 *   \param xl        x position of left image pixel
 *   \param yl        y position of left image pixel
 *   \param xr        x position of right image pixel
 *   \param yr        y position of right image pixel
 *   \param N         NxN neighborhood
 *
 *   \return NCC for pixel neighborhood 
 */
/*======================================================================*/
inline float unaryNCCNeighbor(CMatrix<Color> const &leftImg,
                              CMatrix<Color> const &rightImg,
                              int xl, int yl, int xr, int yr, int N)
{
  float theta = 0.0f;
  int lim = static_cast<int>(N/2);
  float varLeftImg = 0.0f;
  float varRightImg = 0.0f;
  
  Color averagePixelLeftImg;
  Color averagePixelRightImg;
  averagePixelLeftImg = averagePixel(leftImg, xl, yl, N);
  averagePixelRightImg = averagePixel(rightImg, xr, yr, N);

  for (int j = -lim; j < lim; ++j) {
    for (int i = -lim; i < lim; ++i) {
      if (xl+i > 0 && xl+i < leftImg.xSize() &&
          yl+j > 0 && yl+j < leftImg.ySize() &&
          xr+i > 0 && xr+i < rightImg.xSize() &&
          yr+j > 0 && yr+j < rightImg.ySize()) {
        theta += pixelDotProd(
                  pixelDifference(leftImg(xl+i, yl+j), averagePixelLeftImg),            
                  pixelDifference(rightImg(xr+i, yr+j), averagePixelRightImg));
        // Variance of left Image
        varLeftImg += unaryL2Squared(leftImg(xl+i, yl+j),
                                     averagePixelLeftImg);
        // Variance of right Image
        varRightImg += unaryL2Squared(rightImg(xr+i, yr+j),
                                      averagePixelRightImg);
      }
    }
  }

  return theta/std::sqrt(varLeftImg*varRightImg);
}


/*======================================================================*/
/*! 
 *   Potts model for distance of labels a and b. No cost for same label,
 *   constant cost for different labels.
 *
 *   \param a Label of first pixel
 *   \param b Label of second pixel
 *
 *   \return 0 if equal, 1 otherwise
 */
/*======================================================================*/
inline float thetapq(int a, int b)
{
  return (a == b) ? 0.0f : 1.0f;
}


/*======================================================================*/
/*! 
 *   Semi-global Matching between two images
 *
 *   \param result The resulting matrix with disparities
 *   \param leftImg Left Image
 *   \param rightImg Right Image
 *   \param unaryCost Type of unary cost: {pixelWise, L1, L2, NCC}
 *   \param N NxN patch size for L1, L2, NCC unary costs
 *   \param horizontal Horizontal message passing. Set to 1 to activate
 *   \param vertical Vertical message passing. Set to 1 to activate
 *   \param diagonal Diagonal message passing. Set to 1 to activate
 *
 *   \return void, just writes the result in the result matrix
 */
/*======================================================================*/
void sgmCPU(CMatrix<float> &result,
            CMatrix<Color> const &leftImg, CMatrix<Color> const &rightImg,
            int unaryCost, int N,
            bool horizontal, bool vertical, bool diagonal)
{
  /*-----------------------------------------------------------------------
   *  Unary cost computation 
   *-----------------------------------------------------------------------*/
  std::cout << "Precomputing unary costs... \r" << std::flush;
  CTensor<float> unarycosts(
      leftImg.xSize(), leftImg.ySize(), MAX_DISPARITY + 1);
  for (int y = 0; y < leftImg.ySize(); ++y)
  {
    for (int x = 0; x < leftImg.xSize(); ++x)
    {
      for (int i = 0; i <= MAX_DISPARITY; ++i)
      {
        if (x - i < 0) unarycosts(x, y, i) = 1.0e9f;
        else {
          switch(unaryCost) {
            case 0:  // Pixel-wise Euclidean distance
              unarycosts(x, y, i) = unaryEuclidean(leftImg(x, y),
                                                   rightImg(x - i, y));
              break;
            case 1:  // NxN L1 distance
              unarycosts(x, y, i) = unaryLxNeighbor(leftImg, rightImg,
                                                     x, y, x - i, y,
                                                     unaryCost, N);
              break;
            case 2:  // NxN L2 distance
              unarycosts(x, y, i) = unaryLxNeighbor(leftImg, rightImg,
                                                     x, y, x - i, y,
                                                     unaryCost, N);
              break;
            case 3:  // NxN NCC
              unarycosts(x, y, i) = -abs(unaryNCCNeighbor(leftImg,
                                                          rightImg,
                                                           x, y, x - i, y,
                                                           N));
              break;
          }
        }
      }
    }
    std::cout << "Precomputing unary costs... "
              << static_cast<int>((100.0f * y) / leftImg.ySize()) << "% \r"
              << std::flush;
  }
  std::cout << "Precomputing unary costs... 100%" << std::endl;

  /*-----------------------------------------------------------------------
   *  Disparity estimation (message passing)
   *-----------------------------------------------------------------------*/

  /* HORIZONTAL (scanline-wise) message Passing */
  std::vector<CMatrix<float> > MpqsHFCube(leftImg.ySize());  // Horizontal Forward 
  std::vector<CMatrix<float> > MpqsHBCube(leftImg.ySize());  // Horizontal Backward
  std::cout << "Computing HORIZONTAL disparities... \r" << std::flush;
  for (int y = 0; y < leftImg.ySize(); ++y)
  {
    /*---------------------------------------------------------------------
     *  Forward pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsHF(leftImg.xSize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    { 
      MpqsHF(0, j) = 0.0f;
    }
    for (int q = 1; q < leftImg.xSize(); ++q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsHF(q, j) = unarycosts(q - 1, y, 0) + MpqsHF(q - 1, 0) +
              LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(q - 1, y, i) + MpqsHF(q - 1, i) +
              LAMBDA * thetapq(i, j);
          if (cost < MpqsHF(q, j)) {
            MpqsHF(q, j) = cost;
          }
        }
      }
    }
    MpqsHFCube[y] = MpqsHF;

    /*---------------------------------------------------------------------
     *  Backward pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsHB(leftImg.xSize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    {
      MpqsHB(leftImg.xSize() - 1, j) = 0.0f;
    }
    for (int q = leftImg.xSize() - 2; q >= 0; --q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsHB(q, j) = unarycosts(q + 1, y, 0) + MpqsHB(q + 1, 0) +
              LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(q + 1, y, i) + MpqsHB(q + 1, i) +
              LAMBDA * thetapq(i, j);
          if (cost < MpqsHB(q, j)) {
            MpqsHB(q, j) = cost;
          }
        }
      }
    }
    MpqsHBCube[y] = MpqsHB;

    std::cout << "Computing HORIZONTAL disparities... "
              << static_cast<int>((100.0f * y) / leftImg.ySize()) << "% \r"
              << std::flush;
  }
  std::cout << "Computing HORIZONTAL disparities...100%" << std::endl;
   

  /* VERTICAL (scanline-wise) message Passing */
  std::vector<CMatrix<float> > MpqsVFCube(leftImg.xSize());  // Vertical Forward
  std::vector<CMatrix<float> > MpqsVBCube(leftImg.xSize());  // Vertical Backward
  std::cout << "Computing VERTICAL disparities... \r" << std::flush;
  for (int x = 0; x < leftImg.xSize(); ++x)
  {
    /*---------------------------------------------------------------------
     *  Forward pass (Top to Bottom)
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsVF(leftImg.ySize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    {
      MpqsVF(0, j) = 0.0f;
    } 
    for (int q = 1; q < leftImg.ySize(); ++q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsVF(q, j) = unarycosts(x, q - 1, 0) + MpqsVF(q - 1, 0) +
              LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(x, q - 1, i) + MpqsVF(q - 1, i) +
              LAMBDA * thetapq(i, j);
          if (cost < MpqsVF(q, j)) {
            MpqsVF(q, j) = cost;
          } 
        }
      }
    }
    MpqsVFCube[x] = MpqsVF;

    /*---------------------------------------------------------------------
     *  Backward pass (Bottom to Top)
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsVB(leftImg.ySize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    {
      MpqsVB(leftImg.ySize() - 1, j) = 0.0f;
    }
    for (int q = leftImg.ySize() - 2; q >= 0; --q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsVB(q, j) = unarycosts(x, q + 1, 0) + MpqsVB(q + 1, 0) +
              LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(x, q + 1, i) + MpqsVB(q + 1, i) +
              LAMBDA * thetapq(i, j);
          if (cost < MpqsVB(q, j)) {
            MpqsVB(q, j) = cost;
          } 
        }
      }
    }
    MpqsVBCube[x] = MpqsVB;

    std::cout << "Computing VERTICAL disparities... "
              << static_cast<int>((100.0f * x) / leftImg.xSize()) << "% \r"
              << std::flush;
  }
  std::cout << "Computing VERTICAL disparities...100%" << std::endl;


  /* DIAGONAL message Passing */
  std::vector<CMatrix<float> > MpqsDBRCube(leftImg.ySize()); // Diagonal to Bottom Right
  std::vector<CMatrix<float> > MpqsDBLCube(leftImg.ySize()); // Diagonal to Bottom Left
  std::vector<CMatrix<float> > MpqsDTLCube(leftImg.ySize()); // Diagonal to Top Left
  std::vector<CMatrix<float> > MpqsDTRCube(leftImg.ySize()); // Diagonal to Top Right 

  // Initialize top row of disparities matrices
  std::cout << "Computing DIAGONAL disparities... \r" << std::flush;
  CMatrix<float> MpqsDBR(leftImg.xSize(), MAX_DISPARITY + 1);
  CMatrix<float> MpqsDBL(leftImg.xSize(), MAX_DISPARITY + 1);
  CMatrix<float> MpqsDTL(leftImg.xSize(), MAX_DISPARITY + 1);
  CMatrix<float> MpqsDTR(leftImg.xSize(), MAX_DISPARITY + 1);
  for (int x = 0; x < leftImg.xSize(); ++x)
  {
    for(int j = 0; j <= MAX_DISPARITY; ++j) 
    {
      MpqsDBR(0, j) = 0.0f;
      MpqsDBL(0, j) = 0.0f;
      MpqsDTL(0, j) = 0.0f;
      MpqsDTR(0, j) = 0.0f;
    }
  }
  MpqsDBRCube[0] = MpqsDBR;
  MpqsDBLCube[0] = MpqsDBL;
  MpqsDTLCube[leftImg.ySize()-1] = MpqsDTL;
  MpqsDTRCube[leftImg.ySize()-1] = MpqsDTR;

  for (int y = 1; y < leftImg.ySize(); ++y)
  {
    /*---------------------------------------------------------------------
     *  To Bottom Right pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsDBR(leftImg.xSize(), MAX_DISPARITY + 1);
    // Initialize dispaties matrix
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    { 
      MpqsDBR(0, j) = 0.0f;
    }
    for (int q = 1; q < leftImg.xSize(); ++q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsDBR(q, j) = unarycosts(q - 1, y - 1, 0) + MpqsDBRCube[y-1](q - 1, 0) +
              LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(q - 1, y - 1, i) + MpqsDBRCube[y-1](q - 1, i) +
              LAMBDA * thetapq(i, j);
          if (cost < MpqsDBR(q, j)) {
            MpqsDBR(q, j) = cost;
          }
        }
      }
    }
    MpqsDBRCube[y] = MpqsDBR;

    /*---------------------------------------------------------------------
     *  To Bottom Left pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsDBL(leftImg.xSize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    {
      MpqsDBL(leftImg.xSize() - 1, j) = 0.0f;
    }
    for (int q = leftImg.xSize() - 2; q >= 0; --q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsDBL(q, j) = unarycosts(q + 1, y - 1, 0) + MpqsDBLCube[y-1](q + 1, 0) +
              LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(q + 1, y - 1, i) + MpqsDBLCube[y-1](q + 1, i) +
              LAMBDA * thetapq(i, j);
          if (cost < MpqsDBL(q, j)) {
            MpqsDBL(q, j) = cost;
          }
        }
      }
    }
    MpqsDBLCube[y] = MpqsDBL;

    /*---------------------------------------------------------------------
     *  To Top Left pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsDTL(leftImg.xSize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    {
      MpqsDTL(leftImg.xSize() - 1, j) = 0.0f;
    }
    for (int q = leftImg.xSize() - 2; q >= 0; --q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsDTL(q, j) = unarycosts(q + 1, leftImg.ySize() - y, 0)
                        + MpqsDTLCube[leftImg.ySize() - y](q + 1, 0)
                        + LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(q + 1, leftImg.ySize(), i) 
                       + MpqsDTLCube[leftImg.ySize() - y](q + 1, i) 
                       + LAMBDA * thetapq(i, j);
          if (cost < MpqsDTL(q, j)) {
            MpqsDTL(q, j) = cost;
          }
        }
      }
    }
    MpqsDTLCube[leftImg.ySize()-y-1] = MpqsDTL;

    /*---------------------------------------------------------------------
     *  To Top Right pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsDTR(leftImg.xSize(), MAX_DISPARITY + 1);
    // Initialize dispaties matrix
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    { 
      MpqsDTR(0, j) = 0.0f;
    }
    for (int q = 1; q < leftImg.xSize(); ++q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsDTR(q, j) = unarycosts(q - 1, leftImg.ySize() - y, 0)
                        + MpqsDTRCube[leftImg.ySize() - y](q - 1, 0)
                        + LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(q - 1, leftImg.ySize() - y, i)
                       + MpqsDTRCube[leftImg.ySize() - y](q - 1, i)
                       + LAMBDA * thetapq(i, j);
          if (cost < MpqsDTR(q, j)) {
            MpqsDTR(q, j) = cost;
          }
        }
      }
    }
    MpqsDTRCube[leftImg.ySize()-y-1] = MpqsDTR;

    std::cout << "Computing DIAGONAL disparities... "
              << static_cast<int>((100.0f * y) / leftImg.ySize()) << "% \r"
              << std::flush;
  }
  std::cout << "Computing DIAGONAL disparities...100%" << std::endl;


  /*---------------------------------------------------------------------
   *  Decision
   *---------------------------------------------------------------------*/
  std::cout << "Computing DECISIONS... \r" << std::flush;
  for (int y = 0; y < leftImg.ySize(); ++y) 
  {
    for (int x = 0; x < leftImg.xSize(); ++x)
    {
      int minIndex = 0;
      float minCost = unarycosts(x, y, 0) 
                      + horizontal*(MpqsHFCube[y](x, 0)
                                    + MpqsHBCube[y](x, 0))
                      + vertical*(MpqsVFCube[x](y, 0)
                                  + MpqsVBCube[x](y, 0))
                      + diagonal*(MpqsDBRCube[y](x, 0)
                                 + MpqsDBLCube[y](x, 0)
                                 + MpqsDTLCube[y](x, 0)
                                 + MpqsDTRCube[y](x, 0));
      for (int i = 1; i <= MAX_DISPARITY; ++i)
      {
        float cost = unarycosts(x, y, i) 
                     + horizontal*(MpqsHFCube[y](x, i)
                                  + MpqsHBCube[y](x, i))
                     + vertical*(MpqsVFCube[x](y, i)
                                + MpqsVBCube[x](y, i))
                     + diagonal*(MpqsDBRCube[y](x, i)
                                + MpqsDBLCube[y](x, i)
                                + MpqsDTLCube[y](x, i)
                                + MpqsDTRCube[y](x, i));
        if (cost < minCost)
        {
          minCost = cost;
          minIndex = i;
        }
      }
      result(x, y) = static_cast<float>(minIndex);
    }
    std::cout << "Computing DECISIONS... "
              << static_cast<int>((100.0f * y) / leftImg.ySize()) << "% \r"
              << std::flush;
  }
  std::cout << "Computing DECISIONS... 100%" << std::endl;  

}


int main(int argc, char** argv)
{
  if (argc < 4)
  {
    std::cerr << "usage: " << argv[0] << " <path to left image> <path to right "
         << "image> <disparity output path>" << std::endl;
    exit(1);
  }

  /*-----------------------------------------------------------------------
   *  Read rectified left and right input image and put them into
   *  Color CMatrices
   *-----------------------------------------------------------------------*/
  CTensor<unsigned char> tmp;
  tmp.readFromPPM(argv[1]);
  CMatrix<Color> leftImg;
  CTensorToColorCMatrix(leftImg, tmp);
  tmp.readFromPPM(argv[2]);
  CMatrix<Color> rightImg;
  CTensorToColorCMatrix(rightImg, tmp);
  
  /*-----------------------------------------------------------------------
   *  Prepare output disparity map
   *-----------------------------------------------------------------------*/
  CMatrix<float> result(leftImg.xSize(), leftImg.ySize());
  
  // Matrix of all options (horizontal, vertical, diagonal)
  int options[4][3][3] = {{{1, 0, 0}, {1, 1, 0}, {1, 1, 1}},
                          {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}},
                          {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}},
                          {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}}
                         };

  int N = 7;  // NxN patches
  timer::start("SGM (CPU)");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      timer::start("SGM (CPU)");
      int unaryCost = i;
      int horizontal = options[i][j][0];
      int vertical = options[i][j][1];
      int diagonal = options[i][j][2];
      sgmCPU(result, leftImg, rightImg, unaryCost, N,
             horizontal, vertical, diagonal);

      std::string option (NumberToString(i));
      option.append(NumberToString(horizontal));
      option.append(NumberToString(vertical));
      option.append(NumberToString(diagonal));
      std::string resultFile (argv[3]);
      resultFile.insert(0, "-");
      resultFile.insert(0, option); 
      result.writeToFloatFile(resultFile.c_str());

      std::string dispEPE ("../bin/disp-epe ");
      dispEPE.append(resultFile.c_str());
      dispEPE.append(" ");
      dispEPE.append("1-gt.float3");

      std::cout << "Option " << option << std::endl;
      system(dispEPE.c_str());

      std::string floatToPGM ("../bin/float3-to-pgm ");
      floatToPGM.append(resultFile);
      floatToPGM.append(" ");
      floatToPGM.append(resultFile);
      floatToPGM.erase(floatToPGM.length()-6);
      floatToPGM.append("pgm");
      system(floatToPGM.c_str());


      timer::stop("SGM (CPU)");
      timer::printToScreen(std::string(), timer::AUTO_COMMON, timer::ELAPSED_TIME);
    }
  }
  // sgmCPU(result, leftImg, rightImg, 1, 7, 1, 1, 0);
  // Parameters
  // Result Image
  // Left Image
  // Right Image
  // unaryCost {0: PixelWise Euclidean; 1: L1 NxN; 2: L2 NxN; 3: NCC NxN}
  // N NxN patch size
  // horizontal, vertical, diagonal Message Passing

  
  timer::stop("SGM (CPU)");
  timer::printToScreen(std::string(), timer::AUTO_COMMON, timer::ELAPSED_TIME);

  return 0;
}
