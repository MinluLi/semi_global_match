#include "CTensor.h"
#include "timer.h"

#include <cmath>

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
 *   \param N         NxN neighborhood
 *   \param option    1: L1-norm; 2: L2-norm squared
 *
 *   \return Lx-norm for pixel neighborhood
 */
/*======================================================================*/
inline float unaryLxNeighbor(CMatrix<Color> const &leftImg,
                             CMatrix<Color> const &rightImg,
                             int xl, int yl, int xr, int yr, int N,
                             int option)
{
  float theta = 0.0f;
  int lim = static_cast<int>(N/2);
  for (int j = -lim; j < lim; ++j) {
    for (int i = -lim; i < lim; ++i) {
        if (xl+i > 0 && xl+i < leftImg.xSize() &&
            yl+j > 0 && yl+j < leftImg.ySize() &&
            xr+i > 0 && xr+i < rightImg.xSize() &&
            yr+j > 0 && yr+j < rightImg.ySize()) {
          if (option == 1)
            theta += unaryL1(leftImg(xl+i, yl+j), rightImg(xr+i, yr+j));
          if (option == 2)
            theta += unaryL2Squared(leftImg(xl+i, yl+j), rightImg(xr+i, yr+j));
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

void sgmCPU(CMatrix<float> &result,
            CMatrix<Color> const &leftImg, CMatrix<Color> const &rightImg)
{
  /*-----------------------------------------------------------------------
   *  Unary cost computation (Currently simple L2 color distance)
   *
   *  ** ToDo: Extend this to block-matching using L2, L1 and NCC metrics
   *  **       optional: try different block shapes (and shape combinations)
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
        // else unarycosts(x, y, i) = unaryEuclidean(leftImg(x, y), rightImg(x - i, y));
        else unarycosts(x, y, i) = unaryLxNeighbor(leftImg, rightImg, x, y, x - i, y, 7, 2);
        // else unarycosts(x, y, i) = -abs(unaryNCCNeighbor(leftImg, rightImg, x, y, x - i, y, 7));
      }
    }
    std::cout << "Precomputing unary costs... "
              << static_cast<int>((100.0f * y) / leftImg.ySize()) << "% \r"
              << std::flush;
  }
  std::cout << "Precomputing unary costs... 100%" << std::endl;

  /*-----------------------------------------------------------------------
   *  Disparity estimation (Scanline-wise message passing)
   *  ** ToDo: for Bachelors: Extend this to horizontal and vertical message
   *  **                      passing
   *  **       for Masters: Extend this to 8 directions
   *  **                    (i.e. 0,22.5,45,67.5,90,112.5,135,157.5 degrees)
   *-----------------------------------------------------------------------*/

  /* HORIZONTAL (scanline-wise) message Passing */
  std::vector<CMatrix<float> > MpqsF_(leftImg.ySize());
  std::vector<CMatrix<float> > MpqsB_(leftImg.ySize());

  std::cout << "Computing HORIZONTAL disparities... \r" << std::flush;
  for (int y = 0; y < leftImg.ySize(); ++y)
  {
    /*---------------------------------------------------------------------
     *  Forward pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsF(leftImg.xSize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    { 
      MpqsF(0, j) = 0.0f;
    }
    for (int q = 1; q < leftImg.xSize(); ++q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsF(q, j) = unarycosts(q - 1, y, 0) + MpqsF(q - 1, 0) +
              LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(q - 1, y, i) + MpqsF(q - 1, i) +
              LAMBDA * thetapq(i, j);
          if (cost < MpqsF(q, j)) {
            MpqsF(q, j) = cost;
          }
        }
      }
    }
    MpqsF_[y] = MpqsF;

    /*---------------------------------------------------------------------
     *  Backward pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsB(leftImg.xSize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
    {
      MpqsB(leftImg.xSize() - 1, j) = 0.0f;
    }
    for (int q = leftImg.xSize() - 2; q >= 0; --q)
    {
      for (int j = 0; j <= MAX_DISPARITY; ++j)
      {
        MpqsB(q, j) = unarycosts(q + 1, y, 0) + MpqsB(q + 1, 0) +
              LAMBDA * thetapq(0, j);
        for (int i = 1; i <= MAX_DISPARITY; ++i)
        {
          float cost = unarycosts(q + 1, y, i) + MpqsB(q + 1, i) +
              LAMBDA * thetapq(i, j);
          if (cost < MpqsB(q, j)) {
            MpqsB(q, j) = cost;
          }
        }
      }
    }
    MpqsB_[y] = MpqsB;
    std::cout << "Computing HORIZONTAL disparities... "
              << static_cast<int>((100.0f * y) / leftImg.ySize()) << "% \r"
              << std::flush;
  }
  std::cout << "Computing HORIZONTAL disparities...100%" << std::endl;

    
  /* VERTICAL (scanline-wise) message Passing */
  std::vector<CMatrix<float> > MpqsVF_(leftImg.xSize());
  std::vector<CMatrix<float> > MpqsVB_(leftImg.xSize());

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
    MpqsVF_[x] = MpqsVF;

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
    MpqsVB_[x] = MpqsVB;
    std::cout << "Computing VERTICAL disparities... "
              << static_cast<int>((100.0f * x) / leftImg.xSize()) << "% \r"
              << std::flush;
  }
  std::cout << "Computing VERTICAL disparities...100%" << std::endl;


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
                      + MpqsF_[y](x, 0)
                      + MpqsB_[y](x, 0)
                      + MpqsVF_[x](y, 0)
                      + MpqsVB_[x](y, 0);
      for (int i = 1; i <= MAX_DISPARITY; ++i)
      {
        float cost = unarycosts(x, y, i) 
                     + MpqsF_[y](x, i)
                     + MpqsB_[y](x, i)
                     + MpqsVF_[x](y, i)
                     + MpqsVB_[x](y, i);
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
  
  timer::start("SGM (CPU)");
  sgmCPU(result, leftImg, rightImg);
  timer::stop("SGM (CPU)");

  result.writeToFloatFile(argv[3]);

  timer::printToScreen(
      std::string(), timer::AUTO_COMMON, timer::ELAPSED_TIME);

  return 0;
}
