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
 *   Compute distance of given pixels. This is a very basic L2 color
 *   comparison and has to be extended to block matching
 *
 *   \param a The first pixel
 *   \param b The second pixel
 *
 *   \return L2-distance of a and b
 */
/*======================================================================*/
inline float unary(Color const &a, Color const &b)
{
  return std::sqrt((static_cast<float>(a.x) - static_cast<float>(b.x)) *
                   (static_cast<float>(a.x) - static_cast<float>(b.x)) +
                   (static_cast<float>(a.y) - static_cast<float>(b.y)) *
                   (static_cast<float>(a.y) - static_cast<float>(b.y)) +
                   (static_cast<float>(a.z) - static_cast<float>(b.z)) *
                   (static_cast<float>(a.z) - static_cast<float>(b.z)));
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
        else unarycosts(x, y, i) = unary(leftImg(x, y), rightImg(x - i, y));
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
  std::cout << "Computing disparities... \r" << std::flush;
  for (int y = 0; y < leftImg.ySize(); ++y)
  {
    /*---------------------------------------------------------------------
     *  Forward pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsF(leftImg.xSize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j) MpqsF(0, j) = 0.0f;
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
          if (cost < MpqsF(q, j)) MpqsF(q, j) = cost;
        }
      }
    }

    /*---------------------------------------------------------------------
     *  Backward pass
     *---------------------------------------------------------------------*/
    CMatrix<float> MpqsB(leftImg.xSize(), MAX_DISPARITY + 1);
    for (int j = 0; j <= MAX_DISPARITY; ++j)
        MpqsB(leftImg.xSize() - 1, j) = 0.0f;
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
          if (cost < MpqsB(q, j)) MpqsB(q, j) = cost;
        }
      }
    }
    
    /*---------------------------------------------------------------------
     *  Decision
     *---------------------------------------------------------------------*/
    for (int q = 0; q < leftImg.xSize(); ++q)
    {
      int minIndex = 0;
      float minCost = unarycosts(q, y, 0) + MpqsF(q, 0) + MpqsB(q, 0);
      for (int i = 1; i <= MAX_DISPARITY; ++i)
      {
        float cost = unarycosts(q, y, i) + MpqsF(q, i) + MpqsB(q, i);
        if (cost < minCost)
        {
          minCost = cost;
          minIndex = i;
        }
      }
      result(q, y) = static_cast<float>(minIndex);
    }

    std::cout << "Computing disparities... "
              << static_cast<int>((100.0f * y) / leftImg.ySize()) << "% \r"
              << std::flush;
  }
  std::cout << "Computing disparities... 100%" << std::endl;  
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
