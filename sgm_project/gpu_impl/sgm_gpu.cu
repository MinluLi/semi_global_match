#include "CTensor.h"
#include "CMatrix.h"
#include "CVector.h"
#include "NMath.h"
#include "timer.h"

#include <cmath>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <stdlib.h>

// Block dimension of CUDA kernels
#define BLOCK_DIM 16

// NxN Patch size
static const int N_PATCH = 7;
static const int DEV = N_PATCH/2;

// Image Textures
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texLeft;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRight;

/*-------------------------------------------------------------------------
 *  Throw errors in CUDA function calls
 *-------------------------------------------------------------------------*/
#define CHECK_CUDA_ERROR(msg) _CHECK_CUDA_ERROR(__FILE__,__LINE__,msg)
inline void _CHECK_CUDA_ERROR(
    char const * const filename, int line, std::string const &msg)
{
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    std::stringstream err;
    err << filename << ":" << line << ": ";
    if (msg != "") err << msg << " - ";
    err << cudaGetErrorString(error);
    throw std::runtime_error(err.str().c_str());
  }
}

/*-------------------------------------------------------------------------
 *  Convert integer to std::string 
 *-------------------------------------------------------------------------*/
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
__device__ inline float unaryL2Squared(Color const &a, Color const &b)
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
 *   Compute euclidean L2 distance of given pixels.
 *
 *   \param a The first pixel
 *   \param b The second pixel
 *
 *   \return L2-distance of a and b
 */
/*======================================================================*/
__device__ inline float unaryEuclidean(Color const &a, Color const &b)
{
 return std::sqrt(unaryL2Squared(a, b));
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
__device__ inline float4 pixelDifference(Color const &a, Color const &b)
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
__device__ inline float pixelDotProd(float4 const &a, float4 const &b)
{
  return a.x * b.x +
         a.y * b.y +
         a.z * b.z;
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
__device__ inline float thetapq(int a, int b)
{
  return (a == b) ? 0.0f : 1.0f;
}


/*======================================================================*/
/*! 
 *   Compute unary costs with Pixel Wise Euclidean distance
 *   between two images
 *
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param leftImgD Left image aligned
 *   \param rightImgD Right image aligned
 *   \param xSize xSize of original left image
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void unaryCostEuclideanKernel(float* unaryCostsCubeD,
    Color* leftImg, Color* rightImg, int xSize, int ySize)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int d = blockDim.z * blockIdx.z + threadIdx.z;
  if( x >= xSize || y >= ySize )
    return;

  // Position of the thread in the unary costs cube array
  int offset = d + x*(MAX_DISPARITY+1) + y*(MAX_DISPARITY+1)*xSize;
  // Left and Right image pixel
  Color* imgL = (Color*)((char*) leftImg + y*xSize) + x;
  Color* imgR = (Color*)((char*) rightImg + y*xSize) + x-d;

  unaryCostsCubeD[offset] = unaryEuclidean(*imgL, *imgR);
}


/*======================================================================*/
/*! 
 *   Compute unary costs with L1-norm for pixel neighborhood
 *
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param xSize xSize of original left image
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void unaryCostL1NormKernel(float* unaryCostsCubeD,
    int xSize, int ySize)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int d = blockDim.z * blockIdx.z + threadIdx.z;
  if( x >= xSize || y >= ySize )
    return;

  // Position of the thread in the unary costs cube array
  int offset = d + x*(MAX_DISPARITY+1) + y*(MAX_DISPARITY+1)*xSize;

  // Position of the pixel in the Left and Right texture
  float uL = (float) x/ (float) xSize;
  float vL = (float) y/ (float) ySize;
  float uR = uL - (float) d / (float) xSize;
  float vR = vL;

  // Read Left image block to shared memory
  __shared__ Color sharedMemLeft[BLOCK_DIM+N_PATCH-1][BLOCK_DIM+N_PATCH-1];
  int row = threadIdx.y;
  int col = threadIdx.x;

  sharedMemLeft[row+DEV][col+DEV] = tex2D(texLeft, uL, vL); // all threads
  if (row == 0) { // upper side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+i][col+DEV] = tex2D(texLeft, uL, vL-(float)(DEV-i)/(float)ySize);
    }
    if (col == 0) { // corner 0,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+i][col+j] = tex2D(texLeft, uL-(float)(DEV-i)/(float)xSize,
                                                       vL-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 0,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+i][col+DEV+j+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                       vL-(float)(DEV-j)/(float)ySize);
        }
      }
    }
  } else if (row == BLOCK_DIM-1) { // bottom side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV+i+1][col+DEV+1] = tex2D(texLeft, uL, vL+(float)(i+1)/(float)ySize);
    }
    if (col == 0) { // corner 1,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+DEV+i+1][col+j] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                             vL-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 1,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+DEV+i+1][col+DEV+j+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                                   vL+(float)(j+1)/(float)ySize);
        }
      }
    }
  }
  if (col == 0) { // left side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV][col+i] = tex2D(texLeft, uL-(float)(DEV-i)/(float)xSize, vL);
    }
  } else if (col == BLOCK_DIM-1) { // right side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV][col+i+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize, vL);
    }
  }

  // Read Right image block to shared memory
  __shared__ Color sharedMemRight[BLOCK_DIM+N_PATCH-1][BLOCK_DIM+N_PATCH-1];

  sharedMemRight[row+DEV][col+DEV] = tex2D(texRight, uR, vR); // all threads
  if (row == 0) { // upper side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+i][col+DEV] = tex2D(texRight, uR, vR-(float)(DEV-i)/(float)ySize);
    }
    if (col == 0) { // corner 0,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+i][col+j] = tex2D(texRight, uR-(float)(DEV-i)/(float)xSize,
                                                       vR-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 0,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+i][col+DEV+j+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                       vR-(float)(DEV-j)/(float)ySize);
        }
      }
    }
  } else if (row == BLOCK_DIM-1) { // bottom side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV+i+1][col+DEV+1] = tex2D(texRight, uR, vR+(float)(i+1)/(float)ySize);
    }
    if (col == 0) { // corner 1,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+DEV+i+1][col+j] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                             vR-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 1,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+DEV+i+1][col+DEV+j+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                                   vR+(float)(j+1)/(float)ySize);
        }
      }
    }
  }
  if (col == 0) { // left side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV][col+i] = tex2D(texRight, uR-(float)(DEV-i)/(float)xSize, vR);
    }
  } else if (col == BLOCK_DIM-1) { // right side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV][col+i+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize, vR);
    }
  }

  __syncthreads();

  // Compute unary cost by reading the block shared memory
  float theta = 0.0f;
  Color colorL, colorR;
  row += DEV;
  col += DEV;

  for (int j = -DEV; j < DEV; ++j) {
    for (int k = -DEV; k < DEV; ++k) {
      colorL = sharedMemLeft[row+j][col+k];
      colorR = sharedMemRight[row+j][col+k];
      theta += abs(static_cast<float>(colorL.x) - static_cast<float>(colorR.x)) +
               abs(static_cast<float>(colorL.y) - static_cast<float>(colorR.y)) +
               abs(static_cast<float>(colorL.z) - static_cast<float>(colorR.z));
    }
  }
  unaryCostsCubeD[offset] = theta;
}


/*======================================================================*/
/*! 
 *   Compute unary costs with L2-norm for pixel neighborhood
 *
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param xSize xSize of original left image
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void unaryCostL2NormKernel(float* unaryCostsCubeD,
    int xSize, int ySize)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int d = blockDim.z * blockIdx.z + threadIdx.z;
  if( x >= xSize || y >= ySize )
    return;

  // Position of the thread in the unary costs cube array
  int offset = d + x*(MAX_DISPARITY+1) + y*(MAX_DISPARITY+1)*xSize;

  // Position of the pixel in the Left and Right texture
  float uL = (float) x/ (float) xSize;
  float vL = (float) y/ (float) ySize;
  float uR = uL - (float) d / (float) xSize;
  float vR = vL;

  // Read Left image block to shared memory
  __shared__ Color sharedMemLeft[BLOCK_DIM+N_PATCH-1][BLOCK_DIM+N_PATCH-1];
  int row = threadIdx.y;
  int col = threadIdx.x;

  sharedMemLeft[row+DEV][col+DEV] = tex2D(texLeft, uL, vL); // all threads
  if (row == 0) { // upper side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+i][col+DEV] = tex2D(texLeft, uL, vL-(float)(DEV-i)/(float)ySize);
    }
    if (col == 0) { // corner 0,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+i][col+j] = tex2D(texLeft, uL-(float)(DEV-i)/(float)xSize,
                                                       vL-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 0,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+i][col+DEV+j+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                       vL-(float)(DEV-j)/(float)ySize);
        }
      }
    }
  } else if (row == BLOCK_DIM-1) { // bottom side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV+i+1][col+DEV+1] = tex2D(texLeft, uL, vL+(float)(i+1)/(float)ySize);
    }
    if (col == 0) { // corner 1,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+DEV+i+1][col+j] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                             vL-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 1,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+DEV+i+1][col+DEV+j+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                                   vL+(float)(j+1)/(float)ySize);
        }
      }
    }
  }
  if (col == 0) { // left side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV][col+i] = tex2D(texLeft, uL-(float)(DEV-i)/(float)xSize, vL);
    }
  } else if (col == BLOCK_DIM-1) { // right side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV][col+i+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize, vL);
    }
  }

  // Read Right image block to shared memory
  __shared__ Color sharedMemRight[BLOCK_DIM+N_PATCH-1][BLOCK_DIM+N_PATCH-1];

  sharedMemRight[row+DEV][col+DEV] = tex2D(texRight, uR, vR); // all threads
  if (row == 0) { // upper side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+i][col+DEV] = tex2D(texRight, uR, vR-(float)(DEV-i)/(float)ySize);
    }
    if (col == 0) { // corner 0,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+i][col+j] = tex2D(texRight, uR-(float)(DEV-i)/(float)xSize,
                                                       vR-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 0,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+i][col+DEV+j+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                       vR-(float)(DEV-j)/(float)ySize);
        }
      }
    }
  } else if (row == BLOCK_DIM-1) { // bottom side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV+i+1][col+DEV+1] = tex2D(texRight, uR, vR+(float)(i+1)/(float)ySize);
    }
    if (col == 0) { // corner 1,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+DEV+i+1][col+j] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                             vR-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 1,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+DEV+i+1][col+DEV+j+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                                   vR+(float)(j+1)/(float)ySize);
        }
      }
    }
  }
  if (col == 0) { // left side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV][col+i] = tex2D(texRight, uR-(float)(DEV-i)/(float)xSize, vR);
    }
  } else if (col == BLOCK_DIM-1) { // right side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV][col+i+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize, vR);
    }
  }

  __syncthreads();

  // Compute unary cost by reading the block shared memory
  float theta = 0.0f;
  Color colorL, colorR;
  row += DEV;
  col += DEV;

  for (int j = -DEV; j < DEV; ++j) {
    for (int k = -DEV; k < DEV; ++k) {
      colorL = sharedMemLeft[row+j][col+k];
      colorR = sharedMemRight[row+j][col+k];
      theta += unaryL2Squared(colorL, colorR);
    }
  }
  unaryCostsCubeD[offset] = theta;
}


/*======================================================================*/
/*! 
 *   Compute unary costs with NCC for pixel neighborhood
 *
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param xSize xSize of original left image
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void unaryCostNCCKernel(float* unaryCostsCubeD,
    int xSize, int ySize)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int d = blockDim.z * blockIdx.z + threadIdx.z;
  if( x >= xSize || y >= ySize )
    return;

  // Position of the thread in the unary costs cube array
  int offset = d + x*(MAX_DISPARITY+1) + y*(MAX_DISPARITY+1)*xSize;

  // Position of the pixel in the Left and Right texture
  float uL = (float) x/ (float) xSize;
  float vL = (float) y/ (float) ySize;
  float uR = uL - (float) d / (float) xSize;
  float vR = vL;

  // Read Left image block to shared memory
  __shared__ Color sharedMemLeft[BLOCK_DIM+N_PATCH-1][BLOCK_DIM+N_PATCH-1];
  int row = threadIdx.y;
  int col = threadIdx.x;

  sharedMemLeft[row+DEV][col+DEV] = tex2D(texLeft, uL, vL); // all threads
  if (row == 0) { // upper side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+i][col+DEV] = tex2D(texLeft, uL, vL-(float)(DEV-i)/(float)ySize);
    }
    if (col == 0) { // corner 0,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+i][col+j] = tex2D(texLeft, uL-(float)(DEV-i)/(float)xSize,
                                                       vL-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 0,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+i][col+DEV+j+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                       vL-(float)(DEV-j)/(float)ySize);
        }
      }
    }
  } else if (row == BLOCK_DIM-1) { // bottom side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV+i+1][col+DEV+1] = tex2D(texLeft, uL, vL+(float)(i+1)/(float)ySize);
    }
    if (col == 0) { // corner 1,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+DEV+i+1][col+j] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                             vL-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 1,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemLeft[row+DEV+i+1][col+DEV+j+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize,
                                                                   vL+(float)(j+1)/(float)ySize);
        }
      }
    }
  }
  if (col == 0) { // left side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV][col+i] = tex2D(texLeft, uL-(float)(DEV-i)/(float)xSize, vL);
    }
  } else if (col == BLOCK_DIM-1) { // right side
    for (int i = 0; i < DEV; ++i) {
      sharedMemLeft[row+DEV][col+i+1] = tex2D(texLeft, uL+(float)(i+1)/(float)xSize, vL);
    }
  }

  // Read Right image block to shared memory
  __shared__ Color sharedMemRight[BLOCK_DIM+N_PATCH-1][BLOCK_DIM+N_PATCH-1];

  sharedMemRight[row+DEV][col+DEV] = tex2D(texRight, uR, vR); // all threads
  if (row == 0) { // upper side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+i][col+DEV] = tex2D(texRight, uR, vR-(float)(DEV-i)/(float)ySize);
    }
    if (col == 0) { // corner 0,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+i][col+j] = tex2D(texRight, uR-(float)(DEV-i)/(float)xSize,
                                                       vR-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 0,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+i][col+DEV+j+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                       vR-(float)(DEV-j)/(float)ySize);
        }
      }
    }
  } else if (row == BLOCK_DIM-1) { // bottom side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV+i+1][col+DEV+1] = tex2D(texRight, uR, vR+(float)(i+1)/(float)ySize);
    }
    if (col == 0) { // corner 1,0
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+DEV+i+1][col+j] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                             vR-(float)(DEV-j)/(float)ySize);
        }
      }
    } else if (col == BLOCK_DIM-1) { // corner 1,1
      for (int i = 0; i < DEV; ++i) {
        for (int j = 0; j < DEV; ++j) {
          sharedMemRight[row+DEV+i+1][col+DEV+j+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize,
                                                                   vR+(float)(j+1)/(float)ySize);
        }
      }
    }
  }
  if (col == 0) { // left side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV][col+i] = tex2D(texRight, uR-(float)(DEV-i)/(float)xSize, vR);
    }
  } else if (col == BLOCK_DIM-1) { // right side
    for (int i = 0; i < DEV; ++i) {
      sharedMemRight[row+DEV][col+i+1] = tex2D(texRight, uR+(float)(i+1)/(float)xSize, vR);
    }
  }

  __syncthreads();

  /**** Compute unary cost by reading the block shared memory ****/
  Color colorL, colorR;
  row += DEV;
  col += DEV;

  // Average left and right pixel
  Color avgPixelLeftImg, avgPixelRightImg;
  float avgPixelLeftImgX = 0.0f;
  float avgPixelLeftImgY = 0.0f;
  float avgPixelLeftImgZ = 0.0f;
  float avgPixelRightImgX = 0.0f;
  float avgPixelRightImgY = 0.0f;
  float avgPixelRightImgZ = 0.0f;

  for (int j = -DEV; j < DEV; ++j) {
    for (int k = -DEV; k < DEV; ++k) {
      colorL = sharedMemLeft[row+j][col+k];
      colorR = sharedMemRight[row+j][col+k];
      avgPixelLeftImgX += colorL.x;
      avgPixelLeftImgY += colorL.y;
      avgPixelLeftImgZ += colorL.z;
      avgPixelRightImgX += colorR.x;
      avgPixelRightImgY += colorR.y;
      avgPixelRightImgZ += colorR.z;
    }
  }

  avgPixelLeftImg.x = avgPixelLeftImgX/(N_PATCH*N_PATCH);
  avgPixelLeftImg.y = avgPixelLeftImgY/(N_PATCH*N_PATCH);
  avgPixelLeftImg.z = avgPixelLeftImgZ/(N_PATCH*N_PATCH);
  avgPixelRightImg.x = avgPixelRightImgX/(N_PATCH*N_PATCH);
  avgPixelRightImg.y = avgPixelRightImgY/(N_PATCH*N_PATCH);
  avgPixelRightImg.z = avgPixelRightImgZ/(N_PATCH*N_PATCH);

  // Compute NCC between pixels
  float theta = 0.0f;
  float varLeftImg = 0.0f;
  float varRightImg = 0.0f;

  for (int j = -DEV; j < DEV; ++j) {
    for (int k = -DEV; k < DEV; ++k) {
      colorL = sharedMemLeft[row+j][col+k];
      colorR = sharedMemRight[row+j][col+k];

      theta += pixelDotProd(
                pixelDifference(colorL, avgPixelLeftImg),            
                pixelDifference(colorR, avgPixelRightImg));
      // Variance of left Image
      varLeftImg += unaryL2Squared(colorL, avgPixelLeftImg);
      // Variance of right Image
      varRightImg += unaryL2Squared(colorR, avgPixelRightImg);
    }
  }
  unaryCostsCubeD[offset] = theta/std::sqrt(varLeftImg*varRightImg);
}


/*======================================================================*/
/*! 
 *   Compute message passing in Horizontal Forward pass
 *
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param MqsHFCubeD Cube with message passing costs
 *   \param xSize xSize of original left image 
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void MPHFKernel(float* unaryCostsCubeD, float* MqsHFCubeD,
                           int xSize, int ySize)
{
  int d = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if( d > MAX_DISPARITY+1  || y >= ySize )
    return;

  // Shared memory to store intermediate results along the columns
  __shared__ float Mpq[MAX_DISPARITY+1];
  __shared__ float unaryCostsSharedMem[MAX_DISPARITY+1];
  // Position of the thread in the message passing cube array  
  int xdSize = xSize*(MAX_DISPARITY+1); 
  int offsetMP = d + 0*(MAX_DISPARITY+1) + y*xdSize; // x=0
  MqsHFCubeD[offsetMP] = 0.0f;
  Mpq[d] = 0.0f;
  __syncthreads();

  // Loop over the columns to pass the messages
  int offsetUC = d + 0*(MAX_DISPARITY+1) + y*xdSize;
  for (int x = 1; x < xSize; ++x) {
    offsetMP += MAX_DISPARITY+1;
    unaryCostsSharedMem[d] = unaryCostsCubeD[offsetUC];
    Mpq[d] = Mpq[d] + unaryCostsSharedMem[d] + LAMBDA*thetapq(d, 0);
    __syncthreads();

    float minCost = Mpq[d];
    for (int j = 1; j <= MAX_DISPARITY+1; ++j) {
      float cost = Mpq[j] + unaryCostsSharedMem[j] + LAMBDA*thetapq(j, d);
      if (cost < minCost) {
        minCost = cost;
      }
    }
    MqsHFCubeD[offsetMP] = minCost;
    Mpq[d] = minCost;
    offsetUC += MAX_DISPARITY+1;
    __syncthreads();
  }
}


/*======================================================================*/
/*! 
 *   Compute message passing in Horizontal Backward pass
 *
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param MqsHBCubeD Cube with message passing costs
 *   \param xSize xSize of original left image 
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void MPHBKernel(float* unaryCostsCubeD, float* MqsHBCubeD,
                           int xSize, int ySize)
{
  int d = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if( d > MAX_DISPARITY+1  || y >= ySize )
    return;

  // Shared memory to store intermediate results along the columns
  __shared__ float Mpq[MAX_DISPARITY+1];
  __shared__ float unaryCostsSharedMem[MAX_DISPARITY+1];
  // Position of the thread in the message passing cube array  
  int xdSize = xSize*(MAX_DISPARITY+1); 
  int offsetMP = d + (xSize-1)*(MAX_DISPARITY+1) + y*xdSize; // x=0
  MqsHBCubeD[offsetMP] = 0.0f;
  Mpq[d] = 0.0f;
  __syncthreads();

  // Loop over the columns to pass the messages
  int offsetUC = d + (xSize-1)*(MAX_DISPARITY+1) + y*xdSize;
  for (int x = xSize-2; x >= 0; --x) {
    offsetMP -= MAX_DISPARITY+1;
    unaryCostsSharedMem[d] = unaryCostsCubeD[offsetUC];
    Mpq[d] = Mpq[d] + unaryCostsSharedMem[d] + LAMBDA*thetapq(d, 0);
    __syncthreads();

    float minCost = Mpq[d];
    for (int j = 1; j <= MAX_DISPARITY+1; ++j) {
      float cost = Mpq[j] + unaryCostsSharedMem[j] + LAMBDA*thetapq(j, d);
      if (cost < minCost) {
        minCost = cost;
      }
    }
    MqsHBCubeD[offsetMP] = minCost;
    Mpq[d] = minCost;
    offsetUC -= MAX_DISPARITY+1;
    __syncthreads();
  }
}


/*======================================================================*/
/*! 
 *   Compute message passing in Vertical Forward pass
 *
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param MqsVFCubeD Cube with message passing costs
 *   \param xSize xSize of original left image 
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void MPVFKernel(float* unaryCostsCubeD, float* MqsVFCubeD,
                           int xSize, int ySize)
{
  int d = blockDim.x * blockIdx.x + threadIdx.x;
  int x = blockDim.y * blockIdx.y + threadIdx.y;
  if( d > MAX_DISPARITY+1  || x >= xSize )
    return;

  // Shared memory to store intermediate results along the columns
  __shared__ float Mpq[MAX_DISPARITY+1];
  __shared__ float unaryCostsSharedMem[MAX_DISPARITY+1];
  // Position of the thread in the message passing cube array
  int xdSize = xSize*(MAX_DISPARITY+1); 
  int offsetMP = d + x*(MAX_DISPARITY+1) + 0*xdSize; // y=0
  MqsVFCubeD[offsetMP] = 0.0f;
  Mpq[d] = 0.0f;
  __syncthreads();

  // Loop over the rows to pass the messages
  int offsetUC = d + x*(MAX_DISPARITY+1) + 0*xdSize;
  for (int y = 1; y < ySize; ++y) {
    offsetMP += xdSize;
    unaryCostsSharedMem[d] = unaryCostsCubeD[offsetUC];
    Mpq[d] = Mpq[d] + unaryCostsSharedMem[d] + LAMBDA*thetapq(d, 0);
    __syncthreads();

    float minCost = Mpq[d];
    for (int j = 1; j <= MAX_DISPARITY+1; ++j) {
      float cost = Mpq[j] + unaryCostsSharedMem[j] + LAMBDA*thetapq(j, d);
      if (cost < minCost) {
        minCost = cost;
      }
    }
    MqsVFCubeD[offsetMP] = minCost;
    Mpq[d] = minCost;
    offsetUC += xdSize;
    __syncthreads();
  }
}


/*======================================================================*/
/*! 
 *   Compute message passing in Vertical Backward pass
 *
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param MqsVBCubeD Cube with message passing costs
 *   \param xSize xSize of original left image 
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void MPVBKernel(float* unaryCostsCubeD, float* MqsVBCubeD,
                           int xSize, int ySize)
{
  int d = blockDim.x * blockIdx.x + threadIdx.x;
  int x = blockDim.y * blockIdx.y + threadIdx.y;
  if( d > MAX_DISPARITY+1  || x >= ySize )
    return;

  // Shared memory to store intermediate results along the columns
  __shared__ float Mpq[MAX_DISPARITY+1];
  __shared__ float unaryCostsSharedMem[MAX_DISPARITY+1];
  // Position of the thread in the message passing cube array  
  int xdSize = xSize*(MAX_DISPARITY+1); 
  int offsetMP = d + x*(MAX_DISPARITY+1) + (ySize-1)*xdSize; // y=xSize
  MqsVBCubeD[offsetMP] = 0.0f;
  Mpq[d] = 0.0f;
  __syncthreads();

  // Loop over the rows to pass the messages
  int offsetUC = d + x*(MAX_DISPARITY+1) + (ySize-1)*xdSize;
  for (int y = ySize-2; y >= 0; --y) {
    offsetMP -= xdSize;
    unaryCostsSharedMem[d] = unaryCostsCubeD[offsetUC];
    Mpq[d] = Mpq[d] + unaryCostsSharedMem[d] + LAMBDA*thetapq(d, 0);
    __syncthreads();

    float minCost = Mpq[d];
    for (int j = 1; j <= MAX_DISPARITY+1; ++j) {
      float cost = Mpq[j] + unaryCostsSharedMem[j] + LAMBDA*thetapq(j, d);
      if (cost < minCost) {
        minCost = cost;
      }
    }
    MqsVBCubeD[offsetMP] = minCost;
    Mpq[d] = minCost;
    offsetUC -= xdSize;
    __syncthreads();
  }
}


/*======================================================================*/
/*! 
 *   Compute decision of all incoming HORIZONTAL messages to every pixel
 *
 *   \param resultD Image with resulting disparity
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param MqsHFCubeD Cube with message passing costs
 *   \param MqsHBCubeD Cube with message passing costs
 *   \param xSize xSize of original left image 
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void decisionHKernel(float* resultD, float* unaryCostsCubeD,
                float* MqsHFCubeD, float* MqsHBCubeD, int xSize, int ySize)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if( x >= xSize || y >= ySize )
    return;
  
  int xdSize = xSize*(MAX_DISPARITY+1); 
  int offsetUC = 0 + x*(MAX_DISPARITY+1) + y*xdSize;

  float minCost = unaryCostsCubeD[offsetUC] + MqsHFCubeD[offsetUC] +
                  MqsHBCubeD[offsetUC];
  int minIndex = 0;
  float cost = 0.0f;
  for (int d = 1; d <= MAX_DISPARITY; ++d) {
    offsetUC += 1;
    cost = unaryCostsCubeD[offsetUC] + MqsHFCubeD[offsetUC] +
           MqsHBCubeD[offsetUC];
    if(cost < minCost) {
      minCost = cost;
      minIndex = d;
    }
  }

  int offset = x + y*xSize;
  resultD[offset] = minIndex;
}


/*======================================================================*/
/*! 
 *   Compute decision of all incoming HORIZONTAL + VERTICAL
 *   messages to every pixel
 *
 *   \param resultD Image with resulting disparity
 *   \param unaryCostsCubeD Cube with unary costs
 *   \param MqsHFCubeD Cube with Horizontal Forwards message passing costs
 *   \param MqsHBCubeD Cube with Horizontal Backwards message passing costs
 *   \param MqsVFCubeD Cube with Vertical Forwards message passing costs
 *   \param MqsVBCubeD Cube with Vertical Backwards message passing costs
 *   \param xSize xSize of original left image 
 *   \param ySize ySize of original left image 
 *
 *   \return None
 */
/*======================================================================*/
__global__ void decisionHVKernel(float* resultD, float* unaryCostsCubeD,
                float* MqsHFCubeD, float* MqsHBCubeD,
                float* MqsVFCubeD, float* MqsVBCubeD,
                int xSize, int ySize)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if( x >= xSize || y >= ySize )
    return;
  
  int xdSize = xSize*(MAX_DISPARITY+1); 
  int offsetUC = 0 + x*(MAX_DISPARITY+1) + y*xdSize;

  float minCost = unaryCostsCubeD[offsetUC] + MqsHFCubeD[offsetUC] +
                  MqsHBCubeD[offsetUC] + MqsVFCubeD[offsetUC] + 
                  MqsVBCubeD[offsetUC];
  int minIndex = 0;
  float cost = 0.0f;
  for (int d = 1; d <= MAX_DISPARITY; ++d) {
    offsetUC += 1;
    cost = unaryCostsCubeD[offsetUC] + MqsHFCubeD[offsetUC] +
           MqsHBCubeD[offsetUC] + MqsVFCubeD[offsetUC] + 
           MqsVBCubeD[offsetUC];
    if(cost < minCost) {
      minCost = cost;
      minIndex = d;
    }
  }

  int offset = x + y*xSize;
  resultD[offset] = minIndex;
}



/*======================================================================*/
/*! 
 *   Main Function
 */
/*======================================================================*/
int main(int argc, char** argv)
{
  if (argc < 4)
  {
    std::cerr << "usage: " << argv[0] << " <path to left image> <path to right "
         << "image> <disparity output path>" << std::endl;
    exit(1);
  }
  std::string outputFile(argv[3]);
  // unaryCosts map
  std::map<int, std::string> unaryCostsMap;
  unaryCostsMap.insert(std::make_pair(1, "PixelWise Euclidean"));
  unaryCostsMap.insert(std::make_pair(2, "L1"));
  unaryCostsMap.insert(std::make_pair(3, "L2"));
  unaryCostsMap.insert(std::make_pair(4, "NCC"));
  // msgPassOption map (horizontal, vertical, diagonal)
  std::map<int, std::string> msgPassOptionMap;
  msgPassOptionMap.insert(std::make_pair(1, "horizontal"));
  msgPassOptionMap.insert(std::make_pair(2, "horizontal + vertical"));
  msgPassOptionMap.insert(std::make_pair(3, "horizontal + vertical + diagonal"));

  /*-----------------------------------------------------------------------
   *  Menu with unary cost and message passing options
   *-----------------------------------------------------------------------*/
  int unaryCostOption = 0;
  while (unaryCostOption < 1 || unaryCostOption > 4) {
    std::cout << "**************************************************" << std::endl;
    std::cout << "Unary cost options:" << std::endl;
    std::cout << "1 - Pixel-wise euclidean" << std::endl;
    std::cout << "2 - L1 NxN patch" << std::endl;
    std::cout << "3 - L2 NxN patch" << std::endl;
    std::cout << "4 - NCC NxN patch" << std::endl;
    std::cin >> unaryCostOption;
  }
  std::cout << std::endl;
  
  int msgPassingOption = 0;
  std::cout << std::endl;
  while (msgPassingOption < 1 || msgPassingOption > 3) {
    std::cout << "Message Passing Options:" << std::endl;
    std::cout << "1 - Horizontal" << std::endl;
    std::cout << "2 - Horizontal + Vertical" << std::endl;
    std::cout << "3 - Horizontal + Vertical + Diagonal" << std::endl;
    std::cin >> msgPassingOption;
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
  
  /*-----------------------------------------------------------------------
   * Compute Semi-Global Matching on the GPU
   *-----------------------------------------------------------------------*/
  try {
    /*************** Unary Costs Computation *******************************/
    // Allocate matrices on the device
    Color* leftImgD;
    Color* rightImgD;
    float* resultD;

    // Allocate and copy memory of left, right and result from host to device
    cudaMalloc(&leftImgD, sizeof(Color)*leftImg.size());
    CHECK_CUDA_ERROR("Could not allocate device memory for left image");
    cudaMemcpy2D((void*) leftImgD, sizeof(Color)*leftImg.xSize(), (void*) leftImg.data(),
                 sizeof(Color)*leftImg.xSize(), sizeof(Color)*leftImg.xSize(),
                 leftImg.ySize(), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR("Could not copy input left image from host to device");

    cudaMalloc(&rightImgD, sizeof(Color)*rightImg.size());
    CHECK_CUDA_ERROR("Could not allocate device memory for left image");
    cudaMemcpy2D((void*) rightImgD, sizeof(Color)*rightImg.xSize(), (void*) rightImg.data(),
                 sizeof(Color)*rightImg.xSize(), sizeof(Color)*rightImg.xSize(),
                 rightImg.ySize(), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR("Could not copy input right image from host to device");

    cudaMalloc((void**) &resultD, sizeof(float)*leftImg.size());
    CHECK_CUDA_ERROR("Could not allocate device memory for result");

    // Bind the memory to the Left and Right images Textures
    cudaBindTexture2D( 0, texLeft, leftImgD, leftImg.xSize(), leftImg.ySize(), sizeof(Color)*leftImg.xSize());
    CHECK_CUDA_ERROR("Could not bind the left image to the texture");

    cudaBindTexture2D( 0, texRight, rightImgD, rightImg.xSize(), rightImg.ySize(), sizeof(Color)*rightImg.xSize());
    CHECK_CUDA_ERROR("Could not bind the right image to the texture");
    
    // Setup texture parameters 
    texLeft.filterMode = cudaFilterModePoint;
    texLeft.addressMode[0] = cudaAddressModeClamp;
    texLeft.addressMode[1] = cudaAddressModeClamp;
    texLeft.normalized = true;
    
    texRight.filterMode = cudaFilterModePoint;
    texRight.addressMode[0] = cudaAddressModeClamp;
    texRight.addressMode[1] = cudaAddressModeClamp;
    texRight.normalized = true;

    // Allocate memory for a unary costs Cube array in the device
    float* unaryCostsCubeD; // (d, x, y)
    cudaMalloc((void**)&unaryCostsCubeD, sizeof(float)*(MAX_DISPARITY+1)*leftImg.xSize()*leftImg.ySize());
    CHECK_CUDA_ERROR("Could not allocate device memory for unary costs cube");

    // Define Kernel blocks and Grid of blocks for unary costs
    dim3 blockUC(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 gridUC(std::ceil((float) leftImg.xSize()/(float) BLOCK_DIM),
                        std::ceil((float) leftImg.ySize()/(float) BLOCK_DIM), MAX_DISPARITY+1);

    /***************************
      Unitary Costs Kernels 
     ***************************/
    timer::start("SGM (GPU)");
    timer::start("Unary Costs (GPU)");
    switch(unaryCostOption) {
      case 1: unaryCostEuclideanKernel<<<gridUC, blockUC>>>(unaryCostsCubeD, leftImgD, rightImgD,
                                                    leftImg.xSize(), leftImg.ySize());
              break;
      case 2: unaryCostL1NormKernel<<<gridUC, blockUC>>>(unaryCostsCubeD,
                                                 leftImg.xSize(), leftImg.ySize());
              break;
      case 3: unaryCostL2NormKernel<<<gridUC, blockUC>>>(unaryCostsCubeD,
                                                 leftImg.xSize(), leftImg.ySize());
              break;
      case 4: unaryCostNCCKernel<<<gridUC, blockUC>>>(unaryCostsCubeD,
                                              leftImg.xSize(), leftImg.ySize());
              break;
    }

    cudaDeviceSynchronize(); 
    CHECK_CUDA_ERROR("Unitary Costs Kernels task has failed");
    timer::stop("Unary Costs (GPU)");

    
    /****** Horizontal Message Passing ******/
    float* MqsHFCubeD; // Horizontal Forward
    float* MqsHBCubeD; // Horizontal Backward
    /*************** Message Passing Computation *******************************/
    if (msgPassingOption == 1 || msgPassingOption == 2 || msgPassingOption == 3) {
      // Allocate memory for a message passing Cube array in the device (d, x, y)
      cudaMalloc((void**)&MqsHFCubeD, sizeof(float)*(MAX_DISPARITY+1)*leftImg.xSize()*leftImg.ySize());
      CHECK_CUDA_ERROR("Could not allocate device memory for horizontal forward cube");

      cudaMalloc((void**)&MqsHBCubeD, sizeof(float)*(MAX_DISPARITY+1)*leftImg.xSize()*leftImg.ySize());
      CHECK_CUDA_ERROR("Could not allocate device memory for horizontal backward cube");

      // Message Passing Kernerls 
      // Define Kernel blocks and Grid of blocks for HORIZONTAL message passing
      dim3 blockMPH(MAX_DISPARITY+1, 1, 1);
      dim3 gridMPH(1, leftImg.ySize(), 1);
      
      // Create streams to run the message passing kernels in parallel
      // cudaStream_t stream1, stream2;
      // cudaStreamCreate(&stream1);
      // cudaStreamCreate(&stream2);
      
      timer::start("Message Passing Horizontal (GPU)");
     // Message passing horizontal forward kernel
      MPHFKernel<<<gridMPH, blockMPH>>>(unaryCostsCubeD, MqsHFCubeD,
                                        leftImg.xSize(), leftImg.ySize());
      // Message passing horizontal backward kernel
      MPHBKernel<<<gridMPH, blockMPH>>>(unaryCostsCubeD, MqsHBCubeD,
                                        leftImg.xSize(), leftImg.ySize());
      timer::stop("Message Passing Horizontal (GPU)");
    }
 
    /****** Vertical Message Passing ******/
    float* MqsVFCubeD; // Vertical Forward
    float* MqsVBCubeD; // Vertical Backward
    if (msgPassingOption == 2 || msgPassingOption == 3) {
      // Allocate memory for a message passing Cube array in the device (d, x, y)
      cudaMalloc((void**)&MqsVFCubeD, sizeof(float)*(MAX_DISPARITY+1)*leftImg.xSize()*leftImg.ySize());
      CHECK_CUDA_ERROR("Could not allocate device memory for horizontal forward cube");

      cudaMalloc((void**)&MqsVBCubeD, sizeof(float)*(MAX_DISPARITY+1)*leftImg.xSize()*leftImg.ySize());
      CHECK_CUDA_ERROR("Could not allocate device memory for horizontal backward cube");

      // Message Passing Kernerls
      // Define Kernel blocks and Grid of blocks for HORIZONTAL message passing
      dim3 blockMPV(MAX_DISPARITY+1, 1, 1);
      dim3 gridMPV(1, leftImg.xSize(), 1);
      
      // Create streams to run the message passing kernels in parallel
      // cudaStream_t stream1, stream2;
      // cudaStreamCreate(&stream1);
      // cudaStreamCreate(&stream2);
      
      timer::start("Message Passing Vertical (GPU)");
     // Message passing vertical forward kernel
      MPVFKernel<<<gridMPV, blockMPV>>>(unaryCostsCubeD, MqsVFCubeD,
                                        leftImg.xSize(), leftImg.ySize());
      // Message passing vertical backward kernel
      MPVBKernel<<<gridMPV, blockMPV>>>(unaryCostsCubeD, MqsVBCubeD,
                                        leftImg.xSize(), leftImg.ySize());
      timer::stop("Message Passing Vertical (GPU)");
    }

    cudaDeviceSynchronize(); 
    CHECK_CUDA_ERROR("A message passing task has failed");

 

    /*************** Decision Computation *******************************/
    dim3 blockDec(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 gridDec(std::ceil((float) leftImg.xSize()/(float) BLOCK_DIM),
                        std::ceil((float) leftImg.ySize()/(float) BLOCK_DIM), 1);

    timer::start("Decision (GPU)");
    switch(msgPassingOption) {
      case 1: // Decision kernel Horizontal
              decisionHKernel<<<gridDec, blockDec>>>(resultD, unaryCostsCubeD,
                        MqsHBCubeD, MqsHFCubeD, leftImg.xSize(), leftImg.ySize());
              break;
      case 2: // Decision kernel Horizontal + Vertical
              decisionHVKernel<<<gridDec, blockDec>>>(resultD, unaryCostsCubeD,
                                MqsHBCubeD, MqsHFCubeD, MqsVFCubeD, MqsVBCubeD,
                                leftImg.xSize(), leftImg.ySize());
              break;
    }
    cudaDeviceSynchronize(); 
    CHECK_CUDA_ERROR("Decision function has failed");
    timer::stop("Decision (GPU)");

    // Copy resulting disparity map from device to host
    cudaMemcpy2D((void*)result.data(), sizeof(float)*leftImg.xSize(),
                 (void*)resultD, sizeof(float)*leftImg.xSize(), sizeof(float)*leftImg.xSize(), leftImg.ySize(),
                 cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR("Could not copy the result from device to host");
    timer::stop("SGM (GPU)");
    timer::printToScreen(std::string(), timer::AUTO_COMMON, timer::ELAPSED_TIME);

    // Free allocated memory
    cudaFree(leftImgD);
    cudaFree(rightImgD);
    cudaFree(resultD);
    cudaFree(unaryCostsCubeD);
    cudaFree(MqsHFCubeD);
    cudaFree(MqsHBCubeD);
  } catch(std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }


  /*---------------------------------------------------------------------
   *  Write results to output file 
   *  Format: [imgNumber]-[unaryCostOption]-[msgPassingOption]-gt.float3
   *---------------------------------------------------------------------*/
  std::string resultFile ("-gt.float3");
  std::string imgFileNum("");
  imgFileNum.append(outputFile.substr(0, outputFile.size()-10));
  resultFile.insert(0, NumberToString(msgPassingOption)); 
  resultFile.insert(0, NumberToString(unaryCostOption));
  resultFile.insert(0, "-");
  resultFile.insert(0, imgFileNum);
  // Write result to file
  result.writeToFloatFile(resultFile.c_str());

 /*---------------------------------------------------------------------
  *  Write result to terminal
  *---------------------------------------------------------------------*/
  std::cout << std::endl;
  if (unaryCostOption == 1) {
    std::cout << "Unary cost: " << unaryCostsMap[unaryCostOption];
  } else {
    std::cout << "Unary cost: " << unaryCostsMap[unaryCostOption];
    std::cout << " " << N_PATCH << "x" << N_PATCH << " patches";
  }

  // Compute EPE against ground truth
  // Overall EPE
  std::string dispEPE ("../bin/disp-epe ");
  dispEPE.append(resultFile);
  dispEPE.append(" ");
  dispEPE.append(outputFile);
  std::cout << ", Msg Passing: " << msgPassOptionMap[msgPassingOption] << std::endl;
  std::cout << dispEPE << std::endl;
  if (system(dispEPE.c_str()) == -1) {
    std::cerr << "Couldn't run disp-epe command" << std::endl;
    return 1;
  }

  // Non-occluded EPE
  std::string dispEPEocc(dispEPE);
  dispEPEocc.append(" ");
  dispEPEocc.append(imgFileNum);
  dispEPEocc.append("-occ.pgm");
  std::cout << dispEPEocc << std::endl;
  if (system(dispEPEocc.c_str()) == -1) {
    std::cerr << "Couldn't run disp-epe command" << std::endl;
    return 1;
  }

  // Generate PGM image from resulting disparity map
  std::cout << std::endl << "Generate PGM image from file" << std::endl;
  std::string floatToPGM ("../bin/float3-to-pgm ");
  floatToPGM.append(resultFile);
  floatToPGM.append(" ");
  floatToPGM.append(resultFile);
  floatToPGM.erase(floatToPGM.length()-6);
  floatToPGM.append("pgm");
  std::cout << floatToPGM << std::endl;
  if (system(floatToPGM.c_str()) == -1) {
    std::cerr << "Couldn't run float3-to-pgm command" << std::endl;
    return 1;
  }

  return 0;
}
