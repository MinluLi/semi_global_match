#include <cfloat>
#include <iostream>
#include "timer.h"
#include <stdexcept>
#include <sstream>

const int BLOCK_SIZE = 32;

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

/*
 *  Transposes matrix A
 *  A     input matrix
 *  B     output matrix
 *  nRows number of rows of A
 *  nCols number of columns of A
 */
void transpose_matrix_cpu( const float* A, float* B, int nRows, int nCols)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int row = 0; row < nRows; ++row)
      for (int col = 0; col < nCols; ++col)
          B[col * nRows + row] = A[row * nCols + col];
}

/*
 *  Transposes matrix A
 *  A     input matrix
 *  B     output matrix
 *  nRows number of rows of A
 *  nCols number of columns of A
 */
__global__ void transpose_matrix_naive(
    const float* A, float* B, int nRows, int nCols)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (col >= nCols || row >= nRows) return;

  B[col * nRows + row] = A[row * nCols + col];
}


__global__ void transpose_matrix_smem(
    const float* A, float* B, int nRows, int nCols)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (col >= nCols || row >= nRows) return;

  /*
   * Fill in matrix transpose code using shared memory for
   * coalesced global memory access. For optimum performance also avoid
   * bank conflicts in shared memory
   */
  // Set shared memory of a submatrix of A
  __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];

  // Copy from global memory to shared memory row-wise -> row-wise
  As[threadIdx.x + threadIdx.y * BLOCK_SIZE] = A[row * nCols + col];

  // Wait here until the whole submatrix is filled
  __syncthreads();

  // Copy from shared memory to global memory column-wise -> row-wise
  // Ensures that access to global memory is contiguous for
  // countiguous threads
  col = blockDim.x * blockIdx.y + threadIdx.x;
  row = blockDim.y * blockIdx.x + threadIdx.y;
  nCols = nRows; 
  B[row * nCols + col] = As[threadIdx.x * BLOCK_SIZE + threadIdx.y];

}

int main( int, char** )
{
  try
  {
    int nRows = 1024, nCols = 2048;
    float *h_A, *h_B, *cpu_B;
    
    h_A = (float*) malloc(sizeof(float) * nRows * nCols);
    h_B = (float*) malloc(sizeof(float) * nRows * nCols);
    cpu_B = (float*) malloc(sizeof(float) * nRows * nCols);
    
    // initialize matrix A
    for( int i = 0; i < nRows * nCols; ++i )
        h_A[i] = (2.f * rand()) / RAND_MAX - 1; // random numbers in [-1,1]
    
    // allocate matrices on the device and initialize A
    float *d_A, *d_B;
    cudaMalloc((void**) &d_A, sizeof(float) * nRows * nCols);
    CHECK_CUDA_ERROR("Could not allocate device memory for input matrix");
    cudaMalloc((void**) &d_B, sizeof(float) * nRows * nCols);
    CHECK_CUDA_ERROR("Could not allocate device memory for transposed matrix");
    cudaMemcpy((void*) d_A, (void*) h_A, sizeof(float) * nRows * nCols,
               cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR("Could not copy matrix to device");

    // setup a 2D grid with 2D blocks
    dim3 block(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 grid(nCols/BLOCK_SIZE,nRows/BLOCK_SIZE,1);

    timer::start("naive transpose");
    transpose_matrix_naive<<<grid,block>>>(d_A, d_B, nRows, nCols);
    timer::stop("naive transpose"); // this function blocks until the kernel finished
    CHECK_CUDA_ERROR("Error in naive transpose kernel");

    timer::start("shared memory transpose");
    transpose_matrix_smem<<<grid,block>>>(d_A, d_B, nRows, nCols);
    timer::stop("shared memory transpose"); // this function blocks until the kernel finished
    CHECK_CUDA_ERROR("Error in shared memory transpose kernel");

    timer::start("cpu transpose");
    transpose_matrix_cpu(h_A, cpu_B, nRows, nCols);
    timer::stop("cpu transpose");

    timer::printToScreen();

    // copy result back
    cudaMemcpy((void*) h_B, (void*) d_B, sizeof(float) * nRows * nCols,
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR("Could not copy result matrix to host");

    // check result
    for( int row = 0; row < nRows; ++row )
        for( int col = 0; col < nCols; ++col )
        {
          float a = h_A[row * nCols + col];
          float b = cpu_B[col * nRows + row];
          if(std::abs(a-b) > 1e-4f)
          {
            std::cout << "Something went wrong at element ("
                      << row << ", " << col << ")"
                      << "  original = " << a
                      << "  transposed = " << b
                      << "  difference = " << std::abs(a-b)
                      << std::endl;
          }
          
          float cpu_value = cpu_B[row*nRows+col];
          float gpu_value = h_B[row*nRows+col];
          if(std::abs(cpu_value-gpu_value) > 1e-4f)
          {
            std::cout << "Something went wrong at element ("
                      << row << ", " << col << ")"
                      << "  GPU result = " << gpu_value
                      << "  CPU result = " << cpu_value
                      << "  difference = " << std::abs(cpu_value-gpu_value)
                      << std::endl;
          }
        }

    free(h_A);
    free(h_B);
    free(cpu_B);
    cudaFree(d_A);
    cudaFree(d_B);
  }
  catch (std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
