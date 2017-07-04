#include <cfloat>
#include <iostream>
#include "timer.h"

#define BLOCK_SIZE 16 
#define MATRIX_SIZE 1024 

/*
 *  Multiplies two square matrices A and B.
 *  A    first operand
 *  B    second operand
 *  C    result matrix
 *  N    the number of rows/columns of the matrices
 */
__global__ void matrix_mul_naive_kernel( const float* A, const float* B, float* C, int N )
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if( col >= N || row >= N )
    return;

  float value = 0;
  for( int i = 0; i < N; ++i )
    value += A[row*N+i] * B[i*N+col];

  C[row*N+col] = value;
}


__global__ void matrix_mul_smem_kernel( const float* A, const float* B, float* C, int N )
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if( col >= N || row >= N )
    return;

  float value = 0;

  // Alocate space for submatrices in shared memory
  // Each thread in the block has access to this memory
  __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE];

  // Compute multiplication of submatrices of A and B
  for (int i = 0; i < N/BLOCK_SIZE; i++) {
    
    // Load submatrices from device memory to shared memory
    // Each thread from the block will load one element from the
    // corresponding submatrix A and one element from the submatrix B
    As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[row*N + i*BLOCK_SIZE + threadIdx.x];
    Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(threadIdx.y + i*BLOCK_SIZE)*N + col];

    // Sync threads before moving to the computation step
    // Ensure that submatrices were loaded before trying to compute
    // the matrix multiplication, which requires accessing to submatrices
    __syncthreads();

    // Multiply submatrices
    for (int j = 0; j < BLOCK_SIZE; j++) {
        value += As[threadIdx.y*BLOCK_SIZE + j] * Bs[j*BLOCK_SIZE + threadIdx.x];
    }

    // Synchronize threads to ensure computation is done 
    // before loading two new submatrices in the next iteration
    __syncthreads();
  }

  C[row*N+col] = value;
}


// Matrix multiplication on the CPU for comparison
void matrix_mul_cpu( const float* A, const float* B, float* C, int N )
{
#pragma omp parallel for
  for( int row = 0; row < N; ++row )
  for( int col = 0; col < N; ++col )
  {
    float value = 0;
    for( int i = 0; i < N; ++i )
      value += A[row*N+i] * B[i*N+col];

    C[row*N+col] = value;
  }
}


int main( int, char** )
{
  int N = MATRIX_SIZE; // the number of rows/columns of the matrices
  float* h_A = (float*) malloc(sizeof(float)*N*N);
  float* h_B = (float*) malloc(sizeof(float)*N*N);
  float* h_C = (float*) malloc(sizeof(float)*N*N);
  float* cpu_C = (float*) malloc(sizeof(float)*N*N); // result matrix for the CPU method

  // initialize matrices A and B
  for( int i = 0; i < N*N; ++i )
  {
    h_A[i] = (2.f*rand())/RAND_MAX - 1; // random numbers in [-1,1]
    h_B[i] = (2.f*rand())/RAND_MAX - 1;
  }

  // allocate matrices on the device and initialize them
  float *d_A, *d_B, *d_C;
  cudaMalloc( (void**)&d_A, sizeof(float)*N*N );
  cudaMalloc( (void**)&d_B, sizeof(float)*N*N );
  cudaMalloc( (void**)&d_C, sizeof(float)*N*N );
  cudaMemcpy( (void*)d_A, (void*)h_A, sizeof(float)*N*N, cudaMemcpyHostToDevice );
  cudaMemcpy( (void*)d_B, (void*)h_B, sizeof(float)*N*N, cudaMemcpyHostToDevice );

  // setup a 2D grid with 2D blocks
  dim3 block(BLOCK_SIZE,BLOCK_SIZE,1);
  dim3 grid(N/BLOCK_SIZE,N/BLOCK_SIZE,1);

  timer::start("naive matrix multiplication");
  matrix_mul_naive_kernel<<<grid,block>>>( d_A, d_B, d_C, N );
  timer::stop("naive matrix multiplication"); // this function blocks until the kernel is finished
  
  timer::start("shared memory matrix multiplication");
  matrix_mul_smem_kernel<<<grid,block>>>( d_A, d_B, d_C, N );
  timer::stop("shared memory matrix multiplication"); // this function blocks until the kernel finished
  
  timer::start("cpu matrix multiplication");
  matrix_mul_cpu( h_A, h_B, cpu_C, N );
  timer::stop("cpu matrix multiplication");

  timer::printToScreen();


  // copy result back
  cudaError_t status = cudaMemcpy( (void*)h_C, (void*)d_C, sizeof(float)*N*N, cudaMemcpyDeviceToHost );
  if (status != cudaSuccess) {
    std::cout << "Copy operation failed\n";
    return 0;
  }

  // check result
  for( int row = 0; row < N; ++row )
  for( int col = 0; col < N; ++col )
  {
    float cpu_value = cpu_C[row*N+col];
    float gpu_value = h_C[row*N+col];
    if( std::abs(cpu_value-gpu_value) > 1e-4f )
    // This test is stricter but requires the compiler option '-fmad=false'
    /*if( std::abs(cpu_value-gpu_value) > FLT_EPSILON )*/
    {
      std::cout << "Something went wrong at element (" << row << ", " << col << ")"
                << "  GPU result = " << gpu_value
                << "  CPU result = " << cpu_value
                << "  difference = " << std::abs(cpu_value-gpu_value) << std::endl;
    }
  }

  // don't forget to free the memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(cpu_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
