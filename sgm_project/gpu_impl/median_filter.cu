#include "CMatrix.h"
#include "CTensor.h"

#define BLOCK_DIM 16

texture<float,cudaTextureType2D,cudaReadModeElementType> texRef;

/*
 * Detects Errors in Cuda function calls
 */
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


// 3x3 Median filter kernel
__global__ void median_filter_kernel(float *img, int x_size, int y_size) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  // Position of the thread in the image
  int offset = x + y*gridDim.x*blockDim.x;
  if (x >= x_size || y >= y_size) {
    return;
  }

  // Texture is accessed with normalized coordinates
  float u = (float) x/ (float) x_size;
  float v = (float) y/ (float) y_size;

  // Reserve space for shared memory
  __shared__ float image_shared_memory[BLOCK_DIM+2][BLOCK_DIM+2];

  // Threads write to shared memory
  image_shared_memory[threadIdx.y+1][threadIdx.x+1] = tex2D(texRef, u, v);

  if (threadIdx.y == 0) { // upper side
    image_shared_memory[threadIdx.y][threadIdx.x+1] = tex2D(texRef, u, v-1.0/y_size);
  } 
  if (threadIdx.y == BLOCK_DIM-1) { // bottom side
    image_shared_memory[threadIdx.y+2][threadIdx.x+1] = tex2D(texRef, u, v+1.0/y_size);
  }
  if (threadIdx.x == 0) { // left
    image_shared_memory[threadIdx.y+1][threadIdx.x] = tex2D(texRef, u-1.0/x_size, v);
  }
  if (threadIdx.x == BLOCK_DIM-1) { // right side
    image_shared_memory[threadIdx.y+1][threadIdx.x+2] = tex2D(texRef, u+1.0/x_size, v);
  }
  if (threadIdx.y == 0 && threadIdx.x == 0) { // corner 0,0
    image_shared_memory[threadIdx.y][threadIdx.x] = tex2D(texRef, u-1.0/x_size, v-1.0/y_size);
  }
  if (threadIdx.x == BLOCK_DIM-1 && threadIdx.y == 0) { // corner 1,0
    image_shared_memory[threadIdx.y][threadIdx.x+2] = tex2D(texRef, u+1.0/x_size, v-1.0/y_size);
  }
  if (threadIdx.x == 0 && threadIdx.y == BLOCK_DIM-1) { // corner 0,1
    image_shared_memory[threadIdx.y+2][threadIdx.x] = tex2D(texRef, u-1.0/x_size, v+1.0/y_size);
  }
  if (threadIdx.x == BLOCK_DIM-1 && threadIdx.y == BLOCK_DIM-1) { // corner 1,1
    image_shared_memory[threadIdx.y+2][threadIdx.x+2] = tex2D(texRef, u+1.0/x_size, v+1.0/y_size);
  }
  __syncthreads();

  // Build array to compute the median from shared memory
  float pixels[3*3];
  for (int i=0; i < 3; i++) {
    for (int j=0; j < 3; j++) {
      pixels[i*3+j] = image_shared_memory[threadIdx.y+i][threadIdx.x+j];
    }
  }
  
  // Sort values with bubblesort
  float tmp;
  for (int i = 0; i < 3*3; i++) {
    for (int j = 0; j < 3*3 - 1; j++) {
      if (pixels[j] > pixels[j+1]) {
        tmp = pixels[j+1];
        pixels[j+1] = pixels[j];
        pixels[j] = tmp;
      }
    }
  }

  // Take median value of an array of size 9
  img[offset] = pixels[(3*3)/2];
}


int main( int argc, char** argv )
{
  CMatrix<float> img;
  img.readFromPGM("girl_sp-noise.pgm");
  CMatrix<float> result(img.xSize(), img.ySize());

  try {
    // Allocate memory in the Device for input and result images
    float* d_in_img;
    float* d_result_img;

    size_t pitch;
    cudaMallocPitch(&d_in_img, &pitch, sizeof(float)*img.xSize(), img.ySize());
    CHECK_CUDA_ERROR("Could not allocate device memory for input image");

    cudaMalloc((void**) &d_result_img, sizeof(float)*img.size());
    CHECK_CUDA_ERROR("Could not allocate device memory for output image");

    // Copy memory to global device memory
    cudaMemcpy2D((void*) d_in_img,
                 pitch, 
                 (void*) img.data(),
                 sizeof(float)*img.xSize(),
                 sizeof(float)*img.xSize(),
                 img.ySize(),
                 cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR("Could not copy input image from host to device");
        
    // Bind input image memory to a texture
    cudaBindTexture2D( 0, texRef, d_in_img, img.xSize(), img.ySize(), sizeof(float)*img.xSize());
    CHECK_CUDA_ERROR("Could not bind the input image to the texture");
    
    // Setup the texture parameters 
    texRef.filterMode = cudaFilterModePoint;
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.normalized = true;

    // Run median filter kernel
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid(std::ceil((float) img.xSize()/(float) BLOCK_DIM),
              std::ceil((float) img.ySize()/(float) BLOCK_DIM),
              1);
    
    median_filter_kernel<<<grid, block>>>(d_result_img, img.xSize(), img.ySize());
    cudaDeviceSynchronize(); // forces buffer flush
    
    // Copy resulting image from device to host
    cudaMemcpy2D((void*)result.data(),
                 sizeof(float)*img.xSize(),
                 (void*)d_result_img,
                 pitch,
                 sizeof(float)*img.xSize(),
                 img.ySize(),
                 cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR("Could not copy the resulting image from device to host");
  }
  catch(std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Write resulting image to file
  result.writeToPGM("girl_noise_removed.pgm");

  return 0;
}

