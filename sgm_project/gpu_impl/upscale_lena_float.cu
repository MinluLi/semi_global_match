#include "CTensor.h"

#define THREADS_PER_BLOCK_DIM 16

texture<uchar4,cudaTextureType2D,cudaReadModeNormalizedFloat> texRef;

typedef uchar4 Color;

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


/*
 *  CTensor stores the three colors separately in a 3D array
 *  We want to store the image in a 2D matrix where each element holds the RGB value
 */
void CTensorToColorCMatrix(CMatrix<Color>& out, const CTensor<unsigned char>& in)
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


/*
 *  The inverse function to CTensorToColorMatrix()
 */
void ColorCMatrixToCTensor(CTensor<unsigned char>& out, const CMatrix<Color>& in)
{
  out.setSize(in.xSize(), in.ySize(), 3);
  for( int y = 0; y < out.ySize(); ++y )
  for( int x = 0; x < out.xSize(); ++x )
  {
    out(x,y,0) = in(x,y).x; // R
    out(x,y,1) = in(x,y).y; // G
    out(x,y,2) = in(x,y).z; // B
  }
}



__global__ void upscale_kernel( Color* img, int x_size, int y_size )
{
  // 
  // 1. Compute the normalized texture coordinates
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x >= x_size || y >= y_size) {
    return;
  }
 
  float u = (float) x/ (float) x_size;
  float v = (float) y/ (float) y_size;

  // 2. Read a texel as normalized float
  float4 texel = tex2D(texRef, u, v);

  // 3. Round to integer. Use the function lroundf()
  long color_x = lroundf(texel.x*255);
  long color_y = lroundf(texel.y*255);
  long color_z = lroundf(texel.z*255);
  
  int offset = x + y*gridDim.x*blockDim.x;
  Color* ptr = (Color*)((char*)img) + offset;
  Color color = *ptr;
  color.x = color_x;
  color.y = color_y;
  color.z = color_z;
  *ptr = color;
  
}


int main( int argc, char** argv )
{
  // Read the image from disk
  CTensor<unsigned char> tmp;
  tmp.readFromPPM("lena_small.ppm");

  // Store the image in an appropriate format
  CMatrix<Color> img;
  CTensorToColorCMatrix(img, tmp);

  CMatrix<Color> result(512,300);

  try {
    //
    // 1. Allocate memory for the input image and the result 
    Color* d_in_img;
    Color* d_result_img;

    cudaMalloc((void**) &d_in_img, sizeof(Color)*img.size());
    CHECK_CUDA_ERROR("Could not allocate device memory for input matrix");

    cudaMalloc((void**) &d_result_img, sizeof(Color)*result.size());
    CHECK_CUDA_ERROR("Could not allocate device memory for result matrix");

    // 2. Copy the input image to global device memory
    cudaMemcpy2D((void*) d_in_img,
                 sizeof(Color)*img.xSize(),
                 (void*) img.data(),
                 sizeof(Color)*img.xSize(),
                 sizeof(Color)*img.xSize(),
                 img.ySize(),
                 cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR("Could not copy input image from host to device");

    // 3. Bind the memory to the texture using cudaBindTexture()
    cudaBindTexture2D( 0, texRef, d_in_img, img.xSize(), img.ySize(), sizeof(Color)*img.xSize());
    CHECK_CUDA_ERROR("Could not bind the input image to the texture");
    
    // 4. Setup the texture parameters 
    texRef.filterMode = cudaFilterModeLinear;
    texRef.addressMode[0] = cudaAddressModeBorder;
    texRef.addressMode[1] = cudaAddressModeBorder;
    texRef.normalized = true;
        
    // 5. Define the grid and the block. Run the kernel
    dim3 block(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM, 1);
    dim3 grid(std::ceil((float) result.xSize()/(float) THREADS_PER_BLOCK_DIM),
              std::ceil((float) result.ySize()/(float) THREADS_PER_BLOCK_DIM),
              1);
    
    upscale_kernel<<<grid, block>>>(d_result_img, result.xSize(), result.ySize());

    // 6. Copy the result to the host
    cudaMemcpy2D((void*) result.data(),
                 sizeof(Color)*result.xSize(),
                 (void*) d_result_img,
                 sizeof(Color)*result.xSize(),
                 sizeof(Color)*result.xSize(), result.ySize(),
                 cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR("Could not copy result image from device to host");

    // Convert to tensor and save
    ColorCMatrixToCTensor(tmp, result);
    tmp.writeToPPM("lena_large.ppm");

    // Unbind the texture. Only a limited number of textures can be bound at the 
    // same time.
    cudaUnbindTexture( texRef );
    CHECK_CUDA_ERROR("Could not Unbind Texture");

    // Free allocated memory
    cudaFree(d_in_img);
    cudaFree(d_result_img);

  }                      
  catch(std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }


  return 0;
}
