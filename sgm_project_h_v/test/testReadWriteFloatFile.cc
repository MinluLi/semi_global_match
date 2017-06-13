#include <CMatrix.h>
#include <cstdlib>

int main(int, char **)
{
  std::cout << "Testing float Array write/read... " << std::flush;
  CMatrix<float> A(10, 20);
  for (int y = 0; y < A.ySize(); ++y)
      for (int x = 0; x < A.xSize(); ++x)
          A(x, y) = static_cast<float>(std::rand()) /
              static_cast<float>(RAND_MAX);
  
  A.writeToFloatFile("debug.dat");

  CMatrix<float> B;
  B.readFromFloatFile("debug.dat");
  
  if (A.xSize() != B.xSize() || A.ySize() != B.ySize())
  {
    std::cout << "FAILED" << std::endl;
    std::cerr << "Shape mismatch: Written (" << A.xSize() << "," << A.ySize()
              << "), Read: (" << B.xSize() << "," << B.ySize() << ")"
              << std::endl;
    exit(1);
  }

  for (int y = 0; y < A.ySize(); ++y)
  {
    for (int x = 0; x < A.xSize(); ++x)
    {
      if (A(x, y) != B(x, y))
      {
        std::cout << "FAILED" << std::endl;
        std::cerr << "Error at position (" << x << "," << y << "): Wrote "
                  << A(x, y) << " but read " << B(x, y) << std::endl;
        exit(1);
      }
    }
  }           

  std::cout << "OK" << std::endl;

  return 0;
}
