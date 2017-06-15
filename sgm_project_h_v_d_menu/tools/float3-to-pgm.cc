#include <CMatrix.h>
#include <iostream>
#include <cstdlib>
using namespace std;


int main(int argc, char* argv[])
{
    if(argc<3)
    {
        cerr << "usage: float3-to-pgm <file.float3> <file.pgm>" << endl;
        exit(-1);
    }

    CMatrix<float> mat;
    mat.readFromFloatFile(argv[1]);
    mat.normalize(0,255);
    mat.writeToPGM(argv[2]);
    exit(0);
}
