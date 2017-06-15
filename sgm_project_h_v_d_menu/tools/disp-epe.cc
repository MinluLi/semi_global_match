#include <CMatrix.h>
#include <iostream>
#include <cstdlib>
#include <cassert>

using namespace std;

int main(int argc, char* argv[])
{
    if(argc<3)
    {
        cerr << "usage: disp-epe <disp.float3> <gt.float3> [occ.pgm]" << endl;
        exit(-1);
    }

    CMatrix<float> disp;
    disp.readFromFloatFile(argv[1]);

    CMatrix<float> gt;
    gt.readFromFloatFile(argv[2]);

    assert(disp.size() == gt.size());

    double epe = 0;
    if(argc==3)
    {
        for(int i=0; i<disp.size(); i++)
            epe += fabs(disp.data()[i] - gt.data()[i]);
        epe /= disp.size();
    }
    else
    {
        CMatrix<float> occ;
        occ.readFromPGM(argv[3]);

        int n = 0;
        for(int i=0; i<disp.size(); i++)
            if(occ.data()[i]==0)
            {
                epe += fabs(disp.data()[i] - gt.data()[i]);
                n++;
            }

        epe /= n;
    }

    cout << epe << endl;
    exit(0);
}
