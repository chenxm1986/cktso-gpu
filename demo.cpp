#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cktso.h"
#include "cktso-gpu.h"

bool ReadMtxFile(const char file[], int &n, int *&ap, int *&ai, double *&ax)
{
    FILE *fp = fopen(file, "r");
    if (NULL == fp)
    {
        printf("Cannot open file \"%s\".\n", file);
        return false;
    }

    char buf[256] = "\0";
    bool first = true;
    int pc = 0;
    int ptr = 0;
    while (fgets(buf, 256, fp) != NULL)
    {
        const char *p = buf;
        while (*p != '\0')
        {
            if (' ' == *p || '\t' == *p || '\r' == *p || '\n' == *p) ++p;
            else break;
        }

        if (*p == '\0') continue;
        else if (*p == '%') continue;
        else
        {
            if (first)
            {
                first = false;
                int r, c, nz;
                sscanf(p, "%d %d %d", &r, &c, &nz);
                if (r != c)
                {
                    printf("Matrix is not square because row = %d and column = %d.\n", r, c);
                    fclose(fp);
                    return false;
                }

                n = r;
                ap = new int [n + 1];
                ai = new int [nz];
                ax = new double [nz];
                if (NULL == ap || NULL == ai || NULL == ax)
                {
                    printf("Malloc for matrix failed.\n");
                    fclose(fp);
                    return false;
                }
                ap[0] = 0;
            }
            else
            {
                int r, c;
                double v;
                sscanf(p, "%d %d %lf", &r, &c, &v);
                --r;
                --c;
                ai[ptr] = r;
                ax[ptr] = v;
                if (c != pc)
                {
                    ap[c] = ptr;
                    pc = c;
                }
                ++ptr;
            }
        }
    }
    ap[n] = ptr;

    fclose(fp);
    return true;
}

double L2NormOfResidual(const int n, const int ap[], const int ai[], const double ax[], const double x[], const double b[], bool row0_col1)
{
    if (row0_col1)
    {
        double *bb = new double [n];
        memcpy(bb, b, sizeof(double) * n);
        for (int i = 0; i < n; ++i)
        {
            const double xx = x[i];
            const int start = ap[i];
            const int end = ap[i + 1];
            for (int p = start; p < end; ++p)
            {
                bb[ai[p]] -= xx * ax[p];
            }
        }
        double s = 0.;
        for (int i = 0; i < n; ++i)
        {
            s += bb[i] * bb[i];
        }
        delete []bb;
        return sqrt(s);
    }
    else
    {
        double s = 0.;
        for (int i = 0; i < n; ++i)
        {
            double r = 0.;
            const int start = ap[i];
            const int end = ap[i + 1];
            for (int p = start; p < end; ++p)
            {
                const int j = ai[p];
                r += ax[p] * x[j];
            }
            r -= b[i];
            s += r * r;
        }
        return sqrt(s);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: benchmark <mtx file>\n");
        printf("Example: benchmark add20.mtx\n");
        return -1;
    }

    int ret;
    int n;
    int *ap = NULL;
    int *ai = NULL;
    double *ax = NULL;
    ICktSo inst_cpu = NULL;
    ICktSoGpu inst_gpu = NULL;
    int *iparm_cpu, *iparm_gpu;
    const long long *oparm_cpu, *oparm_gpu;
    double *b = NULL;
    double *x = NULL;

    if (!ReadMtxFile(argv[1], n, ap, ai, ax)) goto EXIT;

    b = new double [n + n];
    x = b + n;
    if (NULL == b)
    {
        printf("Malloc for b and x failed.\n");
        goto EXIT;
    }
    for (int i = 0; i < n; ++i)
    {
        b[i] = (double)rand() / RAND_MAX * 100.;
        x[i] = 0.;
    }

    ////////////////////////////////////////////////////////////////////
    //create cpu solver instance
    ret = CKTSO_CreateSolver(&inst_cpu, &iparm_cpu, &oparm_cpu);
    if (ret < 0)
    {
        printf("Failed to create solver instance, return code = %d.\n", ret);
        goto EXIT;
    }
    iparm_cpu[0] = 1;//enable timer

    //cpu symbolic analysis
    inst_cpu->Analyze(false, n, ap, ai, ax, 0);
    printf("Analysis time = %g s.\n", oparm_cpu[0] * 1e-6);

    //cpu factorization
    inst_cpu->Factorize(ax, false);
    printf("CPU factorization time = %g s.\n", oparm_cpu[1] * 1e-6);

    //sort factors by cpu solver instance to reduce gpu accelerator initialization time
    inst_cpu->SortFactors(true);
    printf("CPU sort time = %g s.\n", oparm_cpu[3] * 1e-6);

    ////////////////////////////////////////////////////////////////////
    //create gpu accelerator instance
    ret = CKTSO_CreateGpuAccelerator(&inst_gpu, &iparm_gpu, &oparm_gpu, 0);
    if (ret != 0)
    {
        printf("Failed to create gpu accelerator instance, return code = %d.\n", ret);
        goto EXIT;
    }
    iparm_gpu[0] = 1;//enable timer

    //initialize gpu accelerator data
    ret = inst_gpu->InitializeGpuAccelerator(inst_cpu);
    if (ret != 0)
    {
        printf("Failed to initialize gpu accelerator, return code = %d.\n", ret);
        goto EXIT;
    }
    printf("GPU accelerator initialization time = %g s.\n", oparm_gpu[0] * 1e-6);
    printf("GPU memory usage = %g GB.\n", (double)oparm_gpu[4] / 1024. / 1024. / 1024.);

    //change ax values
    for (int i = 0; i < ap[n]; ++i) ax[i] *= (double)rand() / RAND_MAX * 2.;

    //refactorize matrix on gpu
    ret = inst_gpu->GpuRefactorize(ax);
    if (ret != 0)
    {
        printf("Failed to refactorize matrix on gpu, return code = %d.\n", ret);
        goto EXIT;
    }
    printf("GPU refactorization time = %g s.\n", oparm_gpu[1] * 1e-6);

    //solve on gpu
    ret = inst_gpu->GpuSolve(b, x, false);
    if (ret != 0)
    {
        printf("Failed to solve on gpu, return code = %d.\n", ret);
        goto EXIT;
    }
    printf("GPU solving time = %g s.\n", oparm_gpu[2] * 1e-6);

    //calculate error of solution
    printf("Residual = %g.\n", L2NormOfResidual(n, ap, ai, ax, x, b, false));

    ret = inst_gpu->GpuSolve(b, x, true);
    if (ret != 0)
    {
        printf("Failed to solve on gpu, return code = %d.\n", ret);
        goto EXIT;
    }
    printf("GPU transposed solving time = %g s.\n", oparm_gpu[2] * 1e-6);

    //calculate error of solution
    printf("Residual = %g.\n", L2NormOfResidual(n, ap, ai, ax, x, b, true));

EXIT:
    delete []ap;
    delete []ai;
    delete []ax;
    delete []b;
    inst_cpu->DestroySolver();
    inst_gpu->DestroyGpuAccelerator();
    return 0;
}
