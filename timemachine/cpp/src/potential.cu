#include <iostream>

#include "potential.hpp"
#include "gpu_utils.cuh"
#include "surreal.cuh"

namespace timemachine {

void Potential::execute_host(
    const int N,
    const int P,
    const double *h_x, // [N,3]
    const double *h_p, // [P,]
    const double *h_box, // [3, 3]
    const double lambda, // [1]
    unsigned long long *h_du_dx, // [N,3]
    double *h_du_dp, // [P]
    double *h_du_dl, //
    double *h_u) {

    double *d_x;
    double *d_p;
    double *d_box;

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_x, N*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, h_x, N*D*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_p, P*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_p, h_p, P*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_box, D*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_box, h_box, D*D*sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_du_dx; // du/dx
    double *d_du_dp;
    double *d_du_dl; // du/dl
    double *d_u = nullptr; // u

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMalloc(&d_du_dx, N*D*sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_du_dx, 0, N*D*sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_du_dp, P*sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_du_dp, 0, P*sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_du_dl, sizeof(double)));
    gpuErrchk(cudaMemset(d_du_dl, 0, sizeof(double)));
    gpuErrchk(cudaMalloc(&d_u, sizeof(double)));
    gpuErrchk(cudaMemset(d_u, 0, sizeof(double)));


    this->execute_device(
        N,
        P,
        d_x, 
        d_p,
        d_box,
        lambda,
        d_du_dx,
        d_du_dp,
        d_du_dl,
        d_u,
        static_cast<cudaStream_t>(0)
    );

    gpuErrchk(cudaMemcpy(h_du_dx, d_du_dx, N*D*sizeof(*h_du_dx), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dx));
    gpuErrchk(cudaMemcpy(h_du_dp, d_du_dp, P*sizeof(*h_du_dp), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dp));
    gpuErrchk(cudaMemcpy(h_du_dl, d_du_dl, sizeof(*h_du_dl), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dl));
    gpuErrchk(cudaMemcpy(h_u, d_u, sizeof(*h_u), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_box));

};



void Potential::execute_host_du_dx(
    const int N,
    const int P,
    const double *h_x, // [N,3]
    const double *h_p, // [P,]
    const double *h_box, // [3, 3]
    const double lambda, // [1]
    unsigned long long *h_du_dx) {

    double *d_x;
    double *d_p;
    double *d_box;

    const int D = 3;

    gpuErrchk(cudaMalloc(&d_x, N*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, h_x, N*D*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_p, P*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_p, h_p, P*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_box, D*D*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_box, h_box, D*D*sizeof(double), cudaMemcpyHostToDevice));

    unsigned long long *d_du_dx; // du/dx

    // very important that these are initialized to zero since the kernels themselves just accumulate
    gpuErrchk(cudaMalloc(&d_du_dx, N*D*sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_du_dx, 0, N*D*sizeof(unsigned long long)));

    this->execute_device(
        N,
        P,
        d_x, 
        d_p,
        d_box,
        lambda,
        d_du_dx,
        nullptr,
        nullptr,
        nullptr,
        static_cast<cudaStream_t>(0)
    );

    gpuErrchk(cudaMemcpy(h_du_dx, d_du_dx, N*D*sizeof(*h_du_dx), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_du_dx));
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_box));

};

}

