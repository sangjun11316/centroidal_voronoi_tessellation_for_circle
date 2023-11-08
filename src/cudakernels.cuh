#ifndef CUDAKERNELS_CUH
#define CUDAKERNELS_CUH

#include <cuda_runtime.h>

#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/sort.h"
#include "thrust/copy.h"
#include "thrust/fill.h"
#include "thrust/reduce.h"
#include "thrust/functional.h"
#include "thrust/iterator/zip_iterator.h"

#include "cooperative_groups.h"

#include "helper_cuda.h"
#include "helper_math.h"

namespace cg = cooperative_groups;

/////////////////
//   globals   //
/////////////////
// parallel reduction
__device__ void warpReduceAdd(volatile double *sdata, int tid) { // 'volatile' is important
  if (tid+32 < blockDim.x) sdata[tid] += sdata[tid + 32];
  if (tid+16 < blockDim.x) sdata[tid] += sdata[tid + 16];
  if (tid+ 8 < blockDim.x) sdata[tid] += sdata[tid +  8];
  if (tid+ 4 < blockDim.x) sdata[tid] += sdata[tid +  4];
  if (tid+ 2 < blockDim.x) sdata[tid] += sdata[tid +  2];
  if (tid+ 1 < blockDim.x) sdata[tid] += sdata[tid +  1];
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__ void reorderDataAndFindCellStartD(int     *cellStart,         // output
                                             int     *cellEnd,           // output
                                             int     *gridParticleHash,  // input
                                             int     *gridParticleIndex, // input
                                             size_t   numParticles) {
  // handle to thread block group
  cg::thread_block       cta = cg::this_thread_block();
  extern __shared__ int  sharedHash[]; // dynamic allocation -> blockSize + 1 elements
  int                    index = blockIdx.x * blockDim.x + threadIdx.x;
  int                    hash;

  if (index < numParticles) {
    hash = gridParticleHash[index];

    // Load hash data into shared memory
    sharedHash[threadIdx.x + 1] = hash;

    // first thread in block must load neighbor particle hash
    if (index > 0 && threadIdx.x == 0) {
      sharedHash[0] = gridParticleHash[index - 1];
    }
  }

  cg::sync(cta);

  if (index < numParticles) {

    if (index == 0 || hash != sharedHash[threadIdx.x]) {
      cellStart[hash] = index;

      if (index > 0) cellEnd[sharedHash[threadIdx.x]] = index;
    }

    if (index == numParticles - 1) {
      cellEnd[hash] = index + 1;
    }
  }
}

__global__ void updateRefD(float *ref, float *ji, float *energy, int *area,  // outputs
                           float *query, int *cellStart, int *cellEnd,
                           int *gridQueryIndex, float *alpha, float *beta, 
                           int numRef, int numQuery) {

  size_t indexRef = blockIdx.x * blockDim.x + threadIdx.x;
  if (indexRef >= numRef) return;

  int startIndex = cellStart[indexRef];
  if (startIndex < 0) return;

  int endIndex = cellEnd[indexRef];
  int nQuery = endIndex - startIndex;

  float xRef = ref[0*numRef + indexRef];
  float yRef = ref[1*numRef + indexRef];

  // get avg of query points
  float xQueryAvg = 0.;
  float yQueryAvg = 0.;
  float _energy   = 0.;
  for (int i=startIndex; i<endIndex; i++) {
    int queryIndex = gridQueryIndex[i];
    float xQuery = query[0*numQuery + queryIndex];
    float yQuery = query[1*numQuery + queryIndex];
    xQueryAvg += xQuery;
    yQueryAvg += yQuery;
    _energy   += (xRef - xQuery)*(xRef - xQuery) + (yRef - yQuery)*(yRef - yQuery);
  }
  xQueryAvg = xQueryAvg / (float)nQuery;
  yQueryAvg = yQueryAvg / (float)nQuery;

  // update ref points
  float xnew = (  (alpha[0]*ji[indexRef] + beta[0]) * xRef
                + (alpha[1]*ji[indexRef] + beta[1]) * xQueryAvg  ) / (ji[indexRef] + 1.);

  float ynew = (  (alpha[0]*ji[indexRef] + beta[0]) * yRef
                + (alpha[1]*ji[indexRef] + beta[1]) * yQueryAvg  ) / (ji[indexRef] + 1.);

  //printf("xref %e, xnew %e, yref %e, ynew %e, ji %e \n", ref[0*numRef+indexRef], xnew, ref[1*numRef+indexRef], ynew, ji[indexRef]);

  ref[0*numRef + indexRef] = xnew;
  ref[1*numRef + indexRef] = ynew;
  ji[indexRef]    += 1.;
  energy[indexRef] = _energy;
  area[indexRef]   = nQuery;
 
}

// Round a / b to nearest higher integer value
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads) {
  numThreads = min(blockSize, n);
  //numThreads = std::min(blockSize, n);
  numBlocks  = iDivUp(n, numThreads);
}

void reorderDataAndFindCellStart(int    *cellStart, int *cellEnd,
                                 int    *gridParticleHash, int *gridParticleIndex,
                                 size_t numParticles, int numCells) {
  int numThreads, numBlocks;
  computeGridSize(numParticles, 256, numBlocks, numThreads);

  // set all cells to empty
  checkCudaErrors(cudaMemset(cellStart, -1, numCells * sizeof(int)));

  int smemSize = sizeof(int) * (numThreads + 1);
  reorderDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>> (cellStart, cellEnd, 
                                                                     gridParticleHash, gridParticleIndex,
                                                                     numParticles);

  // check if Error
  getLastCudaError("Kernel execution failed : reorderDataAndFindCellStartD");
}

void updateRef(float *ref, float *ji, float *energy, int *area,
               float *query, int *cellStart, int *cellEnd, 
               int *gridQueryIndex, float *alpha, float *beta, 
               int numRef, int numQuery) {

  int numThreads, numBlocks;
  computeGridSize(numRef, 256, numBlocks, numThreads);

  updateRefD<<<numBlocks, numThreads>>> (ref, ji, energy, area,
                                         query, cellStart, cellEnd,
                                         gridQueryIndex, alpha, beta, 
                                         numRef, numQuery);

  // check if Error
  getLastCudaError("Kernel execution failed : updateRefD");
}


#endif
