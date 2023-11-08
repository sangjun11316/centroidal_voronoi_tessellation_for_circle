#include "main.h"

// Program derived from Garcia's kNN implementation
// https://github.com/vincentfpgarcia/kNN-CUDA
/**
 * Initializes randomly the reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 */

// https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
template<typename T>
T uniform_random(T range_from, T range_to) {
    std::random_device                  rd;
    std::mt19937                        gen(rd());
    std::uniform_int_distribution<T>    distr(range_from, range_to);
    return distr(gen);
}

float uniform_random_float(float range_from, float range_to) {
    std::random_device                rd;
    std::mt19937                      gen(rd());
    std::uniform_real_distribution<>  distr(range_from, range_to);
    return distr(gen);
}

// Salome .dat file related
std::vector<std::string> split(std::string input, char delimiter) {
    std::vector<std::string> answer;
    std::stringstream        ss(input);
    std::string              temp;

    while (std::getline(ss, temp, delimiter)) {
        answer.push_back(temp);
    }

    return answer;
}

// https://www.geeksforgeeks.org/split-string-by-space-into-vector-in-cpp-stl/
std::vector<std::string> split2(std::string input, char delimiter) {
    std::vector<std::string> answer;

    int start, end;
    start = end = 0;

    while ((start = input.find_first_not_of(delimiter, end)) != std::string::npos) {
        end = input.find(delimiter, start);
        answer.push_back(input.substr(start,end-start));
    }

    return answer;
}

int read_pygmshHeader(std::string filename) {
  int numRef;

  std::ifstream            myFile(filename);
  std::string              line;
  std::vector<std::string> lineSplit;

  if (!myFile) {
    std::cout << "ERROR: failed to open file " << filename << std::endl;
    exit(-1);
  }

  std::getline(myFile, line);
  lineSplit = split(line, ' ');
  numRef = std::stoi(lineSplit[0]);

  return numRef;
}

void initializePygmsh(std::string filename, float *ref, int numRef) {
  std::ifstream            myFile(filename);
  std::string              line;
  std::vector<std::string> lineSplit;

  if (!myFile) {
    std::cout << "ERROR: failed to open file " << filename << std::endl;
    exit(-1);
  }

  std::getline(myFile, line); // dummy reader for header

  for (int i=0; i<numRef; i++) {
    std::getline(myFile, line); 
    lineSplit = split2(line, ' ');
    ref[0*numRef + i] = std::stof(lineSplit[0]);
    ref[1*numRef + i] = std::stof(lineSplit[1]);
  }

}

int estimateCubic_numRef(float Radius, int targetNref) {

  float spacing = sqrtf(M_PI) * Radius * sqrt(1./(float)targetNref);

  int _numberHalf = (int) ( Radius / spacing / 2. ); 

  srand(1973);
  float f_jitter = 1.;

  int count = 0;
  for (int i=-5*_numberHalf; i<5*_numberHalf; i++) {

    for (int j=-5*_numberHalf; j<5*_numberHalf; j++) {
      float jitter_x = (float)(rand() / (double)RAND_MAX) - 0.5;
      float xtmp = spacing * ((float)i + f_jitter*jitter_x);

      float jitter_y = (float)(rand() / (double)RAND_MAX) - 0.5;
      float ytmp = spacing * ((float)j + f_jitter*jitter_y);

      float rtmp = sqrtf(xtmp*xtmp + ytmp*ytmp);
      if (rtmp <= Radius) {
        count += 1;
      }

    }
  }

  return count;
}

void initializeCubic_Ref(float *ref,
                         float Radius, int targetNref, int numNref) {

  float *ref_tmp = (float *) malloc(numNref * 2 * sizeof(float));;
  float spacing = sqrtf(M_PI) * Radius * sqrt(1./(float)targetNref);

  int _numberHalf = (int) ( Radius / spacing / 2. ); 

  srand(1973);
  float f_jitter = 1.;

  int count = 0;
  for (int i=-5*_numberHalf; i<5*_numberHalf; i++) {

    for (int j=-5*_numberHalf; j<5*_numberHalf; j++) {
      float jitter_x = (float)(rand() / (double)RAND_MAX) - 0.5;
      float xtmp = spacing * ((float)i + f_jitter*jitter_x);

      float jitter_y = (float)(rand() / (double)RAND_MAX) - 0.5;
      float ytmp = spacing * ((float)j + f_jitter*jitter_y);

      float rtmp = sqrtf(xtmp*xtmp + ytmp*ytmp);
      if (rtmp <= Radius) {
        ref_tmp[0*numNref + count] = xtmp;
        ref_tmp[1*numNref + count] = ytmp;

        count += 1;
      }

    }
  }

  // shuffle for better visualization of the elements
  std::vector<int> v(numNref);
  std::iota(std::begin(v), std::end(v), 0);
  //std::random_shuffle(v.begin(), v.end(), [&](int i) {return std::rand() % i;});
  std::shuffle(std::begin(v), std::end(v), std::default_random_engine());

  for (int i=0; i<numNref; i++) {
    ref[0*numNref + i] = ref_tmp[0*numNref + v[i]];
    ref[1*numNref + i] = ref_tmp[1*numNref + v[i]];
  } 

  free(ref_tmp);
}

void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    srand(1973);

    float Radius = 1.;

    int   randMethod = 0;

    // Generate random reference points
    for (int ipt=0; ipt<ref_nb; ipt++) {
        //for (int d=0; d<dim; d++) {
            float a, b;
            if (randMethod == 0) {
                a = (float)(rand() / (double)RAND_MAX);
                b = (float)(rand() / (double)RAND_MAX);
            }else if (randMethod == 1) {
                a = (float)(uniform_random(0,ref_nb) / (double)ref_nb);
                b = (float)(uniform_random(0,ref_nb) / (double)ref_nb);
            }else if (randMethod == 2) {
                a = uniform_random_float(0.,1.);
                b = uniform_random_float(0.,1.);
            }else {
                printf("ERROR: wrong random sampling method");
            }

            if (a >= b) {
                float tmp = b;
                b = a;
                a = tmp;
            }

            //ref[d*ref_nb + ipt] = 10. * (float)(rand() / (double)RAND_MAX);
            ref[0*ref_nb + ipt] = b*Radius*cosf(2.*M_PI*a/b);
            ref[1*ref_nb + ipt] = b*Radius*sinf(2.*M_PI*a/b);
        //}
    }

    // Generate random query points
    for (int ipt=0; ipt<query_nb; ipt++) {
        //for (int d=0; d<dim; d++) {
            float a, b;
            if (randMethod == 0) {
                a = (float)(rand() / (double)RAND_MAX);
                b = (float)(rand() / (double)RAND_MAX);
            }else if (randMethod == 1) {
                a = (float)(uniform_random(0,query_nb) / (double)query_nb);
                b = (float)(uniform_random(0,query_nb) / (double)query_nb);
            }else if (randMethod == 2) {
                a = uniform_random_float(0.,1.);
                b = uniform_random_float(0.,1.);
            }else {
                printf("ERROR: wrong random sampling method");
            }

            if (a >= b) {
              float tmp = b;
              b = a;
              a = tmp;
            }

            //query[d*query_nb + ipt] = 10. * (float)(rand() / (double)RAND_MAX);
            query[0*query_nb + ipt] = b*Radius*cosf(2.*M_PI*a/b);
            query[1*query_nb + ipt] = b*Radius*sinf(2.*M_PI*a/b);
        //}
    }
}

//https://stackoverflow.com/questions/22425283/how-could-we-generate-random-numbers-in-cuda-c-with-different-seed-on-each-run
__global__ void initCurandD(curandState *crstate, unsigned long seed, int numPoints){

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numPoints) return;

    curand_init(seed, index, 0, &crstate[index]);

}

void initCurand(curandState *crstate, unsigned long seed, int numPoints) {

  int numThreads, numBlocks;
  computeGridSize(numPoints, 256, numBlocks, numThreads);

  initCurandD<<<numBlocks, numThreads>>> (crstate, seed, numPoints);

  // check if Error
  getLastCudaError("Kernel execution failed : initCurandD");

}


__global__ void randomSampleCircleD(curandState *crstate, float *dArr, float Radius, int numPoints) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numPoints) return;

  float a = 1. - curand_uniform(&crstate[index]);  // to include zero, and exclude one
  float b = 1. - curand_uniform(&crstate[index]);

  //double a = 1. - curand_uniform_double(&crstate[index]);
  //double b = 1. - curand_uniform_double(&crstate[index]);

  if (a >= b) {
    float tmp = b;
    b = a;
    a = tmp;
  }

  dArr[0*numPoints + index] = b*Radius*cosf(2.*CUDART_PI*a/b);
  dArr[1*numPoints + index] = b*Radius*sinf(2.*CUDART_PI*a/b);

}

void randomSampleCircle(curandState *crstate, float *dArr, float Radius, int numPoints) {

  int numThreads, numBlocks;
  computeGridSize(numPoints, 256, numBlocks, numThreads);

  randomSampleCircleD<<<numBlocks, numThreads>>> (crstate, dArr, Radius, numPoints);

  // check if Error
  getLastCudaError("Kernel execution failed : randomSampleCircleD");

}

void copyQueryLocal(float *query_local, int query_nb_local, 
                    float *query,       int query_nb, 
                    int ichunk, int dim) {

    for (int ipt=0; ipt<query_nb_local; ipt++) {
        for (int id=0; id<dim; id++) {

            query_local[id*query_nb_local + ipt] = query[id*query_nb + (ichunk*query_nb_local + ipt)];
        }
    }
}

__global__ void copyQueryLocalDeviceD(float *d_query_local, int query_nb_local, 
                                      float *d_query,       int query_nb,
                                      int ichunk, int dim) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= query_nb_local) return;

  d_query_local[0*query_nb_local + index] = d_query[0*query_nb + (ichunk*query_nb_local + index)]; 
  d_query_local[1*query_nb_local + index] = d_query[1*query_nb + (ichunk*query_nb_local + index)]; 

}

void copyQueryLocalDevice(float *d_query_local, int query_nb_local, 
                          float *d_query,       int query_nb,
                          int ichunk, int dim) {

  int numThreads, numBlocks;
  computeGridSize(query_nb_local, 256, numBlocks, numThreads);

  copyQueryLocalDeviceD<<<numBlocks, numThreads>>> (d_query_local, query_nb_local,
                                                    d_query, query_nb,
                                                    ichunk, dim);

  // check if Error
  getLastCudaError("Kernel execution failed : copyQueryLocalDeviceD");
}

__global__ void copyQueryModDeviceD(float *d_query_mod, int query_nb_mod, 
                                    float *d_query,     int query_nb,
                                    int query_nb_local, int ichunk, int dim) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= query_nb_mod) return;

  d_query_mod[0*query_nb_mod + index] = d_query[0*query_nb + (ichunk*query_nb_local + index)]; 
  d_query_mod[1*query_nb_mod + index] = d_query[1*query_nb + (ichunk*query_nb_local + index)]; 

}

void copyQueryModDevice(float *d_query_mod, int query_nb_mod, 
                        float *d_query,     int query_nb,
                        int query_nb_local, int ichunk, int dim) {

  int numThreads, numBlocks;
  computeGridSize(query_nb_mod, 256, numBlocks, numThreads);

  copyQueryModDeviceD<<<numBlocks, numThreads>>> (d_query_mod, query_nb_mod,
                                                  d_query, query_nb,
                                                  query_nb_local, ichunk, dim);

  // check if Error
  getLastCudaError("Kernel execution failed : copyQueryModDeviceD");
}

void pushKnnLocal(float *knn_dist,       int *knn_index,
                  float *knn_dist_local, int *knn_index_local, 
                  int ichunk, int query_nb, int query_nb_local, int k) {

    for (int ipt=0; ipt<query_nb_local; ipt++) {
        for (int ik=0; ik<k; ik++) {
            knn_dist [ik*query_nb + (ichunk*query_nb_local + ipt)] = knn_dist_local [ik*query_nb_local + ipt];
            knn_index[ik*query_nb + (ichunk*query_nb_local + ipt)] = knn_index_local[ik*query_nb_local + ipt];
        }
    }
}

__global__ void pushKnnLocalDeviceD(float *d_knn_dist,         int *d_knn_index,  int *d_knn_index2,
                                    float *d_dist_localPitch,  int *d_index_localPitch,
                                    int ichunk, int query_nb, int query_nb_local, int k) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= query_nb_local) return;

  d_knn_dist  [0*query_nb + (ichunk*query_nb_local + index)] = d_dist_localPitch [0*query_nb_local + index]; 
  d_knn_index [0*query_nb + (ichunk*query_nb_local + index)] = d_index_localPitch[0*query_nb_local + index]; 
  d_knn_index2[0*query_nb + (ichunk*query_nb_local + index)] = d_index_localPitch[0*query_nb_local + index]; 
}

void pushKnnLocalDevice(float *d_knn_dist,        int *d_knn_index, int *d_knn_index2,
                        float *d_dist_localPitch, int *d_index_localPitch,
                        int ichunk, int query_nb, int query_nb_local, int k) {

  int numThreads, numBlocks;
  computeGridSize(query_nb_local, 256, numBlocks, numThreads);

  pushKnnLocalDeviceD<<<numBlocks, numThreads>>> (d_knn_dist, d_knn_index, d_knn_index2,
                                                  d_dist_localPitch, d_index_localPitch,
                                                  ichunk, query_nb, query_nb_local, k);

  // check if Error
  getLastCudaError("Kernel execution failed : pushKnnLocalDeviceD");
}

__global__ void pushKnnModDeviceD(float *d_knn_dist,         int *d_knn_index,  int *d_knn_index2,
                                  float *d_dist_modPitch,  int *d_index_modPitch,
                                  int ichunk, int query_nb, int query_nb_local, int query_nb_mod, int k) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= query_nb_mod) return;

  d_knn_dist  [0*query_nb + (ichunk*query_nb_local + index)] = d_dist_modPitch [0*query_nb_local + index]; 
  d_knn_index [0*query_nb + (ichunk*query_nb_local + index)] = d_index_modPitch[0*query_nb_local + index]; 
  d_knn_index2[0*query_nb + (ichunk*query_nb_local + index)] = d_index_modPitch[0*query_nb_local + index]; 
}

void pushKnnModDevice(float *d_knn_dist,        int *d_knn_index, int *d_knn_index2,
                      float *d_dist_modPitch, int *d_index_modPitch,
                      int ichunk, int query_nb, int query_nb_local, int query_nb_mod, int k) {

  int numThreads, numBlocks;
  computeGridSize(query_nb_mod, 256, numBlocks, numThreads);

  pushKnnModDeviceD<<<numBlocks, numThreads>>> (d_knn_dist, d_knn_index, d_knn_index2,
                                                d_dist_modPitch, d_index_modPitch,
                                                ichunk, query_nb, query_nb_local, query_nb_mod, k);

  // check if Error
  getLastCudaError("Kernel execution failed : pushKnnModDeviceD");
}

void checkDeviceMemory() {
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

  if ( cudaSuccess != cuda_status ){
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(1);
  }

  double free_db = (double)free_byte ;
  double total_db = (double)total_byte ;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

int main(int argc, char **argv) {

    // Parameters
    int       ref_nb_target   = atoi(argv[1]);
    int       query_ref_ratio = atoi(argv[2]);
    int       max_iter        = atoi(argv[3]);
    const int dim      = 2;
    const int k        = 1;

    float     Radius   = 1.;

    // Display
    printf("Centroidal Voronoi Tessellation for a circle\n");
    printf("- Radius %e \n", Radius);
    printf("PARAMETERS\n");
    printf("- Number reference points : %d\n",   ref_nb_target);
    printf("-  query / ref ratio      : %d\n",   query_ref_ratio);
    printf("- Dimension of points     : %d\n",   dim);
    printf("- Number of neighbors     : %d\n\n", k);

    // estimate ref_nb (assuming cubic regular grid)
    printf("Estimations\n");
    // Cubic-mesh based
    int ref_nb_cubic = estimateCubic_numRef(Radius, ref_nb_target);
    printf("- ref_nb_cubic  : %d \n", ref_nb_cubic);
    printf("  -> ref_nb is replaced by this value\n");
    int ref_nb = ref_nb_cubic;

    /* PyGmsh based
    std::string file_pygmsh = "points_pygmsh.dat";
    int ref_nb_pygmsh = read_pygmshHeader(file_pygmsh);
    printf("ref_nb_pygmsh %d \n", ref_nb_pygmsh);
    int ref_nb = ref_nb_pygmsh;
    */

    //int ref_nb = ref_nb_target;

    int query_nb = query_ref_ratio * ref_nb;

    // Some estimations
    int  max_size = 1000000000;

    int query_nb_local = max_size / ref_nb;
    int query_chunk;
    int query_nb_mod;
    if (query_nb <= query_nb_local) {
        query_nb_local = query_nb;
        query_chunk    = 1;
        query_nb_mod   = 0;
    }else { // with bigger problem, I have to divide 'query' points (sample points)
        query_chunk    = query_nb / query_nb_local;
        query_nb_mod   = query_nb % query_nb_local;
    }

    printf("max_size       %d \n", max_size);
    printf("ref_nb         %d \n", ref_nb);
    printf("query_nb       %d \n", query_nb);
    printf("query_nb_local %d \n", query_nb_local);
    printf("query_chunk    %d \n", query_chunk);
    printf("query_nb_mod   %d \n", query_nb_mod);

    // Allocate input points and output k-NN distances / indexes
    float *ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float *query      = (float*) malloc(query_nb * dim * sizeof(float));
    float *knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   *knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));

    int   *area       = (int*)   malloc(ref_nb * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index ) {
        printf("Error: Memory allocation error in CPU\n"); 

        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    //printf("Initializing\n");
    //initialize_data(ref, ref_nb, query, query_nb, dim);
    initializeCubic_Ref(ref, Radius, ref_nb_target, ref_nb);
    //initializePygmsh(file_pygmsh, ref, ref_nb);
    //printf("  ... done\n");

    // Allocate for GPU
    thrust::device_vector<float> d_ref      (ref, ref + dim*ref_nb);
    thrust::device_vector<float> d_query    (query, query + dim*query_nb);
    thrust::device_vector<float> d_knn_dist (knn_dist,  knn_dist+query_nb);
    thrust::device_vector<int>   d_knn_index(knn_index, knn_index+query_nb);
    thrust::device_vector<int>   d_knn_index2(d_knn_index.begin(), d_knn_index.end()); // to store the original order

    thrust::device_vector<float> d_query_local     (dim*query_nb_local);
    thrust::device_vector<float> d_knn_dist_local  (query_nb_local);
    thrust::device_vector<int>   d_knn_index_local (query_nb_local);

    thrust::device_vector<float> d_query_mod    (dim*query_nb_mod);
    thrust::device_vector<float> d_knn_dist_mod (query_nb_mod);
    thrust::device_vector<int>   d_knn_index_mod(query_nb_mod);

    // needed for sorting & Du's CVT algorithm
    thrust::device_vector<int>  d_queryIndex(query_nb);
    thrust::device_vector<int>  d_cellStart(ref_nb);
    thrust::device_vector<int>  d_cellEnd(ref_nb);

    thrust::device_vector<float> d_alpha(2); d_alpha[0] = 0.5; d_alpha[1] = 0.5;
    thrust::device_vector<float> d_beta (2); d_beta [0] = 0.5; d_beta [1] = 0.5;
    thrust::device_vector<float> d_ji(ref_nb); thrust::fill(d_ji.begin(), d_ji.end(), 1.);

    // for diagnostics
    thrust::device_vector<float> d_energy(ref_nb);
    thrust::device_vector<int>   d_area  (ref_nb);

    // Allocate Pitches in Advance
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);

    // Return variables
    cudaError_t err0, err1, err2, err3;

    // Allocate global memory
    float * d_refPitch         = NULL;
    float * d_query_localPitch = NULL;
    float * d_dist_localPitch  = NULL;
    int   * d_index_localPitch = NULL;

    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes_local;
    size_t  dist_pitch_in_bytes_local;
    size_t  index_pitch_in_bytes_local;
    err0 = cudaMallocPitch((void**)&d_refPitch,   &ref_pitch_in_bytes,   ref_nb   * size_of_float, dim);
    err1 = cudaMallocPitch((void**)&d_query_localPitch, &query_pitch_in_bytes_local, query_nb_local * size_of_float, dim);
    err2 = cudaMallocPitch((void**)&d_dist_localPitch,  &dist_pitch_in_bytes_local,  query_nb_local * size_of_float, ref_nb);
    err3 = cudaMallocPitch((void**)&d_index_localPitch, &index_pitch_in_bytes_local, query_nb_local * size_of_int,   k);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        printf("ERROR: Memory allocation error in knn_cuda_global (locals)\n");
        cudaFree(d_refPitch);
        cudaFree(d_query_localPitch);
        cudaFree(d_dist_localPitch);
        cudaFree(d_index_localPitch); 
        return false;
    }

    // Deduce pitch values
    size_t ref_pitch         = ref_pitch_in_bytes         / size_of_float;
    size_t query_pitch_local = query_pitch_in_bytes_local / size_of_float;
    size_t dist_pitch_local  = dist_pitch_in_bytes_local  / size_of_float;
    size_t index_pitch_local = index_pitch_in_bytes_local / size_of_int;

    // Check pitch values
    if (query_pitch_local != dist_pitch_local || query_pitch_local != index_pitch_local) {
        printf("ERROR: Invalid pitch value (locals)\n");
        cudaFree(d_refPitch);
        cudaFree(d_query_localPitch);
        cudaFree(d_dist_localPitch);
        cudaFree(d_index_localPitch); 
        return false; 
    }

    float * d_query_modPitch   = NULL;
    float * d_dist_modPitch    = NULL;
    int   * d_index_modPitch   = NULL;

    size_t  query_pitch_in_bytes_mod;
    size_t  dist_pitch_in_bytes_mod;
    size_t  index_pitch_in_bytes_mod;
    err1 = cudaMallocPitch((void**)&d_query_modPitch, &query_pitch_in_bytes_mod, query_nb_mod * size_of_float, dim);
    err2 = cudaMallocPitch((void**)&d_dist_modPitch,  &dist_pitch_in_bytes_mod,  query_nb_mod * size_of_float, ref_nb);
    err3 = cudaMallocPitch((void**)&d_index_modPitch, &index_pitch_in_bytes_mod, query_nb_mod * size_of_int,   k);
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        printf("ERROR: Memory allocation error in knn_cuda_global (mods)\n");
        cudaFree(d_query_modPitch);
        cudaFree(d_dist_modPitch);
        cudaFree(d_index_modPitch); 
        return false;
    }

    // Deduce pitch values
    size_t query_pitch_mod = query_pitch_in_bytes_mod / size_of_float;
    size_t dist_pitch_mod  = dist_pitch_in_bytes_mod  / size_of_float;
    size_t index_pitch_mod = index_pitch_in_bytes_mod / size_of_int;

    // Check pitch values
    if (query_pitch_mod != dist_pitch_mod || query_pitch_mod != index_pitch_mod) {
        printf("ERROR: Invalid pitch value (mods)\n");
        cudaFree(d_query_modPitch);
        cudaFree(d_dist_modPitch);
        cudaFree(d_index_modPitch); 
        return false; 
    }

    // Initialize cuRands
    curandState *d_crStateRef; 
    curandState *d_crStateQuery;
    checkCudaErrors( cudaMalloc((void**)&d_crStateRef,   ref_nb  *sizeof(curandState)) );   
    checkCudaErrors( cudaMalloc((void**)&d_crStateQuery, query_nb*sizeof(curandState)) );   

    initCurand(d_crStateRef,   1, ref_nb);
    initCurand(d_crStateQuery, 2, query_nb);

    checkDeviceMemory();

    //////////////////////////
    //    File control      //
    //////////////////////////
    std::string filename_energy = "energy.dat";

    try {
        if (std::filesystem::remove(filename_energy))
            std::cout << "previous " << filename_energy << " has been deleted.\n";
        else
            std::cout << "no previous " << filename_energy << ", do nothing.\n";
    }
    catch(const std::filesystem::filesystem_error& err) {
        std::cout << "filesystem error: " << err.what() << '\n';
    }

    //////////////////////////
    //    Iteration Loop    //
    //////////////////////////

    //randomSampleCircle(d_crStateRef, thrust::raw_pointer_cast(d_ref.data()), Radius, ref_nb);

    for (int iloop=0; iloop<max_iter; iloop++) {
        printf("%d th iteration \n", iloop);

        randomSampleCircle(d_crStateQuery, thrust::raw_pointer_cast(d_query.data()), Radius, query_nb);

        // Execute k-NN search
        struct timeval tic;
        gettimeofday(&tic, NULL);

        //--- Nearest search
        for (int ichunk=0; ichunk < query_chunk; ichunk++) {
            copyQueryLocalDevice(thrust::raw_pointer_cast(d_query_local.data()), query_nb_local, 
                                 thrust::raw_pointer_cast(d_query.data()), query_nb,
                                 ichunk, dim);        

            knn_cuda_global(d_refPitch,         thrust::raw_pointer_cast(d_ref.data()), ref_nb, 
                            d_query_localPitch, thrust::raw_pointer_cast(d_query_local.data()), query_nb_local,
                            ref_pitch_in_bytes, query_pitch_in_bytes_local,
                            dist_pitch_in_bytes_local, index_pitch_in_bytes_local,
                            dim, k, 
                            d_dist_localPitch, d_index_localPitch);

            pushKnnLocalDevice(thrust::raw_pointer_cast(d_knn_dist.data()), 
                               thrust::raw_pointer_cast(d_knn_index.data()), thrust::raw_pointer_cast(d_knn_index2.data()),
                               d_dist_localPitch, d_index_localPitch,
                               ichunk, query_nb, query_nb_local, k);
        }

        if (query_nb_mod > 0) {

            copyQueryModDevice(thrust::raw_pointer_cast(d_query_mod.data()), query_nb_mod, 
                               thrust::raw_pointer_cast(d_query.data()), query_nb,
                               query_nb_local,query_chunk, dim);

            //knn_cuda_global(dev_refPitch, ref_nb, dev_query_mod, query_nb_mod, dim, k, dev_knn_dist_modPitch, dev_knn_index_modPitch);
            knn_cuda_global(d_refPitch,         thrust::raw_pointer_cast(d_ref.data()), ref_nb, 
                            d_query_modPitch,   thrust::raw_pointer_cast(d_query_mod.data()), query_nb_mod,
                            ref_pitch_in_bytes, query_pitch_in_bytes_mod,
                            dist_pitch_in_bytes_mod, index_pitch_in_bytes_mod,
                            dim, k, 
                            d_dist_modPitch, d_index_modPitch);

            pushKnnModDevice(thrust::raw_pointer_cast(d_knn_dist.data()),
                             thrust::raw_pointer_cast(d_knn_index.data()), thrust::raw_pointer_cast(d_knn_index2.data()),
                             d_dist_modPitch, d_index_modPitch,
                             query_chunk, query_nb, query_nb_local, query_nb_mod, k);
        }

        //--- Construct Wi arrays
        // sort (only for k=1 case)
        thrust::sequence(d_queryIndex.begin(), d_queryIndex.end());

        thrust::sort_by_key(d_knn_index2.begin(),
                            d_knn_index2.end(),
                            d_queryIndex.begin());


        reorderDataAndFindCellStart(thrust::raw_pointer_cast(d_cellStart.data()),
                                    thrust::raw_pointer_cast(d_cellEnd.data()),
                                    thrust::raw_pointer_cast(d_knn_index2.data()),
                                    thrust::raw_pointer_cast(d_queryIndex.data()),
                                    query_nb, ref_nb);


        //--- update Ref's
        if (max_iter > 1) {  // use max_iter == 1 for diagnostics for the initial condition
            updateRef(thrust::raw_pointer_cast(d_ref.data()),
                      thrust::raw_pointer_cast(d_ji.data()),
                      thrust::raw_pointer_cast(d_energy.data()), 
                      thrust::raw_pointer_cast(d_area.data()),   
                      thrust::raw_pointer_cast(d_query.data()),
                      thrust::raw_pointer_cast(d_cellStart.data()),
                      thrust::raw_pointer_cast(d_cellEnd.data()),
                      thrust::raw_pointer_cast(d_queryIndex.data()),
                      thrust::raw_pointer_cast(d_alpha.data()),
                      thrust::raw_pointer_cast(d_beta.data()),
                      ref_nb, query_nb);
        }

        // append energy
        float avgEnergy = thrust::reduce(d_energy.begin(), d_energy.end(), 0., thrust::plus<float>());
        avgEnergy = avgEnergy / (float)ref_nb;
        std::ofstream energyFile;
        energyFile.open(filename_energy, std::ios_base::app);
        energyFile << std::setw(20) << std::setprecision(IOPREC) << std::scientific << avgEnergy << std::endl;
        cudaDeviceSynchronize();

        struct timeval toc;
        gettimeofday(&toc, NULL);
        double elapsed_time = toc.tv_sec - tic.tv_sec;
        elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
        printf("  ...done in %8.5f seconds \n", elapsed_time);

    }

    thrust::copy(d_ref.begin(), d_ref.end(), ref);
    thrust::copy(d_query.begin(), d_query.end(), query);
    thrust::copy(d_knn_dist.begin(), d_knn_dist.end(), knn_dist);
    thrust::copy(d_knn_index.begin(), d_knn_index.end(), knn_index);

    thrust::copy(d_area.begin(), d_area.end(), area);

    // Dump data
    dump_points(ref, ref_nb, query, query_nb, dim, -1);
    dump_area  (area, ref_nb, -1);
    dump_knn   (knn_dist, knn_index, query_nb, dim, k, -1);

    // Deallocate memory (!!! Need to add cudaFree's !!!)
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    return EXIT_SUCCESS;
}
