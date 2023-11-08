#include "io.h"

//#define n_padding 6

size_t n_padding = 6;

void dump_points(float *ref,   int    ref_nb,
                 float *query, int    query_nb,
                 int dim, int iter) {
    std::string filename_ref = "points_ref";
    std::string filename_query = "points_query";

    std::ofstream outputFile;

    if (iter < 0) {
        filename_ref   += ".dat";
        filename_query += ".dat";
    }else {
        filename_ref +=    std::string(n_padding - std::min(n_padding, std::to_string(iter).length()), '0')
                         + std::to_string(iter)
                         + ".dat";

        filename_query +=   std::string(n_padding - std::min(n_padding, std::to_string(iter).length()), '0')
                          + std::to_string(iter)
                          + ".dat";
    }

    // dump ref points
    std::cout << " > Dump ref's to " << filename_ref << std::endl;

    outputFile.open(filename_ref);
    if (outputFile.is_open()) {

        outputFile << ref_nb << std::endl;
        outputFile << dim    << std::endl;

        for (int ipt=0; ipt<ref_nb; ipt++) {
            for (int d=0; d<dim; d++) {
                outputFile << std::setw(20) << std::setprecision(IOPREC) << std::scientific << ref[d*ref_nb + ipt];
            }
            outputFile << std::endl;
        }
        outputFile.close();
    }

    std::cout << "   ...done " << std::endl;

    // dump query points
    std::cout << " > Dump query's to " << filename_query << std::endl;

    outputFile.open(filename_query);
    if (outputFile.is_open()) {

        outputFile << query_nb << std::endl;
        outputFile << dim    << std::endl;

        for (int ipt=0; ipt<query_nb; ipt++) {
            for (int d=0; d<dim; d++) {
                outputFile << std::setw(20) << std::setprecision(IOPREC) << std::scientific << query[d*query_nb + ipt];
            }
            outputFile << std::endl;
        }
        outputFile.close();
    }

    std::cout << "   ...done " << std::endl;

}


void dump_area(int  *area,  int  ref_nb,
               int iter) {
    std::string filename_area = "area_ref";

    std::ofstream outputFile;

    if (iter < 0) {
        filename_area  += ".dat";
    }else {
        filename_area  +=   std::string(n_padding - std::min(n_padding, std::to_string(iter).length()), '0')
                          + std::to_string(iter)
                          + ".dat";
    }

    // dump area (ref's)
    std::cout << " > Dump area (ref, int) to " << filename_area << std::endl;

    outputFile.open(filename_area);
    if (outputFile.is_open()) {

        for (int ipt=0; ipt<ref_nb; ipt++) {
            outputFile << std::setw(20) <<  area[ipt];
            outputFile << std::endl;
        }
        outputFile.close();
    }

    std::cout << "   ...done " << std::endl;
}

void dump_knn(float *knn_dist, int *knn_index,
              int query_nb, int dim, int k, int iter) {
    std::string filename = "knn_query";

    std::ofstream outputFile;

    if (iter < 0) { 
        filename += ".dat";
    }else {
        filename +=   std::string(n_padding - std::min(n_padding, std::to_string(iter).length()), '0')
                    + std::to_string(iter)
                    + ".dat";
    }

    // dump knn
    std::cout << " > Dump knn result " << filename << std::endl;

    outputFile.open(filename);
    if (outputFile.is_open()) {

        outputFile << query_nb << std::endl;
        outputFile << dim      << std::endl;
        outputFile << k        << std::endl;

        for (int ipt=0; ipt<query_nb; ipt++) {

            for (int ik=0; ik<k; ik++) {
                outputFile << std::setw(20) << knn_index[ik*query_nb + ipt];
            }
            outputFile << std::endl;

            for (int ik=0; ik<k; ik++) {
                outputFile << std::setw(20) << std::setprecision(IOPREC) << std::scientific << knn_dist[ik*query_nb + ipt];
            }
            outputFile << std::endl;

        }
        outputFile.close();
    }

    std::cout << "   ...done " << std::endl;

}
