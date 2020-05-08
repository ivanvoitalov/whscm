// compile with: g++ weighted_edgelist_generator.cpp -o weighted_edgelist_generator -std=c++11 -fopenmp -O3

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <thread>
#include <omp.h>

// function to sample a random lambda from the double power-law
// distribution with the switching point at lambda = lam_c
// and the power-law exponents alpha1, aplha2
double sample_lambda(const double &alpha1, const double &alpha2,
                     const double &lam_c, const double &u){
    double A1 = (alpha2 - 1.) * (alpha1 - 1.) / ( alpha2 - 1. + std::pow(lam_c, 1.-alpha1)*(alpha1 - alpha2) );
    double A2 = A1 * std::pow(lam_c, alpha2 - alpha1);

    double lam = -1.0;
    if ( u <= A1 * (1. - std::pow(lam_c, 1.-alpha1)) / (alpha1-1.) ){
        lam = std::pow(1. - u*(alpha1-1.) / A1, 1./(1.-alpha1));
    }
    else{
        lam = std::pow(std::pow(lam_c, 1.-alpha2) - ((alpha2-1.)/A2) * ( u - (A1 / (alpha1 - 1.) ) * ( 1.-std::pow(lam_c,1.-alpha1) ) ), 1./(1.-alpha2));
    }

    return lam;
}

// function to compute connection probability between the two nodes
// with the parameters lambda, mu
double get_connection_probability(const double &lam_i, const double &lam_j,
                                  const double &mu_i, const double &mu_j,
                                  const double &R){
    return 1.0 / ( 1.0 + std::exp(2.0*R) * (mu_i + mu_j) / (lam_i * lam_j) );
}

// function to generate a weighted edge list
// according to the double power-law model
int generate_weighted_edgelist(const uint32_t &n, const double &R, const double &a,
                               const double &alpha1, const double &alpha2,
                               const double &beta1, const double &beta2, const uint32_t &seed,
                               const std::string &output_file_path,
                               const std::string &output_coords_file_path,
                               const uint32_t &n_threads, const uint32_t &verbosity){
    // check number of threads
    uint32_t correct_n_threads = 1;
    uint32_t max_n_threads = std::thread::hardware_concurrency();
    if (n_threads > max_n_threads){
        if (verbosity == 1)
            std::cout << "WARNING: Input number of threads exceeds the maximum number of threads possible simultaneously." << std::endl;
        correct_n_threads = max_n_threads;
    }
    else
        correct_n_threads = n_threads;

    if (verbosity == 1)
        std::cout << "Number of threads to be used: " << correct_n_threads << "." << std::endl;

    double lam_c = std::pow(2.*a*std::exp(2.0 * R), 1./(2. + beta1));
    if (verbosity == 1)
        std::cout << "Constant lam_c = " << lam_c << std::endl;
    // generate nodes' parameters lambda and mu
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::unordered_map< uint32_t, std::vector<double> > coordinates_map;
    if (verbosity == 1)
        std::cout << "Generating nodes' coordinates." << std::endl;
    for (uint32_t i = 0; i < n; i++){
        double u = uniform_dist(generator);
        double lam = sample_lambda(alpha1, alpha2, lam_c, u);
        double mu = -1.0;
        if (lam <= lam_c)
            mu = a * std::pow(lam, -beta1);
        else{
            mu = a * std::pow(lam_c, beta2-beta1) * std::pow(lam, -beta2);
        }
        coordinates_map[i] = {lam, mu};   
    }

    if (verbosity == 1)
        std::cout << "Coordinates generated." << std::endl;

    // save coordinates file if needed
    if (output_coords_file_path != "NONE"){
        if (verbosity == 1)
            std::cout << "Saving coordinates to: " << output_coords_file_path << std::endl;
        std::ofstream coordinates_output_file(output_coords_file_path.c_str());
        for (uint32_t i = 0; i < n; i++)
            coordinates_output_file << std::fixed << std::setprecision(12) << i << " " << coordinates_map[i][0] << " " << coordinates_map[i][1] << std::endl;
        coordinates_output_file.close();
    }
    else{
        if (verbosity == 1)
            std::cout << "WARNING: output path for the coordinates file is not provided, coordinates will not be saved." << std::endl;
    }

    // all generated links are stored here
    std::vector< std::tuple< uint32_t, uint32_t, double > > all_links;

    if (verbosity == 1)
        std::cout << "Generating links." << std::endl;

    #pragma omp parallel num_threads(correct_n_threads) 
    {
        //initialize random number generators for each thread separately
        uint32_t thread_id = omp_get_thread_num();
        std::mt19937 thread_generator (seed + thread_id + 1);
        std::uniform_real_distribution<double> thread_uniform_dist(0.0, 1.0);
        //thread-local vector of vectors storing generated links
        std::vector< std::tuple< uint32_t, uint32_t, double > > thread_links;
        #pragma omp for schedule(dynamic)
            for (uint32_t i = 0; i < n; i++){
                double lam_i = coordinates_map[i][0];
                double mu_i = coordinates_map[i][1];
                for (uint32_t j = i + 1; j < n; j++){
                    double lam_j = coordinates_map[j][0];
                    double mu_j = coordinates_map[j][1];
                    double p_ij = get_connection_probability(lam_i, lam_j,
                                                             mu_i, mu_j, R);
                    double u = thread_uniform_dist(thread_generator);
                    if (u <= p_ij){
                        // draw a link weight
                        std::exponential_distribution<double> expon_dist(mu_i + mu_j);
                        double w_ij = expon_dist(thread_generator);
                        std::tuple< uint32_t, uint32_t, double > link{i, j, w_ij};
                        thread_links.push_back(link);
                    }
                }
            }
        // aggregate all generated links
        #pragma omp critical
        {
            for (auto it = thread_links.begin(); it != thread_links.end(); ++it){
                all_links.push_back(*it);
            }
        }
    }

    // write generated edge list
    if (verbosity == 1)
        std::cout << "Writing links to the output file." << std::endl;

    std::ofstream edgelist_output_file(output_file_path.c_str());
    for (auto it = all_links.begin(); it != all_links.end(); ++it){
        edgelist_output_file << std::fixed << std::setprecision(12) << std::get<0>(*it) << " " << std::get<1>(*it) << " " << std::get<2>(*it) << std::endl;
    }

    if (verbosity == 1)
        std::cout << "Weighted edge list saved to: " << output_file_path << std::endl;
    
    return 0;
}

// function to generate a weighted edge list
// according to the double power-law model in the eta = 1 case
int generate_weighted_edgelist_eta1(const uint32_t &n, const double &R, const double &a,
                                    const double &alpha1, const double &alpha2,
                                    const double &beta1, const double &beta2, const uint32_t &seed,
                                    const std::string &output_file_path,
                                    const std::string &output_coords_file_path,
                                    const uint32_t &n_threads, const uint32_t &verbosity){
    // check number of threads
    uint32_t correct_n_threads = 1;
    uint32_t max_n_threads = std::thread::hardware_concurrency();
    if (n_threads > max_n_threads){
        if (verbosity == 1)
            std::cout << "WARNING: Input number of threads exceeds the maximum number of threads possible simultaneously." << std::endl;
        correct_n_threads = max_n_threads;
    }
    else
        correct_n_threads = n_threads;

    if (verbosity == 1)
        std::cout << "Number of threads to be used: " << correct_n_threads << "." << std::endl;

    if (verbosity == 1)
        std::cout << "Using the eta = 1 special case generator." << std::endl;
    // generate nodes' parameters lambda and mu
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::unordered_map< uint32_t, std::vector<double> > coordinates_map;
    if (verbosity == 1)
        std::cout << "Generating nodes' coordinates." << std::endl;
    for (uint32_t i = 0; i < n; i++){
        double u = uniform_dist(generator);
        double lam = std::pow(1. - u, 1./(1. - alpha1));
        double mu = a;
        coordinates_map[i] = {lam, mu};   
    }

    if (verbosity == 1)
        std::cout << "Coordinates generated." << std::endl;

    // save coordinates file if needed
    if (output_coords_file_path != "NONE"){
        if (verbosity == 1)
            std::cout << "Saving coordinates to: " << output_coords_file_path << std::endl;
        std::ofstream coordinates_output_file(output_coords_file_path.c_str());
        for (uint32_t i = 0; i < n; i++)
            coordinates_output_file << std::fixed << std::setprecision(12) << i << " " << coordinates_map[i][0] << " " << coordinates_map[i][1] << std::endl;
        coordinates_output_file.close();
    }
    else{
        if (verbosity == 1)
            std::cout << "WARNING: output path for the coordinates file is not provided, coordinates will not be saved." << std::endl;
    }

    // all generated links are stored here
    std::vector< std::tuple< uint32_t, uint32_t, double > > all_links;

    if (verbosity == 1)
        std::cout << "Generating links." << std::endl;

    #pragma omp parallel num_threads(correct_n_threads) 
    {
        //initialize random number generators for each thread separately
        uint32_t thread_id = omp_get_thread_num();
        std::mt19937 thread_generator (seed + thread_id + 1);
        std::uniform_real_distribution<double> thread_uniform_dist(0.0, 1.0);
        //thread-local vector of vectors storing generated links
        std::vector< std::tuple< uint32_t, uint32_t, double > > thread_links;
        #pragma omp for schedule(dynamic)
            for (uint32_t i = 0; i < n; i++){
                double lam_i = coordinates_map[i][0];
                for (uint32_t j = i + 1; j < n; j++){
                    double lam_j = coordinates_map[j][0];
                    double p_ij = 1./(1.0 + 2*a*std::exp(2.*R) / (lam_i * lam_j));
                    double u = thread_uniform_dist(thread_generator);
                    if (u <= p_ij){
                        // draw a link weight
                        std::exponential_distribution<double> expon_dist(2.*a);
                        double w_ij = expon_dist(thread_generator);
                        std::tuple< uint32_t, uint32_t, double > link{i, j, w_ij};
                        thread_links.push_back(link);
                    }
                }
            }
        // aggregate all generated links
        #pragma omp critical
        {
            for (auto it = thread_links.begin(); it != thread_links.end(); ++it){
                all_links.push_back(*it);
            }
        }
    }

    // write generated edge list
    if (verbosity == 1)
        std::cout << "Writing links to the output file." << std::endl;

    std::ofstream edgelist_output_file(output_file_path.c_str());
    for (auto it = all_links.begin(); it != all_links.end(); ++it){
        edgelist_output_file << std::fixed << std::setprecision(12) << std::get<0>(*it) << " " << std::get<1>(*it) << " " << std::get<2>(*it) << std::endl;
    }

    if (verbosity == 1)
        std::cout << "Weighted edge list saved to: " << output_file_path << std::endl;
    
    return 0;
}


// main
int main(int argc, char *argv[]){

    uint32_t n = std::stoi(argv[1]);
    double R = std::stod(argv[2]);
    double a = std::stod(argv[3]);
    double alpha1 = std::stod(argv[4]);
    double alpha2 = std::stod(argv[5]);
    double beta1 = std::stod(argv[6]);
    double beta2 = std::stod(argv[7]);
    uint32_t seed = std::stoi(argv[8]);
    std::string output_file_path = argv[9];
    std::string output_coords_file_path = argv[10];
    uint32_t n_threads = std::stoi(argv[11]);
    uint32_t verbosity = std::stoi(argv[12]);

    // use special case eta = 1 (when beta1 = 0, beta2 = 0, alpha1 = alpha2 = gamma)
    if (beta1 == 0) {
        int result = generate_weighted_edgelist_eta1(n, R, a, alpha1, alpha2, beta1, beta2,
                                                     seed, output_file_path, output_coords_file_path,
                                                     n_threads, verbosity);
    }


    else{
        int result = generate_weighted_edgelist(n, R, a, alpha1, alpha2, beta1, beta2,
                                                seed, output_file_path, output_coords_file_path,
                                                n_threads, verbosity);
    }

    
}