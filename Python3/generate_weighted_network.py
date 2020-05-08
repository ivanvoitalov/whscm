import argparse
import numpy as np
import os
import whscm_parameters_solver as ps
import subprocess
import sys
import time

# ==========================
# ========== Main ==========
# ==========================

def main():
    parser = argparse.ArgumentParser(description =
                                     '''Script to generate weighted networks
                                     with given power-law degree and strength
                                     distributions.''')
    
    # required arguments
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-n', '--num_nodes', type = int,
                               help = 'Target number of nodes in the network.',
                               required = True)
    required_args.add_argument('-g', '--gamma', type = float,
                               help = '''Target power-law exponent of the degree
                               distribution.''',
                               required = True)
    required_args.add_argument('-e', '--eta', type = float,
                               help = '''Target strength-degree scaling
                               exponent.''',
                               required = True)
    required_args.add_argument('-o', '--output_file', type = str,
                               help = '''Path to the output file where the
                               resulting weighted edge list will be stored.''',
                               required = True)


    # optional arguments
    parser.add_argument('--params_output', type = str,
                        help = '''Output file path for the nodes\' hidden variables
                                  used to generate the network. If not specified,
                                  hidden variables will not be saved.''')
    parser.add_argument('--n_threads', type = int, default = 1,
                        help = '''Number of threads to use for network generation.
                                If larger than the number of cores available,
                                will be set to the number of available cores.
                                (default = 1)''')
    parser.add_argument('-R', type = float,
                        help = '''\'R\' parameter used to generate the network.
                                Must be provided along with the \'a\' parameter.''')
    parser.add_argument('-a', type = float,
                        help = '''\'a\' parameter used to generate the network.
                                Must be provided along with the \'R\' parameter.''')
    parser.add_argument('-k', type = float,
                        help = '''Target average degree of the network.
                                Must be provided along with the \'sigma0\' parameter.''')
    parser.add_argument('--sigma0', type = float,
                        help = '''Target value of expected strength as a function of
                                degree k rescaled by the k^{eta}. 
                                Sets the \'base level\' of the strength-degree
                                correlation curve.
                                Must be provided along with the \'k\' parameter.''')
    parser.add_argument('--solver', type = int, default = 0,
                        help = '''Type of the solver to find the (R, a) parameters.
                                  If type = 0, then the approximate solver will be used
                                  (fast option, may be not accurate for some combinations
                                  of input parameters).
                                  If type = 1, then the exact solver will be used
                                  (slow option, should be more accurate).
                                  Also see the solver performance notes at the code
                                  repository. (default = 0)''')
    parser.add_argument('-s', type = int, default = 1,
                        help = '''Seed to use for random number generator.
                                  (default = 1)''')
    parser.add_argument('-v', type = int, default = 0,
                        help = '''Flag to control the output verbosity.
                                  Set to 1 if detailed output is needed.
                                  (default = 0)''')



    args = parser.parse_args()

    # check arguments for consistency
    if args.solver != 0 and args.solver != 1:
        parser.error('''The solver parameter should be either
                     0 (fast and approximate) or 
                     1 (slow and accurate).''')
    if (args.s + args.n_threads >= 4294967295):
        parser.error('''Random seed should fit C++ uint32 type,
                        please select smaller value.''')
    if (args.v != 0) and (args.v != 1):
        parser.error('Verbosity flag should be 0 or 1.')
    if args.s <= 0:
        parser.error('Random seed should be an integer greater or equal to 1.')
    if args.num_nodes <= 0:
        parser.error('Number of nodes should be greater than 0.')
    if args.gamma <= 2.0:
        parser.error('Target gamma should be greater than 2.')
    if args.eta < 1.0:
        parser.error('''Target eta should be less or equal to 1.
                        (only super-linear scaling between strength and
                        degrees is supported)''')

    if not (args.R or args.a or args.k or args.sigma0):
        print("Parameters (-R, -a) or (-k, --sigma0) should be provided.")
        sys.exit(2)
    if (args.R or args.a) and (args.k or args.sigma0):
        print('''Parameters (-R, -a) are mutually exclusive with
                 parameters (-k, --sigma0), please pick either pair.''')
        sys.exit(2)
    if (args.R and not args.a) or (args.a and not args.R):
        if args.k and args.sigma0:
            pass
        else:
            print('''Parameters (-R, -a) or (-k, --sigma0) should be
                     provided in pairs.''')
            sys.exit(2)
    if (args.k and not args.sigma0) or (args.sigma0 and not args.k):
        if args.R and args.a:
            pass
        else:
            print('''Parameters (-R, -a) or (-k, --sigma0) should be
                     provided in pairs.''')
            sys.exit(2)

    if args.n_threads <= 0:
        parser.error('Number of threads should be greater than 0.')

    
    
    if os.path.basename(args.output_file) == "":
        parser.error('Please specify the output file path.')
    # check if the directory to which file should be saved exists
    edgelist_dir_path = os.path.abspath(os.path.dirname(args.output_file))
    if edgelist_dir_path != "":
        if not os.path.isdir(edgelist_dir_path):
            parser.error('''Directory of the provided output edge list path
                            does not exist.''')
    else:
        pass

    if args.params_output:
        if os.path.basename(args.params_output) == "":
            parser.error('''Please specify the output file path for the
                            parameters file (\'--params_output\' flag).''')
        param_file_dir_path = os.path.abspath(os.path.dirname(args.params_output))
        if param_file_dir_path != "":
            if not os.path.isdir(param_file_dir_path):
                parser.error('''Directory of the provided output parameters file
                                path does not exist.''')
        else:
            pass

    # find all the model parameters
    gamma_target = args.gamma
    eta_target = args.eta
    n = args.num_nodes

    if eta_target == 1.0:
        alpha1 = gamma_target
        alpha2 = gamma_target
        beta1 = 0.0
        beta2 = 0.0
    else:
        alpha1 = 1. + (gamma_target - 1.) * eta_target
        beta1 = (gamma_target - (gamma_target - 2.)/gamma_target) * (eta_target - 1.)
        alpha2 = 1. + ((alpha1 - 1.) / (1. + beta1)) * (1. + (gamma_target-2.)*(1. - 1./eta_target))
        beta2 = alpha2 - 1. + eta_target*(alpha2 - 1.)/(gamma_target - 1.)
        

    if args.R:
        # run generator code with given R and a
        R = args.R
        a = args.a
    else:
        # find R and a parameters from k and A
        if args.k > (n-1):
            parser.error("Target average degree should be less than n.")
            sys.exit(2)
        if args.k < 1.0:
            parser.error("Target average degree should be larger than 1.")
            sys.exit(2)
        if args.sigma0 <= 0.0:
            parser.error("Target sigma0 should be positive.")
            sys.exit(2)

        print("Searching for (R, a) solutions...")
        if args.solver == 0:
            if args.v == 1:
                R, a = ps.get_solution_approx(n, args.k, args.sigma0, gamma_target, eta_target,
                                              verbose = 1)
            else:
                R, a = ps.get_solution_approx(n, args.k, args.sigma0, gamma_target, eta_target,
                                              verbose = 0)
        elif args.solver == 1:
            if args.v == 1:
                R, a = ps.get_solution(n, args.k, args.sigma0, gamma_target, eta_target,
                                       verbose = 1)
            else:
                R, a = ps.get_solution(n, args.k, args.sigma0, gamma_target, eta_target,
                                       verbose = 0)
        if args.v == 1:
            print("Solutions found: R = %.12f, a = %.12f." % (R, a))

    
    print("******************************")
    print("Generating a weighted network using the following parameters:")
    print("n = %d" % n)
    print("R = %.12f" % R)
    print("a = %.12f" % a)
    print("alpha1 = %.8f" % alpha1)
    print("alpha2 = %.8f" % alpha2)
    print("beta1 = %.8f" % beta1)
    print("beta2 = %.8f" % beta2)
    print("******************************")

    # convert input paths to absolute paths
    output_file_path = os.path.abspath(args.output_file)
    if args.params_output:
        output_coords_file_path = os.path.abspath(args.params_output)
    else:
        output_coords_file_path = "NONE"

    # check if the folder exists and if the binary is in the folder
    python_script_dirname = os.path.dirname(os.path.realpath(__file__))
    parent_dirname = os.path.abspath(os.path.join(python_script_dirname, os.pardir))
    bin_abspath = os.path.abspath(parent_dirname+'/bin/weighted_edgelist_generator.out')
    if os.path.exists(bin_abspath):
        # if compiled, run the C++ generator
        generator_args = tuple(map(str,[n, R, a, alpha1, alpha2, beta1, beta2, args.s, output_file_path, output_coords_file_path, args.n_threads, args.v]))
        t1 = time.time()
        subprocess.call([bin_abspath + ' %s %s %s %s %s %s %s %s %s %s %s %s' % generator_args], shell = True)
        t2 = time.time()
    
    else:
        bin_dirname = os.path.abspath(parent_dirname+"/bin/")
        if not os.path.isdir(bin_dirname):
            subprocess.call('mkdir %s' % bin_dirname, shell = True)
            
        # if not compiled, compile and run
        print('Compiling the C++ code...')
        cpp_abspath = os.path.abspath(parent_dirname+'/src/weighted_edgelist_generator.cpp')
        bin_abspath = os.path.abspath(parent_dirname+'/bin/weighted_edgelist_generator.out')
        generator_args = tuple(map(str,[n, R, a, alpha1, alpha2, beta1, beta2, args.s, output_file_path, output_coords_file_path, args.n_threads, args.v]))
        subprocess.call(['g++ %s -o %s -std=c++11 -fopenmp -O3' % (cpp_abspath, bin_abspath)], shell = True)
        t1 = time.time()
        subprocess.call([bin_abspath + ' %s %s %s %s %s %s %s %s %s %s %s %s' % generator_args], shell = True)
        t2 = time.time()

    if args.v == 1:
        print("Time used for edge list generation: %.8f seconds." % (t2 - t1))
        


if __name__ == '__main__':
    
    t_start = time.time()

    main()

    t_finish = time.time()
    print("Total time elapsed: %.8f seconds." % (t_finish - t_start))
    print("==============================")