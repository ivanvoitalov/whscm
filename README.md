# Weighted Hypersoft Configuration Model (WHSCM)

## Summary
This repository contains the code to generate networks from the _Weighted Hypersoft Configuration Model (WHSCM)_ with the prescribed power-law degree distribution and tunable super-linear scaling between nodes' strengths and degrees.

## Dependencies

The WHSCM generator presented here is a combination of Python scripts and C++ code with the OpenMP support for multi-thread generation of links. To use the code, you need standard libraries shipped with Python and:

* g++ compiler with the C++11 and OpenMP support
* NumPy
* SciPy
* mpmath

## Installation

No installation is required. You may directly launch the Python scripts from the _Python2/Python3_ directories corresponding to the version of Python installed on your computer.

**C++ code compilation.** The C++ code is supposed to compile itself when the script is first launched. The source C++ code may be found under the _src_ directory. The compiled binary file will be placed in the _bin_ directory which is created when the generator is launched for the first time. In case you need to implement any changes to the C++ source code, you may simply remove the _bin_ directory, and the code will be recompiled the next time it is launched.

## Simple Usage Example

Suppose you are given a real weighted network and would like to generate the null model networks from the WHSCM that resemble its basic structural properties and are maximally random. Suppose the size of the network is ![$n = 10^4$](https://render.githubusercontent.com/render/math?math=%24n%20%3D%2010%5E4%24), the power-law degree distribution has the exponent ![$\gamma = 2.5$](https://render.githubusercontent.com/render/math?math=%24%5Cgamma%20%3D%202.5%24) and the average degree is ![$\bar{k} = 10$](https://render.githubusercontent.com/render/math?math=%24%5Cbar%7Bk%7D%20%3D%2010%24). Additionally, as observed from many realistic weighted networks, suppose you observe that nodes' strength scale super-linearly with nodes' degrees, ![$\bar{s}(k) \sim k^{\eta}$](https://render.githubusercontent.com/render/math?math=%24%5Cbar%7Bs%7D(k)%20%5Csim%20k%5E%7B%5Ceta%7D%24), where ![$\eta = 1.5$](https://render.githubusercontent.com/render/math?math=%24%5Ceta%20%3D%201.5%24) is the exponent describing this scaling. Moreover, you observe the "baseline" of the strength-degree scaling curve ![$\bar{s}(k)$](https://render.githubusercontent.com/render/math?math=%24%5Cbar%7Bs%7D(k)%24) such that the expected strength as a function of degree rescaled by the degree approaches some given constant ![$\sigma_0 = 0.1$](https://render.githubusercontent.com/render/math?math=%24%5Csigma_0%20%3D%200.1%24): ![$\bar{s}(k) / k^{\eta} \rightarrow \sigma_0$](https://render.githubusercontent.com/render/math?math=%24%5Cbar%7Bs%7D(k)%20%2F%20k%5E%7B%5Ceta%7D%20%5Crightarrow%20%5Csigma_0%24). Given these parameters that one may measure directly from the network, you may generate a WHSCM null model network using the following command:
```
python Python2/generate_weighted_network.py -n 10000 -g 2.5 -e 1.5 -o output_edgelist.net -k 10.0 --sigma0 0.1
```

This will create a weighted edge list "output_edgelist.net" in the root directory of the repository. Within this file, each line denotes a connected pair of nodes and their link weight in the following format:
```
node_i node_j w_ij
```

If you would like to also save the latent parameters ("coordinates") ![$\lambda, \mu$](https://render.githubusercontent.com/render/math?math=%24%5Clambda%2C%20%5Cmu%24) for each node that are used to generate this WHSCM network instance, you may provide an output path for the coordinates file using the `--params_output` flag. The terminal command would then look as follows:
```
python Python2/generate_weighted_network.py -n 10000 -g 2.5 -e 1.5 -o output_edgelist.net -k 10.0 --sigma0 0.1 --params_output output_coordinates.dat
```

The coordinates file has the following format output format per line:
```
node_label lambda mu
```

## Additional Options

The code supports some additional options that may help users while running the code. The description of all available flags may be accessed by running the following command:
```
python Python2/generate_weighted_network.py --help
```
Here we briefly describe these flags and their functionality.

`-n`. This parameter sets the number of nodes in the WHSCM network.

`-g`. This parameter sets the power-law exponent of the target degree distribution.

`-e`. This parameter sets the target exponent of strength-degree super-linear scaling. If it is set to 1, the special case generator equivalent to the unweighted HSCM is launched.

`-k`. This parameter sets the target average degree of a network.

`--sigma0`. This parameter sets the "baseline" of the strength-degree scaling.

`-o`. This parameter sets the output path of the resulting weighted edge list.

`--params_output`. This parameter sets the output path of the file containing the latent variables ![$\lambda, \mu$](https://render.githubusercontent.com/render/math?math=%24%5Clambda%2C%20%5Cmu%24) for each node.

`--n_threads`. This flag allows to run the network generator in parallel using OpenMP. By default, it is set to 1, however, for faster generation of weighted links, it is advised to set it to the number of available threads.

`-R, -a`. These flags should be used simultaneously instead of `-k` and `--sigma0` flags. If the parameters ![$R, a$](https://render.githubusercontent.com/render/math?math=%24R%2C%20a%24) of the WHSCM (see the paper for reference) are known either from a custom solver or from previous runs of the generator, there is no need to solve for them again given the same input parameters, so you may use them directly instead of specifying the ![$\bar{k}, \sigma_0$](https://render.githubusercontent.com/render/math?math=%24%5Cbar%7Bk%7D%2C%20%5Csigma_0%24) parameters.

`--solver`. This parameter may take the values of 0 and 1. If it is set to 0, then the approximate solver is used to find the ![$R, a$](https://render.githubusercontent.com/render/math?math=%24R%2C%20a%24) parameters of the WHSCM given the ![$\bar{k}, \sigma_0$](https://render.githubusercontent.com/render/math?math=%24%5Cbar%7Bk%7D%2C%20%5Csigma_0%24) parameters. If it is set to 1, then an exact solver is launched on top of the results of the approximate solver to find a more precise estimate of ![$R, a$](https://render.githubusercontent.com/render/math?math=%24R%2C%20a%24). The approximate solver is significantly faster that the exact one, and finds relatively accurate estimates of the ![$R, a$](https://render.githubusercontent.com/render/math?math=%24R%2C%20a%24) parameters.

`-s`. This parameter sets the random seed. Note that it should not exceed the maximum value of the C++ uint32 type minus the number of parallel threads, as each parallel thread is seeded with its unique random number generator. 

`-v`. This parameter equals either 0 or 1. If it is set to 0, the verbosity of the code is set to minimum. If it is set to 1, the verbosity of the code is set to maximum.