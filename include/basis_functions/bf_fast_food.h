// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_
#define INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_

#include "IBasisFunction.h"
extern "C" {
#include "spiral_wht.h"
}
#include <vector>

namespace libgp {
class FastFood: public IBasisFunction {
public:
	virtual ~FastFood();

	/**
	 * Allows to set S, G, RandPI and B.
	 */
	void setExtraParameters(const Eigen::MatrixXd & Extra);

	Eigen::VectorXd computeBasisFunctionVector(const Eigen::VectorXd &x);

	const Eigen::MatrixXd & getInverseOfSigma();

	const Eigen::MatrixXd & getCholeskyOfInvertedSigma();

	const Eigen::MatrixXd & getSigma();

	double getLogDeterminantOfSigma();

	void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2,
			Eigen::VectorXd &grad);

	void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2,
			double kernel_value, Eigen::VectorXd &grad);

	bool gradDiagWrappedIsNull(size_t parameter);

	void gradBasisFunction(SampleSet * sampleSet, const Eigen::MatrixXd &Phi,
			size_t p, Eigen::MatrixXd &Grad);

	bool gradBasisFunctionIsNull(size_t p);

	void compute_dkdx(const Eigen::VectorXd & x, const Eigen::VectorXd & kstar,
			SampleSet * sampleSet, Eigen::MatrixXd & JT);

	void gradiSigma(size_t p, Eigen::MatrixXd & diSigmadp);

	bool gradiSigmaIsNull(size_t p);

	bool sigmaIsDiagonal() {
		return true;
	}
	;

	std::string to_string();

	std::string pretty_print_parameters();

	/**
	 * Returns the sampled scaling matrices.
	 */
	Eigen::MatrixXd getS();

	/**
	 * Returns the sampled Gaussian matrices.
	 */
	Eigen::MatrixXd getG();

	/**
	 * Returns the sampled binary matrices.
	 */
	Eigen::MatrixXd getB();

	/**
	 * Returns the sampled permutation matrices.
	 */
	Eigen::MatrixXd getPI();

	/**
	 * Returns a pointer to the permutation matrix storage.
	 */
	std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> *> getPIs();

	void setS(const Eigen::MatrixXd& S);

	void setG(const Eigen::MatrixXd& G);

	void setB(const Eigen::MatrixXd& B);

	void setPIs(
			std::vector<
					Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> *> PIs);

protected:
	void log_hyper_updated(const Eigen::VectorXd &p);

	virtual bool real_init();

	size_t get_param_dim_without_noise(size_t input_dim,
			size_t num_basis_functions);

private:
	void inline precomputeHe();
	void inline initRandPI();

	/**
	 * Deletes all permutation matrices.
	 */
	void deletePIs();

	/**
	 * Initializes the matrices B,G,S,Pi and He.
	 */
	inline void initializeMatrices();

	/**
	 * Signal amplitude.
	 */
	double sf2;

	/**
	 * The length scales.
	 */
	Eigen::VectorXd ell;

	/**
	 * The weight prior.
	 */
	Eigen::MatrixXd Sigma;

	/**
	 * The inverse weight prior.
	 */
	Eigen::MatrixXd iSigma;

	/**
	 * The Cholesky of the weight prior.
	 */
	Eigen::MatrixXd choliSigma;

	/**
	 * Log of half of the determinant of Sima.
	 */
	double logDetSigma;

	/**
	 * The smallest power of two s.t. input_dim <= 2^next_pow.
	 */
	size_t next_pow;

	/**
	 * The smallest number that is a power of 2 and larger than input dim.
	 */
	size_t next_input_dim;

	/**
	 * Tree for fast hadamard multiplications.
	 */
	Wht * wht_tree;

	/**
	 * Concatenated diagonal random binary matrices.
	 * NOTE: Double matrix because Eigen refuses to mix types... -.-
	 */
	Eigen::MatrixXd b;

	/**
	 * Concatenated diagonal Gaussian random matrices.
	 */
	Eigen::MatrixXd g;

	/**
	 * Concatenated diagonal random scaling matrices.
	 */
	Eigen::MatrixXd s;

	/**
	 * Random permutation matrix.
	 */
	std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> *> PIs;

	/**
	 * Temporary structure used in multiplyW().
	 */
	Eigen::VectorXd x;

	/**
	 * Temporary structure used in multiplyW().
	 */
	Eigen::VectorXd temp;

	/**
	 * Matrix that contains the results of HGPiHBe_d where d in [1, ..., D].
	 */
	Eigen::MatrixXd He;

	/**
	 * Contains log(|Sigma|)/2.
	 */
	double log_determinant_sigma;

	/**
	 * The number of different matrices V.
	 */
	size_t M_intern;
};
}

#endif /* INCLUDE_BASIS_FUNCTIONS_BF_FAST_FOOD_H_ */
