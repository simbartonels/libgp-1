// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "basis_functions/bf_fast_food.h"

#include "cov_factory.h"

#include "cov_se_ard.h"
#include "cov_sum.h"
#include "cov_noise.h"
#include "gp_utils.h"

#include <cmath>

//extern "C" Wht * wht_get_tree(int);
//extern "C" {
//	#include "wht_trees.c"
//}

namespace libgp {

FastFood::~FastFood() {
	while (!PIs.empty()) {
		delete PIs.back();
		PIs.pop_back();
	}
}

Eigen::VectorXd libgp::FastFood::computeBasisFunctionVector(
		const Eigen::VectorXd& x) {
	return multiplyW(x);
}

Eigen::VectorXd FastFood::multiplyW(const Eigen::VectorXd& x_unpadded) {
	Eigen::VectorXd phi(M);
	phi.setZero();
	x.head(input_dim) = x_unpadded.cwiseQuotient(ell);
	x.tail(input_dim).fill(0);

	for (size_t m = 0; m < M_intern; m++) {
		temp.array() = b.row(m).array() * x.array();
		//TODO: uncomment or fix if this is causes the segmentation fault
//		wht_apply(wht_tree, 1, temp.data());
		temp = g.row(m).cwiseProduct((*PIs.at(m)) * temp);
		//TODO: uncomment or fix if this is causes the segmentation fault
//		wht_apply(wht_tree, 1, temp.data());
		temp = s.row(m).cwiseProduct(temp);
		//TODO: uncomment or fix if this is causes the segmentation fault
//		phi.segment(m * input_dim, m * input_dim + input_dim).array() =
//				temp.head(input_dim).array().sin();
//		phi.segment(2 * m * input_dim, 2 * m * input_dim + input_dim).array() =
//				temp.head(input_dim).array().cos();
	}
	return phi;
}

Eigen::MatrixXd libgp::FastFood::getInverseWeightPrior() {
	return iSigma;
}

Eigen::MatrixXd libgp::FastFood::getCholeskyOfInverseWeightPrior() {
	return choliSigma;
}

Eigen::MatrixXd libgp::FastFood::getWeightPrior() {
	return Sigma;
}

double FastFood::getLogDeterminantOfWeightPrior() {
	return log_determinant_sigma;
}

void libgp::FastFood::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& grad) {
}

void libgp::FastFood::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		double kernel_value, Eigen::VectorXd& grad) {
}

void libgp::FastFood::gradBasisFunction(const Eigen::VectorXd& x,
		const Eigen::VectorXd& phi, size_t p, Eigen::VectorXd& grad) {
}

void libgp::FastFood::gradInverseWeightPrior(size_t p,
		Eigen::MatrixXd & diSigmadp) {

}

void libgp::FastFood::set_loghyper(const Eigen::VectorXd& p) {
	CovarianceFunction::set_loghyper(p);
	sf2 = exp(2 * p(input_dim));
	for (size_t i = 0; i < input_dim; i++)
		ell(i) = exp(p(i));
	Sigma.diagonal().fill(sf2 / M / input_dim);
	//contains log(|Sigma|)/2
	log_determinant_sigma = M * input_dim
			* (2 * p(input_dim) - log(M * input_dim));
	iSigma.diagonal().fill(M * input_dim / sf2);
	choliSigma.diagonal().fill(sqrt(M) * sqrt(input_dim) * exp(-p(input_dim)));
	std::cout
			<< "bf_fast_food: internal data structures updated for new hyper-parameters"
			<< std::endl;
}

std::string libgp::FastFood::to_string() {
	return "FastFood";
}

bool libgp::FastFood::real_init() {
	//TODO: check covariance function!

	//length scales + amplitude + noise
	param_dim = input_dim + 1 + 1;
	next_pow = ilogb(input_dim) + 1;
	next_input_dim = pow(2, next_pow);
	assert(next_input_dim >= input_dim);
	assert(pow(2, next_pow - 1) < input_dim);
	M_intern = floor(M / 2 / input_dim);
	if (M_intern == 0)
		M_intern = 1;
	assert(2 * M_intern * input_dim <= M);
	loghyper.resize(get_param_dim());
	ell.resize(input_dim);
	Sigma.resize(M);
	iSigma.resize(M);
	choliSigma.resize(M);
	wht_tree = wht_get_tree(next_pow);
	assert(wht_tree != NULL);
	s.resize(M_intern, next_input_dim);
	g.resize(M_intern, next_input_dim);
	b.resize(M_intern, next_input_dim);
	PIs.resize(M_intern);
	x.resize(next_input_dim);
	temp.resize(next_input_dim);

	for (size_t i = 0; i < M_intern; i++) {
		//TODO: does this need to be a call to new? (see sampleset implementation)
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>* pi =
				new Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>(
						next_input_dim);
		pi->setIdentity();
		//make a random permutation
		std::random_shuffle(pi->indices().data(),
				pi->indices().data() + pi->indices().size());
		PIs.push_back(pi);
		for (size_t d1 = 0; d1 < next_input_dim; d1++) {
			double r = 0;
			for (size_t d2 = 0; d2 < next_input_dim; d2++) {
				double stdn = Utils::randn();
				r += stdn * stdn;
			}
			s.row(i)(d1) = sqrt(r);
			g.row(i)(d1) = Utils::randn();
			b.row(i)(d1) = 2 * Utils::randi(2) - 1;
		}
		s.row(i) /= g.row(i).norm();
	}
	//TODO: maybe this can be skipped, depending on the implementation of wht
	s /= sqrt(next_input_dim);
	std::cout << "bf_fast_food: initialization complete" << std::endl;
	return true;
}

Eigen::MatrixXd FastFood::getS() {
	return s;
}

Eigen::MatrixXd FastFood::getG() {
	return g;
}

Eigen::MatrixXd FastFood::getB() {
	return b;
}

Eigen::MatrixXd FastFood::getPI() {
	Eigen::MatrixXd pi_matrix(M, next_input_dim);
	Eigen::VectorXi temp_vector(next_input_dim);
	for (size_t m = 0; m < M; m++) {
		temp_vector = (*PIs.at(m)).indices();
		for (size_t j = 0; j < next_input_dim; j++) {
			pi_matrix(m, j) = (double) temp_vector(j);
		}
	}
	return pi_matrix;
}
}
