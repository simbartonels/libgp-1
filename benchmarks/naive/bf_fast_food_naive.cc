// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "bf_fast_food_naive.h"

#include "cov_factory.h"

#include "cov_se_ard.h"
#include "cov_sum.h"
#include "cov_noise.h"
#include "gp_utils.h"

#include <cmath>

namespace libgp {

FastFoodNaive::~FastFoodNaive() {
	while (!PIs.empty()) {
		delete PIs.back();
		PIs.pop_back();
	}
	wht_delete(wht_tree);
}

Eigen::VectorXd libgp::FastFoodNaive::computeBasisFunctionVector(
		const Eigen::VectorXd& x_unpadded) {
	Eigen::VectorXd phi = Eigen::VectorXd::Zero(M);

	x.head(input_dim) = x_unpadded.cwiseQuotient(ell);

	//already done in real_init():
	//x.tail(next_input_dim - input_dim).fill(0);

	for (size_t m = 0; m < M_intern; m++) {
		//TODO: it could be efficient to transpose B in general!
		temp.array() = b.row(m).transpose().array() * x.array();
		wht_apply(wht_tree, 1, temp.data());
		temp = g.row(m).transpose().cwiseProduct((*PIs.at(m)) * temp);
		wht_apply(wht_tree, 1, temp.data());
		temp = s.row(m).transpose().cwiseProduct(temp);
		//TODO: is the part below faster than the for loop? if so fix it
		//TODO: read again the documentation for segment!
//		phi.segment((M_intern + m) * input_dim,
//				(M_intern + m) * input_dim + input_dim).array() = temp.head(
//				input_dim).array().cos();
//		phi.segment(m * input_dim, m * input_dim + input_dim).array() = temp.head(
//				input_dim).array().sin();
		for (size_t j = 0; j < input_dim; j++) {
			phi(m * input_dim + j) = cos(temp(j));
			phi((M_intern + m) * input_dim + j) = sin(temp(j));
		}
	}
	return phi;
}

Eigen::VectorXd FastFoodNaive::multiplyW_withStandardBasisVector(size_t dim) {
	//TODO: the result of this can be precomputed!
	Eigen::VectorXd result(M_intern * input_dim);
	x.setZero();
	x(dim) = 1;
	for (size_t m = 0; m < M_intern; m++) {
		//TODO: it could be efficient to transpose B in general!
		temp.array() = b.row(m).transpose().array() * x.array();
		wht_apply(wht_tree, 1, temp.data());
		temp = g.row(m).transpose().cwiseProduct((*PIs.at(m)) * temp);
		wht_apply(wht_tree, 1, temp.data());
		temp = s.row(m).transpose().cwiseProduct(temp);
		result.segment(m * input_dim, input_dim) = temp.head(input_dim);
	}
	return result;
}

const Eigen::MatrixXd & libgp::FastFoodNaive::getInverseOfSigma() {
	return iSigma;
}

const Eigen::MatrixXd & libgp::FastFoodNaive::getCholeskyOfInvertedSigma() {
	return choliSigma;
}

const Eigen::MatrixXd & libgp::FastFoodNaive::getSigma() {
	return Sigma;
}

double FastFoodNaive::getLogDeterminantOfSigma() {
	return log_determinant_sigma;
}

void libgp::FastFoodNaive::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& g) {
	grad(x1, x2, getWrappedKernelValue(x1, x2), g);
}

void libgp::FastFoodNaive::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		double kernel_value, Eigen::VectorXd& grad) {
	//TODO: implement!
}

bool FastFoodNaive::gradDiagWrappedIsNull(size_t parameter) {
	return false;
}

void libgp::FastFoodNaive::gradBasisFunction(SampleSet * sampleSet,
		const Eigen::MatrixXd& Phi, size_t p, Eigen::MatrixXd& Grad) {
	assert(Grad.size() == Phi.size());
	if (p < input_dim) {
		size_t n = sampleSet->size();
		//TODO: use precomputed values
		Eigen::VectorXd z = multiplyW_withStandardBasisVector(p);
		for (size_t i = 0; i < n; i++) {
			//TODO: a vectorized version is faster
			double c = (sampleSet->x(i))(p) / ell(p);
			Grad.col(i).head(M_intern * input_dim) = c
					* z.cwiseProduct(
							Phi.col(i).segment(M_intern * input_dim,
									M_intern * input_dim));
			Grad.col(i).segment(M_intern * input_dim, M_intern * input_dim) = c
					* z.cwiseProduct(-Phi.col(i).head(M_intern * input_dim));
			//can we assume here that the rest of the gradient is already set to zero? yes
//			grad.tail(M - 2 * M_intern * input_dim).setZero();
		}
	} else {
		Grad.setZero();
	}
}

bool FastFoodNaive::gradBasisFunctionIsNull(size_t p) {
	//Phi is independent of noise and length scale
	if (p < input_dim)
		return false;
	return true;
}

void libgp::FastFoodNaive::gradiSigma(size_t p, Eigen::MatrixXd & diSigmadp) {
	if (p == input_dim) {
		//TODO: this could be more efficient
		diSigmadp.setIdentity();
//		dSigmadp.diagonal().fill(2 * sf2 / M_intern / input_dim);
		diSigmadp.diagonal().fill(-2. * M_intern * input_dim / sf2);
		diSigmadp.diagonal().tail(M - 2 * M_intern * input_dim).setZero();
	} else {
		//in an efficient implementation this function will not be called in this case
		diSigmadp.setZero();
	}
}

bool FastFoodNaive::gradiSigmaIsNull(size_t p) {
	//the weight prior depends only on the signal variance
	if (p == input_dim)
		return false;
	return true;
}

void FastFoodNaive::log_hyper_updated(const Eigen::VectorXd &p) {
	sf2 = exp(2 * p(input_dim));
	for (size_t i = 0; i < input_dim; i++)
		ell(i) = exp(p(i));
	Sigma.diagonal().fill(sf2 / M_intern / input_dim);
	Sigma.diagonal().tail(M - 2 * M_intern * input_dim).fill(1);
	//contains log(|Sigma|)/2
	log_determinant_sigma = M_intern * input_dim
			* (2 * p(input_dim) - log(M_intern * input_dim));
	iSigma.diagonal().fill(M_intern * input_dim / sf2);
	iSigma.diagonal().tail(M - 2 * M_intern * input_dim).fill(1);
	choliSigma.diagonal().fill(
			exp(-p(input_dim)) * sqrt(M_intern) * sqrt(input_dim));
	choliSigma.diagonal().tail(M - 2 * M_intern * input_dim).fill(1);
}

std::string libgp::FastFoodNaive::to_string() {
	return "FastFoodNaive";
}

size_t FastFoodNaive::get_param_dim_without_noise(size_t input_dim,
		size_t num_basis_functions) {
	//length scales + amplitude
	//no need to take care of the noise
	return input_dim + 1;
}

bool libgp::FastFoodNaive::real_init() {
	//TODO: check covariance function!

	//next_pow = ilog2(input_dim) + 1;
	int out;
	std::frexp(input_dim - 1, &out);
	next_pow = out;
	next_input_dim = pow(2, next_pow);
//	std::cout << "bf_fast_food: internal dimension " << next_input_dim
//			<< std::endl;
	assert(next_input_dim >= input_dim);
	assert(pow(2, next_pow - 1) < input_dim);
	assert(M >= 2 * input_dim);
	M_intern = floor(M / 2 / input_dim);
//	std::cout << "bf_fast_food: number of V matrices " << M_intern << std::endl;
	assert(2 * M_intern * input_dim <= M);
	ell.resize(input_dim);
	Sigma.resize(M, M);
	Sigma.setZero();
	iSigma.resize(M, M);
	iSigma.setZero();
	choliSigma.resize(M, M);
	choliSigma.setZero();
	wht_tree = wht_get_tree(next_pow);
	assert(wht_tree != NULL);

	s.resize(M_intern, next_input_dim);
	g.resize(M_intern, next_input_dim);
	b.resize(M_intern, next_input_dim);
	//vector will automatically resize, this call just breaks things
	//PIs.resize(M_intern);
	x.resize(next_input_dim);
	x.tail(next_input_dim - input_dim).fill(0);
	temp.resize(next_input_dim);

	for (size_t i = 0; i < M_intern; i++) {
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>* pi =
				new Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>(
						next_input_dim);
		pi->setIdentity();
		//make a random permutation
		std::random_shuffle(pi->indices().data(),
				pi->indices().data() + pi->indices().size());
		PIs.push_back(pi);
		assert(pi == PIs.at(i));
		for (size_t d1 = 0; d1 < next_input_dim; d1++) {
			g.row(i)(d1) = Utils::randn();
			double d = 2 * Utils::randi(2);
			b.row(i)(d1) = d - 1;
			double r = 0;
			for (size_t d2 = 0; d2 < next_input_dim; d2++) {
				double stdn = Utils::randn();
				r += stdn * stdn;
			}
			s.row(i)(d1) = sqrt(r);
		}
		s.row(i) /= g.row(i).norm();
	}
	s /= sqrt(next_input_dim);
//	std::cout << "bf_fast_food: initialization complete" << std::endl;
	return true;
}

Eigen::MatrixXd FastFoodNaive::getS() {
	return s;
}

Eigen::MatrixXd FastFoodNaive::getG() {
	return g;
}

Eigen::MatrixXd FastFoodNaive::getB() {
	return b;
}

Eigen::MatrixXd FastFoodNaive::getPI() {
	Eigen::MatrixXd pi_matrix(M_intern, next_input_dim);
	Eigen::VectorXi temp_vector(next_input_dim);
	for (size_t m = 0; m < M_intern; m++) {
		temp_vector = (*PIs.at(m)).indices();
		for (size_t j = 0; j < next_input_dim; j++) {
			pi_matrix(m, j) = (double) temp_vector(j);
		}
	}
	return pi_matrix;
}


std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> *> FastFoodNaive::getPIs(){
	return PIs;
}
}
