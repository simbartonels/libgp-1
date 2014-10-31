// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "fic_gp.h"
#include "cov_factory.h"
#include "basis_functions/basisf_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {

const double log2pi = log(2 * M_PI);

//TODO: find a way to get the value from the super class
const double initial_L_size = 1000;

FICGaussianProcess::FICGaussianProcess(size_t input_dim, std::string covf_def,
		size_t num_basisf, std::string basisf_def) :
		AbstractGaussianProcess(input_dim, covf_def) {
	BasisFFactory factory;
	//wrap initialized covariance function with basis function
	cf = factory.createBasisFunction(basisf_def, num_basisf, cf);
	cf->loghyper_changed = 0;
	bf = (IBasisFunction *) cf;
	M = bf->getNumberOfBasisFunctions();
	alpha.resize(M);
	L.resize(M, M);
	Lu.resize(M, M);
	Luu.resize(M, M);
	beta.resize(M);
}

FICGaussianProcess::~FICGaussianProcess() {
//	  delete V;
//	  delete isqrtgamma;
}

double FICGaussianProcess::var_impl(const Eigen::VectorXd x_star) {
	return bf->getWrappedKernelValue(x_star, x_star)
			+ k_star.transpose() * L * k_star;
}

void FICGaussianProcess::computeCholesky() {
	//TODO: refactor! this method is too long!
	/*
	 * This method does not compute the Cholesky in the same sense as
	 * the GaussianProcess class does. Here the same thing happens as
	 * in infFITC.m from the gpml toolbox by Rasmussen and Nikisch. The
	 * method computeCholesky is kept for abstraction reasons.
	 */
	size_t n = sampleset->size();

	if (n > isqrtgamma.rows()) {
		isqrtgamma.resize(n);
		dg.resize(n);
		V.resize(M, n);
	}
	//corresponds to Ku in infFITC
	//TODO: it might be necessary to create this matrix on the heap!
	Eigen::MatrixXd Phi(M, n);
	//corresponds to diagK in infFITC
	Eigen::VectorXd k(n);
	for (size_t i = 0; i < n; i++) {
		Eigen::VectorXd xi = sampleset->x(i);
		Eigen::VectorXd phi = bf->computeBasisFunctionVector(xi);
		//TODO: is there a faster operation?
		for (size_t j = 0; j < M; j++) {
			Phi(j, i) = phi(j);
		}
		//TODO: rethink the design here
		//it might be better to seperate basis functions and kernels
		k(i) = bf->getWrappedKernelValue(xi, xi);
	}
	Luu = bf->getCholeskyOfInverseWeightPrior();
	/*
	 * TODO: could we just multiply Phi with sqrt(gamma) HERE instead of using
	 * the inverse later? What's more stable?
	 */
	V = Luu.topLeftCorner(M, M).triangularView<Eigen::Lower>().solve(Phi);
	//noise is already added in k
	dg = k - (V.transpose() * V).diagonal();
//	isqrtgamma = isqrtgamma.cwiseInverse().sqrt();
	isqrtgamma.array() = 1 / dg.array().sqrt();
	V = V * isqrtgamma.asDiagonal();
	// TODO: is it possible to use the self adjoint view here?
	Lu = V * V.transpose() + Eigen::MatrixXd::Identity(M, M);
	Lu.topLeftCorner(M, M) = Lu.llt().matrixL();

	Eigen::MatrixXd iUpsi = bf->getWeightPrior();

	/*
	 * Here we have to divert from the Matlab implementation. in Matlab all
	 * the matrices are upper matrices. Here we have what they are supposed to be:
	 * lower matrices.
	 */
	Eigen::MatrixXd temp = Luu * Lu;
	//the line below does not work. why?
//	L = L.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(M, M));
	L = temp.triangularView<Eigen::Lower>().solve(
			Eigen::MatrixXd::Identity(M, M));
	temp.transpose().triangularView<Eigen::Upper>().solveInPlace(L);
	L = L - iUpsi;
}

void FICGaussianProcess::updateCholesky(const double x[], double y) {
	//Do nothing and just recompute everything.
	//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;
}

void FICGaussianProcess::update_k_star(const Eigen::VectorXd &x_star) {
	k_star.resize(bf->getNumberOfBasisFunctions());
	k_star = bf->computeBasisFunctionVector(x_star);
}

void FICGaussianProcess::update_alpha() {
	size_t n = sampleset->size();
	if (n > r.size()) {
		r.resize(n);
	}
	// Map target values to VectorXd
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	r.array() = y.array() * isqrtgamma.array();
	beta = V * r;
	/*
	 * In the Matlab implementation Luu and Lu are upper matrices and that's
	 * why we need to transpose here.
	 */
	Lu.triangularView<Eigen::Lower>().solveInPlace(beta);
	//alpha = Luu\(Lu\be)
	alpha = Lu.transpose().triangularView<Eigen::Upper>().solve(beta);
	Luu.transpose().triangularView<Eigen::Upper>().solveInPlace(alpha);
}

double FICGaussianProcess::log_likelihood_impl() {
	double t = 0;
	for (size_t i = 0; i < M; i++) {
		t += log(Lu(i, i));
	}
	size_t n = sampleset->size();
	double t2 = 0;
	for(size_t i = 0; i < n; i++){
		t2 += log(dg(i)) + r(i)*r(i) - beta(i)*beta(i);
	}
	//TODO: is this better? or should it be moved to the loop?
	t2=t2+n*log2pi;
	return t + t2/2;
}

Eigen::VectorXd FICGaussianProcess::log_likelihood_gradient_impl() {
	return Eigen::VectorXd::Zero(bf->get_param_dim());
}
}
