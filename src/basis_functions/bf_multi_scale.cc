// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "basis_functions/bf_multi_scale.h"

#include "cov_factory.h"

#include "cov_se_ard.h"
#include "cov_sum.h"
#include "cov_noise.h"

#include <cmath>

namespace libgp {

/*
 * TODO: This class can be made more memory efficient when iUpsi and LUpsi are stored in the same matrix.
 */

bool MultiScale::real_init() {
	//TODO: signal that Multiscale ignores the covariance function!
	CovFactory f;
	CovarianceFunction * expectedCov;
	expectedCov = f.create(input_dim, "CovSum ( CovSEiso, CovNoise)");
	if (cov->to_string() != expectedCov->to_string()) {
		//TODO: signal reason for error!
		return false;
	}

	loghyper.resize(get_param_dim());
	Upsi.resize(M, M);
	LUpsi.resize(M, M);
	iUpsi.resize(M, M);
	U.resize(M, input_dim);
	Uell.resize(M, input_dim);
	ell.resize(input_dim);
	return true;
}

Eigen::VectorXd MultiScale::computeBasisFunctionVector(
		const Eigen::VectorXd & x) {
	Eigen::VectorXd uvx(M);
	for (size_t i = 0; i < M; i++) {
		uvx(i) = g(x, U.row(i), Uell.row(i));
	}
	return uvx;
}

void MultiScale::gradBasisFunction(const Eigen::VectorXd &x,
		const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad) {
//	dAdl(Uvx, sigma(k, d), d, x, V, k)
//	dAdl(A, p, d, x, z, i)
	//	dA(i, :) = ((((z(i, d) - x(:, d))./p).^2-1./p))'.*A(i, :)/2
//    Uvx = p2 * dUvx;
	//Uvx=g(x,vi,si)
	if (p < input_dim) {
		//derivative with respect to the length scales
		/*
		 * Since we add half the length scales to the inducing length scales the derivative with
		 * respect to the length scales is not trivially zero.
		 */
		size_t d = p;
		grad = (U.col(d) - x(d)).cwiseQuotient(Uell.col(d));
		grad.array() = grad.array().square() - Uell.col(d).array().cwiseInverse();
		grad = ell(d) * grad.cwiseProduct(phi) / 2;
//		for (size_t i = 0; i < M; i++) {
//			double t = (U(i, d) - x(d)) / Uell(i, d);
//			t = t * t - 1 / Uell(i, d);
//			grad(i) = t * phi(i) / 2;
//		}
	} else if(p >= input_dim && p < M*input_dim+input_dim){
//        [d, j] = getDimensionAndIndex(di, D, M);
//        p2 = sigma(j, d);
//        Uvx = dAdl(Uvx, p2, d, x, V, j);
//
//        p = p2+sigma(:, d)-ell(d);
//        dUpsi = dAdl(Upsi, p, d, V, V, j);
//
//        % chain rule
//        p2 = p2 - ell(d)/2; % that half has no influence on the gradient
//        Uvx = p2 * Uvx;
//        dUpsi(j, :) = p2 * dUpsi(j, :);
//        dUpsi(:, j) = dUpsi(j, :);
//        dUpsi(j, j) = 2 * dUpsi(j, j);
//        Upsi = dUpsi;
	}
	else {
//        %inducing point derivatives
//        [d, j] = getDimensionAndIndex(di, D, M);
//        dUvx = zeros(size(Uvx));
//        sig = sigma(j, d);
//        dUvx(j, :) = (-V(j, d) + x(:, d))/sig .* Uvx(j, :)';
//        Uvx = dUvx;
//
//        dUpsi = zeros(size(Upsi));
//        p2 = sigma(j, d);
//        p = p2+sigma(:, d)-ell(d);
//        dUpsi(j, :) = (-V(j, d) + V(:, d)) .* Upsi(j, :)' ./p;
//        dUpsi(:, j) = dUpsi(j, :);
//        Upsi = dUpsi;
	}
}

Eigen::MatrixXd MultiScale::getInverseWeightPrior() {
	//TODO: check that upper part of Upsi is same as lower part
	return Upsi;
}

void MultiScale::gradInverseWeightPrior(size_t p, Eigen::MatrixXd & diSigmadp) {

}

Eigen::MatrixXd MultiScale::getCholeskyOfInverseWeightPrior() {
	return LUpsi;
}

Eigen::MatrixXd MultiScale::getWeightPrior() {
	return iUpsi;
}

double MultiScale::getWrappedKernelValue(const Eigen::VectorXd &x1,
		const Eigen::VectorXd &x2) {
	//adding noise has to be incorporated here
	double noise = 0;
	if (x1 == x2)
		noise = sn2;
	return c * g(x1, x2, ell) + noise;
}

void MultiScale::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& grad) {
	grad(x1, x2, getWrappedKernelValue(x1, x2), grad);
}

void MultiScale::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		double kernel_value, Eigen::VectorXd& grad) {
	//TODO: implement
}

void MultiScale::set_loghyper(const Eigen::VectorXd& p) {
	CovarianceFunction::set_loghyper(p);

	for (size_t i = 0; i < input_dim; i++)
		ell(i) = exp(loghyper(i));
	size_t idx = input_dim;
	for (size_t d = 0; d < input_dim; d++) {
		for (size_t m = 0; m < M; m++) {
			/*
			 * For robustness half of the length scales is added to the
			 * inducing length scales.
			 */
			Uell(m, d) = exp(loghyper(idx)) + ell(d) / 2;
			U(m, d) = loghyper(idx + M * input_dim);
			idx++;
		}
	}

	c = exp(loghyper(2 * M * input_dim + input_dim));

	sn2 = exp(2 * loghyper(2 * M * input_dim + input_dim + 1));
	snu2 = 1e-6 * sn2;
	initializeMatrices();
}

void MultiScale::initializeMatrices() {
	/**
	 * Initializes the matrices LUpsi and iUpsi.
	 */
	for (size_t i = 0; i < M; i++) {
		Eigen::VectorXd vi = U.row(i);
		Eigen::VectorXd s = Uell.row(i) - ell.transpose();

		for (size_t j = 0; j < i; j++) {
			Upsi(i, j) = g(vi, U.row(j), s.transpose() + Uell.row(j)) / c;
		}
		Upsi(i, i) = g(vi, U.row(i), s.transpose() + Uell.row(i)) / c + snu2;
	}
	//this division has been moved into the for loop above
	//LUpsi = LUpsi / c;
	Upsi = Upsi.selfadjointView<Eigen::Lower>();

	//TODO: refactor: Is it possible to remove the topLeftCorner-calls?
	LUpsi.topLeftCorner(M, M) = Upsi.topLeftCorner(M, M).selfadjointView<
			Eigen::Lower>().llt().matrixL();
	iUpsi = LUpsi.topLeftCorner(M, M).triangularView<Eigen::Lower>().solve(
			LUpsi.Identity(M, M));
	//TODO: it should be sufficient to transpose here
	LUpsi.topLeftCorner(M, M).triangularView<Eigen::Lower>().adjoint().solveInPlace(
			iUpsi);
}

void MultiScale::set_loghyper(const double p[]) {
	CovarianceFunction::set_loghyper(p);
}

size_t MultiScale::get_param_dim() {
	//1 for length scale
	//1 for noise
	//input_dim length scales
	//M*input_dim inducing inputs
	//M*input_dim corresponding length scales
	return 2 * M * input_dim + input_dim + 1 + 1;
}

Eigen::VectorXd MultiScale::get_loghyper() {
	return loghyper;
}

std::string MultiScale::to_string() {
	return "MultiScale";
}

double MultiScale::g(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		const Eigen::VectorXd& sigma) {
	//TODO: can we make this numerically more stable?
	/*
	 * idea:
	 * - compute s1+s2-s in advance (make matrix)
	 * - compute log of that matrix
	 * - then the division becomes a sum in the exp call below
	 * - however: need to call exp() on sigma!
	 * -> safe this implementation as naive?
	 */
	Eigen::VectorXd delta = x1 - x2;
	double z = delta.cwiseQuotient(sigma).transpose() * delta;
	z = exp(-0.5 * z);
	double p = 1;
	for (size_t i = 0; i < input_dim; i++) {
		p = 2 * M_PI * p * sigma(i);
	}
	p = sqrt(p);
	return z / p;
}

}
