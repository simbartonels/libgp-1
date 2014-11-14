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
	assert(grad.size() == phi.size());
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
		grad.array() = (U.col(d).array() - x(d)) / Uell.col(d).array();
		grad.array() = grad.array().square()
				- Uell.col(d).array().cwiseInverse();
		grad = ell(d) * grad.cwiseProduct(phi) / 4;
//		for (size_t i = 0; i < M; i++) {
//			double t = (U(i, d) - x(d)) / Uell(i, d);
//			t = t * t - 1 / Uell(i, d);
//			grad(i) = t * phi(i) / 2;
//		}
	} else if (p >= input_dim && p < M * input_dim + input_dim) {
//        [d, j] = getDimensionAndIndex(di, D, M);
//        p2 = sigma(j, d);
//        Uvx = dAdl(Uvx, p2, d, x, V, j);
		//	dAdl(A, p, d, x, z, i)
		//	dA(i, :) = ((((z(i, d) - x(:, d))./p).^2-1./p))'.*A(i, :)/2
		grad = Eigen::VectorXd::Zero(M);
		size_t m = (p - input_dim) % M;
		size_t d = (p - input_dim - m) / M;
		double t = (U(m, d) - x(d)) / Uell(m, d);
		grad(m) = (t * t - 1 / Uell(m, d)) * phi(m) / 2;
		//        % chain rule
		//        p2 = p2 - ell(d)/2; % that half has no influence on the gradient
		//        Uvx = p2 * Uvx;
		grad(m) *= (Uell(m, d) - ell(d) / 2);
	} else if (p >= M * input_dim + input_dim
			&& p < 2 * M * input_dim + input_dim) {
//        %inducing point derivatives
//        [d, j] = getDimensionAndIndex(di, D, M);
//        dUvx = zeros(size(Uvx));
//        sig = sigma(j, d);
//        dUvx(j, :) = (-V(j, d) + x(:, d))/sig .* Uvx(j, :)';
//        Uvx = dUvx;
		grad = Eigen::VectorXd::Zero(M);
		size_t m = (p - input_dim) % M;
		size_t d = (p - input_dim - m) / M - input_dim;
		grad(m) = (-U(m, d) + x(d)) / Uell(m, d) * phi(m);
	} else {
		//length scale and noise derivative
		grad = Eigen::VectorXd::Zero(M);
	}
}

Eigen::MatrixXd MultiScale::getInverseWeightPrior() {
	//TODO: check that upper part of Upsi is same as lower part
	return Upsi;
}

void MultiScale::gradInverseWeightPrior(size_t p, Eigen::MatrixXd & diSigmadp) {
	diSigmadp = Eigen::MatrixXd::Zero(M, M);
	if (p < input_dim || p == 2 * M * input_dim + input_dim + 1) {
		// length scale derivatives and noise derivative
		// zero
	} else if (p == 2 * M * input_dim + input_dim) {
		//amplitude derivatives
		diSigmadp = -Upsi;
	} else {
		//derivatives with respect to inducing inputs or inducing length scales
		//TODO: unnecessary memory allocation on the heap!
		Eigen::VectorXd temp(M);
		size_t m = (p - input_dim) % M;
		size_t d = ((p - input_dim - m) / M) % input_dim;
		temp.array() = Uell.col(d).array() + (Uell(m, d) - ell(d));
		if (p < M * input_dim + input_dim) {
			//derivatives for inducing length scales
			//	dAdl(A, p, d, x, z, i)
			//	dA(i, :) = ((((z(i, d) - x(:, d))./p).^2-1./p))'.*A(i, :)/2
			diSigmadp.col(m).array() = ((U(m, d) - U.col(d).array())
					/ temp.array()).square() - 1 / temp.array();
			diSigmadp.col(m).array() *= (Uell(m, d) - ell(d) / 2)
					* Upsi.col(m).array() / 2;
			temp = diSigmadp.col(m);
			diSigmadp.row(m) = temp;
			diSigmadp(m, m) = 2 * diSigmadp(m, m);
		} else {
			diSigmadp.col(m).array() = (-U(m, d) + U.col(d).array())
					* Upsi.col(m).array() / temp.array();
			temp = diSigmadp.col(m);
			diSigmadp.row(m) = temp;
		}
	}
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
		Eigen::VectorXd& g) {
	grad(x1, x2, getWrappedKernelValue(x1, x2), g);
}

void MultiScale::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		double kernel_value, Eigen::VectorXd& grad) {
	//TODO: there is actually no need for a general gradient
	//think about adding a method gradDiag

	grad.segment(input_dim, 2 * M * input_dim).setZero();
	grad(2 * M * input_dim + input_dim) = kernel_value;
	if (x1 == x2) {
		grad.head(input_dim).fill(-kernel_value / 2);
		//noise gradient
		grad(2 * M * input_dim + input_dim + 1) = 2 * kernel_value;
	} else {
		//chain rule already applied
		grad.head(input_dim).array() = ((x1.array() - x2.array()).square()
				/ ell.array() - 1) * kernel_value / 2;
		grad(2 * M * input_dim + input_dim + 1) = 0.;
	}
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
	//this division has been moved into the for-loop above
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
