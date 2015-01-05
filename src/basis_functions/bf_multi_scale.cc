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

size_t MultiScale::get_param_dim_without_noise(size_t input_dim, size_t M) {
	//1 for length scale
	//input_dim length scales
	//M*input_dim inducing inputs
	//M*input_dim corresponding length scales
	//no need to take care of the noise
	return 2 * M * input_dim + input_dim + 1;
}

bool MultiScale::real_init() {
	//TODO: signal that Multiscale ignores the covariance function!
	CovFactory f;
	CovarianceFunction * expectedCov;
	expectedCov = f.create(input_dim, "CovSum ( CovSEard, CovNoise)");
	if (cov->to_string() != expectedCov->to_string()) {
		std::cerr
				<< "MultiScale GPR is only applicable for covariance function: "
				<< expectedCov->to_string() << std::endl;
		return false;
	}

	Upsi.resize(M, M);
	LUpsi.resize(M, M);
	iUpsi.resize(M, M);
	U.resize(M, input_dim);
	Uell.resize(M, input_dim);
	ell.resize(input_dim);
	temp.resize(M);
	temp_input_dim.resize(input_dim);
	UpsiCol.resize(M);
	factors.resize(M);
	delta.resize(input_dim);
	//this assures that previous_p can not correspond to a parameter number
	previous_p = get_param_dim() + 2;
	return true;
}

void MultiScale::putDiagWrapped(SampleSet * sampleSet, Eigen::VectorXd& diag) {
	diag.fill(c_over_ell_det + sn2);
}

Eigen::VectorXd MultiScale::computeBasisFunctionVector(
		const Eigen::VectorXd & x) {
	Eigen::VectorXd uvx(M);
	for (size_t i = 0; i < M; i++) {
		delta = x - U.row(i).transpose();
		double z = delta.transpose().cwiseQuotient(Uell.row(i)) * delta;
		z = exp(-0.5 * z);
		uvx(i) = z / factors(i);
	}
	return uvx;
}

void MultiScale::gradBasisFunction(const Eigen::VectorXd &x,
		const Eigen::VectorXd &phi, size_t p, Eigen::VectorXd &grad) {
	assert(grad.size() == phi.size());
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
	} else if (p >= input_dim && p < 2 * M * input_dim + input_dim) {
		bool lengthScaleDerivative = p < M * input_dim + input_dim;
		//use precomputed values where possible
		if (p != previous_p) {
			grad.setZero();
			setPreviousNumberAndDimensionForParameter(p, lengthScaleDerivative);
		}

		size_t m = previous_m;
		size_t d = previous_d;
		if (lengthScaleDerivative) {
			//length scale derivatives
			double t = (U(m, d) - x(d)) / Uell(m, d);
			grad(m) = (t * t - 1 / Uell(m, d)) * phi(m) / 2;
			grad(m) *= (Uell(m, d) - ell(d) / 2);
		} else {
			//inducing point derivatives
			grad(m) = (-U(m, d) + x(d)) / Uell(m, d) * phi(m);
		}
	} else {
		//amplitude and noise derivative
		grad.setZero();
	}
}

void inline MultiScale::setPreviousNumberAndDimensionForParameter(size_t p,
		bool lengthScaleDerivative) {
	previous_m = (p - input_dim) % M;
	previous_d = (p - input_dim - previous_m) / M;
	if (!lengthScaleDerivative)
		previous_d -= input_dim;
	previous_p = p;
}

bool MultiScale::gradBasisFunctionIsNull(size_t p) {
	return p >= 2 * M * input_dim + input_dim;
}

Eigen::MatrixXd MultiScale::getInverseOfSigma() {
	return Upsi;
}

void MultiScale::gradiSigma(size_t p, Eigen::MatrixXd & dSigmadp) {
	dSigmadp.setZero();
	if (p < input_dim) {
		// length scale derivatives
		// zero
	} else if (p == 2 * M * input_dim + input_dim + 1) {
		//noise derivative
		//little contribution due to the inducing noise
		dSigmadp.diagonal().fill(2 * snu2);
	} else if (p == 2 * M * input_dim + input_dim) {
		//amplitude derivatives
		//we need to subtract the inducing input noise since it is not affected by the amplitude
		dSigmadp = -Upsi + snu2 * Eigen::MatrixXd::Identity(M, M);
	} else {
		//derivatives with respect to inducing inputs or inducing length scales
		//don't call setPrevious...() here. it breaks things in gradBasisFunction()
		size_t m = (p - input_dim) % M;
		size_t d = ((p - input_dim - m) / M) % input_dim;
		temp.array() = Uell.col(d).array() + (Uell(m, d) - ell(d));

		//TODO: unintended copy
		UpsiCol = Upsi.col(m);
		//for the gradients we have to remove the inducing input noise (from the diagonal)
		UpsiCol(m) -= snu2;
		if (p < M * input_dim + input_dim) {
			//derivatives for inducing length scales
			dSigmadp.col(m).array() = ((U(m, d) - U.col(d).array())
					/ temp.array()).square() - temp.cwiseInverse().array();
			dSigmadp.col(m).array() *= (Uell(m, d) - ell(d) / 2)
					* UpsiCol.array() / 2;
			//TODO: can this be more efficient?
			temp = dSigmadp.col(m);
			dSigmadp.row(m) = temp;
			dSigmadp(m, m) *= 2;
		} else {
			//derivatives for inducing inputs
			dSigmadp.col(m).array() = (-U(m, d) + U.col(d).array())
					* UpsiCol.array() / temp.array();
			//TODO: can this be more efficient?
			temp = dSigmadp.col(m);
			dSigmadp.row(m) = temp;
		}
	}
}

bool MultiScale::gradiSigmaIsNull(size_t p) {
	return p < input_dim;
}

Eigen::MatrixXd MultiScale::getCholeskyOfInvertedSigma() {
	return LUpsi;
}

Eigen::MatrixXd MultiScale::getSigma() {
	return iUpsi;
}

double MultiScale::getLogDeterminantOfSigma() {
	return halfLogDetiUpsi;
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
	grad.segment(input_dim, 2 * M * input_dim).setZero();
	grad(2 * M * input_dim + input_dim + 1) = 0.;
	if (x1 == x2) {
		//does not influence the gradient
		kernel_value -= sn2;
		grad(2 * M * input_dim + input_dim + 1) = 2 * sn2;
	}
	grad(2 * M * input_dim + input_dim) = kernel_value;
	//chain rule already applied
	grad.head(input_dim).array() = ((x1.array() - x2.array()).square()
			/ ell.array() - 1) * kernel_value / 2;
}

void MultiScale::gradDiagWrapped(SampleSet * sampleset,
		const Eigen::VectorXd & diagK, size_t parameter,
		Eigen::VectorXd & gradient) {
	if (parameter < input_dim)
		gradient.fill(-c_over_ell_det / 2);
	else if (parameter == 2 * M * input_dim + input_dim)
		gradient.fill(c_over_ell_det);
	else if (parameter == 2 * M * input_dim + input_dim + 1)
		gradient.fill(2 * sn2);
	else
		gradient.setZero();
}

bool MultiScale::gradDiagWrappedIsNull(size_t parameter) {
	return parameter >= input_dim && parameter < 2 * M * input_dim + input_dim;
}

void MultiScale::log_hyper_updated(const Eigen::VectorXd& p) {
	for (size_t i = 0; i < input_dim; i++)
		ell(i) = exp(loghyper(i));
	size_t idx = input_dim;
	factors.fill(1);
	for (size_t d = 0; d < input_dim; d++) {
		for (size_t m = 0; m < M; m++) {
			/*
			 * For robustness half of the length scales is added to the
			 * inducing length scales.
			 */
			Uell(m, d) = exp(loghyper(idx)) + ell(d) / 2;
			U(m, d) = loghyper(idx + M * input_dim);
			factors(m) *= 2 * M_PI * Uell(m, d);
			idx++;
		}
	}
	factors.array() = factors.array().sqrt();
	c = exp(loghyper(2 * M * input_dim + input_dim));
	double ell_determinant_factor = 1;
	for (size_t i = 0; i < input_dim; i++) {
		ell_determinant_factor *= 2 * M_PI * ell(i);
	}
	ell_determinant_factor = sqrt(ell_determinant_factor);
	c_over_ell_det = c / ell_determinant_factor;

	sn2 = exp(2 * loghyper(2 * M * input_dim + input_dim + 1));
	snu2 = 1e-6 * sn2;
	initializeMatrices();
}

void MultiScale::initializeMatrices() {
	/**
	 * Initializes the matrices LUpsi and iUpsi.
	 */
	for (size_t i = 0; i < M; i++) {
		//can not use delta here. would be overwritten in g!
		temp_input_dim = Uell.row(i).transpose() - ell;
		//don't transpose temp in place - it breaks things later
		for (size_t j = 0; j < i; j++) {
			Upsi(i, j) = g(U.row(i), U.row(j), temp_input_dim.transpose() + Uell.row(j)) / c;
		}

		Upsi(i, i) = g(U.row(i), U.row(i), temp_input_dim.transpose() + Uell.row(i)) / c + snu2;
	}

	//this division has been moved into the for-loop above
	//LUpsi = LUpsi / c;
	Upsi = Upsi.selfadjointView<Eigen::Lower>();

	//TODO: refactor: Is it possible to remove the topLeftCorner-calls?
	LUpsi.topLeftCorner(M, M) = Upsi.topLeftCorner(M, M).selfadjointView<
			Eigen::Lower>().llt().matrixL();
	//TODO: is this numerically stable? Probably it's better to make a sum over the logs!
	halfLogDetiUpsi = -log(LUpsi.diagonal().prod());
	iUpsi = LUpsi.topLeftCorner(M, M).triangularView<Eigen::Lower>().solve(
			LUpsi.Identity(M, M));
	//TODO: it should be sufficient to transpose here
	LUpsi.topLeftCorner(M, M).triangularView<Eigen::Lower>().adjoint().solveInPlace(
			iUpsi);
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
	delta = x1 - x2;
	double z = delta.cwiseQuotient(sigma).transpose() * delta;
	z = exp(-0.5 * z);
	double p = 1;
	for (size_t i = 0; i < input_dim; i++) {
		p *= 2 * M_PI * sigma(i);
	}
	p = sqrt(p);
	return z / p;
}

}
