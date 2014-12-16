// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "basis_functions/bf_solin.h"
#include <cmath>

namespace libgp {

libgp::Solin::~Solin() {
}

Eigen::VectorXd libgp::Solin::computeBasisFunctionVector(
		const Eigen::VectorXd& x) {
	Eigen::VectorXd phi(M);
	phi.tail(M - input_dim * M_intern).setZero();
	//    Md = M;
	size_t Md = M_intern;
	phi.head(M_intern).array() = ((x(0) * m.array() + L) / L).sin() / sqrt(L);
	//    for d = 2:D
	for (size_t d = 1; d < input_dim; d++) {
		phi_1D.array() = ((x(d) * m.array() + L) / L).sin() / sqrt(L);
//        t2 = zeros(Md*M, sz);
//        for m = 1:M
		for (size_t j = 1; j < M_intern; j++) {
//            idx = (m-1)*Md+(1:Md);
//            t2(idx, :) = temp * diag(squeeze(Phi(d, m, :)));
			//TODO: since we iterate over M_intern anyway: would it be faster to compute phi_1D(j) here?
			phi.segment(j * Md, Md) = phi.head(Md) * phi_1D(j);
		}
		phi.head(Md).array() *= phi_1D(0);
//        end
//        temp = t2;
//        Md = Md * M;
		Md *= M_intern;
//    end
	}
//    K = temp;
	return phi;
}

Eigen::MatrixXd libgp::Solin::getInverseOfSigma() {
	return iSigma;
}

Eigen::MatrixXd libgp::Solin::getCholeskyOfInvertedSigma() {
	return choliSigma;
}

Eigen::MatrixXd libgp::Solin::getSigma() {
	return Sigma;
}

double libgp::Solin::getLogDeterminantOfSigma() {
	return logDetSigma;
}

void libgp::Solin::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& grad) {
}

void libgp::Solin::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		double kernel_value, Eigen::VectorXd& grad) {
}

void libgp::Solin::gradBasisFunction(const Eigen::VectorXd& x,
		const Eigen::VectorXd& phi, size_t p, Eigen::VectorXd& grad) {
	grad.fill(0);
}

bool libgp::Solin::gradBasisFunctionIsNull(size_t p) {
	//the basis functions are independent of all hyper-parameters
	return true;
}

void libgp::Solin::gradiSigma(size_t p, Eigen::MatrixXd& diSigmadp) {
	if (p < input_dim) {
		counter.fill(1);
		double temp;
		for (size_t i = 0; i < MToTheD; i++) {
			/*
			 * df^-1/dx = -f^-2*df/dx
			 * In our case df/dx=f*c and therefore we get
			 * df^-1/dx=-f^-1*c
			 */
			diSigmadp(i, i) = (ell(p) * piOverLOver2Sqrd * counter(p)
					* counter(p) - 1) * iSigma.diagonal()(i);

			incCounter(counter);
		}
	} else if (p == input_dim){
		diSigmadp.diagonal().head(MToTheD) = -2*iSigma.diagonal().head(MToTheD);
	}
	else
		diSigmadp.setZero();
}

bool libgp::Solin::gradiSigmaIsNull(size_t p) {
	//noise gradient?
	if (p == param_dim - 1)
		return true;
	return false;
}

std::string libgp::Solin::to_string() {
	return "Solin";
}

void libgp::Solin::log_hyper_updated(const Eigen::VectorXd& p) {
	//initialize hyper-parameters
	double temp = 1;
	for (size_t i = 0; i < input_dim; i++) {
		ell(i) = exp(2 * p(i));
		temp *= exp(p(i));
	}
	sf2 = exp(2 * p(input_dim));

	//initialize spectral density
	c = sf2 * pow(2 * M_PI, 0.5 * input_dim) * temp;
	temp = M_PI / L / 2;
	temp *= temp;
	piOverLOver2Sqrd = temp;

	//create Sigma and associated fields
	Eigen::VectorXd lambdaSquared(input_dim);
	logDetSigma = 0;

	counter.fill(1);
	for (size_t i = 0; i < MToTheD; i++) {
		lambdaSquared.array() = piOverLOver2Sqrd
				* counter.array().square().cast<double>();
		//TODO: does this work?
		double value = spectralDensity(lambdaSquared);
		Sigma.diagonal()(i) = value;
		iSigma.diagonal()(i) = 1 / value;
		choliSigma.diagonal()(i) = 1 / sqrt(value);
		logDetSigma += log(value);

		incCounter(counter);
	}

	logDetSigma /= 2;
}

inline double Solin::spectralDensity(const Eigen::VectorXd & lambdaSquared) {
	return c * exp(-lambdaSquared.cwiseProduct(ell).sum() / 2);
}

inline void Solin::incCounter(Eigen::VectorXi & counter) {
	for (size_t idx = 0; idx < input_dim; idx++) {
		double fill = counter(idx) % (M_intern + 1) + 1;
		counter(idx) = fill;
		if (fill > 1)
			break;
	}
}

size_t Solin::get_param_dim_without_noise(size_t input_dim,
		size_t num_basis_functions) {
	//length scales + amplitude
	//no need to take care of the noise
	return input_dim + 1;
}

bool libgp::Solin::real_init() {
	//TODO: make sure that we wrap the right covariance function!

	L = 1.2;

	ell.resize(input_dim);

	M_intern = std::floor(std::pow(M, 1. / input_dim));
	//casts for Visual Studio support
	MToTheD = (size_t) std::pow((double) M_intern, (int) input_dim);

	//TODO: make this integer?
	counter.resize(input_dim);

	m.resize(M_intern);
	phi_1D.resize(M_intern);
	for (size_t i = 1; i <= M_intern; i++)
		m(i - 1) = M_PI * i / 2;
	assert(MToTheD <= M);
	Sigma.resize(M);
	Sigma.diagonal().tail(M - MToTheD).fill(1);
	iSigma.resize(M);
	iSigma.diagonal().tail(M - MToTheD).fill(1);
	choliSigma.resize(M);
	choliSigma.diagonal().tail(M - MToTheD).fill(1);
	return true;
}

}
