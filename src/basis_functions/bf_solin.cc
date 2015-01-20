// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "basis_functions/bf_solin.h"
#include <cmath>

namespace libgp {

libgp::Solin::Solin() : L(1.2), sqrtL(sqrt(1.2)){
}

libgp::Solin::~Solin() {
}

Eigen::VectorXd libgp::Solin::computeBasisFunctionVector(
		const Eigen::VectorXd& x) {
	Eigen::VectorXd phi(M);
	phi.tail(M - input_dim * M_intern).setZero();
	//for this call it needs to be phi and not phi_1D
	phi1D(x(0), phi);
	size_t Md = M_intern;
	for (size_t d = 1; d < input_dim; d++) {
		//it's not faster to compute phi_1D(j) in the loop
		phi1D(x(d), phi_1D);
		//we need to start at 1 as we do not want to overwrite phi.head(Md)
		for (size_t j = 1; j < M_intern; j++)
			phi.segment(j * Md, Md) = phi.head(Md) * phi_1D(j);
		//no we want to overwrite phi.head(Md)
		phi.head(Md).array() *= phi_1D(0);
		Md *= M_intern;
	}
	return phi;
}

inline void Solin::phi1D(const double & xd, Eigen::VectorXd & phi){
	phi.head(M_intern).array() = (m.array() * (xd + L)).sin() / sqrtL;
}

const Eigen::MatrixXd & libgp::Solin::getInverseOfSigma() {
	return iSigma;
}

const Eigen::MatrixXd & libgp::Solin::getCholeskyOfInvertedSigma() {
	return choliSigma;
}

const Eigen::MatrixXd & libgp::Solin::getSigma() {
	return Sigma;
}

double libgp::Solin::getLogDeterminantOfSigma() {
	return logDetSigma;
}

void libgp::Solin::grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
		Eigen::VectorXd& grad) {
	std::cout << "Warning: grad method is not implemented for bf_solin" << std::endl;
	grad.fill(1);
}

bool Solin::gradDiagWrappedIsNull(size_t parameter){
	return false;
}


void libgp::Solin::gradBasisFunction(SampleSet * sampleSet,
		const Eigen::MatrixXd& Phi, size_t p, Eigen::MatrixXd& Grad) {
	Grad.fill(0);
}

bool libgp::Solin::gradBasisFunctionIsNull(size_t p) {
	//the basis functions are independent of all hyper-parameters
	return true;
}

void libgp::Solin::gradiSigma(size_t p, Eigen::MatrixXd& diSigmadp) {
	if (p < input_dim) {
//		counter.fill(1);
//		double temp = ell(p) * piOverLOver2Sqrd;
//		for (size_t i = 0; i < MToTheD; i++) {
//			/*
//			 * df^-1/dx = -f^-2*df/dx
//			 * In our case df/dx=f*c and therefore we get
//			 * df^-1/dx=-f^-1*c
//			 */
//			diSigmadp(i, i) = (temp * counter(p)
//					* counter(p) - 1) * iSigma.diagonal()(i);
//			incCounter(counter);
//		}
		diSigmadp.diagonal().head(MToTheD) = (ell(p) * piOverLOver2Sqrd * indices.col(p).array().square().cast<double>() - 1.) * iSigma.diagonal().head(MToTheD).array();
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
	double temp = 0;
	for (size_t i = 0; i < input_dim; i++) {
		ell(i) = exp(2 * p(i));
		temp += p(i);
	}
	sf2 = exp(2 * p(input_dim));

	//initialize spectral density constants
	c = sf2 * pow(2 * M_PI, 0.5 * input_dim) * exp(temp);
	temp = M_PI / L / 2;
	temp *= temp;
	piOverLOver2Sqrd = temp;

	//create Sigma and associated fields
	Eigen::VectorXd lambdaSquared(input_dim);
	logDetSigma = 0;

	counter.fill(1);
	for (size_t i = 0; i < MToTheD; i++) {
		lambdaSquared.array() = piOverLOver2Sqrd
				* indices.row(i).array().square().cast<double>();
		double value = spectralDensity(lambdaSquared);
		Sigma.diagonal()(i) = value;
		iSigma.diagonal()(i) = 1 / value;
		choliSigma.diagonal()(i) = 1 / sqrt(value);
		logDetSigma += log(value);
	}
	//logDetSigma is supposed to contain half of the lot determinant
	logDetSigma /= 2;
}

inline double Solin::spectralDensity(const Eigen::VectorXd & lambdaSquared) {
	return c * exp(-lambdaSquared.cwiseProduct(ell).sum() / 2);
}

inline void Solin::incCounter(Eigen::VectorXi & counter) {
	for (size_t idx = 0; idx < input_dim; idx++) {
		//this part is not the bottleneck
//		size_t fill = (counter(idx) % M_intern) + 1;
		size_t fill = counter(idx) + 1;
		//overflow ?
		if(fill == M_intern)
			fill = 1;
		counter(idx) = fill;
		//no overflow, no need to increase the next values
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

	ell.resize(input_dim);

	M_intern = std::floor(std::pow(M, 1. / input_dim));
	//casts for Visual Studio support
	MToTheD = (size_t) std::pow((double) M_intern, (int) input_dim);

	counter.resize(input_dim);
	indices.resize(MToTheD, input_dim);
	for(size_t i = 0; i < MToTheD; i++){
		counter.fill(1);
		indices.row(i) = counter;
		incCounter(counter);
	}

	phi_1D.resize(M_intern);
	m.resize(M_intern);
	for (size_t i = 1; i <= M_intern; i++)
		m(i - 1) = M_PI * i / L / 2;
	assert(MToTheD <= M);
	Sigma.resize(M, M);
	Sigma.setZero();
	Sigma.diagonal().tail(M - MToTheD).fill(1);
	iSigma.resize(M, M);
	iSigma.setZero();
	iSigma.diagonal().tail(M - MToTheD).fill(1);
	choliSigma.resize(M, M);
	choliSigma.setZero();
	choliSigma.diagonal().tail(M - MToTheD).fill(1);
	return true;
}

}
