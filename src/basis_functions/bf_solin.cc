// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "basis_functions/bf_solin.h"
#include <cmath>
#include "cov_factory.h"
#include "cov.h"


namespace libgp {

Solin::Solin():piOver2Sqrd(M_PI*M_PI/4){

}

libgp::Solin::~Solin() {
}

void libgp::Solin::setL(Eigen::VectorXd L){
	this->L = L;
	this->sqrtL.array() = L.array().sqrt();
	this->squaredL.array() = L.array().square();
	std::cerr << "bf_solin: INFO: L has been initialized. (L=" << L.transpose() << std::endl;
	log_hyper_updated(loghyper);
}

Eigen::VectorXd libgp::Solin::computeBasisFunctionVector(
		const Eigen::VectorXd& x) {
	Eigen::VectorXd phi(M);
	phi.tail(M - input_dim * M_intern).setZero();
	//for this call it needs to be phi and not phi_1D
	phi1D(x, 0, phi);
	size_t Md = M_intern;
	for (size_t d = 1; d < input_dim; d++) {
		//it's not faster to compute phi_1D(j) in the next loop
		phi1D(x, d, phi_1D);
		//we need to start at 1 as we do not want to overwrite phi.head(Md)
		for (size_t j = 1; j < M_intern; j++)
			phi.segment(j * Md, Md) = phi.head(Md) * phi_1D(j);
		//now we want to overwrite phi.head(Md)
		phi.head(Md).array() *= phi_1D(0);
		Md *= M_intern;
	}
	return phi;
}

inline void Solin::phi1D(const Eigen::VectorXd & x, size_t d, Eigen::VectorXd & phi) {
	phi.head(M_intern).array() = (m.array() * (x(d) + L(d)) / L(d)).sin() / sqrtL(d);
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
	std::cout << "Warning: grad method is not implemented for bf_solin"
			<< std::endl;
	grad.fill(1);
}

bool Solin::gradDiagWrappedIsNull(size_t parameter) {
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
		diSigmadp.diagonal().head(MToTheD).array() = (ell(p) * piOver2Sqrd / squaredL(p)
				* indices.col(p).array().square().cast<double>() - 1.)
				* iSigma.diagonal().head(MToTheD).array();
	} else if (p == input_dim) {
		diSigmadp.diagonal().head(MToTheD) = -2
				* iSigma.diagonal().head(MToTheD);
	} else
		diSigmadp.setZero();
}

bool libgp::Solin::gradiSigmaIsNull(size_t p) {
	//noise gradient?
	if (p == param_dim - 1)
		return true;
	return false;
}

std::string libgp::Solin::to_string() {
	return BF_SOLIN_NAME;
}

std::string Solin::pretty_print_parameters(){
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	std::stringstream ss;
	ss << (2*loghyper.array().exp()).format(CleanFmt);
	return ss.str();
}

void libgp::Solin::log_hyper_updated(const Eigen::VectorXd& p) {
	//initialize hyper-parameters
	for (size_t i = 0; i < input_dim; i++)
		ell(i) = exp(2 * p(i));
	sf2 = exp(2 * p(input_dim));

	//initialize spectral density constants
//	c = sf2 * pow(2 * M_PI, 0.5 * input_dim) * exp(p.head(input_dim).sum());
	log_c = 2 * p(input_dim) + input_dim * log(2. * M_PI) / 2 + p.head(input_dim).sum();
	//create Sigma and associated fields
	Eigen::VectorXd lambdaSquared(input_dim);
	logDetSigma = 0;

	for (size_t i = 0; i < MToTheD; i++) {
		lambdaSquared.array() = piOver2Sqrd
				* indices.row(i).array().square().cast<double>() / squaredL.transpose().array();
		double value = logSpectralDensity(lambdaSquared);
		logDetSigma += value;
//		value = exp(value);
//		if(value == 0.0)
//			value = 1e-200;
		choliSigma.diagonal()(i) = exp(-value/2);
		//TODO: here things get messy
		Sigma.diagonal()(i) = exp(value);
		iSigma.diagonal()(i) = exp(-value);
	}
	//logDetSigma is supposed to contain half of the log determinant
	logDetSigma /= 2;
}

inline double Solin::logSpectralDensity(const Eigen::VectorXd & lambdaSquared) {
//	return c * exp(-lambdaSquared.cwiseProduct(ell).sum() / 2);
	//this is numerically more stable
	return log_c - lambdaSquared.cwiseProduct(ell).sum() / 2;
}

inline void Solin::incCounter(Eigen::VectorXi & counter) {
	for (size_t idx = 0; idx < input_dim; idx++) {
		size_t c = counter(idx) + 1;
		//overflow ?
		if (c == M_intern + 1)
			c = 1;
		counter(idx) = c;
		//no overflow, no need to increase the next values
		if (c > 1)
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
	CovFactory f;
	CovarianceFunction * expectedCov;
	expectedCov = f.create(input_dim, "CovSum ( CovSEard, CovNoise)");
	if (cov->to_string() != expectedCov->to_string()) {
		std::cerr
				<< "This implementation of Solin's Laplace Approximation"
						" is only applicable for covariance function: "
				<< expectedCov->to_string() << std::endl;
		return false;
	}
	ell.resize(input_dim);

	M_intern = std::floor(std::pow(M, 1. / input_dim));
	//casts for Visual Studio support
	MToTheD = (size_t) std::pow((double) M_intern, (int) input_dim);

	counter.resize(input_dim);
	indices.resize(MToTheD, input_dim);
	counter.fill(1);
	for (size_t i = 0; i < MToTheD; i++) {
		indices.row(i) = counter;
		incCounter(counter);
	}

	phi_1D.resize(M_intern);
	m.resize(M_intern);
	for (size_t i = 1; i <= M_intern; i++)
		m(i - 1) = M_PI * i / 2;
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

	L.resize(input_dim);
	L.fill(4);
	std::cerr << "bf_solin: WARNING: L is initialized to default value " << L.transpose() << std::endl;
	sqrtL.resize(input_dim);
	sqrtL.array() = L.array().sqrt();
	squaredL.resize(input_dim);
	squaredL.array() = L.array().square();
	return true;
}

}
