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
		Phi.resize(M, n);
	}
	//corresponds to diagK in infFITC
	Eigen::VectorXd k(n);
	for (size_t i = 0; i < n; i++) {
		Eigen::VectorXd xi = sampleset->x(i);
		Eigen::VectorXd phi = bf->computeBasisFunctionVector(xi);
		//TODO: is there a faster operation?
		for (size_t j = 0; j < M; j++) {
			Phi(j, i) = phi(j);
		}
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
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
//    W = Ku./repmat(sqrt(dg)',nu,1);
	Eigen::MatrixXd W = Phi * isqrtgamma.asDiagonal();
	W = W * W.transpose();
//    W = chol(Kuu+W*W'+snu2*eye(nu))'\Ku; % inv(K) = inv(G) - inv(G)*W'*W*inv(G);
	W = W + bf->getInverseWeightPrior();
	W = W.selfadjointView<Eigen::Lower>().llt().solve(Phi);
//    % = (Avv/sn2)^(-1/2)*Uvx
//    al = (y-m - W'*(W*((y-m)./dg)))./dg;
	const std::vector<double>& targets = sampleset->y();
	size_t n = sampleset->size();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	Eigen::VectorXd al(n);
	al.array() = (y - W.transpose() * (W * (y.array() * isqrtgamma.array()).matrix())).array() * isqrtgamma.array();
//    % = (y - Uvx'*(Avv/sn2)^(-1)*Uvx*Gamma^(-1)*y)*Gamma^(-1)
//    B = iKuu*Ku;
	Eigen::MatrixXd B = bf->getWeightPrior() * Phi;
//    % = Upsi^(-1)*Uvx
//    clear Ku Kuu iKuu; %KRZ - also the line below moved from above.
//    Wdg = W./repmat(dg',nu,1); w = B*al;
	Eigen::MatrixXd Wdg = W * isqrtgamma.asDiagonal();
	Eigen::VectorXd w = B * al;
//
//    % w = Upsi^(-1)*Uvx*Gamma^(-1/2)*y - Upsi^(-1)*Uvx*Uvx'*v
//    %KRZ - free more memory.
	for(size_t i = 0; i < num_params; i++){
//      [ddiagKi,dKuui,dKui] = feval(cov{:}, hyp.cov, x, [], i);  % eval cov deriv
		Eigen::VectorXd ddiagKi = bf->gradDiag(i);
		Eigen::MatrixXd dKuui = bf->gradInverseWeightPrior(i);
		Eigen::MatrixXd dKui = bf->gradBasisFunctionVector(i);
		//TODO: first implement gradients of multi_scale to see how that works!
//      R = 2*dKui-dKuui*B; v = ddiagKi - sum(R.*B,1)';   % diag part of cov deriv
//      % R = 2*dUvx-dUpsi*Upsi^(-1)*Uvx
//      % v = dGamma?
//      dnlZ.cov(i) = (ddiagKi'*(1./dg) +w'*(dKuui*w-2*(dKui*al)) -al'*(v.*al) ...
//                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2;
//      % = tr(dGAmma*Gamma^(-1)) + ...
//    end
//    clear dKui; %KRZ
//    dnlZ.lik = sn2*(sum(1./dg) -sum(sum(W.*W,1)'./(dg.*dg)) -al'*al);
//    % since snu2 is a fixed fraction of sn2, there is a covariance-like term in
//    % the derivative as well
//    dKuui = 2*snu2; R = -dKuui*B; v = -sum(R.*B,1)';   % diag part of cov deriv
//    dnlZ.lik = dnlZ.lik + (w'*dKuui*w -al'*(v.*al)...
//                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2;
	}
	return gradient;
}
}
