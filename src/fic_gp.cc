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
//TODO: this class confuses weight prior and inverse weight prior.
//TODO: bf_multi_scale returns the right things but under the wrong name.

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
	k_star.resize(M);
}

FICGaussianProcess::~FICGaussianProcess() {
//	  delete V;
//	  delete isqrtgamma;
}

double FICGaussianProcess::var_impl(const Eigen::VectorXd x_star) {
	//TODO: as far a I can tell this is the only usage of L
	//=> it's probably sufficient to really use only the Cholesky
	//and not necessary to compute the inverse
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
		k.resize(n);
		isqrtgamma.resize(n);
		dg.resize(n);
		V.resize(M, n);
		Phi.resize(M, n);
	}
	for (size_t i = 0; i < n; i++) {
		//TODO: remove allocation!
		Eigen::VectorXd xi = sampleset->x(i);
		Phi.col(i) = bf->computeBasisFunctionVector(xi);
		k(i) = bf->getWrappedKernelValue(xi, xi);
	}
	Luu = bf->getCholeskyOfInvertedSigma();
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
	// TODO: V*V^T is symmetric and it should be possible to reduce the costs.
	//Especially since V is an Mxn matrix!
	Lu = V * V.transpose() + Eigen::MatrixXd::Identity(M, M);
	Lu.topLeftCorner(M, M) = Lu.llt().matrixL();

	/*
	 * Here we have to divert from the Matlab implementation. in Matlab all
	 * the matrices are upper matrices. Here we have what they are supposed to be:
	 * lower matrices.
	 */
	//TODO: avoid allocation
	Eigen::MatrixXd temp = Luu * Lu;
	//the line below does not work. why?
//	L = L.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(M, M));
	L = temp.triangularView<Eigen::Lower>().solve(
			Eigen::MatrixXd::Identity(M, M));
	temp.transpose().triangularView<Eigen::Upper>().solveInPlace(L);
	L = L - bf->getSigma();
}

void FICGaussianProcess::updateCholesky(const double x[], double y) {
	//Do nothing and just recompute everything.
	//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;
}

void FICGaussianProcess::update_k_star(const Eigen::VectorXd &x_star) {
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
	double t3 = 0;
	//TODO: is this method numerically stable? what are typical values?
	for (size_t i = 0; i < M; i++) {
		t += log(Lu(i, i));
		t3 -= beta(i) * beta(i);
	}
	size_t n = sampleset->size();
	double t2 = 0;
	for (size_t i = 0; i < n; i++) {
		t2 += log(dg(i)) + r(i)*r(i);
	}
	//TODO: is this better? or should it be moved to the loop?
	t2 = t2 + n * log2pi;

	//TODO: the following call should be more efficient than the loops below. Does it compile?
//	t = Lu.diagonal().log().sum()-beta.squaredNorm()+dg.log().sum()+r.squaredNorm()+n*log2pi;
	return t + (t2  + t3) / 2;
}

Eigen::VectorXd FICGaussianProcess::log_likelihood_gradient_impl() {
	//TODO: move allocations to constructor
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
//    W = Ku./repmat(sqrt(dg)',nu,1);
	Eigen::MatrixXd W = Phi * isqrtgamma.asDiagonal();
	//    W = chol(Kuu+W*W'+snu2*eye(nu))'\Ku; % inv(K) = inv(G) - inv(G)*W'*W*inv(G);
	W = (W * W.transpose() + bf->getInverseOfSigma()).selfadjointView<Eigen::Lower>().llt().matrixL().solve(Phi);

	const std::vector<double>& targets = sampleset->y();
	size_t n = sampleset->size();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	Eigen::VectorXd al(n);
	//    al = (y-m - W'*(W*((y-m)./dg)))./dg;
	al = (y - W.transpose() * (W * (y.cwiseQuotient(dg)))).cwiseQuotient(dg);
//    B = iKuu*Ku;
	Eigen::MatrixXd B = bf->getSigma() * Phi;
//    % = Upsi^(-1)*Uvx
//    clear Ku Kuu iKuu; %KRZ - also the line below moved from above.
//    Wdg = W./repmat(dg',nu,1); w = B*al;
	Eigen::MatrixXd Wdg = W * dg.transpose().cwiseInverse().asDiagonal();
	Eigen::VectorXd w = B * al;

	Eigen::MatrixXd ddiagK(n, num_params);
	Eigen::VectorXd t(num_params);
	//TODO: in an SMGPR specific implementation this can be a lot more efficient!
	// 1) ignore all the zero entries
	// 2) use that temp is always the same
	// 3) the gradients for the length scales are all the same
	for (size_t j = 0; j < n; j++) {
		double temp = k(j);
		bf->grad(sampleset->x(j), sampleset->x(j), temp, t);
		ddiagK.row(j) = t;
	}
	Eigen::MatrixXd dKuui(M, M);
	Eigen::MatrixXd dKui(M, n);
	t.resize(M);
	for (size_t i = 0; i < num_params; i++) {
//      [ddiagKi,dKuui,dKui] = feval(cov{:}, hyp.cov, x, [], i);  % eval cov deriv
		bf->gradiSigma(i, dKuui);
		for (size_t j = 0; j < n; j++) {
			bf->gradBasisFunction(sampleset->x(j), Phi.col(j), i, t);
			dKui.col(j) = t;
		}
		//      R = 2*dKui-dKuui*B; v = ddiagKi - sum(R.*B,1)';   % diag part of cov deriv
		//TODO: check if these allocations are necessary and if move them to the constructor
		Eigen::VectorXd doublevec(1);
		Eigen::MatrixXd R = 2 * dKui - dKuui * B;
		Eigen::VectorXd v = ddiagK.col(i).transpose() - R.cwiseProduct(B).colwise().sum();
//      dnlZ.cov(i) = (ddiagKi'*(1./dg) +w'*(dKuui*w-2*(dKui*al)) -al'*(v.*al) ...
//                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2;
		//TODO: some expressions do not depend on i!
		//eg: Wdg.array().square().matrix().colwise().sum()
		doublevec = dg.cwiseInverse().transpose()*ddiagK.col(i) + w.transpose()*(dKuui*w-2*(dKui*al))
				-al.transpose()*(al.cwiseProduct(v))
//				- (R*Wdg.transpose()).cwiseProduct(B*Wdg.transpose()).sum()
				-Wdg.array().square().matrix().colwise().sum()*v
				;
		doublevec(0) -= (R*Wdg.transpose()).cwiseProduct(B*Wdg.transpose()).sum();
		gradient(i) = doublevec(0);
	}
	gradient/=2;
//    clear dKui; %KRZ
//    dnlZ.lik = sn2*(sum(1./dg) -sum(sum(W.*W,1)'./(dg.*dg)) -al'*al);
//    % since snu2 is a fixed fraction of sn2, there is a covariance-like term in
//    % the derivative as well
//    dKuui = 2*snu2; R = -dKuui*B; v = -sum(R.*B,1)';   % diag part of cov deriv
//    dnlZ.lik = dnlZ.lik + (w'*dKuui*w -al'*(v.*al)...
//                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2;
	return gradient;
}
}
