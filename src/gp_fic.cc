// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_fic.h"
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
//	W.resize(M, M);
	BWdg.resize(M, M);
	w.resize(M);
	beta.resize(M);
	k_star.resize(M);
	dKuui.resize(M, M);
	temp.resize(M);
	LuuLu.resize(M, M);
}

FICGaussianProcess::~FICGaussianProcess() {
}

double FICGaussianProcess::var_impl(const Eigen::VectorXd &x_star) {
	return bf->getWrappedKernelValue(x_star, x_star)
			+ k_star.transpose() * L * k_star;
}

void FICGaussianProcess::computeCholesky() {
	/*
	 * This method does not compute the Cholesky in the same sense as
	 * the GaussianProcess class does. Here the same thing happens as
	 * in infFITC.m from the gpml toolbox by Rasmussen and Nikisch. The
	 * method name computeCholesky is kept for abstraction reasons.
	 */
	size_t n = sampleset->size();

	if (n > isqrtgamma.rows()) {
		k.resize(n);
		isqrtgamma.resize(n);
		dg.resize(n);
		V.resize(M, n);
		Phi.resize(M, n);
	}
	for (size_t i = 0; i < n; i++)
		Phi.col(i) = bf->computeBasisFunctionVector(sampleset->x(i));
	bf->putDiagWrapped(sampleset, k);

	/*
	 * TODO: could we just multiply Phi with sqrt(gamma) HERE instead of using
	 * the inverse later? What's more stable?
	 */
	V = bf->getCholeskyOfInvertedSigma().triangularView<Eigen::Lower>().solve(
			Phi);
	//noise is already added in k
	//TODO: symmetric matrix multiplication = potential for speed up
	dg = k - (V.transpose() * V).diagonal();
	//the line below is not faster
//	dg.array() = k.array() - V.array().square().colwise().sum().transpose();

//	isqrtgamma = isqrtgamma.cwiseInverse().sqrt();
	//TODO: first occurence of dg.cwiseInverse() should we save that result? O(n) Operation!
	isqrtgamma.array() = 1 / dg.array().sqrt();

	//the line below would be faster here but not any longer in compute alpha
	//	Lu = V * dg.cwiseInverse().asDiagonal() * V.transpose() + Eigen::MatrixXd::Identity(M, M);
	V = V * isqrtgamma.asDiagonal();
	Lu.setZero();
	Lu.selfadjointView<Eigen::Lower>().rankUpdate(V);
	Lu.diagonal().array()+=1;
	Lu = Lu.llt().matrixL();

	/*
	 * Here we have to divert from the Matlab implementation. in Matlab all
	 * the matrices are upper matrices. Here we have what they are supposed to be:
	 * lower matrices.
	 */
//	LuuLu = bf->getCholeskyOfInvertedSigma().triangularView<Eigen::Lower>() * Lu.triangularView<Eigen::Lower>();
	//TODO: check if the line above is faster
	LuuLu = bf->getCholeskyOfInvertedSigma() * Lu;
	//a solveInPlace with L.setIdentity() is not faster
	L = LuuLu.triangularView<Eigen::Lower>().solve(
			Eigen::MatrixXd::Identity(M, M));
	LuuLu.transpose().triangularView<Eigen::Upper>().solveInPlace(L);
	L = L - bf->getSigma();
}

void FICGaussianProcess::updateCholesky(const double x[], double y) {
	//Do nothing and just recompute everything.
	//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;

	//TODO: give signal to recompute y^Ty (also in gp_deg)
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
	//TODO: r and beta are used only in the computation of llh
	//potential for speed up?
	r.array() = y.array() * isqrtgamma.array();
	beta = V * r;
	//if we change definition of V: beta = V * y / gamma
	//V is also used only here
	/*
	 * In the Matlab implementation Luu and Lu are upper matrices and that's
	 * why we need to transpose here.
	 */
	Lu.triangularView<Eigen::Lower>().solveInPlace(beta);
	//alpha = Luu\(Lu\be)
	alpha = Lu.transpose().triangularView<Eigen::Upper>().solve(beta);
	bf->getCholeskyOfInvertedSigma().transpose().triangularView<Eigen::Upper>().solveInPlace(
			alpha);
}

double FICGaussianProcess::log_likelihood_impl() {
	size_t n = sampleset->size();
	return Lu.diagonal().array().log().sum()
			+ (-beta.squaredNorm() + dg.array().log().sum() + r.squaredNorm()
					+ n * log2pi) / 2;
}

Eigen::VectorXd FICGaussianProcess::log_likelihood_gradient_impl() {
	//TODO: move stuff that is parameter independent to computeAlpha or something
	//TODO: move allocations to constructor
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
	const std::vector<double>& targets = sampleset->y();
	size_t n = sampleset->size();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	//TODO: move to extra function
	if (n > al.size()) {
		al.resize(n);
		W.resize(M, n);
		Wdg.resize(M, n);
		B.resize(M, n);
		ddiagK.resize(n);
		dKui.resize(M, n);
		R.resize(M, n);
		v.resize(n);
		WdgSum.resize(n);
	}

	dKui.setZero();
	dKuui.setZero();
	ddiagK.setZero();

	//    W = Ku./repmat(sqrt(dg)',nu,1);
	//    W = chol(Kuu+W*W'+snu2*eye(nu))'\Ku;
	//TODO: in the computation of V this step could be made faster
	//we misuse BWdg as temporary variable here
	BWdg.setZero();
	BWdg.selfadjointView<Eigen::Lower>().rankUpdate(Phi * isqrtgamma.asDiagonal());
	BWdg+=bf->getInverseOfSigma();
	W = BWdg.selfadjointView<Eigen::Lower>().llt().matrixL().solve(Phi);
//	W =
//			(Phi * dg.cwiseInverse().asDiagonal() * Phi.transpose()
//					+ bf->getInverseOfSigma()).selfadjointView<Eigen::Lower>().llt().matrixL().solve(
//					Phi);

	//    al = (y-m - W'*(W*((y-m)./dg)))./dg;
	al = (y - W.transpose() * (W * (y.cwiseQuotient(dg)))).cwiseQuotient(dg);
//    B = iKuu*Ku;
	B = bf->getSigma() * Phi;
//    % = Upsi^(-1)*Uvx
//    Wdg = W./repmat(dg',nu,1); w = B*al;
	Wdg = W * dg.cwiseInverse().asDiagonal();
	w = B * al;
	BWdg = B * Wdg.transpose();
	WdgSum = Wdg.array().square().matrix().colwise().sum();
	for (size_t i = 0; i < num_params; i++) {
//      [ddiagKi,dKuui,dKui] = feval(cov{:}, hyp.cov, x, [], i);
		bool gradiSigmaIsNull = bf->gradiSigmaIsNull(i);
		bool gradBasisFunctionIsNull = bf->gradBasisFunctionIsNull(i);

		double wdKuuiw;
		if (!gradiSigmaIsNull) {
			bf->gradiSigma(i, dKuui);
			wdKuuiw = (w.transpose() * dKuui * w).sum();
		} else {
//			dKuui.setZero();
			wdKuuiw = 0;
		}

		double wdKuial;
		if (!gradBasisFunctionIsNull) {
			bf->gradBasisFunction(sampleset, Phi, i, dKui);
			wdKuial = 2 * (w.transpose()  * dKui * al).sum();
			//R = 2*dKui-dKuui*B;
			if(!gradiSigmaIsNull)
				R = 2 * dKui - dKuui * B;
			else
				//TODO: do we want to compute that?!
				R = 2 * dKui;
		} else {
			//this branch will be entered only for two parameters (in case of MultiScale)
			if(!gradiSigmaIsNull)
				R = -dKuui * B;
			else
				//TODO: can we avoid that?
				R.setZero();
//			dKui.setZero();
			wdKuial = 0;
		}

		double ddiagK_idg;
		if (!bf->gradDiagWrappedIsNull(i)) {
			bf->gradDiagWrapped(sampleset, k, i, ddiagK);
			ddiagK_idg = ddiagK.cwiseQuotient(dg).sum();
			// v = ddiagKi - sum(R.*B,1)';   % diag part of cov deriv
			v = ddiagK.transpose() - R.cwiseProduct(B).colwise().sum();
		} else {
			ddiagK_idg = 0;
			v = -R.cwiseProduct(B).colwise().sum();
		}

//      dnlZ.cov(i) = (ddiagKi'*(1./dg) +w'*(dKuui*w-2*(dKui*al)) -al'*(v.*al) ...
//                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2;
		gradient(i) = ddiagK_idg
		//TODO: line below can be optimized (if either dKuui or dKui are 0)
				+ wdKuuiw - wdKuial
				//TODO: save al^2?
				- (v.array() * al.array().square()).sum()
				- WdgSum.cwiseProduct(v).sum()
				- (R * Wdg.transpose()).cwiseProduct(BWdg).sum();
	}
	gradient /= 2;
	//noise gradient included in the loop above
	return gradient;
}
}
