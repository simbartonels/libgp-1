#include "util/time_call.h"
#include <Eigen/Dense>

Eigen::MatrixXd Mat1;
Eigen::MatrixXd Mat2;
Eigen::MatrixXd Target;
Eigen::VectorXd diag;

void mult_baseline(){
	Target = Mat1 * Mat2;
}

void mult1(){
	Target = Mat1*Mat1.transpose();
}

void mult1_alternative(){
	size_t n = 500;
	for(size_t i=0; i < n; i++){
		for(size_t j=0; j < i; j++){
			Target(i, j) = Mat1.row(i) * Mat1.row(j).transpose();
			Target(j, i) = Target(i, j);
		}
		Target(i, i) = Mat1.row(i).norm();
	}
}

void mult1_alternative2(){
	size_t n = 500;
	for(size_t i=0; i < n; i++){
		for(size_t j=0; j < i; j++){
			//does not work for whichever reason
			Target(i, j) = Mat1.row(i) * Mat1.col(j);
			Target(j, i) = Target(i, j);
		}
		Target(i, i) = Mat1.row(i).norm();
	}
}

void mult1_alternative3(){
//	Target.setZero();
	//TODO: this is a lot faster! use it!
	Target.selfadjointView<Eigen::Lower>().rankUpdate(Mat1);
	//this part might not even be necessary. check that!
	//TODO: use selfadjointView where possible
	//TODO: is it possible to it as return type!?
	Target.triangularView<Eigen::StrictlyUpper>() = Target.transpose();
	//TODO: MAKE SURE THE RESULT IS INDEED THE SAME!!!
}

void mult2(){
	Target = Mat2.transpose() * Mat2;
}

void mult_diag(){
	Target = Mat1 * diag.asDiagonal() * Mat1.transpose();
}

int main(int argc, char const *argv[]) {
	Mat1.resize(500, 3000);
	Mat1.setRandom();
	Mat2.resize(3000, 500);
	Mat2.setRandom();
	diag.resize(3000);
	diag.setRandom();

	Target.resize(500, 500);
	std::cout << "initialization complete. running benchmark" << std::endl;
//	compare_time(mult_baseline, mult1, 5);
//	compare_time(mult1, mult2, 5);
//	compare_time(mult1, mult_diag, 5);

//	Target.setZero();
	compare_time(mult1_alternative3, mult1, 5);
//	compare_time(mult1, mult1_alternative2, 5);
}



