//
// Created by Austin Clyde on 6/13/17.
//

#include <iostream>
#include "LinearRegression.h"
#include <Eigen/Dense>

LinearRegression::LinearRegression(double learning_rate, unsigned long max_iter, double epsilon) {
  alpha = learning_rate;
  EPSILON = epsilon;
  MAX_ITER = max_iter;
  dim[0] = 0;
  dim[1] = 0;
}

void LinearRegression::fit(Eigen::MatrixXd X, Eigen::VectorXd y) {
  dim[0] = (size_t) X.rows();
  dim[1] = (size_t) X.cols();

  if (dim[0] != y.size() || dim[0] == 0 || dim[1] == 0) {
    std::cout << "ERROR! X is " << dim[0] << " by " << dim[1];
    std::cout << ", and y is " << y.size();
    return;
  }
  gradientDescent(X, y);
}

Eigen::VectorXd LinearRegression::predict(Eigen::MatrixXd X) {
  // check sizes
  return X * theta;
}

double LinearRegression::score(Eigen::MatrixXd X, Eigen::VectorXd y) {
  Eigen::VectorXd preds = predict(X);
  return (preds - y).squaredNorm();
}

void LinearRegression::gradientDescent(Eigen::MatrixXd X, Eigen::VectorXd y) {
  theta.resize(dim[1]);
  Eigen::VectorXd theta_prev(dim[1]);
  theta.fill(1);
  theta_prev.fill(1);
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]", "{", "}\n");
  Eigen::VectorXd h, delta;
  for (int k = 0; k < MAX_ITER && (theta - theta_prev).squaredNorm() < EPSILON; k++) {
    theta_prev = theta;
    delta = 1 / ((double) dim[0]) * (X.transpose() * (X * theta - y));
    //dJ/dtheta => gradient J(theta) = X.T * (X* theta -y)
    // see page 11 http://cs229.stanford.edu/notes/cs229-notes1.pdf
    // delta = 1/((double) dim[0]) * (X.cwiseProduct((X*theta-y).replicate(1,dim[1]))).colwise().sum();

    theta = (theta - (alpha * delta));
    //std::cout << "Iteration " << k << " , Cost:" << computeCost(X,y) << std::endl;
    //std::cout << theta.format(fmt) << std::endl;
  }
}

double LinearRegression::computeCost(Eigen::MatrixXd X, Eigen::VectorXd y) {
  Eigen::IOFormat fmt(8, 0, ", ", "\n", "[", "]", "{", "}\n");
  Eigen::MatrixXd sum = (X * theta - y);
  double s = sum.cwiseProduct(sum).sum();
  return s / ((double) 2 * dim[0]);
}