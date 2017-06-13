//
// Created by Austin Clyde on 6/13/17.
//

#ifndef LINEARREGRESSION_LINEARREGRESSION_H
#define LINEARREGRESSION_LINEARREGRESSION_H

#include <Eigen/Dense>
#include "Classifier.h"

class LinearRegression: public Classifier {

private:
    void gradientDescent(Eigen::MatrixXd X, Eigen::VectorXd y);
    double computeCost(Eigen::MatrixXd X, Eigen::VectorXd y);

protected:
    Eigen::VectorXd theta;
    double EPSILON;
    double alpha;
    unsigned long MAX_ITER;
    size_t dim[2];

public:
    LinearRegression(double learning_rate, unsigned long max_iter, double epsilon);

    virtual void fit(Eigen::MatrixXd X, Eigen::VectorXd y);

    virtual Eigen::VectorXd predict(Eigen::MatrixXd X);

    virtual double score(Eigen::MatrixXd X, Eigen::VectorXd y);

    Eigen::VectorXd getThetas() {return theta;};
};


#endif //LINEARREGRESSION_LINEARREGRESSION_H
