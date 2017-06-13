//
// Created by Austin Clyde on 6/13/17.
//

#ifndef LINEARREGRESSION_CLASSIFIER_H
#define LINEARREGRESSION_CLASSIFIER_H

#include <Eigen/Dense>

class Classifier {

public:
        virtual void fit(Eigen::MatrixXd X, Eigen::VectorXd y) = 0;

        virtual Eigen::VectorXd predict(Eigen::MatrixXd) = 0;

        virtual double score(Eigen::MatrixXd X, Eigen::VectorXd y) = 0;
};


#endif //LINEARREGRESSION_CLASSIFIER_H
