#ifndef RANDOMNUMBERGENERATOR_HPP_
#define RANDOMNUMBERGENERATOR_HPP_

// for random sampling
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <cstdlib>
#include <mutex>
#include <Eigen/Core>


template<typename Dtype>
class RandomNumberGenerator {

 public:

  RandomNumberGenerator() {
  }

  ~RandomNumberGenerator() {
  }

  /* mean =0, std = 1*/
  Dtype sampleNormal() {
    auto dist = boost::random::normal_distribution<Dtype>(0.0, 1.0);
    return dist(rngGenerator);
  }

  /* from -1 to 1*/
  Dtype sampleUniform() {
    auto dist = boost::uniform_real<Dtype>(-1, 1);
    return dist(rngGenerator);
  }

  /* from 0 to 1*/
  Dtype sampleUniform01() {
    auto dist = boost::uniform_real<Dtype>(0, 1);
    return dist(rngGenerator);
  }

  bool forXPercent(float epsilon) {
    auto dist = boost::uniform_real<Dtype>(0, 1);
    return dist(rngGenerator) < epsilon;
  }

  int intRand(const int &min, const int &max) {
    {
      std::uniform_int_distribution<int> distribution(min, max);
      return distribution(rngGenerator);
    }
  }

  /* weighted random sampling where the weights are given by 1, r, r^2, ... */
  int intWeightedRand (const int &max, Dtype weightDecayFtr) {
    Dtype sum = (Dtype(1) - std::pow(weightDecayFtr, max+1)) / (Dtype(1)-weightDecayFtr);
    return std::ceil(log(Dtype(1) - sampleUniform01() * sum * (Dtype(1) - weightDecayFtr)) / log(weightDecayFtr)) - 1;
  }

  template<int dim>
  void sampleVectorInNormalUniform(Dtype *vector) {
    auto dist = boost::uniform_real<Dtype>(-1, 1);
    for (int i = 0; i < dim; i++)
      vector[i] = dist(rngGenerator);
  }

  template<int dim>
  void sampleInUnitSphere(Dtype *vector) {
    sampleVectorInNormalUniform<dim>(vector);
    Dtype sum = 0.0f;

    for (int i = 0; i < dim; i++)
      sum += vector[i] * vector[i];

    Dtype amplitudeOverSum = pow(std::abs(sampleUniform()), Dtype(1.0) / Dtype(dim)) / sqrtf(sum);

    for (int i = 0; i < dim; i++)
      vector[i] = vector[i] * amplitudeOverSum;
  }

  template<int dim>
  void sampleOnUnitSphere(Dtype *vector) {
    sampleVectorInNormalUniform<dim>(vector);
    Dtype sum = 0.0f;

    for (int i = 0; i < dim; i++)
      sum += vector[i] * vector[i];

    for (int i = 0; i < dim; i++)
      vector[i] = vector[i] / sqrtf(sum);
  }

  template<typename Derived>
  void shuffleSTDVector(std::vector<Derived> &order) {
    boost::variate_generator<boost::mt19937 &, boost::uniform_int<> >
        random_number_shuffler(rngGenerator, boost::uniform_int<>());
    std::random_shuffle(order.begin(), order.end(), random_number_shuffler);
  }

  /* this method is opitmized for memory use. The column should be dynamic size */
  template<typename Derived, int Rows, int Cols>
  void shuffleColumns(Eigen::Matrix<Derived, Rows, Cols> &matrix) {
    int colSize = int(matrix.cols());

    /// sampling the order
    std::vector<int> order;
    std::vector<bool> needSuffling(colSize, true);

    order.resize(colSize);
    for (int i = 0; i < colSize; i++) order[i] = i;
    shuffleSTDVector(order);
    Eigen::Matrix<Derived, Rows, 1> memoryCol(matrix.rows());

    int colID;

    for (int colStartID = 0; colStartID < colSize; colStartID++) {
      if (order[colStartID] == colStartID || !needSuffling[colStartID]) continue;

      colID = colStartID;
      memoryCol = matrix.col(colID);
      do {
        matrix.col(colID) = matrix.col(order[colID]);
        needSuffling[colID] = false;
        colID = order[colID];
      } while (colStartID != order[colID]);
      matrix.col(colID) = memoryCol;
      needSuffling[colID] = false;
    }
  }

  std::vector<unsigned> getNrandomSubsetIdx (unsigned nOfElem, unsigned nOfSubElem) {
    std::vector<unsigned> memoryIdx(nOfSubElem);
    ///// randomly sampling memory indeces
    for (unsigned i = 0; i < nOfSubElem; i++) {
      memoryIdx[i] = intRand(0, nOfElem - 1);
      for (unsigned j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }
    return memoryIdx;
  }

  /*you can use this method to make the random samples the same*/
  void seed(uint32_t seed) {
    rngGenerator.seed(seed);
  }

 private:
  boost::random::mt19937 rngGenerator;

};


#endif /* RANDOMNUMBERGENERATOR_HPP_ */
