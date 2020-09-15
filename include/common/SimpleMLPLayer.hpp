

#ifndef SIMPLEMLP_HPP
#define SIMPLEMLP_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include "iostream"
#include <fstream>
#include <cmath>

namespace rai {

namespace FuncApprox {

enum class ActivationType {
  linear,
  relu,
  tanh,
  softsign
};

template<typename Dtype, ActivationType activationType>
struct Activation {
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, -1> &output) {}
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, 1> &output) {}
};

template<typename Dtype>
struct Activation<Dtype, ActivationType::relu> {
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, -1> &output) {
    output = output.cwiseMax(0.0);
  }

  inline void nonlinearity(Eigen::Matrix<Dtype, -1, 1> &output) {
    output = output.cwiseMax(0.0);
  }
};

template<typename Dtype>
struct Activation<Dtype, ActivationType::tanh> {
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, -1> &output) {
    output = output.array().tanh();
  }
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, 1> &output) {
    output = output.array().tanh();
  }
};

template<typename Dtype>
struct Activation<Dtype, ActivationType::softsign> {
  inline void nonlinearity(Eigen::Matrix<Dtype, -1, -1> &output) {
    for (int i = 0; i < output.size(); i++) {
      output[i] = output[i] / (std::abs(output[i]) + 1.0);
    }
  }

  inline void nonlinearity(Eigen::Matrix<Dtype, -1, 1> &output) {
    for (int i = 0; i < output.size(); i++) {
      output[i] = output[i] / (std::abs(output[i]) + 1.0);
    }
  }
};

template<typename Dtype, int StateDim, int ActionDim, ActivationType activationType>
class MLP_fullyconnected {

 public:
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, StateDim, 1> State;

  MLP_fullyconnected(std::vector<int> hiddensizes) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    layersizes.push_back(StateDim);
    layersizes.reserve(layersizes.size() + hiddensizes.size());
    layersizes.insert(layersizes.end(), hiddensizes.begin(), hiddensizes.end());
    layersizes.push_back(ActionDim);
    ///[input hidden output]

    params.resize(2 * (layersizes.size() - 1));
    Ws.resize(layersizes.size() - 1);
    bs.resize(layersizes.size() - 1);
    lo.resize(layersizes.size());
    Stdev.resize(ActionDim);

    for (int i = 0; i < params.size(); i++) {
      int paramSize = 0;

      if (i % 2 == 0) ///W resize
      {
        Ws[i / 2].resize(layersizes[i / 2 + 1], layersizes[i / 2]);
        params[i].resize(layersizes[i / 2] * layersizes[i / 2 + 1]);
      }
      if (i % 2 == 1) ///b resize
      {
        bs[(i - 1) / 2].resize(layersizes[(i + 1) / 2]);
        params[i].resize(layersizes[(i + 1) / 2]);
      }
    }

  }

  bool load_eigen_from_binary(const std::string filename, Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> &data) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
      return false;
    }
    size_t rows = 0;
    size_t cols = 0;
    in.read((char *) (&rows), sizeof(size_t));
    in.read((char *) (&cols), sizeof(size_t));
    data.resize(rows, cols);
    in.read((char *) data.data(), rows * cols * sizeof(Dtype));

    in.close();
    return true;
  }

  void updateParamFromBin(std::string fileName) {
    Eigen::Matrix<Dtype, -1, -1> temp;
    load_eigen_from_binary(fileName, temp);

    size_t pos = 0;

    for (int i = 0; i < params.size(); i++) {
      int paramSize = 0;
      for (size_t j = 0; j < params[i].size(); j++) {
        params[i](j) = temp.data()[pos++];
      }
      if (i % 2 == 0) ///W copy
      {
        memcpy(Ws[i / 2].data(), params[i].data(), sizeof(Dtype) * Ws[i / 2].size());
      }
      if (i % 2 == 1) ///b copy
      {
        memcpy(bs[(i - 1) / 2].data(), params[i].data(), sizeof(Dtype) * bs[(i - 1) / 2].size());
      }
    }
  }

  void updateParamFromBin_noesis(std::string fileName) {
    Eigen::Matrix<Dtype, -1, -1> temp;
    load_eigen_from_binary(fileName, temp);

    for (int i = 0; i < params.size(); i++) {
      int paramSize = 0;

      if (i % 2 == 0) ///W resize
      {
        Ws[i / 2].resize(layersizes[i / 2 + 1], layersizes[i / 2]);
        params[i].resize(layersizes[i / 2] * layersizes[i / 2 + 1]);
      }
      if (i % 2 == 1) ///b resize
      {
        bs[(i - 1) / 2].resize(layersizes[(i + 1) / 2]);
        params[i].resize(layersizes[(i + 1) / 2]);
      }
    }



    /// output layer

    size_t pos = 0;

    // Wo
    for (size_t j = 0; j <params.end()[-2].size(); j++) {
      params.end()[-2](j) = temp.data()[pos++];
    }

    // bo
    for (size_t j = 0; j <params.end()[-1].size(); j++) {
      params.end()[-1](j) = temp.data()[pos++];
    }

    memcpy(Ws.back().data(), params.end()[-2].data(), sizeof(Dtype) * Ws.back().size());
    memcpy(bs.back().data(), params.end()[-1].data(), sizeof(Dtype) * bs.back().size());


    for (int i = 0; i < params.size() - 2; i++) {
      int paramSize = 0;
      for (size_t j = 0; j < params[i].size(); j++) {
        params[i](j) = temp.data()[pos++];
      }
      if (i % 2 == 0) ///W copy
      {
        memcpy(Ws[i / 2].data(), params[i].data(), sizeof(Dtype) * Ws[i / 2].size());
      }
      if (i % 2 == 1) ///b copy
      {
        memcpy(bs[(i - 1) / 2].data(), params[i].data(), sizeof(Dtype) * bs[(i - 1) / 2].size());
      }
    }
  }


  void updateParamFromTxt(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::ifstream indata;
    indata.open(fileName);
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;

    int totalN = 0;
    ///assign parameters
    for (int i = 0; i < params.size(); i++) {
      int paramSize = 0;

      while (std::getline(lineStream, cell, ',')) { ///Read param
        params[i](paramSize++) = std::stod(cell);
        if (paramSize == params[i].size()) break;
      }
      totalN += paramSize;
      if (i % 2 == 0) ///W copy
        memcpy(Ws[i / 2].data(), params[i].data(), sizeof(Dtype) * Ws[i / 2].size());
      if (i % 2 == 1) ///b copy
        memcpy(bs[(i - 1) / 2].data(), params[i].data(), sizeof(Dtype) * bs[(i - 1) / 2].size());
    }

  }

  inline Action forward(State &state) {

    lo[0] = state;

    for (int cnt = 0; cnt < Ws.size() - 1; cnt++) {
      lo[cnt + 1] = Ws[cnt] * lo[cnt] + bs[cnt];
      activation_.nonlinearity(lo[cnt + 1]);
    }

    lo[lo.size() - 1] = Ws[Ws.size() - 1] * lo[lo.size() - 2] + bs[bs.size() - 1]; /// output layer
    return lo.back();
  }


  inline Action forwardtemp(State &state) {
    Eigen::Matrix<Dtype, -1, 1> temp;
    temp = state;

    for (int cnt = 0; cnt < Ws.size() - 1; cnt++) {
      temp = Ws[cnt] * temp + bs[cnt];
      activation_.nonlinearity(temp);
    }
    temp = Ws.back() * temp + bs.back(); /// output layer
    return temp;
  }

 private:
  std::vector<Eigen::Matrix<Dtype, -1, 1>> params;
  std::vector<Eigen::Matrix<Dtype, -1, -1>> Ws;
  std::vector<Eigen::Matrix<Dtype, -1, 1>> bs;
  std::vector<Eigen::Matrix<Dtype, -1, 1>> lo;

  Activation<Dtype, activationType> activation_;
  Action Stdev;

  std::vector<int> layersizes;
  bool isTanh = false;
};

}

}

#endif //SIMPLEMLP_HPP
