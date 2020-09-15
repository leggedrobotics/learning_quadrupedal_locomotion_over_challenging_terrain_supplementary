//
// Created by joonho on 7/28/20.
//

#ifndef BLIND_SANDBOX_INCLUDE_GRAPH_POLICY_H_
#define BLIND_SANDBOX_INCLUDE_GRAPH_POLICY_H_

#include "GraphLoader.hpp"

template<int ActionDim>
class Policy {
 public:
  Policy(std::string model_path,
         std::string param_path, int state_dim, int history_dim, int history_len) {
    graph_.initialize(model_path);
    graph_.loadLP(param_path);

    state_dims_inv.AddDim(1);
    state_dims_inv.AddDim(1);
    state_dims_inv.AddDim(state_dim);

    history_dims_inv.AddDim(1);
    history_dims_inv.AddDim(history_len);
    history_dims_inv.AddDim(history_dim);

    state_tf_tensor = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, state_dims_inv);
    history_tf_tensor = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, history_dims_inv);
  }
  ~Policy() {

  }
  void initialize() {

  }
  void updateStateBuffer(Eigen::Matrix<float, -1, 1> &state) {
    std::memcpy(state_tf_tensor.template flat<float>().data(),
                state.data(),
                sizeof(float) * state_tf_tensor.NumElements());
  }
  void updateHistoryBuffer(Eigen::Matrix<float, -1, -1> &history) {
    std::memcpy(history_tf_tensor.template flat<float>().data(),
                history.data(),
                sizeof(float) * history_tf_tensor.NumElements());
  }

  void getAction(Eigen::Matrix<float, ActionDim, 1> &action) {
    std::vector<tensorflow::Tensor> outputs;
    graph_.run({std::make_pair("state", state_tf_tensor), std::make_pair("history", history_tf_tensor)},
               {"policy/actionDist"},
               {},
               outputs);

    std::memcpy(action.data(),
                outputs[0].template flat<float>().data(),
                sizeof(float) * ActionDim);

  }

  GraphLoader<float> graph_;
  tensorflow::TensorShape history_dims_inv;
  tensorflow::Tensor history_tf_tensor;

  tensorflow::TensorShape state_dims_inv;
  tensorflow::Tensor state_tf_tensor;

};

#endif //BLIND_SANDBOX_INCLUDE_GRAPH_POLICY_H_
