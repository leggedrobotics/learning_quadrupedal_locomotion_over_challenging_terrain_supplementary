# Supplementary materials for "Learning Quadrupedal Locomotion over Challenging Terrain"
###Project page: https://leggedrobotics.github.io/rl-blindloco
###Author: Joonho Lee (jolee@ethz.ch)


####This repo contains
* __trained policy networks for ANYmal c010 and c100 models.__
    * c010 model is not trained with _Slippery Hills_ terrain.
    * c100 is the latest model (added during the revision)
* __ANYmal environments implemented in Raisim simulator.__
    * Height map terrain is included
    * In the visualizer, press 'c' key to sample new command
    * In the visualizer, press space key to re-initialize the robot

    
####Dependencies
* raisim (https://github.com/raisimTech/raisimlib)
* raisimOgre (https://github.com/raisimTech/raisimOgre)
* tensorflow-cpp (https://github.com/leggedrobotics/tensorflow-cpp)

####Notes
* Some paths are hard-coded in _test_c010.cpp_ and _test_c100.cpp_. Be caureful about them.
* This repository is not maintained anymore. If you have a question, send an email to jolee@ethz.ch. 
* You can install tensorflow-cpp locally as follows.
```sh
cd tensorflow/tensorflow
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${YOUR_LOCAL_BUILD_DIRECTORY} -DCMAKE_BUILD_TYPE=Release ..
make install -j
```
