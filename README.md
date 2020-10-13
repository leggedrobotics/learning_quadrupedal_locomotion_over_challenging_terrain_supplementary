# learning_quadrupdal_locomotion_over_challening_terrain_supplementary
Supplementary materials for "Learning Quadrupedal Locomotion over Challenging Terrain"

Author: Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter

This repo contains trained policy network and testing environments implemented in Raisim simulator.


Dependencies

* raisim (https://github.com/raisimTech/raisimlib)
* raisimOgre (https://github.com/junja94/raisimOgre)
* tensorflow-cpp (https://github.com/leggedrobotics/tensorflow-cpp)

Notes

* install tensorflow-cpp
```sh
cd tensorflow/tensorflow
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${YOUR_LOCAL_BUILD_DIRECTORY} -DCMAKE_BUILD_TYPE=Release ..
make install -j
```
