//
// Created by jemin on 5/16/19.
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef _RAISIM_GYM_VISSETUPCALLBACK_HPP
#define _RAISIM_GYM_VISSETUPCALLBACK_HPP

#include <raisim/OgreVis.hpp>
void setupCallback() {
  auto vis = raisim::OgreVis::get();

  /// light
  Ogre::ColourValue l0_color(0.9, 0.9, 0.9);
  Ogre::Vector3 lightdir(0, -5, -1);

  auto node1 = vis->getSceneManager()->getRootSceneNode()->createChildSceneNode();
  auto l1 = vis->getSceneManager()->createLight("l1");
  l1->setType(Ogre::Light::LightTypes::LT_DIRECTIONAL);
  Ogre::ColourValue l1_color(0.8, 0.8, 0.8);

  l1->setDiffuseColour(l0_color);
  l1->setSpecularColour(l0_color);
  l1->setAttenuation(10, 0.01, 1, 0.0);

  Ogre::Vector3 lightdir1(-1, 0, 1);

  lightdir1 = {-1, -1, 0};
  lightdir.normalise();
  node1->attachObject(l1);
  node1->setDirection({lightdir1});
  node1->setPosition(100.0, 0.0, 0.0);
  node1->setPosition(100.0, 100.0, 0.0);

  vis->getLight()->setDiffuseColour(1, 1, 1);
  vis->getLight()->setCastShadows(true);

  vis->setCameraSpeed(300);

  /// load  textures
  vis->addResourceDirectory(vis->getResourceDir() + "/material/checkerboard");
  vis->loadMaterialFile("checkerboard.material");

  /// shdow setting
  vis->getSceneManager()->setShadowTechnique(Ogre::SHADOWTYPE_TEXTURE_ADDITIVE);
  vis->getSceneManager()->setShadowTextureSettings(2048, 3);
//  vis->getSceneManager()->setSkyBox(true, "navajo_white", 300);
  vis->getSceneManager()->setSkyBox(true, "white", 300);

  /// scale related settings!! Please adapt it depending on your map size
//   beyond this distance, shadow disappears
  vis->getSceneManager()->setShadowFarDistance(3);
  // size of contact points and contact forces
  vis->setContactVisObjectSize(0.03, 0.2);
  // speed of camera motion in freelook mode
  vis->getCameraMan()->setTopSpeed(5);

  vis->addVisualObject("disturb_arrow",
                       "arrowMesh",
                       "blueEmit",
                       {0.2, 0.2, 0.5},
                       false,
                       raisim::OgreVis::RAISIM_OBJECT_GROUP);
  vis->addVisualObject("command_arrow1",
                       "arrowMesh",
                       "red",
                       {0.2, 0.2, 0.5},
                       false,
                       raisim::OgreVis::RAISIM_OBJECT_GROUP);
  vis->addVisualObject("command_arrow2",
                       "arrowMesh",
                       "lawn_green",
                       {0.2, 0.2, 0.3},
                       false,
                       raisim::OgreVis::RAISIM_OBJECT_GROUP);
 vis->addVisualObject("goal", "sphereMesh", "redEmit", {0.2, 0.2, 0.2}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);

  for (size_t i = 0; i < 36; i++) {
    std::string name = "footScan_";
    name += std::to_string(i);
    vis->addVisualObject(name, "sphereMesh", "redEmit", {0.015, 0.015, 0.015},
                         false, raisim::OgreVis::RAISIM_OBJECT_GROUP);

  }

  for (size_t i = 0; i < 4; i++) {
    std::string name = "footTarget_";
    name += std::to_string(i);
    vis->addVisualObject(name, "sphereMesh", "green", {0.01, 0.01, 0.01},
                         false, raisim::OgreVis::RAISIM_OBJECT_GROUP);

  }
}


#endif //_RAISIM_GYM_VISSETUPCALLBACK_HPP
