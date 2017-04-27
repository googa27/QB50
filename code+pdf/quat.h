/*
Quat : vecteur de quatre doubles sur lequel on peut faire quelques operations particulieres
*/

#ifndef QUAT
#define QUAT

#include <iostream>
#include "Eigen/Dense"
#include <cmath>

using namespace std;
using namespace Eigen;

class Quat : public Eigen::Vector4d
{

	private:
	
	public:
	Quat();
	Quat(double x, double y, double z, double t);
	Quat prod(Quat& q2);  //Produit
	Quat inv();  //Inverse
	Vector3d conjug(Vector3d x);  //Conjuguer un vecteur de R3 par un quaternion
};

#endif
