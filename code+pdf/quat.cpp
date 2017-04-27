#include "quat.h"


Quat::Quat()
   : Vector4d()
{ ; }

Quat::Quat(double x, double y, double z, double t)
   : Vector4d( x, y, z, t )
{ ; }


Quat Quat::prod(Quat& q2)
{
	Quat p(0.,0.,0.,0.);
	Quat q1 = *this;
	Vector3d v(q1(0),q1(1),q1(2));
	Vector3d w(q2(0),q2(1),q2(2));
	
	Vector3d z = q1(3)*w + q2(3)*v + v.cross(w);
	p(0)=z(0);
	p(1)=z(1);
	p(2)=z(2);
	p(3) = q1(3)*q2(3) - v.dot(w);
	
	return p;
}

Quat Quat::inv()
{
	Quat& q = *this;
	if (q.norm()==0) return Quat(0,0,0,0);
	Quat p(-q(0),-q(1),-q(2),q(3));
	p /= (q.norm())*(q.norm());
	return p;
}


Vector3d Quat::conjug(Vector3d x)
{
	Quat& q = *this;

	Vector3d v(q(0),q(1),q(2));
	double n(q.norm());
	if (n!=0) v *= -1./n;
	Vector3d u(0,0,0);
	u = 2*v.dot(x)*v;
	if (n!=0) u += ((q(3)/n)*(q(3)/n) - v.norm()*v.norm())*x;
	if (n!=0) u += 2*(q(3)/n)*v.cross(x);

	return u;
}
	
	