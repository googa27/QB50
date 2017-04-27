#include "etat.h"
#include <fstream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#define PI 3.14159265

using namespace std;
using namespace Eigen;

Etat::Etat(){
    q = Quat(1,0,0,0);
    b = Vector3d(0,0,0);
    omega = Vector3d(0,0,0);
    qref = Quat(0,1,0,0);
    omegaref = Vector3d(0,0,0);
    lat = 0;
    longitude = 0;
    momMag = Vector3d(0,0,0);
    intensite = Vector3d(0,0,0);
    controle = true;
}


double Etat::getJulianDate()
{
    time_t t = time(0);
    return ( t / 86400.0 ) + 2440587.5;
}

/*
getVecSolRef
Calcule le vecteur solaire théorique dans le réf géocentrique en fonction de la Julian Date
*/
Vector3d Etat::getVecSolRef(double jd){
	double T = (jd - 2451545.)/36525;
	double lambdaSun = 280.4606184 + (36000.77005361 * T);
	double MSun = 357.5277233 + 35999.05034*T;
	double lambdaE = lambdaSun + 1.914666471*sin(MSun*PI/180) + 9.918994643*sin(2*MSun*PI/180);
	double epsilon = 23.439291 - 0.0130042*T;
	Vector3d s(cos(lambdaE*PI/180), cos(epsilon*PI/180)*sin(lambdaE*PI/180), sin(epsilon*PI/180)*sin(lambdaE*PI/180));
    if (s.norm() != 0)
        s /= s.norm();
	
    return s;
}

//Renvoie le vecteur de taille 6 contenant les mesures des 6 capteurs solaires
VectorXd Etat::getCapteursSolaires(){
    VectorXd v = VectorXd(6);
    v << 1.,-1.,0.,0.,0.,0.;
	return v;//Dans l'ordre face x>0, x<0, y>0, y<0, z>0, z<0
}

//Renvoie le vecteur solaire mesuré dans le référentiel du satellite
Vector3d Etat::getVecSol(){
    VectorXd vmes; //Taille 6
    Vector3d vecsol; //Taille 3
    Matrix3d H = Matrix3d::Identity();
    MatrixXd M(3,6);
    H /= 2.;
    M << 1, -1, 0, 0, 0, 0,
         0, 0, 1, -1, 0, 0,
         0, 0, 0, 0, 1, -1;
    // Ici H = (M * M.transpose)^(-1) ou M est la matrice 3x6 des vecteurs normaux aux 6 capteurs
    vmes = getCapteursSolaires();
    vecsol = M*vmes;
    vecsol = H*vecsol;
    if (vecsol.norm() != 0)
        vecsol /= vecsol.norm();
    
    return vecsol;
}

//Renvoie le champ magnétique théorique dans la référentiel géocentrique à partir de la table et des TLE(i,omega,om,nu)
Vector3d Etat::getChampRef(double lat, double longitude,double om,double i, double omega, double nu){
	ifstream fichier("/Users/adrianvalente/Documents/etudes/psc/source1502/ADCS/igrf.txt",ios::in);
	double x,y,z;
	if (fichier){
		string ligne;
		int n = round(longitude)*179+90+floor(lat);
		for (int i = 1; i <= n; i++)
			getline(fichier,ligne);
		stringstream ss(ligne);
		double buf;
		ss >> buf;
		ss >> buf;
		ss >> buf;
		ss >> x;
		ss >> y;
		ss >> z;
		fichier.close();
	}
	else{
		x = y = z = 0.;
	}
	//Passage du référentiel de la table au référentiel géocentrique grâce aux TLE
    //Réf de la table : z vers le centre de la terre, y vers l'est, et x vers le nord (mais tangentiellement au sol)
    //cf http://www.geomag.bgs.ac.uk/data_service/models_compass/igrf.html
	double theta=omega+nu;
	Matrix3d m1;
	m1 << sin(i)*sin(om),cos(i)*cos(om),cos(om),
		  -sin(i)*cos(om),-cos(i)*sin(om),sin(om),
		  cos(i),sin(i),0;
	Matrix3d m2;
	m2 << 1,0,0,
	 	  0,cos(theta),-sin(theta),
	      0,sin(theta),cos(theta);
	return m1*m2*Vector3d(x,y,z);
}

//Renvoie le vecteur de taille 3 contenant la mesure du magnétomètre dans le référentiel du satellite
Vector3d Etat::getChampM(){
	return Vector3d(1.,0.,0.);
}

double Etat::getLat(){
    return 0;
}

double Etat::getLong(){
    return 0.;
}

Vector3d Etat::getOmega(){
    return Vector3d(1,1,1);
}

void Etat::getTLE(double& perih, double &om, double &i, double &nu){
	return;
}

//Donne le quaternion de référence (ie. à viser) pour la position donnée par les coordonnees orbitales
Quat Etat::getQref(double perih, double om, double i, double nu){
    Matrix3d m1;
    m1 << cos(om), -sin(om), 0,
          sin(om), cos(om), 0,
          0,       0,      1;
    double theta=perih+nu;
    Matrix3d m2;
    m2 << 1, 0, 0,
        0, cos(i), -sin(i),
        0, sin(i), cos(i);
    Matrix3d m3;
    m3 << cos(theta), -sin(theta), 0,
        sin(theta), cos(theta), 0,
                 0,          0, 1;
    Matrix3d m4;
    m4 << 0,-1,0,
          1,0,0,
          0,0,1;
    Matrix3d mtot = m1*m2*m3*m4;
    
    Quat q;
    q(3) = sqrt(1.+mtot(0,0)+mtot(1,1)+mtot(2,2))/2.;
    if (q(3)!=0)
    {
        q(0) = (mtot(2,1)-mtot(1,2))/(4.*q(3));
        q(1) = (mtot(0,2)-mtot(2,0))/(4*q(3));
        q(2) = (mtot(1,0)-mtot(0,1))/(4*q(3));
    } else{ //Si la trace est nulle, il faut trouver l'axe de rotation...
        if (mtot(0,0)>0){
            q(0)=1;
            q(1)=q(2)=0;
        } else if (mtot(1,1)>0){
            q(1)=1;
            q(0)=q(3)=0;
        } else {
            q(2)=1;
            q(1)=q(0)=0;
        }
    }
    return q;
}


Vector3d Etat::actualiser(){

	//Initialisation : on recupere les donnees des capteurs et des tables
    double julianDate = getJulianDate();
    lat = getLat();   //A FAIRE : recuperer les donnees du GPS
    longitude = getLong();
	omega = getOmega(); //A FAIRE : recuperer vitesse de rotation du gyro
					//Vecteur rotation instantanée du satellite par rapport au ref geocentrique, exprimé dans le référentiel du satellite
	//TLE
	//A faire : fonction pour recuperer les donnees des TLE
	perih=0;//argument du perihelie
	om=0;//Longitude du point ascendant croisant le plan equatorial
	i=0;//Inclinaison de l'orbite par rapport au plan equatorial
	nu=90;//Anomalie (pos du satellite)
	//getTLE(&perih, &om, &i, &nu);
    //Passage en radians
    om = om*PI/180;
    perih = perih*PI/180;
    nu = nu*PI/180;
    i = i*PI/180;
	qref = getQref(perih, om, i, nu);
	//Champs Mags
	Vector3d champRef = getChampRef(lat, longitude,om,i,perih,nu); //dans le ref GEOCENTRIQUE
	champRef = q.conjug(champRef);  //On le passe immediatement dans le ref SATELLITE
    Vector3d champRefNorme = champRef;
    if (champRef.norm() != 0 )
        champRefNorme =  champRef/champRef.norm();
	Vector3d champM = getChampM();//A FAIRE : Champ mesuré dans le ref SATELLITE
    Vector3d champMNorme = champM;
    if (champM.norm() != 0)
        champMNorme = champM/champM.norm();
	//Capteurs solaires. Les vecteurs sont directement normalises
	Vector3d vecSolRef = getVecSolRef(julianDate);//Vecteur solaire dans le ref GEOCENTRIQUE
	vecSolRef = q.conjug(vecSolRef);
	if (vecSolRef.norm() != 0) vecSolRef /= vecSolRef.norm();
    Vector3d vecSol(0.,0.,0.);
    vecSol = getVecSol(); //Vecteur solaire mesuré dans le ref SATELLITE
	if (vecSol.norm() != 0) vecSol /= vecSol.norm();
    
	//Calcul des quaternions (dans physic.m)
	Vector3d GrandOmega = kBi*champMNorme.cross(champRefNorme) + kSi*vecSol.cross(vecSolRef);  //Intermediaire de calcul (omega dans le rapport de Valentin)
	Vector3d vtmp = omega - b + kPi*GrandOmega;
	Quat qtmp(vtmp(0),vtmp(1),vtmp(2),0.);
	q += (0.5*q.prod(qtmp) + (1-q.norm()*q.norm())*q)*dt;
	b -= kIi*GrandOmega*dt;
	
	//Controle (l. 540)	calcul du moment genere
	Quat dq;
	Vector3d m(0.,0.,0.);
	if (controle==true && champM.norm()!=0) {
        Quat qtmp = q.inv();
		Quat dq = qref.prod(qtmp);
		Vector3d dq13(dq(0),dq(1),dq(2));
		m = (-0.000048*champM.cross(omega-b-omegaref) - 0.0000003*champM.cross(dq13))/(champM.norm()*champM.norm());
	}
	
	//Calcul intensite l.325
	intensite = Vector3d(m(0)/SurfaceX, m(1)/SurfaceY, m(2)/SurfaceZ);
    return intensite;
}
