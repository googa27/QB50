/*
Representation de l'etat du systeme a chaque instant
*/
#ifndef ETAT
#define ETAT

#include <iostream>
#include "Eigen/Dense"
#include <cmath>
#include "quat.h"

using namespace std;
using namespace Eigen;

//QUELQUES CONSTANTES

//Surfaces equivalentes des bobines (en incluant la multiplication par le nb spires)
const double SurfaceX = 62.5;
const double SurfaceY = 62.5;
const double SurfaceZ = 62.5;

//Pas de temps
const double dt = 0.001;

//Gains
const double kPi = 1, kSi = 1, kBi = 1, kIi = 1;


class Etat{
	private : 
	Quat q;  //Quat observe, référentiel : géocentrique, capteurs solaires + position ET difference magnétometre / table champ magnétique terrestre
	Vector3d b;  //Biais du gyro (à mesurer une fois pour toute une fois le satellite construit, dépend de la température peut-être)
	Vector3d omega; //Vitesse de rotation observee par les gyro, référentiel : géocentrique
	Quat qref;  //Quat de reference devant etre atteint calculé a partir de la position, quat de passage du ref géocentrique au ref orbital
	Vector3d omegaref;  //Vitesse de rotation à atteindre : soit constante 2PI/Torbite, soit depend de la position
	//Vector3d posSat;  //Position du satellite dans le ref geocentrique (quel système de coordonnées), a partir des TLE/GPS
	double lat;  //Latitude et longitude (a recuperer avec les tle)
	double longitude;
	Vector3d momMag;  //Référentiel : satellite, calcule à partir de ce qui precede
	Vector3d intensite;  //Intensités dans les 3 bobines, calcule a partir de momMag et des caracteristiques des bobines
	bool controle;  
	//Donnees des TLE
	double perih;//argument du perihelie
	double om;//Longitude du point ascendant croisant le plan equatorial
	double i;//Inclinaison de l'orbite par rapport au plan equatorial
	double nu;//Anomalie (pos du satellite)
	
	
	public : 
	Etat();
	
	Vector3d actualiser();
    static double getJulianDate();
    static Vector3d ECEFtoECI(double jd, Vector3d vEcef);
	static Vector3d getChampRef(double lat, double longitude,double om,double i, double omega, double nu);  //Fournir le champ mag theorique a partir des tables (dans le ref geo) en nT
	static Vector3d getChampM(); //Recuperer donnees du magnetometre
	static Vector3d getOmega(); //Recuperer donnees du gyrometre
	static Vector3d getVecSolRef(double jd); //Fournir le vecteur solaire theorique en fonction de la position du sat (dans le ref geo)
	static void getTLE(double& perih, double &om, double &i, double &nu);
    
	static Vector3d getVecSol(); //Recuperer donnees capteurs solaires et donner la direction du soleil dans le ref du satellite
    static VectorXd getCapteursSolaires();
	Quat getQref(double om, double i, double omega, double nu);
	double getLat();
	double getLong();
};

//Capteurs solaires + position donne quaternion observé
//Capteurs solaires : f(R^3) = Im f inclus dans R^9 et f injective, on inverse f et on a le vecteur solaire dans le ref du satellite, on a aussi le vecteur solaire dans le ref géocentrique : le produit vectoriel de ces deux vecteurs normalisés donne le vecteur de changement de ref et donc l'attitude du satellite
//On fait pareil avec le magnétomètre : le champ mesuré est dans le ref du satellite, et celui dans la table est dans le ref géocentrique
//On donne un poids à chacune de ces mesures (seuil pour l'utilisation des capteurs solaires), et on en deduit le quaternion observé (ligne 623 fusionsoleil.m)

#endif
