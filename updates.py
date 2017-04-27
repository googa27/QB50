import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time as tm
import calendar as cld
from pyquaternion import Quaternion
import tlm_transcription2 as tr
import TLE_parser_norad as tlepar
import ephem
import datetime

SurfaceX = 62.5
SurfaceY = 62.5
SurfaceZ = 62.5

dt = 0.01

kPi = 1
kSi = 1
kBi = 1
kIi = 1

#using namespace std;
#using namespace Eigen;

#weird operation written in quat.cpp
#need to complete
#this seems to be the rotation of vector x by quaternion quat
def rotate(quat, x):
    v = quat.vector
    n = quat.norm
    if (n!=0):
        v *= -1./n
    u = 2*v.dot(x)*v
    if (n!=0):
        u += ((quat[0]/n)*(quat[0]/n) - la.norm(v)*la.norm(v))*x;
    if (n!=0):
        u += 2*(quat[0]/n)*np.cross(v, x);
    return u;

class Etat:
    def __init__(self):
        self.q = Quaternion(0, 1, 0, 0)
        self.b = np.zeros(3)
        self.omega = np.zeros(3)
        self.qref = Quaternion(0, 0, 1, 0)
        self.omegaref = np.zeros(3)
        self.lat = 0
        self.longitude = 0
        self.momMag = np.zeros(3)
        self.intensite = np.zeros(3)
        self.controle = True
        self.datafile = open('tlm.3', 'r')
        self.currentline = 'initialized'
        self.currentdata = []
        self.gyrometer = np.zeros(3)
        self.magnetometer = np.zeros(3)
        self.capteursolaire = np.zeros(6)

        file = open("TLE_AO73.txt", 'r')
        #TLE reading
        self.name = file.readline()
        self.line1 = file.readline()
        self.line2 = file.readline()
        
        file.close()

        
    def update_data(self, datastring):
        self.currentline = datastring
        self.currentdata = tr.line_to_data(self.currentline)
        self.gyrometer = self.currentdata[-3]
        self.magnetometer = self.currentdata[-2]
        self.capteursolaire = self.currentdata[-1]

##        #updates the data to the next reading line
##    def update_data(self):
##        while(self.currentdata == [] or self.currentdata == 'line not recognized'):
##            self.currentline = self.datafile.readline()
##            self.currentdata = tr.line_to_data(self.currentline)
##        self.gyrometer = self.currentdata[-3]
##        self.magnetometer = self.currentdata[-2]
##        self.capteursolaire = self.currentdata[-1]

    def printstate(self):
        print('gyrometer: ', self.gyrometer)
        print('magnetometer: ', self.magnetometer)
        print('capteurs solaires: ', self.capteursolaire)

    #revisar juliandate
    def getJulianDate(self):
        t = cld.timegm(tm.gmtime())
        return (t / 86400.0) + 2440587.5

    #jd = julian date (cree valente)
    #retrurs solar vercor in geocentric  S_i
    def  getVecSolRef(self, jd):
        T = (jd - 2451545.)/36525
        lambdaSun = (280.4606184
                     + (36000.77005361 * T))
        MSun = (357.5277233
                + 35999.05034*T)
        lambdaE = (lambdaSun
                   + 1.914666471*np.sin(MSun*np.pi/180)
                   + 9.918994643*np.sin(2*MSun*np.pi/180))
        epsilon = (23.439291
                   - 0.0130042*T)
        s = np.array([np.cos(lambdaE*np.pi/180),
                      np.cos(epsilon*np.pi/180)*np.sin(lambdaE*np.pi/180),
                      np.sin(epsilon*np.pi/180)*np.sin(lambdaE*np.pi/180)])
        if (la.norm(s) != 0):
            s = s/la.norm(s)
        return s;
    #solo pasale lo que dice el captor solar
    def getCapteursSolaires(self):
        return self.capteursolaire

    def getVecSol(self):
        vmes = np.zeros(6)
        vecsol = np.zeros(3)
        H = np.eye(3)
        H = H/2
        M = np.array([[1, -1, 0, 0, 0 ,0],
                      [0, 0, 1, -1, 0 ,0],
                      [0, 0, 0, 0, 1,-1]])
        vmes = self.getCapteursSolaires()
        vecsol = np.dot(M, vmes)
        vecsol = np.dot(H, vecsol)
        if(la.norm(vecsol)!= 0):
            vecsol = vecsol/la.norm(vecsol)
        return vecsol


    #this file gets the magnetic field of the earth from the table
    def getChampRef(self, lat, longitude, om, i, omega, nu):
        fichier = open('igrf.txt', 'r')
        x, y, z = 0, 0, 0
        #ifstream fichier("/Users/adrianvalente/Documents/etudes/psc/source1502/ADCS/igrf.txt",ios::in);
        n = int(round(longitude)*179+90+np.floor(lat))
        for k in range(1, n+1):
            ligne = fichier.readline()
            ligne = ligne.split()
            #ATENCION
        x = float(ligne[3])
        y = float(ligne[4])
        z = float(ligne[5])
        mag_aux = np.array([x, y, z])
        fichier.close()
        theta = omega + nu
        m1 = np.array([[np.sin(i)*np.sin(om),np.cos(i)*np.cos(om),np.cos(om)],
		  [-np.sin(i)*np.cos(om),-np.cos(i)*np.sin(om),np.sin(om)],
		  [np.cos(i),np.sin(i),0]])
        m2 = np.array([[1,0,0],
                       [0,np.cos(theta),-np.sin(theta)],
                       [0,np.sin(theta),np.cos(theta)]])
        return np.dot(m1,np.dot(m2, mag_aux))

    #solo pasa el magnetometro
    def getChampM(self):
        return self.magnetometer

    # ver getlong
    def getLat(self):
        tle_rec = ephem.readtle(self.name,self.line1,self.line2)
        tle_rec.compute('2014/1/1 01:00')#insert the time here
        return float(tle_rec.sublat)
    # encontrar formula con tle
    def getLong(self):
        tle_rec = ephem.readtle(self.name,self.line1,self.line2)
        tle_rec.compute('2014/1/1 01:00')#insert the time here
        return float(tle_rec.sublong) 

    # solo pasa lo que esta en el gyrometro
    def getOmega(self):
        return self.gyrometer

    #a partir de tle, recuperar los cuatro angulos que estan como argumento
    #(no se supone que tenga como argumento eso, se supone que teine que retornarlos)

    #perih: argument du perihelie
    #om: Longitude du point ascendant croisant le plan equatorial
    #i: Inclinaison de l'orbite par rapport au plan equatorial
    #nu: Anomalie (pos du satellite)
    def getTLE(self):
        out = tlepar.parse_tle(self.name, self.line1, self.line2)
        return out['arg_of_perigee'], out['ra_of_asc_node'], out['inclination'], out['mean_anomaly']

    def getQref(self, perih, om, i, nu):
        m1 = np.array([[np.cos(om), -np.sin(om), 0],
                       [np.sin(om), np.cos(om), 0],
                       [0,       0,      1]])
        theta=perih+nu
        m2 = np.array([[1, 0, 0],
                       [0, np.cos(i), -np.sin(i)],
                       [0, np.sin(i), np.cos(i)]])
                      
        m3 = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])

        m4 = np.array([[0,-1,0],
                       [1,0,0],
                       [0,0,1]])
                      
        mtot = np.dot(m1,np.dot(m2,np.dot(m3, m4)))
    
        q = Quaternion(0)
        q[0] = np.sqrt(1.+mtot[0,0]+mtot[1,1]+mtot[2,2])/2.
        if (q[0]!=0):
            q[1] = (mtot[2,1]-mtot[1,2])/(4.*q[0])
            q[2] = (mtot[0,2]-mtot[2,0])/(4.*q[0])
            q[3] = (mtot[1,0]-mtot[0,1])/(4.*q[0])
        else: #Si la trace est nulle, il faut trouver l'axe de rotation...
            if (mtot[0,0]>0):
                q[1] = 1
                q[2] = 0
                q[3] = 0
            elif (mtot[1,1]>0):
                q[2] = 1
                q[1] = 0
                q[0] = 0
            else:
                q[3] = 1
                q[2] = 0
                q[1] = 0
        return q;


    #actualiza self.quaternion una sola vez con los ultimos datos recbidos
    def actualiser(self):
           #Initialisation : on recupere les donnees des capteurs et des tables
        julianDate = self.getJulianDate()
        lat = self.getLat()   #A FAIRE : recuperer les donnees du GPS
        longitude = self.getLong()
        omega = self.getOmega() #A FAIRE : recuperer vitesse de rotation du gyro
                                            #Vecteur rotation instantanée du satellite par rapport au ref geocentrique, exprimé dans le référentiel du satellite
            #TLE
            #A faire : fonction pour recuperer les donnees des TLE
        ##perih=0#argument du perihelie
        ##om=0#Longitude du point ascendant croisant le plan equatorial
        ##i=0#Inclinaison de l'orbite par rapport au plan equatorial
        ##nu=90#Anomalie (pos du satellite)
            #getTLE(&perih, &om, &i, &nu);
        #Passage en radians
        perih, om, i, nu = self.getTLE()
        om = om*np.pi/180
        perih = perih*np.pi/180
        nu = nu*np.pi/180
        i = i*np.pi/180
        qref = self.getQref(perih, om, i, nu)
            #Champs Mags
        champRef = self.getChampRef(lat, longitude,om,i,perih,nu) #dans le ref GEOCENTRIQUE
        champRef = rotate(self.q, champRef)  #On le passe immediatement dans le ref SATELLITE
        champRefNorme = champRef
        if (la.norm(champRef) != 0 ):
            champRefNorme =  champRef/la.norm(champRef)
        champM = self.getChampM()#A FAIRE : Champ mesuré dans le ref SATELLITE
        champMNorme = champM
        if (la.norm(champM) != 0):
            champMNorme = champM/la.norm(champM)
            #Capteurs solaires. Les vecteurs sont directement normalises
        vecSolRef = self.getVecSolRef(julianDate)#Vecteur solaire dans le ref GEOCENTRIQUE
        vecSolRef = rotate(self.q, vecSolRef)
        if (la.norm(vecSolRef) != 0):
            vecSolRef = vecSolRef/la.norm(vecSolRef)
        vecSol = self.getVecSol() #Vecteur solaire mesuré dans le ref SATELLITE
        if (la.norm(vecSol) != 0):
            vecSol /= la.norm(vecSol)
            #Calcul des quaternions (dans physic.m)
        GrandOmega = kBi*np.cross(champMNorme,champRefNorme) + kSi*np.cross(vecSol, vecSolRef)  #Intermediaire de calcul (omega dans le rapport de Valentin)
        vtmp = self.omega - self.b + kPi*GrandOmega
        qtmp = Quaternion(scalar = 0, vector = vtmp)
        self.q += (0.5*self.q*qtmp + (1-self.q.norm*self.q.norm)*self.q)*dt
        self.b -= kIi*GrandOmega*dt   
            #Controle (l. 540)	calcul du moment genere
        dq = Quaternion()
        m = np.zeros(3)
        if (self.controle==True and la.norm(champM)!=0):
            qtmp = self.q.inverse
            dq = qref*qtmp
            dq13 = dq.vector
            m = (-0.000048*np.cross(champM, omega-self.b-self.omegaref) - 0.0000003*np.cross(champM, dq13)/(la.norm(champM)*la.norm(champM)))
    
            #Calcul intensite l.325
        intensite = np.array([m[0]/SurfaceX, m[1]/SurfaceY, m[2]/SurfaceZ])
        return intensite;

#-----------------------------------------------------------------------------

e = Etat()
string = 'ON0FR5>TLM/1: <UI>:%01000101@000221;00000091978c07074e085c63\n'
e.update_data(string)
N = 10000
real = np.empty(N)
x = np.empty(N)
y = np.empty(N)
z = np.empty(N)
mag = np.empty(N)
axisx = np.empty(N)
axisy = np.empty(N)
axisz = np.empty(N)
angle = np.empty(N)
for n in range(N):
    e.actualiser()
    quat = e.q
##    real[n] = quat[0]
##    x[n] = quat[1]
##    y[n] = quat[2]
##    z[n] = quat[3]
##    mag[n] = quat.norm
    axis = quat.axis
    axisx[n] = axis[0]
    axisy[n] = axis[1]
    axisz[n] = axis[2]
    angle[n] = quat.angle

##plt.plot(real, label = 'real')
##plt.plot(x, label = 'x')
##plt.plot(y, label = 'y')
##plt.plot(z, label = 'z')
plt.plot(axisx, label = 'axisx')
plt.plot(axisy, label = 'axisy')
plt.plot(axisz, label = 'axisz')
plt.plot(angle, label = 'angle')
##plt.plot(mag, label = 'norm')

plt.legend()

plt.show()


#e.update_data('ON0FR5>TLM/1: <UI>:%01000101@000221;00000091978c07074e085c63')
