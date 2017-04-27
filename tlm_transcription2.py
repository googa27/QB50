import numpy as np
import matplotlib.pyplot as plt
from binascii import hexlify

def test(file):
    line = file.readline()
    out = decoupage(line)
    return out

def test2(file):
    decoup = test(file)
    if(decoup != ['no me interesa']):
        return traiter(decoup)
    return ['no me interesa']

def split_custom(string, characters):
    for ch in characters:
        if string.find(ch) >= 0:
             return string.split(ch)
    return ['no me interesa']

#separates file into relevant data
def decoupage(string):
    out = split_custom(string, '%')
    if len(out)<2:
        return out
    out += split_custom(out.pop(-1), '@')
    out += split_custom(out.pop(-1), ';')
    while out[-1] != '\n':
        aux = out.pop(-1)
        out += [aux[:2], aux[2:]]
    return out

def gyro_reorientation(vec):
    return np.array([-vec[0], vec[1], -vec[2]])

def mag_reorientation(vec):
    return np.array([-vec[1], -vec[0], -vec[2]])


#gyrometer with bias. There should be some movement
#returns list of vectors with translated information
#FORMAT: [DATE, HOUR, GYROMETER, MAGNETOMETER, SOLAR CAPTORS (+x,-x,+y,-y,+z,-z)]
def traiter(data):
    out = []
    aux = [data[1][:4], data[1][4:6], data[1][6:]]
    out.append('/'.join(aux[::-1]))
    aux = [data[2][:2], data[2][2:4], data[2][4:]]
    out.append(':'.join(aux))
    aux = []
    for i in range(3, 6):
        aux.append((int(data[i], 16))*0.14)#degrees, MODIFIED
    out.append(gyro_reorientation(np.array(aux)))
    aux = []
    for i in range(6, 9):
        aux.append((int(data[i], 16) - 2**7)*0.29)#micro tesla
    out.append(mag_reorientation(np.array(aux)))
    aux = []
    for i in range(9, 15):
        aux.append(np.exp(int(data[i], 16)*12.89*1E-3))#micro ampere?
    out.append(np.array(aux))
    return out


def line_to_data(line):
    aux = decoupage(line)
    if aux == ['no me interesa']:
        return 'line not recognized'
    return traiter(aux)

if __name__ == '__main__':

    #-------------------------------------------------------------------------

##    file = open('tlm.3', 'r')
##    y = test(file)
##
##    for i in range(100):
##        if y != ['no me interesa']:
##            print(y)
##        y = test(file)
##    file.close()
    #-------------------------------------------------------------------------
    print('#-------------------------------------------------------------------------')

    file = open('tlm.3', 'r')

    for i in range(200):
        line = file.readline()
        print(line_to_data(line))
    file.close()  
    ##t = []
    ##
    ##girometer = []
    ##
    ##magnetometer = []
    ##
    ##solar_pannel_pos =[]
    ##
    ##solar_pannel_neg = []
    ##
    ##file = open('tlm.3', 'r')
    ##x = file.readline()
    ##y = decoupage(x)
    ##
    ##if y != ['no me interesa']:
    ##        #y = traiter(y)
    ##        t.append(y[2])
    ##        girometer.append([y[3], y[4], y[5]])
    ##        magnetometer.append([y[6], y[7], y[8]])
    ##        solar_pannel_pos.append([y[9], y[11], y[13]])
    ##        solar_pannel_neg.append([y[10], y[12], y[14]])
    ##while x != '':
    ##    if y != ['no me interesa']:
    ##        print(y)
    ##    x = file.readline()
    ##    y = decoupage(x)
    ##    if y != ['no me interesa']:
    ##        traiter(y)
    ##        t.append(y[2])
    ##        girometer.append([y[3], y[4], y[5]])
    ##        magnetometer.append([y[6], y[7], y[8]])
    ##        solar_pannel_pos.append([y[9], y[11], y[13]])
    ##        solar_pannel_neg.append([y[10], y[12], y[14]])
    ##
    ##file.close()



    ##t = np.asarray(t)
    ##girometer = np.asarray(girometer)
    ##magnetometer = np.asarray(magnetometer)
    ##solar_pannel_pos = np.asarray(solar_pannel_pos)
    ##solar_pannel_neg = np.asarray(solar_pannel_neg)
    ##
    ##plt.figure(1)
    ##
    ##plt.subplot(221)
    ##plt.plot(girometer[:, 0], 'b-', girometer[:, 1],
    ##         'g-', girometer[:, 2], 'r-')
    ##
    ##plt.subplot(222)
    ##plt.plot(magnetometer[:, 0], 'b-', magnetometer[:, 1],
    ##         'g-', magnetometer[:, 2], 'r-')
    ##
    ##plt.subplot(223)
    ##plt.plot(solar_pannel_pos[:, 0], 'b-', solar_pannel_pos[:, 1],
    ##         'g-', solar_pannel_pos[:, 2], 'r-')
    ##plt.subplot(224)
    ##plt.plot(solar_pannel_neg[:, 0], 'b-', solar_pannel_neg[:, 1],
    ##         'g-', solar_pannel_neg[:, 2], 'r-')
    ##plt.show()
    ##
    ##comparison = ['63', '63', '6d', '6e', '6f', '6e', '6e', '6f', '6e', '63', '62', '91', '62', '63', '63', '64', '6a', '64', '64', '63', '63','63']
    ##
    ##file = open('tlm.3', 'r')
