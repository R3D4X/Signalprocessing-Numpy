# import packages for scientific python and plotting
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

############################################# Model ################################################
#
#   y(t) = a + b*t + e(t) a: konst signal b*t: drift fehler e(t): mw freies weisses rauschen
#
#   y[n] = (1 nT) * (a) + e
#                   (b)
#
####################################################################################################

debug = 1;
netzbrumm = 0;

N = 200;
T = 0.1; #T = ta = 1s

a = -20; #Real Value
b = 1; #Drift

time = np.linspace(0,N*T,N);
if(netzbrumm == 1):
    y = a + b*time +1*np.random.randn(N) + 5*np.sin(2*np.pi*50*time); #Messung mit Netzbrumm
else:
    y = a + b*time +1*np.random.randn(N); #Messung

a_0 = np.zeros(2); #Parametervektor Anfangsschaetung ist 0
a_1 = np.zeros(2); #Buffer fuer parameterberechnung

p_0 = 100000 * np.identity(2); #Anfangsschaetzfehlerkovarianz ist sehr gross, da wir keine Werte haben
p_1 = np.zeros((2,2)); #Buffer fuer kovar berechnung

phi = np.zeros(2); #Basisvektor
k = np.zeros(2); #Gewichtungsfaktorvektor

for n in range(len(y)):
    if(debug == 1):
        print("In Iteration No. " + str(n));

    phi[0] = 1; #Basis von a (konstantes Messobjekt)
    phi[1] = n*T; #Basis von b (Drift)
    if(debug == 1):
        print("Phi = " + str(phi));

    q = np.matmul(phi,np.matmul(p_0,np.transpose(phi)))
    q = 1 + q;
    q = 1/q;
    if(debug == 1):
        print("Q = " + str(q));
    k = np.matmul(p_0,np.transpose(phi));
    k = q*k;
    if(debug == 1):
        print("K = " + str(k));

    #Guete vorfaktor fuer gewichtungsfaktor
    Q = 1 + phi[0]**2*p_0[0][0] + phi[0]*phi[1]*(p_0[0][1] + p_0[1][0]) + phi[1]**2*p_0[1][1];
    if(debug == 1):
        print("Q = " + str(Q));

    #Gewichtungsfaktor
    k[0] = 1/Q * (phi[0]*p_0[0][0] + phi[1]*p_0[0][1]);
    k[1] = 1/Q * (phi[0]*p_0[1][0] + phi[1]*p_0[1][1]);
    if(debug == 1):
        print("K = " + str(k));

    a_1 = k * (y[n] - np.matmul(phi,a_0));
    a_1 = np.add(a_0,a_1);
    if(debug == 1):
        print("A_1 = " + str(a_1));

    #neue Parameterschaetzung
    a_1[0] = a_0[0] + k[0] * (y[n] - phi[0]*a_0[0] - phi[1]*a_0[1]);
    a_1[1] = a_0[1] + k[1] * (y[n] - phi[0]*a_0[0] - phi[1]*a_0[1]);
    a_0 = a_1;
    if(debug == 1):
        print("Schaetzwert = " + str(a_0));

    #Kovarianz berechnen mit Matrixoperationen
    p_1[0][0] = phi[0]*phi[0];
    p_1[0][1] = phi[0]*phi[1];
    p_1[1][0] = phi[1]*phi[0];
    p_1[1][1] = phi[1]*phi[1];
    p_1 = np.add(p_1, np.linalg.inv(p_0));
    p_1 = np.linalg.inv(p_1);
    if(debug == 1):
        print("p_1:");
        print(p_1);

    #neue kovarianzmatrix
    p_1[0][0] =  p_0[0][0]*(1-k[0]*phi[0]) - p_0[1][0]*k[0]*phi[1];
    p_1[0][1] =  p_0[0][1]*(1-k[0]*phi[0]) - p_0[1][1]*k[0]*phi[1];
    p_1[1][0] =  p_0[1][0]*(1-k[1]*phi[1]) - p_0[0][0]*k[1]*phi[0];
    p_1[1][1] =  p_0[1][1]*(1-k[1]*phi[1]) - p_0[0][1]*k[1]*phi[0];
    p_0 = np.tile(p_1,1);
    if(debug == 1):
        print("Kovarianzmatrix:");
        print(p_0);

print("Done");
print("Schaetzwert = " + str(a_0));
print("Echte Werte:");
print("a = " + str(a));
print("b = " + str(b));
print("Kovarianzmatrix:");
print(p_0);

estimate = a_0[0] + a_0[1]*time; #Schaetzung des Messverlaufs

plt.plot(y,label='Messung');
plt.plot(estimate,label='Schaetzung');
plt.xlabel("Index");
plt.title("Schaetzung des Messsignal");
plt.xlabel("Index");
plt.legend();
plt.show();

