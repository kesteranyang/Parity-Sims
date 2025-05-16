import numpy as np
from hmmlearn import hmm
from scipy import integrate

from matplotlib import pyplot as plt, cm
from scipy import integrate
import csv
import seaborn as sns
import pandas as pd
from scipy.special import kn
from matplotlib import colors
import random 
from sympy import pi, cos, sin, atan2
import scipy.signal as signal
from scipy.optimize import curve_fit
import scipy.fft
from scipy import signal
from tqdm import tqdm
import math
from scipy import special
from statistics import NormalDist
import waveAndLike
import time as tm
import multiprocessing
from decimal import Decimal
from scipy.special import factorial
#import likelihoodcheck
SSF=0.99#0.95
r=1 # P01/P10
P01=r*((1-SSF)/(1+r))
P10=(1-SSF)/(1+r)
#edep=0


def test():
    hbar_ev=6.5821*1e-16 #eVs
    hbar_js=1.054571817*1e-34 #Js
    Ej=hbar_ev*(2*np.pi)*6.25*1e9 #J
    Ec=hbar_ev*(2*np.pi)*250*1e6 #J
    supcon_gap_Al=180*1e-6 #eV
    Ege=np.sqrt(8*Ej*Ec)-Ec #J 
    wq=Ege/hbar_ev
    fq=wq/(2*np.pi)
    #print('fq=',fq*1e-6,'MHz')
    adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
    bdf=((Ej/Ec/2)**(3/4))
    cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
    ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
    dwq=adf*bdf*cdf*ddf
    dfq=dwq/(2*np.pi)

    half_x_pi_len=100*1e-9
    half_y_pi_len=100*1e-9
    rolen=2*1e-6
    idlen=1/(4*dfq)
    actlen=2*1e-6
    ramseymeas_len=half_x_pi_len + idlen + half_y_pi_len + rolen + actlen

    binwidth=0.1*1e-3
    N=int(binwidth/ramseymeas_len)
    maxtime=1.0*1e-3
    num_pts=int(maxtime/binwidth)
    seqnum= N
    #print(N)
    
    return binwidth

def probCalc(SSF, base_par_rate, Edeposited, bsT1,   roseq):
    ssf=SSF
    hbar_ev=6.5821*1e-16 #eVs
    hbar_js=1.054571817*1e-34 #Js
    Ej=hbar_ev*(2*np.pi)*9.945*1e9 #J
    Ec=hbar_ev*(2*np.pi)*390*1e6 #J
    supcon_gap_Al=180*1e-6 #eV
    Ege=np.sqrt(8*Ej*Ec)-Ec #J 
    wq=Ege/hbar_ev
    fq=wq/(2*np.pi)
    #print('fq=',fq*1e-6,'MHz')
    adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
    bdf=((Ej/Ec/2)**(3/4))
    cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
    ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
    dwq=adf*bdf*cdf*ddf
    dfq=dwq/(2*np.pi)
    
    kb_ev=8.6173303*1e-5 #eV/K
    T=20*1e-3 #K
    r0=0.018/1e-9 #Hz
    s0=1e-6/1e-9 #Hz
    ncp=4*1e24 #m^-3
    V_island=2400*((1e-6)**3) #m^3
    phonon_to_qp_eff=0.57
    base_Par_Rate=base_par_rate    #Hz
    
    Edeposited=Edeposited*1e-3 #eV
    xinduced=Edeposited*phonon_to_qp_eff / (ncp*V_island*supcon_gap_Al)
    
    decayRateCoeff=(16*Ej/(hbar_ev*np.pi)) * np.sqrt(Ec/(8*Ej)) * (supcon_gap_Al/(2*hbar_ev*wq))
    baseT1=bsT1*1e-6 #s
    #print('baseT1',baseT1)
    baseDecayRate=1/baseT1
    #print('base non qp decay rate=',baseDecayRate)
    parRateCoeff_v1=(16*Ej / (hbar_ev * np.pi))  *  np.sqrt(kb_ev*T/(2*np.pi*supcon_gap_Al))
    
    
    half_x_pi_len=100*1e-9
    half_y_pi_len=100*1e-9
    rolen=2*1e-6
    idlen=1/(4*dfq)
    actlen=2*1e-6
    ramseymeas_len=half_x_pi_len + idlen + half_y_pi_len + rolen + actlen

    binwidth=0.1*1e-3
    N=int(binwidth/ramseymeas_len)
    maxtime=5.0*1e-3
    num_pts=int(len(roseq)/N) #50ns timestep
    #E_depo_timestream=np.zeros(num_pts) #ns Energy depo into the island
    #E_depo_timestream[0]=EeV #eV
    
    time=np.linspace(0,maxtime,num_pts) #ns
    Ptotal  =np.zeros(len(time))

    P_oddnoid_noro      =np.zeros(len(time))
    P_oddnoid_noro0     =np.zeros(len(time))
    P_oddnoid_noro1     =np.zeros(len(time))
    P_oddnoid_decro     =np.zeros(len(time))
    P_oddnoid_decro0    =np.zeros(len(time))
    P_oddnoid_decro1    =np.zeros(len(time))
    P_oddnoid_excro     =np.zeros(len(time))
    P_oddnoid_excro0    =np.zeros(len(time))
    P_oddnoid_excro1    =np.zeros(len(time))
    P_oddeventid_noro   =np.zeros(len(time))
    P_oddeventid_noro0  =np.zeros(len(time))
    P_oddeventid_noro1  =np.zeros(len(time))
    P_oddeventid_decro  =np.zeros(len(time))
    P_oddeventid_decro0 =np.zeros(len(time))
    P_oddeventid_decro1 =np.zeros(len(time))
    P_oddeventid_excro  =np.zeros(len(time))
    P_oddeventid_excro0 =np.zeros(len(time))
    P_oddeventid_excro1 =np.zeros(len(time))
    P_oddtotal          =np.zeros(len(time))
    
    P_odd0              =np.zeros(len(time))
    P_odd1              =np.zeros(len(time))
    P_oddg              =np.zeros(len(time))
    P_odde              =np.zeros(len(time))
    
    P_eveng              =np.zeros(len(time))
    P_evene              =np.zeros(len(time))
    
    P_evennoid_noro      =np.zeros(len(time))
    P_evennoid_noro0     =np.zeros(len(time))
    P_evennoid_noro1     =np.zeros(len(time))
    P_evennoid_decro     =np.zeros(len(time))
    P_evennoid_decro0    =np.zeros(len(time))
    P_evennoid_decro1    =np.zeros(len(time))
    P_evennoid_excro     =np.zeros(len(time))
    P_evennoid_excro0    =np.zeros(len(time))
    P_evennoid_excro1    =np.zeros(len(time))
    P_eveneventid_noro   =np.zeros(len(time))
    P_eveneventid_noro0  =np.zeros(len(time))
    P_eveneventid_noro1  =np.zeros(len(time))
    P_eveneventid_decro  =np.zeros(len(time))
    P_eveneventid_decro0 =np.zeros(len(time))
    P_eveneventid_decro1 =np.zeros(len(time))
    P_eveneventid_excro  =np.zeros(len(time))
    P_eveneventid_excro0 =np.zeros(len(time))
    P_eveneventid_excro1 =np.zeros(len(time))
    P_eventotal          =np.zeros(len(time))
    P_even0              =np.zeros(len(time))
    P_even1              =np.zeros(len(time))
    
    Pjump        =np.zeros(len(time))
    Expectedjumps=np.zeros(len(time))

    ramseyxqp_v1=np.zeros(len(time))
    ramseyParRate_v1=np.zeros(len(time))
    ramseydecayrate=np.zeros(len(time))
    ramseyexcrate=np.zeros(len(time))
    dephasingrate=np.zeros(len(time))
    Prob_dec_x=np.zeros(len(time))
    Prob_dec_idt=np.zeros(len(time))
    Prob_dec_y=np.zeros(len(time))
    Prob_dec_ro=np.zeros(len(time))
    Prob_Par_meas=np.zeros(len(time))
    Prob_exc_x=np.zeros(len(time))
    Prob_exc_idt=np.zeros(len(time))
    Prob_exc_y=np.zeros(len(time))
    Prob_exc_ro=np.zeros(len(time))
    Prob_exc_deph_idt =np.zeros(len(time))
    Prob_g_none_idt   =np.zeros(len(time))
    Prob_g_idt        =np.zeros(len(time))
    Prob_none_idt     =np.zeros(len(time))
    Prob_deph_idt=np.zeros(len(time))
    Prob_dec_or_exc_meas=np.zeros(len(time))
    ro=np.zeros(len(time))
    Prob_none_meas=np.zeros(len(time))
    all=np.zeros(len(time))
    full=np.zeros(len(time))
    Pnojump=np.zeros(len(time))
    lamda=np.zeros(len(time))
    Pe   =np.zeros(len(time))
    Po   =np.zeros(len(time))
    PestartPe=np.zeros(len(time))
    PestartPo=np.zeros(len(time))
    PostartPo=np.zeros(len(time))
    PostartPe=np.zeros(len(time))
    # pos=0
    # neg=0
    
    for i,ti in enumerate(time):
        
        #rolen=j*1e-6
        # for i in tqdm(range(1000)):
        
        x0_v1= base_Par_Rate / (parRateCoeff_v1 + decayRateCoeff +  (decayRateCoeff * np.exp(-1*hbar_ev*wq/(kb_ev*T))))
        #print('baseParRate',baseParRate)
        #print('xo_v1',x0_v1)
    
        g_v1 = ((2*r0*x0_v1 + s0)*(2*r0*x0_v1 + s0) - s0*s0) / (4*r0)
        tss_v1= 1 / (2*r0*x0_v1 + s0)
        rprime_v1= r0*tss_v1*xinduced / (1 + r0*tss_v1*xinduced)
     
        
        ramseyxqp_v1[i]=((xinduced * (1-rprime_v1))/ (np.exp(ti/tss_v1) - rprime_v1)) + x0_v1
        ramseydecayrate_qp=decayRateCoeff * ramseyxqp_v1[i]
        ramseydecayrate[i]=ramseydecayrate_qp + baseDecayRate
        ramseyexcrate[i]=ramseydecayrate_qp  * np.exp(-1*hbar_ev*wq/(kb_ev*T))
        ramseyParRate_v1_nostate_change= parRateCoeff_v1 * ramseyxqp_v1[i]
        ramseyParRate_v1[i]=ramseyParRate_v1_nostate_change +  ramseydecayrate_qp + ramseyexcrate[i]
        dephasingrate[i]=0
        basePar= (parRateCoeff_v1 + decayRateCoeff +  (decayRateCoeff * np.exp(-1*hbar_ev*wq/(kb_ev*T))))* x0_v1
        #print('baseParQP',basePar)
        
        Prob_dec_idt[i]=1-np.exp(-1*idlen*ramseydecayrate[i])
        
        
        Prob_dec_ro[i] =1-np.exp(-1*rolen*ramseydecayrate[i])
        #print(Prob_dec_ro[i])
       
        Prob_exc_idt[i]=1-np.exp(-1*idlen*ramseyexcrate[i])
        
        Prob_exc_ro[i]=1-np.exp(-1*rolen*ramseyexcrate[i])
        Prob_deph_idt[i]=1-np.exp(-1*idlen*dephasingrate[i])
        Prob_Par_meas[i]=1-np.exp(-1*ramseymeas_len*ramseyParRate_v1[i])
        #ro[i]=(Prob_dec_ro[i]*(1-Prob_exc_ro[i]) + Prob_exc_ro[i]*(1-Prob_dec_ro[i]) + 2*Prob_dec_ro[i]*Prob_exc_ro[i] + (1-Prob_dec_ro[i])*(1-Prob_exc_ro[i]))
        Prob_dec_or_exc_meas[i]= ((Prob_dec_idt[i]*(1-Prob_exc_idt[i]) + Prob_exc_idt[i]*(1-Prob_dec_idt[i]) + 2*Prob_dec_idt[i]*Prob_exc_idt[i]))*(Prob_dec_ro[i]*(1-Prob_exc_ro[i]) + Prob_exc_ro[i]*(1-Prob_dec_ro[i]) + 2*Prob_dec_ro[i]*Prob_exc_ro[i] + (1-Prob_dec_ro[i])*(1-Prob_exc_ro[i]))  + (1-Prob_dec_idt[i])*(1-Prob_exc_idt[i]) *(Prob_dec_ro[i]*(1-Prob_exc_ro[i]) + Prob_exc_ro[i]*(1-Prob_dec_ro[i]) + 2*Prob_dec_ro[i]*Prob_exc_ro[i] + (1-Prob_dec_ro[i])*(1-Prob_exc_ro[i]))
        #Prob_dec_or_exc_meas[i]= ((Prob_dec_idt[i]*(1-Prob_exc_idt[i]) + Prob_exc_idt[i]*(1-Prob_dec_idt[i]) + 2*(Prob_dec_idt[i]*(1-Prob_exc_idt[i]) * Prob_exc_idt[i]*(1-Prob_dec_idt[i]))))*(Prob_dec_ro[i]*(1-Prob_exc_ro[i]) + Prob_exc_ro[i]*(1-Prob_dec_ro[i]) + 2*(Prob_dec_ro[i]*(1-Prob_exc_ro[i]) + Prob_exc_ro[i]*(1-Prob_dec_ro[i])) + (1-Prob_dec_ro[i])*(1-Prob_exc_ro[i]))  + (1-Prob_dec_idt[i])*(1-Prob_exc_idt[i]) * (Prob_dec_ro[i]*(1-Prob_exc_ro[i]) + Prob_exc_ro[i]*(1-Prob_dec_ro[i]) + 2*(Prob_dec_ro[i]*(1-Prob_exc_ro[i]) + Prob_exc_ro[i]*(1-Prob_dec_ro[i]))) 
        Prob_none_meas[i]=(1-Prob_dec_idt[i])*(1-Prob_exc_idt[i])*(1-Prob_dec_ro[i])*(1-Prob_exc_ro[i])
        all[i]= Prob_dec_or_exc_meas[i]#+Prob_none_meas[i]
        Ce2_idt=0.5
        Cg2_idt=0.5
        ##Cases during idt when qubit is excited: only dec, only deph, nothing, some combo of dec and deph
    
    
        #during readout: when qubit is excited only dec or nothing and when qubit is in g, only exc or nothing
        #When nothing happens in idt, during ro, the qubit is excited when parity is odd and qubit is in g when parity is even
        #When nothing happens in idt for odd parity, the Ce2=1 so there can be dec or nothing
        Ce2oddRO_noneidt=1
        Cg2oddRO_noneidt=0
        Ce2evenRO_noneidt=0
        Cg2evenRO_noneidt=1
        #Odd parity
        #Odd parity nothing in idt, dec in ro
        
    
        #cases where stuff happen in idt
        #Odd parity, dec in idt, will cause Ce2=0.5 during ro so dec or exc can hapen during ro
        Ce2RO_decidt=0.5
        Cg2RO_decidt=0.5
        
        # excitation in idt will also casue the same results decay in idt.
        Ce2RO_excidt=0.5
        Ce2RO_dec_or_excidt=0.5
        Cg2RO_excidt=0.5
        Cg2RO_dec_or_excidt=0.5
    
        #Nothing-idt, nothing-ro
        P_oddnoid_noro[i]      =(Ce2_idt*(1-(Prob_dec_idt[i] + Prob_deph_idt[i])) + Cg2_idt*(1-(Prob_exc_idt[i] + Prob_deph_idt[i]))) * (Ce2oddRO_noneidt*(1-Prob_dec_ro[i]) + Cg2oddRO_noneidt*(1-Prob_exc_ro[i]))
        P_oddnoid_noro0[i]     =P_oddnoid_noro[i]*(1-ssf) #measurement noise
        P_oddnoid_noro1[i]     =P_oddnoid_noro[i]*ssf #correct measurement
        #Nothing-id, dec  ro
        P_oddnoid_decro[i]     =(Ce2_idt*(1-(Prob_dec_idt[i] + Prob_deph_idt[i])) + Cg2_idt*(1-(Prob_exc_idt[i] + Prob_deph_idt[i]))) * (Ce2oddRO_noneidt*(Prob_dec_ro[i]))
        P_oddnoid_decro0[i]    = P_oddnoid_decro[i]*ssf #correct measurement
        P_oddnoid_decro1[i]    = P_oddnoid_decro[i]*(1-ssf) #measurement noise
        #Nothing-id, exc ro
        P_oddnoid_excro[i]     =(Ce2_idt*(1-(Prob_dec_idt[i] + Prob_deph_idt[i])) + Cg2_idt*(1-(Prob_exc_idt[i] + Prob_deph_idt[i]))) * ( Cg2oddRO_noneidt*(Prob_exc_ro[i]))
        P_oddnoid_excro0[i]    = P_oddnoid_excro[i]*(1-ssf)
        P_oddnoid_excro1[i]    = P_oddnoid_excro[i]*ssf
        #dec or exc id, nothing-ro
        P_oddeventid_noro[i]   =(Ce2_idt*(Prob_dec_idt[i] + Prob_deph_idt[i]) + Cg2_idt*(Prob_exc_idt[i] + Prob_deph_idt[i])) * (Ce2RO_dec_or_excidt*(1-Prob_dec_ro[i]) + Cg2RO_dec_or_excidt*(1-Prob_exc_ro[i]))
        P_oddeventid_noro0[i]  =P_oddeventid_noro[i]*0.5
        P_oddeventid_noro1[i]  =P_oddeventid_noro[i]*0.5
        #dec or exc id, dec ro
        P_oddeventid_decro[i]  =(Ce2_idt*(Prob_dec_idt[i] + Prob_deph_idt[i]) + Cg2_idt*(Prob_exc_idt[i] + Prob_deph_idt[i])) * (Ce2RO_dec_or_excidt*(Prob_dec_ro[i]))
        P_oddeventid_decro0[i] =P_oddeventid_decro[i]*ssf
        P_oddeventid_decro1[i] =P_oddeventid_decro[i]*(1-ssf)
        #dec or exc id, exc ro
        P_oddeventid_excro[i]  =(Ce2_idt*(Prob_dec_idt[i] + Prob_deph_idt[i]) + Cg2_idt*(Prob_exc_idt[i] + Prob_deph_idt[i])) * (Cg2RO_dec_or_excidt*(Prob_exc_ro[i]))
        P_oddeventid_excro0[i] =P_oddeventid_excro[i]*(1-ssf)
        P_oddeventid_excro1[i] =P_oddeventid_excro[i]*ssf
        P_odd0[i]              =P_oddnoid_noro0[i] + P_oddnoid_decro0[i] + P_oddnoid_excro0[i] + P_oddeventid_noro0[i] + P_oddeventid_decro0[i] + P_oddeventid_excro0[i]
        P_odd1[i]              =P_oddnoid_noro1[i] + P_oddnoid_decro1[i] + P_oddnoid_excro1[i] + P_oddeventid_noro1[i] + P_oddeventid_decro1[i] + P_oddeventid_excro1[i]
        P_oddtotal[i]          =P_odd0[i] + P_odd1[i]#P_oddeventid_noro0[i] + P_oddeventid_noro1[i] + P_oddnoid_noro0[i] + P_oddnoid_noro1[i] + P_oddnoid_decro0[i] + P_oddnoid_decro1[i] + P_oddnoid_excro0[i] + P_oddnoid_excro1[i] + P_oddeventid_decro0[i] + P_oddeventid_decro1[i] + P_oddeventid_excro0[i] + P_oddeventid_excro1[i]
        
        P_oddg[i]              =P_oddnoid_decro[i] + 0.5*P_oddeventid_noro[i] + P_oddeventid_decro[i]
        P_odde[i]              =P_oddnoid_noro[i] + 0.5*P_oddeventid_noro[i] + P_oddeventid_excro[i]
        
        #######
        Ce2oddRO_noneidt=1
        Cg2oddRO_noneidt=0
        Ce2evenRO_noneidt=0
        Cg2evenRO_noneidt=1
        P_evennoid_noro[i]      =(Ce2_idt*(1-(Prob_dec_idt[i] + Prob_deph_idt[i])) + Cg2_idt*(1-(Prob_exc_idt[i] + Prob_deph_idt[i]))) * (Ce2evenRO_noneidt*(1-Prob_dec_ro[i]) + Cg2evenRO_noneidt*(1-Prob_exc_ro[i]))
        P_evennoid_noro0[i]     =P_evennoid_noro[i]*ssf #correct measurement
        P_evennoid_noro1[i]     =P_evennoid_noro[i]*(1-ssf) #measurement noise
        #Nothing-id, dec  ro
        P_evennoid_decro[i]     =(Ce2_idt*(1-(Prob_dec_idt[i] + Prob_deph_idt[i])) + Cg2_idt*(1-(Prob_exc_idt[i] + Prob_deph_idt[i]))) * (Ce2evenRO_noneidt*(Prob_dec_ro[i]))
        P_evennoid_decro0[i]    = P_evennoid_decro[i]*ssf #correct measurement
        P_evennoid_decro1[i]    = P_evennoid_decro[i]*(1-ssf) #measurement noise
        #Nothing-id, exc ro
        P_evennoid_excro[i]     =(Ce2_idt*(1-(Prob_dec_idt[i] + Prob_deph_idt[i])) + Cg2_idt*(1-(Prob_exc_idt[i] + Prob_deph_idt[i]))) * ( Cg2evenRO_noneidt*(Prob_exc_ro[i]))
        P_evennoid_excro0[i]    = P_evennoid_excro[i]*(1-ssf)
        P_evennoid_excro1[i]    = P_evennoid_excro[i]*ssf
        #dec or exc id, nothing-ro
        P_eveneventid_noro[i]   =(Ce2_idt*(Prob_dec_idt[i] + Prob_deph_idt[i]) + Cg2_idt*(Prob_exc_idt[i] + Prob_deph_idt[i])) * (Ce2RO_dec_or_excidt*(1-Prob_dec_ro[i]) + Cg2RO_dec_or_excidt*(1-Prob_exc_ro[i]))
        P_eveneventid_noro0[i]  =P_eveneventid_noro[i]*0.5
        P_eveneventid_noro1[i]  =P_eveneventid_noro[i]*0.5
        #dec or exc id, dec ro
        P_eveneventid_decro[i]  =(Ce2_idt*(Prob_dec_idt[i] + Prob_deph_idt[i]) + Cg2_idt*(Prob_exc_idt[i] + Prob_deph_idt[i])) * (Ce2RO_dec_or_excidt*(Prob_dec_ro[i]))
        P_eveneventid_decro0[i] =P_eveneventid_decro[i]*ssf
        P_eveneventid_decro1[i] =P_eveneventid_decro[i]*(1-ssf)
    #dec or exc id, exc ro
        P_eveneventid_excro[i]  =(Ce2_idt*(Prob_dec_idt[i] + Prob_deph_idt[i]) + Cg2_idt*(Prob_exc_idt[i] + Prob_deph_idt[i])) * (Cg2RO_dec_or_excidt*(Prob_exc_ro[i]))
        P_eveneventid_excro0[i] =P_eveneventid_excro[i]*(1-ssf)
        P_eveneventid_excro1[i] =P_eveneventid_excro[i]*ssf
        P_even0[i]              =P_evennoid_noro0[i] + P_evennoid_decro0[i] + P_evennoid_excro0[i] + P_eveneventid_noro0[i] + P_eveneventid_decro0[i] + P_eveneventid_excro0[i]
        P_even1[i]              =P_evennoid_noro1[i] + P_evennoid_decro1[i] + P_evennoid_excro1[i] + P_eveneventid_noro1[i] + P_eveneventid_decro1[i] + P_eveneventid_excro1[i]
        P_eventotal[i]          =P_even0[i] + P_even1[i]#P_oddeventid_noro0[i] + P_oddeventid_noro1[i] + P_oddnoid_noro0[i] + P_oddnoid_noro1[i] + P_oddnoid_decro0[i] + P_oddnoid_decro1[i] + P_oddnoid_excro0[i] + P_oddnoid_excro1[i] + P_oddeventid_decro0[i] + P_oddeventid_decro1[i] + P_oddeventid_excro0[i] + P_oddeventid_excro1[i]
        P_eveng[i]              =P_evennoid_noro[i] + 0.5*P_eveneventid_noro[i] + P_eveneventid_decro[i]
        P_evene[i]              =P_evennoid_excro[i] + 0.5*P_eveneventid_noro[i] + P_eveneventid_excro[i]
        #prob of a jump is that of (0.5*Podd0 + 0.5Peven0)*Prob(parityswitch)*(Podd1 + Peven1)*2 The factor of 2 is for 01 and the 10 order.
        #The factor of 0.5 is the probability that the qubit is in the odd parity state at the begining of the ramsey measurement. Same for the even parity state.
        #Po=0.5
        #Pe=0.5
        #Pjump[i]=2*(P_odd0[i] + P_even0[i])*(P_odd1[i] + P_even1[i])
        #Pjump[i]  =(Oddstart*P_odd0[i]*(1-Prob_Par_meas[i])*P_odd1[i] + Evenstart*P_even0[i]*(1-Prob_Par_meas[i])*P_even1[i] + Oddstart*P_odd0[i]*Prob_Par_meas[i]*P_even1[i] + Evenstart*P_even0[i]*Prob_Par_meas[i]*P_odd1[i])
        #Pnojump[i]=(Oddstart*P_odd_0[i]*(1-Prob_Par_meas[i])*P_odd_0[i] + Evenstart*P_even0[i]*(1-Prob_Par_meas[i])*P_even0[i] + Oddstart*P_odd1[i]*Prob_Par_meas[i]*P_even1[i] + Evenstart*P_even1[i]*Prob_Par_meas[i]*P_odd1[i])
        #Pjump[i]  =2*(Oddstart*P_odd0[i]*(1-Prob_Par_meas[i])*P_odd1[i] + Evenstart*P_even0[i]*(1-Prob_Par_meas[i])*P_even1[i] + Oddstart*P_odd0[i]*Prob_Par_meas[i]*P_even1[i] + Evenstart*P_even0[i]*Prob_Par_meas[i]*P_odd1[i])
        #Pnojump[i]=(Oddstart*P_odd_0[i]*(1-Prob_Par_meas[i])*P_odd_0[i] + Oddstart*P_odd_1[i]*(1-Prob_Par_meas[i])*P_odd_1[i] + Evenstart*P_even0[i]*(1-Prob_Par_meas[i])*P_even0[i] + Evenstart*P_even1[i]*(1-Prob_Par_meas[i])*P_even1[i] + 2*Oddstart*P_odd1[i]*Prob_Par_meas[i]*P_even1[i]  + 2*Oddstart*P_odd0[i]*Prob_Par_meas[i]*P_even0[i])
        lamda[i]=integrate.simpson(ramseyParRate_v1[:i+1], time[:i+1])
        evenNum_ofswts = (pow(lamda[i],0) * np.exp(-1*lamda[i]) / math.factorial(0)) + (pow(lamda[i],2) * np.exp(-1*lamda[i]) / math.factorial(2)) + (pow(lamda[i],4) * np.exp(-1*lamda[i]) / math.factorial(4)) + (pow(lamda[i],6) * np.exp(-1*lamda[i]) / math.factorial(6)) + (pow(lamda[i],8) * np.exp(-1*lamda[i]) / math.factorial(8)) + (pow(lamda[i],10) * np.exp(-1*lamda[i]) / math.factorial(10))
        PestartPe[i]   = 0.5*evenNum_ofswts#1-Prob_Par_meas[i]
        PostartPe[i]   = 0.5*(1-evenNum_ofswts)#0.5*(pow(lamda[i],1) * np.exp(-1*lamda[i]) / math.factorial(1)) + (pow(lamda[i],3) * np.exp(-1*lamda[i]) / math.factorial(3)) + (pow(lamda[i],5) * np.exp(-1*lamda[i]) / math.factorial(5)) + (pow(lamda[i],7) * np.exp(-1*lamda[i]) / math.factorial(7)) + (pow(lamda[i],9) * np.exp(-1*lamda[i]) / math.factorial(9)) + (pow(lamda[i],11) * np.exp(-1*lamda[i]) / math.factorial(11))#1-Prob_Par_meas[i]
        PostartPo[i]   =0.5*evenNum_ofswts
        PestartPo[i]   =0.5*(1-evenNum_ofswts)
        Pe[i]          =PestartPe[i]+PostartPe[i]
        Po[i]          =PostartPo[i]+PestartPo[i]#PostartPo[i]+PostartPe[i]
        #Pfull[i]       =Pe[i]+Po[i]
        Pjump[i]        =Pe[i]*P_even0[i]*P_odd1[i]*Prob_Par_meas[i]+ Po[i]*P_odd1[i]*P_even0[i]*Prob_Par_meas[i] + Pe[i]*P_even1[i]*P_odd0[i]*Prob_Par_meas[i]+ Po[i]*P_odd0[i]*P_even1[i]*Prob_Par_meas[i] + Pe[i]*P_even0[i]*P_even1[i]*(1-Prob_Par_meas[i])  + Po[i]*P_odd1[i]*P_odd0[i]*(1-Prob_Par_meas[i]) + Pe[i]*P_even1[i]*P_even0[i]*(1-Prob_Par_meas[i])  + Po[i]*P_odd0[i]*P_odd1[i]*(1-Prob_Par_meas[i]) 
        Pnojump[i]      =Pe[i]*P_even0[i]*P_odd0[i]*Prob_Par_meas[i]+ Po[i]*P_odd1[i]*P_even1[i]*Prob_Par_meas[i] + Pe[i]*P_even1[i]*P_odd1[i]*Prob_Par_meas[i]+ Po[i]*P_odd0[i]*P_even0[i]*Prob_Par_meas[i] + Pe[i]*P_even0[i]*P_even0[i]*(1-Prob_Par_meas[i])  + Po[i]*P_odd1[i]*P_odd1[i]*(1-Prob_Par_meas[i]) + Pe[i]*P_even1[i]*P_even1[i]*(1-Prob_Par_meas[i])  + Po[i]*P_odd0[i]*P_odd0[i]*(1-Prob_Par_meas[i]) 
        full[i]         =Pjump[i]+Pnojump[i]
        Expectedjumps[i]=N*Pjump[i]
    #print(max((0.5*P_even1+0.5*P_odd0)))
    #max_errorprob.append(max((0.5*P_even1+0.5*P_odd0)))
    return Pjump, Expectedjumps, binwidth, time, ramseydecayrate, Prob_Par_meas,  P_oddg, P_odde, P_eveng, P_evene, ramseymeas_len, N#



###########Viterbi HMM

def ViterbiHMM(ssf, basePar, edep, baseT1, N, roseqfull):
    ##### Viterbi HMM
    ssf=SSF
    hbar_ev=6.5821*1e-16 #eVs
    hbar_js=1.054571817*1e-34 #Js
    Ej=hbar_ev*(2*np.pi)*6.25*1e9 #J
    Ec=hbar_ev*(2*np.pi)*250*1e6 #J
    supcon_gap_Al=180*1e-6 #eV
    Ege=np.sqrt(8*Ej*Ec)-Ec #J 
    wq=Ege/hbar_ev
    fq=wq/(2*np.pi)
    #print('fq=',fq*1e-6,'MHz')
    adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
    bdf=((Ej/Ec/2)**(3/4))
    cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
    ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
    dwq=adf*bdf*cdf*ddf
    dfq=dwq/(2*np.pi)
    
    Pjump, Expectedjumps, binwidth, probcalctime, ramseydecayrate, Prob_Par_meas,  P_oddg, P_odde, P_eveng, P_evene, ramseymeas_len, N=probCalc(ssf, basePar, edep, baseT1,  roseqfull)
    ###
    states=['odg','ode','evg','eve'] ### Hidden states
    #laststate=3
    numOfstates=len(states) ### num of hidden states
    observations=[0,1] ### readout values
    numOfobsv=len(observations) ### number of readout value types
    initial_prob=[0.25,0.25,0.25,0.25]
    most_prob_hidstate_seqFULL=[]
    
    odg_odg=np.zeros(len(probcalctime))
    odg_ode=np.zeros(len(probcalctime))
    odg_evg=np.zeros(len(probcalctime))
    odg_eve=np.zeros(len(probcalctime))
    ode_odg=np.zeros(len(probcalctime))
    ode_ode=np.zeros(len(probcalctime))
    ode_evg=np.zeros(len(probcalctime))
    ode_eve=np.zeros(len(probcalctime))
    evg_odg=np.zeros(len(probcalctime))
    evg_ode=np.zeros(len(probcalctime))
    evg_evg=np.zeros(len(probcalctime))
    evg_eve=np.zeros(len(probcalctime))
    eve_odg=np.zeros(len(probcalctime))
    eve_ode=np.zeros(len(probcalctime))
    eve_evg=np.zeros(len(probcalctime))
    eve_eve=np.zeros(len(probcalctime))

    for i in range(len(probcalctime)):
        ########## Transition Probabilities For hidden states
        odg_odg[i]= (1-Prob_Par_meas[i]) * P_oddg[i] #0.5 * P_oddg[i] *
        odg_ode[i]= (1-Prob_Par_meas[i]) * P_odde[i] #0.5 * P_oddg[i] *
        odg_evg[i]= (Prob_Par_meas[i])   * P_eveng[i] #0.5 * P_oddg[i] *
        odg_eve[i]= (Prob_Par_meas[i])   * P_evene[i] #0.5 * P_oddg[i] *
        ode_odg[i]= (1-Prob_Par_meas[i]) * P_oddg[i]  ##0.5 * P_odde[i] *
        ode_ode[i]= (1-Prob_Par_meas[i]) * P_odde[i]  ##0.5 * P_odde[i] *
        ode_evg[i]= (Prob_Par_meas[i])   * P_eveng[i]  ##0.5 * P_odde[i] *
        ode_eve[i]= (Prob_Par_meas[i])   * P_evene[i]  ##0.5 * P_odde[i] *
        evg_odg[i]= (Prob_Par_meas[i])  * P_oddg[i]  ### 0.5 * P_eveng[i] *
        evg_ode[i]= (Prob_Par_meas[i])  * P_odde[i]  ### 0.5 * P_eveng[i] *
        evg_evg[i]= (1-Prob_Par_meas[i])  * P_eveng[i]  ### 0.5 * P_eveng[i] *
        evg_eve[i]= (1-Prob_Par_meas[i])  * P_evene[i]  ### 0.5 * P_eveng[i] *
        eve_odg[i]= (Prob_Par_meas[i])  * P_oddg[i]  ###  0.5 * P_evene[i] *
        eve_ode[i]= (Prob_Par_meas[i])  * P_odde[i]  ###  0.5 * P_evene[i] *
        eve_evg[i]= (1-Prob_Par_meas[i])  * P_eveng[i]  ###  0.5 * P_evene[i] *
        eve_eve[i]= (1-Prob_Par_meas[i])  * P_evene[i]  ###  0.5 * P_evene[i] *
    
        ####transition matrix
        #Transition to: odg   , ode  , evg  , eve 
        T_matrix=[ [odg_odg[i], odg_ode[i], odg_evg[i], odg_eve[i]],   ####odg Transition from these to the states listed horizontally
                   [ode_odg[i], ode_ode[i], ode_evg[i], ode_eve[i]],   ####ode Transition from these to the states listed horizontally
                   [evg_odg[i], evg_ode[i], evg_evg[i], evg_eve[i]],   ####evg Transition from these to the states listed horizontally
                   [eve_odg[i], eve_ode[i], eve_evg[i], eve_eve[i]]]   ####eve Transition from these to the states listed horizontally
        
        ####emission_matrix     0   ,    1
        emission_matrix=[  [ssf     , (1-ssf) ], ####odg
                           [(1-ssf) , ssf     ], ####ode
                           [ssf     , (1-ssf) ], ####evg
                           [(1-ssf) , ssf     ]] ####eve
        #for i in ran    
        

        if i!=0:
            initial_prob=[0,0,0,0]
            initial_prob[laststate]=1

        ####observation sequence
        half_x_pi_len=100*1e-9
        half_y_pi_len=100*1e-9
        rolen=2*1e-6
        idlen=1/(4*dfq)
        actlen=2*1e-6
        ramseymeas_len=half_x_pi_len + idlen + half_y_pi_len + rolen + actlen
    
        binwidth=0.1*1e-3
        N=int(binwidth/ramseymeas_len)
        maxtime=1.0*1e-3
        num_pts=int(maxtime/binwidth)
        seqnum= N   #int(binwidth/ramseymeas_len)
        roseq=np.array(roseqfull[i*seqnum:int((i+1)*seqnum)], dtype='int').reshape(-1, 1)
        #print('i=',i, 'roseq',roseq)
        #print(np.array(readout[i*21:int((i+1)*21)]))
        model = hmm.CategoricalHMM(n_components=numOfstates)
        #initial_prob[-1]=initial_prob[]
        model.startprob_ = initial_prob
        model.transmat_ = T_matrix
        model.emissionprob_ = emission_matrix
        #print('i=',i, 'i*19=',i*19)
        log_probability, hidden_states = model.decode(roseq,
        										lengths = len(roseq),
        										algorithm ='viterbi' )
        laststate=hidden_states[-1]
        mostprobableParseq=np.zeros(len(hidden_states))
        for j in range(len(hidden_states)):
            if hidden_states[j]<2:
                mostprobableParseq[j]=1
            else:
                mostprobableParseq[j]=-1
        most_prob_hidstate_seqFULL=most_prob_hidstate_seqFULL+mostprobableParseq.tolist()
        #likelihood.append(log_probability)

    return most_prob_hidstate_seqFULL
############


###############################################
####counting jumps

def countjumps(output):
    jumps=0
    for i in range(len(output)):
        if i==0:
            pass
        else:
            if np.abs(output[i]-output[i-1])>0:
                jumps=1+jumps
    return jumps


#############################################################################################
########Waveform
def Makewaveform(seq,binwidth, ramseymeas_len):
    numrepsInBin=int(binwidth/ramseymeas_len)
    numBins=int(len(seq)/numrepsInBin)
    wvfm=np.zeros(numBins)
    for i in range(numBins):
        jumpsWafm=countjumps(seq[int(i*numrepsInBin):int((i+1)*numrepsInBin)])
        wvfm[i]=jumpsWafm
    return wvfm, numrepsInBin
        


###########################################
############likelihood
def likelihoodEdep(ssf, basePar,  baseT1,   roseq):
    
    energydeps= np.linspace(0,600,13)#np.array([0,25,50,100,150,])#
    likes=np.zeros(len(energydeps))
    for i in range(len(energydeps)):
        
        Edeposited=energydeps[i]
        Pjump, Expectedjumps, binwidth, time, ramseydecayrate, Prob_Par_meas,  P_oddg, P_odde, P_eveng, P_evene, ramseymeas_len, N=probCalc(ssf, basePar, Edeposited, baseT1,  roseq)
        
        #N=numrepsInBin
        
        waveform, numrepsInBin=Makewaveform(roseq,binwidth, ramseymeas_len)
        Prob_si_jps=np.zeros(len(waveform))
        #N=numrepsInBin
        log_Prob_si_jps=np.zeros(len(waveform))
        #print('len(Pjump)',len(Pjump))
        #print('len(waveform)',len(waveform))
        for j in range(len(waveform)):
            ###Prob that we see si jumps at ti
            si=waveform[j]
            Combi_N_si=math.factorial(int(N)) / (math.factorial(int(si)) * math.factorial(int(N-si)))
            Prob_si_jps[j]=Combi_N_si*(pow(Pjump[j], int(si)))*(pow((1-Pjump[j]),(int(N-si))))###Prob that we see si jumps at ti
            log_Prob_si_jps[j]=np.log(Prob_si_jps[j])
        likelihood=sum(log_Prob_si_jps)
        likes[i]=likelihood
    mostprobenergydep=energydeps[np.argmax(likes)]

    
    return mostprobenergydep





######################
#gates
###################
def rotateStateVector(stateVector_x,stateVector_y,stateVector_z, rotAngle, axis):
    #Rx(theta)V=[Vx, cos(theta)*Vy - sin(theta)*Vz, sin(theta)*Vy + cos(theta)*Vz]
    #Ry(theta)V=[cos(theta)*Vx + sin(theta)*Vz, Vy , -sin(theta)*Vx + cos(theta)*Vz]
    #Rz(theta)V=[Vx, cos(theta)*Vy - sin(theta)*Vz, sin(theta)*Vy + cos(theta)*Vz]
    
    
    #rz=[]
    
    if axis=="x" or axis=="X":
        fin_stateVector_x=stateVector_x
        fin_stateVector_y= math.cos(rotAngle)*stateVector_y - math.sin(rotAngle)*stateVector_z
        fin_stateVector_z= math.sin(rotAngle)*stateVector_y + math.cos(rotAngle)*stateVector_z
        
    elif axis=="y" or axis=="Y":
        fin_stateVector_x=math.cos(rotAngle)*stateVector_x +  math.sin(rotAngle)*stateVector_z
        fin_stateVector_y=stateVector_y
        fin_stateVector_z=-1*math.sin(rotAngle)*stateVector_x  + math.cos(rotAngle)*stateVector_z
    
    elif axis=="z" or axis=="Z":
        fin_stateVector_x=math.cos(rotAngle)*stateVector_x - math.sin(rotAngle)*stateVector_y
        fin_stateVector_y=math.sin(rotAngle)*stateVector_x + math.cos(rotAngle)*stateVector_y 
        fin_stateVector_z=stateVector_z
    else:
        print("Wrong name! Use 'X' or 'Y' or 'Z'")
    
    fstVec=[fin_stateVector_x, fin_stateVector_y, fin_stateVector_z]
    
    return fstVec

def pigate(x,y,z):
    #stateVector_x=x
    #stateVector_y=y
    #stateVector_z=z
    fV=rotateStateVector(x,y,z, math.pi, 'x')
    roValue=-1
    return fV, roValue

def x_2gate(x,y,z):
    #stateVector_x=x
    #stateVector_y=y
    #stateVector_z=z
    fV=rotateStateVector(x,y,z, math.pi/2, 'x')
    roValue=-1
    return fV, roValue

def y_2gate(x,y,z,par,phi_ng, delta_ph):
    fV__=rotateStateVector(x,y,z, par*(math.pi/2 + phi_ng +  delta_ph), 'z')
    x__=fV__[0]
    y__=fV__[1]
    z__=fV__[2]
    
    fV=rotateStateVector(x__,y__,z__, math.pi/2, 'y')
    roValue=-1
    return fV, roValue

def ro_measure(x,y,z,  P10, P01):
    theta=math.atan2(np.sqrt(x**2 + y**2),z)
    probg=(np.cos(theta/2))**2
    
    if random.uniform(0, 1)<probg: #grd state
        if random.uniform(0, 1)<P10: #mistake from noise
            roValue=1
        else:
            roValue=0 #correct
    else:
        if random.uniform(0, 1)<P01:
            roValue=0 #mistake
        else:
            roValue=1 #correct
    
    fV=[x,y,z]
    return fV, roValue
            

    

def Gate(gateName, x,y,z,par, phi_ng, delta_ph, P10, P01):
    prevState=[x,y,z]
    
    if gateName=="pi" or gateName=='PI':
        finstateVector, rdoValue=pigate(x,y,z)
    
    elif gateName=='x_2' or gateName=='X_2':
        finstateVector, rdoValue=x_2gate(x,y,z)
        
    elif gateName=='y_2' or gateName=='Y_2':
        finstateVector, rdoValue=y_2gate(x,y,z,par,phi_ng, delta_ph)
        
    elif gateName=='MEASURE' or gateName=='measure' or gateName=='Measure':
        finstateVector, rdoValue=ro_measure(x,y,z,   P10, P01)
        
    elif gateName=='idle' or gateName=='wait':
        finstateVector=[x,y,z]
        rdoValue=-1
    
    elif gateName=='active_reset':
        finstateVector=[0,0,1]
        rdoValue=0
    else:
        print(gateName,'is a Wrong gatename')
    return prevState,finstateVector, rdoValue





#SSF=0.95
#r=1 # P01/P10
#P01=r*((1-SSF)/(1+r))
#P10=(1-SSF)/(1+r)

def rates(edep,maxtime_ms, xqpbase, T1baseus, T2baseus):
    hbar_ev=6.5821*1e-16 #eVs
    hbar_js=1.054571817*1e-34 #Js
    Ej= hbar_ev*(2*np.pi)*7.0*1e9 #J
    Ec= hbar_ev*(2*np.pi)*300*1e6 #J
    planConst = 4.136 * 1e-15 #eVs 
    supcon_gap_Al=180*1e-6 #eV
    Ege=np.sqrt(8*Ej*Ec)-Ec #J 
    wq=Ege/hbar_ev
    fq=wq/(2*np.pi)
    #print('fq=',fq*1e-6,'MHz')
    adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
    bdf=((Ej/Ec/2)**(3/4))
    cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
    ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
    dwq=adf*bdf*cdf*ddf
    dfq=dwq/(2*np.pi)
    # print('dfq', dfq)
    # idlelen= (1/(4*dfq))*1e9 #- y_2len
    # print('idlelen',idlelen)
    
    heV_Hz=4.136*(1e-15) #eV/Hz
    h_bareV_Hz=heV_Hz/(2*np.pi)
    kb_ev=8.6173303*1e-5
    kbeV_K=8.617*(1e-5) #eV/K
    #Ej=6.25*1e9
    #Ec=250*1e6
    
      
    #print('Ej/Ec',Ej/Ec)
    
    
    gp0=43.5*1e9
    gp=43.5*1e9 
    EjeV=Ej#*(heV_Hz)
    EceV=Ec#*(heV_Hz)
    gpeV=supcon_gap_Al #gp*heV_Hz
    gp0eV=supcon_gap_Al #gp0*heV_Hz
    #print('gp0meV',gp0eV*1e3)
    #print('gpeV for 50nm Al thickness',gpeV)
    
    #r0=0.018/1e-9 #Hz
    #s0=1e-6/1e-9 #Hz
    ncp=4*1e24 #m^-3
    V_island= 1000*((1e-6)**3)  #2400*((1e-6)**3) #m^3
    phonon_to_qp_eff=0.57
    
    
    delta_f10=dfq
    
    
    T= 20*1e-3#20*1e-3  
    parRateCoeff_v1 = ((16*Ej) / (hbar_ev * np.pi))  *  np.sqrt((kb_ev*T)/(2*np.pi*supcon_gap_Al)) #16*(Ej*heV_Hz/(h_bareV_Hz*np.pi))*np.sqrt(kbeV_K*T/(2*np.pi*gpeV))
    decayRateCoeff = ((16*Ej)/(hbar_ev*np.pi)) * np.sqrt(Ec/(8*Ej)) * (supcon_gap_Al/(2*hbar_ev*wq))
    #base_xqp=1e-6
    basePar=0#0.1#17000 #Hz
    
    baseT1= T1baseus #100 #20#1e40 #us
    baseT1        = (baseT1*1e-6) * 1e9 #ns
    base_dec_rate =1/(baseT1) #1/ns base deco rate = ratefromqpbase +  ratefromext
    
    
    T2= (T2baseus*1e-6)*1e9 #(100*1e-6)*1e9 # (100*1e-6)*1e9#(5*1e-6)*1e9
    
    r0qp=  0.005  #0.0181  #0.018 
    s0qp=  1e-6  #1.1*1e-6 #1e-6 
    #gqp=(r0qp*xqp0*xqp0 + xqp0*s0qp)s
    val= (3*(1e4) )/ ((1e-6) * ((1e-6)**3))
    # supcon_gap_Alpaper=275*1e-6
    xqp0= xqpbase #1e-8  #base_dec_rate_fromxqp0 /  (np.sqrt((2*2*np.pi*fq*gpeV)/(np.pi*np.pi*h_bareV_Hz))*1e-9)
    # nqp=xqp0*2*supcon_gap_Alpaper*val
    # Ejpaper=20*1e-6
    # Ecpaper=225*1e-6
    # g= (4*Ejpaper)/supcon_gap_Alpaper
    # print('g=',g)
    # dE= np.abs(Ecpaper - (Ejpaper/2))
    
    # ratefromPaper =  ((g*2*np.pi)/(planConst*4*np.pi))* (nqp/val)  * np.sqrt( dE/(2*supcon_gap_Alpaper) )  
    # print('xqp0=',xqp0, 'nqp per um=xqp0*ncp=', nqp/((1e6)**3), ' ratefromPaper=',  ratefromPaper, 'Hz')

    
    base_dec_rate_fromxqp0    =  xqp0 * decayRateCoeff #(np.sqrt((2*2*np.pi*fq*gpeV)/(np.pi*np.pi*h_bareV_Hz))*1e-9)
    #base_Parity_rate = (basePar) * 1e-9 #1/ns #500 #Hz base par rate
    base_Parity_rate = base_dec_rate_fromxqp0 + np.exp((-1*hbar_ev*wq)/(kb_ev*T))*base_dec_rate_fromxqp0 + parRateCoeff_v1 *1e-9*xqp0 
    # print('xqp0=',xqp0, 'nqp per um=xqp0*ncp', (xqp0*ncp)/((1e6)**3), 'base_Parity_rate from nostate change + dec + exc = ', base_Parity_rate * 1e9, 'Hz')
    # print('xqp0=',xqp0, 'nqp per um=xqp0*ncp', (xqp0*ncp)/((1e6)**3), 'nostate change = ', coef*1e-9*xqp0 * 1e9, 'Hz')
    # print('xqp0=',xqp0, 'nqp per um=xqp0*ncp', (xqp0*ncp)/((1e6)**3), 'dec = ', base_dec_rate_fromxqp0 * 1e9, 'Hz')
    # print('xqp0=',xqp0, 'nqp per um=xqp0*ncp', (xqp0*ncp)/((1e6)**3), 'exc = ', np.exp(-1*h_bareV_Hz*wq/(kbeV_K*T))*base_dec_rate_fromxqp0 * 1e9, 'Hz')
    nqp0=xqp0*ncp
    
    EeV=edep*1e-3 #eV
    eps=  0.6 #0.57
    E_joules=EeV*(1.602*1e-19) #J 
    timechunks=1e6 #ns Contains 20 pulse sequences each of which is about 3.1us.
    numRatepts=0.1
    maxtime=maxtime_ms*1e-3*1e9   #numRatepts*timechunks #ns
    num_time_pts= int(maxtime/50)  #int(maxtime/50) #50ns timestep
    

    time=np.linspace(0,maxtime,num_time_pts) #ns
    delta_t=time[1] - time[0]
    Esss = np.zeros(len(time))
    Esss[0]=EeV
    
    
    
    
    rate_P=np.zeros(len(time)) 
    rate_D=np.zeros(len(time))
    rate_Exc=np.zeros(len(time))
    rate_dph=np.zeros(len(time))
    rate_Dqp=np.zeros(len(time))
    rate_Exc_qp=np.zeros(len(time))
    rate_P_no_State_change=np.zeros(len(time))
    
    xqps = np.zeros(len(time))
    
    
    delta_xqps=np.zeros(len(time))
    
    
    
        #print('idlewait',idlewait, 'Pargatetimes',Pargatetimes)
    
    
    
    rate_dph=np.zeros(len(time))
    
    RateP_t=[] 
    #print('Rate
    RateT1_t=[]
    
    fullprate=np.zeros(len(time))
    full_rateDecqp=np.zeros(len(time))

    for i in range((len(time))):
        
        
        # xqp0= base_Parity_rate / (2*coef*1e-9 + (np.sqrt((2*2*np.pi*fq*gpeV)/(np.pi*np.pi*h_bareV_Hz))*1e-9) +  ((np.sqrt((2*2*np.pi*fq*gpeV)/(np.pi*np.pi*h_bareV_Hz))*1e-9) * np.exp(-1*h_bareV_Hz*wq/(kbeV_K*T))))
        # nqp0=xqp0*ncp
        # if i==0:
            
        #     print('nqp0 in 1/m^3 = ', nqp0)
        #     print('nqp0 in 1/um^3 = ', nqp0/(1e18))
        #     print('from dir calc 1/um^3 = e*E/V*gap = ', (eps*Esss[0]/(V_island*gpeV))/1e18)
        gqp= (r0qp*xqp0*xqp0 + xqp0*s0qp)
        if i==0:
            #xinduced=Edeposited*phonon_to_qp_eff / (ncp*V_island*supcon_gap_Al)
            meanCPsbroken=eps*Esss[0]/(2*gpeV)
            Fano=0.2
            sdCPsbroken= np.sqrt(Fano* meanCPsbroken)
            Numqpgenerated= 2 * np.random.normal(loc=meanCPsbroken , scale=sdCPsbroken) 
            xqps[i]= Numqpgenerated/(ncp*V_island) + xqp0
            delta_xqps[i]=xqps[i]-xqp0
           
    
        else:
            xqps[i]=xqps[i-1]
                
            loss=   (r0qp*(xqps[i-1]**2) + s0qp*xqps[i-1])*delta_t #np.random.poisson(((r0qp*(xqps[i-1]**2) + s0qp*xqps[i-1])*delta_t))
            gain=   gqp*delta_t #np.random.poisson(gqp*delta_t)
            #print('- loss + gain',- loss + gain)
            xqps[i] = xqps[i] - loss + gain + Esss[i]/(ncp*V_island*gpeV)
            #print('Esss[i]/(gp*volume*ncp)',Esss[i]/(gpeV*volume*ncp))
                
            #xqps[i] = np.abs(xqps[i])
            #delta_xqps[i]=np.abs(xqps[i]-xqp0)
            if xqps[i]-xqp0 <0:
                #print('!!!Neg deltaxqp!!!')fdcvrxccvdvdcfcdvxrsvdrxrxcvd
                delta_xqps[i]=0
            else:
                delta_xqps[i]=xqps[i]-xqp0
    
        rate_Dqp[i]              = delta_xqps[i] * decayRateCoeff * 1e-9   #np.sqrt((2*wq*gpeV)/(np.pi*np.pi*h_bareV_Hz))*1e-9
        rate_Dqp_base            = xqp0 * decayRateCoeff * 1e-9    #np.sqrt((2*wq*gpeV)/(np.pi*np.pi*h_bareV_Hz))*1e-9
        full_rateDecqp[i]        = rate_Dqp_base  +  rate_Dqp[i]
        rate_D[i]                = base_dec_rate  +  rate_Dqp_base + rate_Dqp[i]
        rate_Exc_nonqp           = base_dec_rate  *  np.exp((-1*hbar_ev*wq)/(kb_ev*T))  #0.0001* 1e-9 
        rate_Exc_qpbase          = rate_Dqp_base  *  np.exp((-1*hbar_ev*wq)/(kb_ev*T))  
        rate_Exc_qp[i]           =  rate_Dqp[i]   *  np.exp((-1*hbar_ev*wq)/(kb_ev*T)) + rate_Exc_qpbase
        rate_Exc[i]              = rate_Exc_nonqp + rate_Exc_qp[i]
        rate_P_no_State_change[i]= ( xqp0  +  delta_xqps[i]) * parRateCoeff_v1 *1e-9 
        rate_dph[i]= 1/T2 - 0.5*base_dec_rate #1e-40# #1.36*1e-9 #base_dephasing_rate
        
        #rate_P[i]=rate_P_no_State_change[i] 
        fullprate[i]=(np.sqrt((2*wq*gpeV)/(np.pi*np.pi*h_bareV_Hz))*1e-9  +  np.exp((-1*hbar_ev*wq)/(kb_ev*T))*np.sqrt((2*(wq)*gpeV)/(np.pi*np.pi*h_bareV_Hz))*1e-9    +   2*parRateCoeff_v1*1e-9)*delta_xqps[i] + base_Parity_rate #+ rate_P[i] + rate_Dqp + rate_Dqp_base + (rate_Dqp 
    
    # print('base_dec_rate + rate_Dqp_base  +  rate_Dqp', (base_dec_rate + rate_Dqp_base  +  rate_Dqp)*1e9 )
    # print('rate_P_no_State_change', rate_P_no_State_change*1e9, 'Hz')
    # print('rate_Exc_nonqp  + rate_Exc_qp', (rate_Exc_nonqp +  rate_Exc_qp)*1e9 )
    np.savez(f'{edep}meVrates', xqps=xqps, rate_D=rate_D, rate_Exc=rate_Exc,  fullprate=fullprate, rate_dph=rate_dph, rate_P_no_State_change=rate_P_no_State_change, rate_Exc_qp=rate_Exc_qp, rate_Exc_nonqp=rate_Exc_nonqp, rate_Dqp_base=rate_Dqp_base, rate_Dqp=rate_Dqp, base_dec_rate=base_dec_rate, full_rateDecqp=full_rateDecqp, xqp0=xqp0)
    return fullprate, rate_D, rate_Dqp, rate_Dqp_base, full_rateDecqp, rate_P_no_State_change, xqps, gqp



# def NonQPDephGeneration(para):
#     ssf=para[0]
#     edep=para[1]
#     maxtime_ms=para[2]
#     hbar_ev=6.5821*1e-16 #eVs
#     hbar_js=1.054571817*1e-34 #Js
#     Ej=hbar_ev*(2*np.pi)*9.945*1e9 #J
#     Ec=hbar_ev*(2*np.pi)*390*1e6 #J
#     supcon_gap_Al=180*1e-6 #eV
#     Ege=np.sqrt(8*Ej*Ec)-Ec #J 
#     wq=Ege/hbar_ev
#     fq=wq/(2*np.pi)
#     #print('fq=',fq*1e-6,'MHz')
#     adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
#     bdf=((Ej/Ec/2)**(3/4))
#     cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
#     ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
#     dwq=adf*bdf*cdf*ddf
#     dfq=dwq/(2*np.pi)
#     delta_f10=dfq
    
#     timechunks=1e6 #ns Contains 20 pulse sequences each of which is about 3.1us.
#     numRatepts=0.1
#     maxtime=maxtime_ms*1e-3*1e9   #numRatepts*timechunks #ns
#     num_time_pts=int(maxtime/50) #50ns timestep
#     #E_depo_timestream=np.zeros(num_pts) #ns Energy depo into the island
#     #E_depo_timestream[0]=EeV #eV
    
#     time=np.linspace(0,maxtime,num_time_pts) #ns
#     ratefile=np.load(f'{edep}meVrates.npz')
#     xqps=ratefile['xqps']
#     rate_D=ratefile['rate_D']
#     #print('rate_D',rate_D)
#     rate_Exc=ratefile['rate_Exc']
    
#     fullprate=ratefile['fullprate']
#     rate_dph=ratefile['rate_dph']
#     rate_P_no_State_change=ratefile['rate_P_no_State_change']
#     rate_Exc_qp=ratefile['rate_Exc_qp']
#     rate_Exc_nonqp=ratefile['rate_Exc_nonqp']
#     rate_Dqp_base=ratefile['rate_Dqp_base']
#     rate_Dqp=ratefile['rate_Dqp']
#     base_dec_rate=ratefile['base_dec_rate']
    
#     delta_t=time[1] - time[0]
#     #print('delta_t',delta_t)
#     maxPhaseerror=2*delta_t*1e-9*delta_f10
    
#     #print('parity=',parityy, 'delta_t=',delta_t, 'ns')
    
#     Pargates=["x_2","idle","y_2","measure","active_reset"]
    
#     idlelen= (1/(4*delta_f10))*1e9 #- y_2len
#     #print('idlelen',idlelen)
#     #print('idlelen s',idlelen*1e-9)
    
    
#     non_QPdeph_event   =np.zeros((len(time)))
#     # paritydecay =np.zeros((len(time)))
#     # parityexc   =np.zeros((len(time)))
#     # parity      =np.zeros((len(time)))
#     # tun_rando=random.uniform(0, 1)
#     for i in range((len(time))):
#         #print(time[i])
#         #print('whichGate',whichGate)
#         ####################################
#         if i==0:
#             t_nonQPdeph=-1*np.log(1-random.uniform(0, 1))*(1/rate_dph[i])
#             NTnQPdeph=t_nonQPdeph/(1/rate_dph[i])
            
         
#         ############ Non QP Dephasing it's time ##########################
#         if i!=0:
            
#             if NTnQPdeph*(1/rate_dph[i])<delta_t:
#                 #### When it's time to switch parity, restart counntdown and do it ###########
#                 #print('partity switch!!!!!!!!  ti=',time[i])
#                 t_nonQPdeph=(-1)*np.log(1-random.uniform(0, 1))*(1/rate_dph[i])
#                 NTnQPdeph=t_nonQPdeph/(1/rate_dph[i])
                
#                 non_QPdeph_event[i]=1   
                
#             else:
#                 NTnQPdeph= NTnQPdeph - delta_t/(1/rate_dph[i])
                

#     return time, non_QPdeph_event





# def TunnelingGeneration(para):
#     ssf=para[0]
#     edep=para[1]
#     maxtime_ms=para[2]
#     hbar_ev=6.5821*1e-16 #eVs
#     hbar_js=1.054571817*1e-34 #Js
#     Ej=hbar_ev*(2*np.pi)*9.945*1e9 #J
#     Ec=hbar_ev*(2*np.pi)*390*1e6 #J
#     supcon_gap_Al=180*1e-6 #eV
#     Ege=np.sqrt(8*Ej*Ec)-Ec #J 
#     wq=Ege/hbar_ev
#     fq=wq/(2*np.pi)
#     #print('fq=',fq*1e-6,'MHz')
#     adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
#     bdf=((Ej/Ec/2)**(3/4))
#     cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
#     ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
#     dwq=adf*bdf*cdf*ddf
#     dfq=dwq/(2*np.pi)
#     delta_f10=dfq
    
#     timechunks=1e6 #ns Contains 20 pulse sequences each of which is about 3.1us.
#     numRatepts=0.1
#     maxtime=maxtime_ms*1e-3*1e9   #numRatepts*timechunks #ns
#     num_time_pts=int(maxtime/50) #50ns timestep
#     #E_depo_timestream=np.zeros(num_pts) #ns Energy depo into the island
#     #E_depo_timestream[0]=EeV #eV
    
#     time=np.linspace(0,maxtime,num_time_pts) #ns
#     ratefile=np.load(f'{edep}meVrates.npz')
#     xqps=ratefile['xqps']
#     rate_D=ratefile['rate_D']
#     rate_Exc=ratefile['rate_Exc']
#     rate_P=ratefile['rate_P']
#     fullprate=ratefile['fullprate']
#     rate_dph=ratefile['rate_dph']
#     rate_P_no_State_change=ratefile['rate_P_no_State_change']
#     rate_Exc_qp=ratefile['rate_Exc_qp']
#     rate_Exc_nonqp=ratefile['rate_Exc_nonqp']
#     rate_Dqp_base=ratefile['rate_Dqp_base']
#     rate_Dqp=ratefile['rate_Dqp']
#     base_dec_rate=ratefile['base_dec_rate']
#     full_rateDecqp=ratefile['full_rateDecqp']
#     delta_t=time[1] - time[0]
#     #print('delta_t',delta_t)
#     maxPhaseerror=2*delta_t*1e-9*delta_f10
    
#     #print('parity=',parityy, 'delta_t=',delta_t, 'ns')
    
#     Pargates=["x_2","idle","y_2","measure","active_reset"]
    
#     idlelen= (1/(4*delta_f10))*1e9 #- y_2len
#     #print('idlelen',idlelen)
#     #print('idlelen s',idlelen*1e-9)
    
    
#     tun_event   =np.zeros((len(time)))
#     paritynone =np.zeros((len(time)))
#     paritydecay =np.zeros((len(time)))
#     parityexc   =np.zeros((len(time)))
#     parity      =np.zeros((len(time)))
#     tun_rando=random.uniform(0, 1)
#     for i in range((len(time))):
#         #print(time[i])
#         #print('whichGate',whichGate)
#         ####################################
#         if i==0:
#             t_event_par=-1*np.log(1-random.uniform(0, 1))*(1/fullprate[i])
#             NTP=t_event_par/(1/fullprate[i])
#             if random.uniform(0, 1)<0.5:  #randomly start with even or odd parity with 50-50 chance
#                 parity[i]=-1  #Set parity to 1 at t0
                
#             else:
#                 parity[i]=1
            
          
            
          
#         ############ Tunneling or Switching parity when it's time ##########################
#         if i!=0:
            
#             if NTP*(1/fullprate[i])<delta_t:
#                 #### When it's time to switch parity, restart counntdown and do it ###########
#                 #print('partity switch!!!!!!!!  ti=',time[i])
#                 t_event_par=(-1)*np.log(1-random.uniform(0, 1))*(1/fullprate[i])
#                 NTP=t_event_par/(1/fullprate[i])
                
#                 parity[i]=parity[i-1]*(-1)
#                 tun_event[i]=1

#                 noChangeParratio=rate_P[i]/fullprate[i]
#                 decParratio=full_rateDecqp[i]/fullprate[i]
#                 excParratio=rate_Exc_qp[i]/fullprate[i]
#                 # print('noChangeParratio',noChangeParratio)
#                 # print('decParratio',decParratio)
#                 # print('excParratio',excParratio)
#                 Par_rando=random.uniform(0, 1)
#                 if Par_rando<noChangeParratio:
#                     #print('partity switch-No change!!!!!!!!')
#                     pass
#                 elif Par_rando>noChangeParratio and Par_rando < (noChangeParratio + decParratio):
#                     paritydecay[i]=1
                    

#                 elif Par_rando>(noChangeParratio + decParratio) and Par_rando<1:
#                     parityexc[i]=1
                    
                
#             else:
#                 NTP= NTP - delta_t/(1/fullprate[i])
#                 parity[i]=parity[i-1]

#     return time, tun_event, paritynone   ,paritydecay ,parityexc   ,parity,       fullprate

def countjumps(output):
    
    flips=0
    laststate=output[0]
    for i in range(len(output)):
        if output[i]!=laststate:
            flips=flips+1
        laststate=output[i]
    return flips

def Make_wvfm_1s(seq, seqT, binwidth, tstep):
    fulltime=seqT[-1]-seqT[0]
    numbins=int(fulltime/binwidth)
    numPerbin=len(seq)/numbins
    wvfm=np.zeros((numbins))

    for i in range(numbins):
        wvfm[i]=sum(seq[int(i*numPerbin):int((i+1)*numPerbin)])
    
    return wvfm, numPerbin, numbins

def Make_wvfm(seq, seqT, binwidth, tstep):
    fulltime=seqT[-1]-seqT[0]
    numbins=int(fulltime/binwidth)
    numPerbin=len(seq)/numbins
    wvfm=np.zeros((numbins))

    for i in range(numbins):
        wvfm[i] = countjumps(seq[int(i*numPerbin):int((i+1)*numPerbin)])
    
    return wvfm, numPerbin, numbins



def sims(para):#(edep):edep,maxtime_ms
    #print('starting sims')
    start_time=tm.time()
    SSF=para[0]
    edep=para[1]
    maxtime_ms=para[2]

    hbar_ev=6.5821*1e-16 #eVs
    hbar_js=1.054571817*1e-34 #Js
    Ej=hbar_ev*(2*np.pi)*7.0*1e9 #J
    Ec=hbar_ev*(2*np.pi)*300*1e6 #J
    
    supcon_gap_Al=180*1e-6 #eV
    Ege=np.sqrt(8*Ej*Ec)-Ec #J 
    wq=Ege/hbar_ev
    fq=wq/(2*np.pi)
    #print('fq=',fq*1e-6,'MHz')
    adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
    bdf=((Ej/Ec/2)**(3/4))
    cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
    ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
    dwq=adf*bdf*cdf*ddf
    dfq=dwq/(2*np.pi)
    # print('dfq', dfq)
    
    
    heV_Hz=4.136*(1e-15) #eV/Hz
    h_bareV_Hz=heV_Hz/(2*np.pi)
    kbeV_K=8.617*(1e-5) #eV/K
    
    delta_ng=0
    ng=0
    delta_f10=dfq*np.cos(2*np.pi*(ng + delta_ng))
    #print('delta_f10 MHz',delta_f10*1e-6)
    timechunks=1e6 #ns Contains 20 pulse sequences each of which is about 3.1us.
    numRatepts=0.1
    maxtime=maxtime_ms*1e-3*1e9   #numRatepts*timechunks #ns
    num_time_pts= int(maxtime/50) #int(maxtime/50) #50ns timestep
    #E_depo_timestream=np.zeros(num_pts) #ns Energy depo into the island
    #E_depo_timestream[0]=EeV #eV
    
    time=np.linspace(0,maxtime,num_time_pts) #ns
    #time, parity, fullprate = TunnelingGeneration(para)
    ratefile=np.load(f'{edep}meVrates.npz')
    xqps=ratefile['xqps']
    rate_D=ratefile['rate_D']
    rate_Exc=ratefile['rate_Exc']
    
    fullprate=ratefile['fullprate']
    #print('fullprate',fullprate)
    rate_dph=ratefile['rate_dph']
    rate_P_no_State_change=ratefile['rate_P_no_State_change']
    # print('rate_P_no_State_change',rate_P_no_State_change)
    # print('rate_P_no_State_change in Hz',rate_P_no_State_change*1e9)
    rate_Exc_qp=ratefile['rate_Exc_qp']
    rate_Exc_nonqp=ratefile['rate_Exc_nonqp']
    fullExc_QP=rate_Exc_qp + rate_Exc_nonqp
    rate_Dqp_base=ratefile['rate_Dqp_base']
    rate_Dqp=ratefile['rate_Dqp']
    # print('rate_Dqp in Hz',rate_Dqp*1e9)
    fullDec_QP=rate_Dqp_base + rate_Dqp
    # print('fullDec_QP Max in MHz',fullDec_QP[0]*1e3)
    #print('fullDec_QP',fullDec_QP)
    base_dec_rate=ratefile['base_dec_rate']
    # print('base_dec_rate ',base_dec_rate)
    xqp0 = ratefile['xqp0']
    charge_jump_rate= 1.35*1e-3*1e-9 # Hz
    #print('base_dec_rate',base_dec_rate)

    #time,non_QPdeph_event = NonQPDephGeneration(para)
    # time,non_QPExc_event = NonQPExcGeneration(para)
    # time,non_QPDecay_event = NonQPDecayGeneration(para)
    #time, tun_event, paritynone ,paritydecay ,parityexc , parity, fullprate = TunnelingGeneration(para)
    
    delta_t=time[1] - time[0]
    #print('delta_t',delta_t)
    maxPhaseerror=2*delta_t*1e-9*delta_f10
    #parityy=-1
    #print('parity=',parityy, 'delta_t=',delta_t, 'ns')
    
    Pargates=["x_2","idle","y_2","measure","active_reset"]
    
    idlelen= (1/(4*delta_f10))*1e9 #- y_2len
    # print('idlelen',idlelen)
    measurelen= 200 #2000 #ns
    active_resetlen=500#2000#(2*1e-6)*1e9 #ns
    x_2len=100 #ns
    y_2len=100#idlelen#40 #ns
    idlewait=  idlelen #int(idlelen/delta_t)*idlelen #-y_2len
    
    # print('idlewait',idlewait)
    # print('delta_f10',delta_f10)
    Pargatetimes=[x_2len, x_2len+idlewait, x_2len+idlewait+y_2len, x_2len+idlewait+y_2len+measurelen, x_2len+idlewait+y_2len+measurelen+active_resetlen]
    X2time=x_2len
    Y2time=x_2len+idlewait+y_2len
    idtime=x_2len+idlewait
    Actime=x_2len+idlewait+y_2len+measurelen+active_resetlen
    ParSeqlen=x_2len+idlelen+y_2len+measurelen+active_resetlen
    #print('ParSeqlen',ParSeqlen)
    numSeq=int(maxtime/ParSeqlen)
    
    whichSeq=0
    whichGate=0
    #print('whichSeq',whichSeq)
    start_time=tm.time()
    
    readout=np.zeros(len(time))
    readoutTime=np.zeros(len(time))
    pigate1=np.zeros(len(time))
    x_2gate1    =np.zeros(len(time))
    y_2gate1    =np.zeros(len(time))
    measureGate1=np.zeros(len(time))
    activereset1=np.zeros(len(time))
    ExcStateProb=np.zeros(len(time))
    phi=0
    phis=np.zeros(len(time))
    parityAtRO=[]
    x_2gate1=np.zeros(len(time))
    y_2gate1=np.zeros(len(time))
    measureGate1=np.zeros(len(time))
    activereset1=np.zeros(len(time))
    dephasing=np.zeros(len(time))
    decoherence=[]#np.zeros(len(time))
    decoherenceTime=[]
    excitation=[]#np.zeros(len(time))
    excitationTime=[]
    parityflip=[]#np.zeros(len(time))
    parityflipTime=[]
    startgt=0
    start=0
    parDecReadout=[]
    parDecReadout_correct=[]
    parDecReadoutTime=[]
    parDecExcProb=[]
    fine=[]
    preve=[]
    gt=[]
    rateT1atRead=[]
    rateTpatRead=[]
    rateT1excatRead=[]
    rateTphiRead=[]
    thetaTest=[]#np.zeros(len(time))
    parIdleVec=[]
    stateVector=np.zeros((len(time),4))
    readout=np.zeros((len(time)))
    dec           =np.zeros((len(time)))
    idt           =np.zeros((len(time)))
    dec_idt       =np.zeros((len(time)))
    exc_idt       =np.zeros((len(time)))
    roevent       =np.zeros((len(time)))
    dec_ro        =np.zeros((len(time)))
    exc_ro        =np.zeros((len(time)))
    ssf_ro_err    =np.zeros((len(time)))
    dec_idt_wrg_ro=np.zeros((len(time)))
    exc_idt_wrg_ro=np.zeros((len(time)))
    dec_ro_wrg_ro =np.zeros((len(time)))
    exc_ro_wrg_ro =np.zeros((len(time)))
    ev_0          =np.zeros((len(time)))
    od_0          =np.zeros((len(time)))
    ev_1          =np.zeros((len(time)))
    od_1          =np.zeros((len(time)))
    ev_g          =np.zeros((len(time)))
    od_g          =np.zeros((len(time)))
    ev_e          =np.zeros((len(time)))
    od_e          =np.zeros((len(time)))
    parity    =np.zeros((len(time)))
    ro=np.zeros((int(5*1e-3/(4.4*1e-6))+1))
    par = random.choice([1, -1])
    stepIsLiveDec=False
    stepIsLiveQPDec=False
    stepIsLivenonPar=False
    stepIsLiveExc=False
    stepIsLiveQPexc=False
    n_par       =np.zeros((len(time)))
    n_par_narrow=np.zeros((len(time)))
    n_nonqpdec  =np.zeros((len(time)))
    n_nonqpexc  =np.zeros((len(time)))
    n_qpdec     =np.zeros((len(time)))
    n_qpexc     =np.zeros((len(time)))
    n_dec       =np.zeros((len(time)))
    n_exc       =np.zeros((len(time)))
    n_ssferr    = 0
    n_evg       =np.zeros((len(time)))
    n_eve       =np.zeros((len(time)))
    n_ev0       =np.zeros((len(time)))
    n_ev1       =np.zeros((len(time)))
    n_odg       =np.zeros((len(time)))

    n_ev       =np.zeros((len(time)))
    n_od       =np.zeros((len(time)))
    
    n_ode       =np.zeros((len(time)))
    n_od0       =np.zeros((len(time)))
    n_od1       =np.zeros((len(time)))
    n_flip      =np.zeros((len(time)))
    n_ro        = 0 #np.zeros((10))
    pos=0
    neg=0
    nodd_nonqpdec=0
    neven_nonqpdec=0
    nonqpdec=np.zeros((len(time)))
    qpdec=np.zeros((len(time)))
    phi_ng=0
    ngs = np.zeros((len(time)))
    n_ph = np.zeros((len(time)))
    delta_ph=0
    decNonQPtimes=[]
    decQPtimes=[]
    tpars=[]
    nerr=0
    nmeas=0
    roerr=np.zeros((3000))
    #print('starting par',par)
    #tun_rando=random.uniform(0, 1)
    for i in range((len(time))):
        
        if par==1:
            pos=pos+1
        else:
            neg=neg+1
        #print(time[i])
        #print('whichGate',whichGate)
        ####################################
        if par==1:
            #print('even')
            n_ev[i]=1
        elif par==-1:
            #print('odd')
            n_od[i]=1
        else:
            print('par not -1 or 1')
        if i==0:
            stateVector[i][0]=0
            stateVector[i][1]=0
            stateVector[i][2]=1
            stateVector[i][3]=par
            parity[i]=par
            #print(f'i={i}','stateVector[i][3]',stateVector[i][3])
            
            rotime=Pargatetimes[3]
            
            #stepIsLiveDec=False
            
        elif i>0:
            
            stateVector[i][0]=stateVector[i-1][0]
            stateVector[i][1]=stateVector[i-1][1]
            stateVector[i][2]=stateVector[i-1][2]
            stateVector[i][3]=par          
            parity[i]=par         #stateVector[i-1][3]
            
        
        ############################################################
        
        ############################################################
        if whichSeq<=numSeq:
            gateApplyTime=whichSeq*ParSeqlen + Pargatetimes[whichGate]
            # if i>98000:
            #     print(whichGate, time[i])
            
            #print('gateApplyTime',gateApplyTime, Pargates[whichGate])
        if whichGate==3:
            rotime=gateApplyTime
        if whichGate==0:
            X2time=gateApplyTime
        if whichGate==1:
            idtime=gateApplyTime
        if whichGate==2:
            Y2time=gateApplyTime
        if whichGate==4:
            Actime=gateApplyTime
            delta_ph=0
    
    
        x=stateVector[i][0]
        y=stateVector[i][1] 
        z=stateVector[i][2]
        if np.abs(gateApplyTime - time[i])<delta_t:
            
            prevSt, finVector, readoutValue = Gate(Pargates[whichGate], x,y,z,stateVector[i][3], phi_ng, delta_ph, P10, P01)
            #print('prevSt',prevSt,'finVector',finVector)
            stateVector[i][0]=finVector[0]
            stateVector[i][1]=finVector[1]
            stateVector[i][2]=finVector[2]
            finSt=finVector
    
            whichGate=whichGate+1
            if whichGate==len(Pargates):
                whichGate=0
                whichSeq=whichSeq+1

            # if whichGate==1:
            #     idt[i]=1

            if whichGate==2: #parity for this measurement round.
                ParForMeas=stateVector[i][3]
            
            if whichGate==3:
                roevent[i]=1
        thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])
        ExcStateProb[i]=(sin(thetaNow/2))**2
        ########################################################################################################
        ######################## Dephasing from Charge jumps and other influences ########################
        if i==0:
            t_event_charge=(-1)*np.log(1 - random.uniform(0, 1))*(1/charge_jump_rate )
            NTcharge = t_event_charge/(1/charge_jump_rate)
        if i!=0:
            if NTcharge*(1/charge_jump_rate)<delta_t:
                delta_ng = delta_ng + random.uniform(-0.5, 0.5)
                t_event_charge=(-1)*np.log(1 - random.uniform(0, 1))*(1/charge_jump_rate)
                NTcharge = t_event_charge/(1/charge_jump_rate)
                delta_f10=dfq*np.cos(2*np.pi*(ng + delta_ng))
                phi_ng=2*np.pi*delta_f10*idlelen*1e9
                ngs[i]=ng
                # print('charge',charge)
                
            else:
                NTcharge = NTcharge - delta_t/(1/charge_jump_rate)

        if i==0:
            t_event_dph=(-1)*np.log(1 - random.uniform(0, 1))*(1/rate_dph[i] )
            NTdph = t_event_dph/(1/rate_dph[i])
        if i!=0:
            if NTdph*(1/rate_dph[i])<delta_t:
                delta_ph =   rate_dph[i]*2*np.pi * idlelen * random.uniform(-0.5, 0.5)
                # print('delta_ph = ',delta_ph )
                # print('rate_dph[i] = ',rate_dph[i] )
                # print('base_dec_rate = ',base_dec_rate )
                t_event_dph=(-1)*np.log(1 - random.uniform(0, 1))*(1/rate_dph[i])
                NTdph = t_event_dph/(1/rate_dph[i])
                # print('dph',dph)
                n_ph[i]=1
                
            else:
                NTdph = NTdph - delta_t/(1/rate_dph[i])
                delta_ph =  0
        #
        ####################Do Decay Stuff##############################################################################
        ####################Do Decay Stuff##############################################################################
        ####################Check if the qubit is not in the ground state##########################################################
        if np.abs(time[i] - X2time)<delta_t and time[i] > X2time and ExcStateProb[i]>0.01:  # and (whichGate==1 or whichGate==3):
           # stepIsLiveDec=False
            t_event_decNonQP=(-1)*np.log(1-random.uniform(0, 1))*(1/base_dec_rate)
            
            # if t_event_decNonQP< idtime-time[i] < 1000: #
            #     print('t_event_decNonQP for idt',t_event_decNonQP, 'time till gate-end=', idtime-time[i])
            #     print('X2time at Tau calc',X2time)
            NT1NonQP=t_event_decNonQP/(1/base_dec_rate)
            
            t_event_decQP=(-1)*np.log(1-random.uniform(0, 1))*(1/fullDec_QP[i])
            # if t_event_decQP< idtime-time[i]:
            #     print('t_event_decQP for idt',t_event_decQP, 'time till gate-end=', idtime-time[i])
                
            NT1QP=t_event_decQP/(1/fullDec_QP[i])
            #print('NT1QP=',NT1QP)
            stepIsLiveDec=True
            
            
            # print('NT1NonQP',NT1NonQP)
            # print('NT1QP',NT1QP)
        if np.abs(time[i] - Y2time)<delta_t and time[i] > Y2time and ExcStateProb[i]>0.01:  # and (whichGate==1 or whichGate==3):
            
           # stepIsLiveDec=False
            t_event_decNonQP=-1*np.log(1-random.uniform(0, 1))*(1/base_dec_rate)
            # if t_event_decNonQP< 1000: #rotime-time[i]:
            #     print('t_event_decNonQP for rot',t_event_decNonQP, 'time till gate-end=', rotime-time[i])
            #     print('Y2time at Tau calc',Y2time)
            NT1NonQP=t_event_decNonQP/(1/base_dec_rate)
            
            t_event_decQP=-1*np.log(1-random.uniform(0, 1))*(1/fullDec_QP[i])
            # if t_event_decQP< rotime-time[i]:
            #     print('t_event_decQP for rot',t_event_decQP, rotime-time[i])
            NT1QP=t_event_decQP/(1/fullDec_QP[i])
            #print('NT1QP=',NT1QP)
            stepIsLiveDec=True
            
        if whichGate==0 or whichGate==2 or whichGate==4:
            stepIsLiveDec=False

                
        if stepIsLiveDec==True:
            if whichGate==1:
                if NT1NonQP*(1/base_dec_rate)>delta_t:
                    NT1NonQP=NT1NonQP  - delta_t/(1/base_dec_rate)
                                            
                else:
                    if ExcStateProb[i]>0.01: #excited
                        stateVector[i][0]=0
                        stateVector[i][1]=0
                        stateVector[i][2]=1
                        stepIsLiveDec=False
                        dec[i]= 1
                        nonqpdec[i]= 1
                        # nonqpdec=nonqpdec+1
                        
                        
                        # if stateVector[i][3]==1:
                        #     nodd_nonqpdec=nodd_nonqpdec+1
                        # elif stateVector[i][3]==-1:
                        #     neven_nonqpdec=neven_nonqpdec+1
                        
                        
                        # decNonQPtimes.append(t_event_decNonQP) #(time[i] - X2time)

                if NT1QP*(1/fullDec_QP[i])>delta_t:
                    NT1QP=NT1QP  - delta_t/(1/fullDec_QP[i])
                    
        
                else:
                    if ExcStateProb[i]>0.01: #excited
                        stateVector[i][0]=0
                        stateVector[i][1]=0
                        stateVector[i][2]=1
                        par=(-1)*par
                        n_par[i]= 1
                        #print('from', -par, 'to',  par, 'at t=', time[i]*1e-3, 'us')
                        stateVector[i][3]=par               #stateVector[i][3]
                        parity[i]=par
                        dec[i]= 1
                        qpdec[i]= 1
                        # print('qp dec')
                        stepIsLiveDec=False
                        # if whichGate==1:
                        # dec_idt[i]=1
                        # decQPtimes.append(t_event_decQP)   #(time[i] - X2time)
                                               

            if whichGate==3:
                if NT1NonQP*(1/base_dec_rate)>delta_t:
                    NT1NonQP=NT1NonQP  - delta_t/(1/base_dec_rate)
                                            
                else:
                    if ExcStateProb[i]>0.01: #excited
                        stateVector[i][0]=0
                        stateVector[i][1]=0
                        stateVector[i][2]=1
                        stepIsLiveDec=False
                        dec[i]= 1
                        nonqpdec[i]= 1
                        # nonqpdec=nonqpdec+1
                        
                        
                        # if stateVector[i][3]==1:
                        #     nodd_nonqpdec=nodd_nonqpdec+1
                        # elif stateVector[i][3]==-1:
                        #     neven_nonqpdec=neven_nonqpdec+1
                        
                        
                        # decNonQPtimes.append(t_event_decNonQP) #(time[i] - X2time)

                if NT1QP*(1/fullDec_QP[i])>delta_t:
                    NT1QP=NT1QP  - delta_t/(1/fullDec_QP[i])
                    
        
                else:
                    if ExcStateProb[i]>0.01: #excited
                        stateVector[i][0]=0
                        stateVector[i][1]=0
                        stateVector[i][2]=1
                        par=(-1)*par
                        n_par[i]= 1
                        #print('from', -par, 'to',  par, 'at t=', time[i]*1e-3, 'us')
                        stateVector[i][3]=par               #stateVector[i][3]
                        parity[i]=par
                        dec[i]= 1
                        qpdec[i]= 1
                        # print('qp dec')
                        stepIsLiveDec=False
                        # if whichGate==1:
                        # dec_idt[i]=1
                        # decQPtimes.append(t_event_decQP)   #(time[i] - X2time)
                        
            ###############################################################
       


         ####################Do Exc Stuff##############################################################################
        ####################Check if the qubit is not in the ground state##########################################################
        if np.abs(time[i] - X2time)<delta_t and time[i] > X2time and ExcStateProb[i]<0.99 :  # and (whichGate==1 or whichGate==3):
           # stepIsLiveExc=False
            t_event_excNonQP=(-1)*np.log(1-random.uniform(0, 1))*(1/rate_Exc_nonqp)
            NExcNonQP=t_event_excNonQP/(1/rate_Exc_nonqp)
            t_event_ExcQP=-1*np.log(1-random.uniform(0, 1))*(1/fullExc_QP[i])
            NExcQP=t_event_ExcQP/(1/fullExc_QP[i])
            stepIsLiveExc=True

        if np.abs(time[i] - Y2time)<delta_t and time[i] > Y2time and ExcStateProb[i]<0.99 :  # and (whichGate==1 or whichGate==3):
           # stepIsLiveExc=False
            t_event_excNonQP=(-1)*np.log(1-random.uniform(0, 1))*(1/rate_Exc_nonqp)
            NExcNonQP=t_event_excNonQP/(1/rate_Exc_nonqp)
            t_event_ExcQP=-1*np.log(1-random.uniform(0, 1))*(1/fullExc_QP[i])
            NExcQP=t_event_ExcQP/(1/fullExc_QP[i])
            stepIsLiveExc=True


        # if stepIsLiveExc==False and (whichGate==1 or whichGate==3):
        #     t_event_excNonQP=(-1)*np.log(1-random.uniform(0, 1))*(1/rate_Exc_nonqp)
        #     NExcNonQP=t_event_excNonQP/(1/rate_Exc_nonqp)
        #     t_event_ExcQP=-1*np.log(1-random.uniform(0, 1))*(1/fullExc_QP[i])
        #     NExcQP=t_event_ExcQP/(1/fullExc_QP[i])
        #     stepIsLiveExc=True
        if whichGate==0 or whichGate==2 or whichGate==4:
            stepIsLiveExc=False

        if stepIsLiveExc==True and (whichGate==1 or whichGate==3):

            if NExcNonQP*(1/rate_Exc_nonqp)>delta_t:
                NExcNonQP=NExcNonQP  - delta_t/(1/rate_Exc_nonqp)

            else:
                if ExcStateProb[i]<0.99: #excited
                    pass
                else:
                    stateVector[i][0]=0
                    stateVector[i][1]=0
                    stateVector[i][2]=-1
                    stepIsLiveExc=False
                    
                    n_nonqpexc[i]= 1
                    n_exc[i]= 1
                    # print('exc')
                    if whichGate==1:
                        exc_idt[i]=1
                    elif whichGate==3:
                        exc_ro[i]=1

            if NExcQP*(1/fullExc_QP[i])>delta_t:
                NExcQP=NExcQP  - delta_t/(1/fullExc_QP[i])
    
            else:
                if ExcStateProb[i]<0.99: #excited
                    pass
                else:
                    stateVector[i][0]=0
                    stateVector[i][1]=0
                    stateVector[i][2]=-1
                    par=(-1)*par
                    #print('from', -par, 'to',  par, 'at t=', time[i]*1e-3, 'us')
                    parity[i]=par
                    stateVector[i][3]=par     #  stateVector[i-1][3]
                    n_exc[i]= 1
                    n_par[i]= 1
                    n_qpexc[i]= 1
                    n_par_narrow[i]= 1
                    stepIsLiveexc=False
                    # print('exc')
                    
                    if whichGate==1:
                        exc_idt[i]=1
                    elif whichGate==3:
                        exc_ro[i]=1
                    #print('switch exc', 'parity prev',stateVector[i-1][3],'parity now', 'whichGate',whichGate)



        ####################Do Parity Switching Where Nothing happens##############################################################################
        ##########################################################################################################################################
        if stepIsLivenonPar==False:
            t_event_par00=(-1)*np.log(1-random.uniform(0, 1))*(1/rate_P_no_State_change[i])
            NTP00=t_event_par00/(1/rate_P_no_State_change[i])
            # print('t_event_par00',t_event_par00)
            stepIsLivenonPar=True
            
          
            
          
        ############ Tunneling or Switching parity when it's time ##########################
        if NTP00*(1/rate_P_no_State_change[i]) > delta_t:
            NTP00= NTP00 - delta_t/(1/rate_P_no_State_change[i])
            
                
        else:
            #### When it's time to switch parity, restart counntdown and do it ###########
            #print('switch no', 'parity prev',stateVector[i-1][3],'parity now', 'whichGate',whichGate)
            # print('par')
            # print('par happened!! for t_event_par00=', t_event_par00)
            tpars.append(t_event_par00)
            stepIsLivenonPar=False
            n_par_narrow[i]= 1
            n_par[i]= 1
            par=(-1)*par
            #print('from', -par, 'to',  par, 'at t=', time[i]*1e-3, 'us')
            parity[i]=par                    #ity[i-1]*(-1)
            stateVector[i][3]=par              # stateVector[i-1][3]


        
        ########################################################################################
        
                
        
            
            
        if i!=0:
            
           
            if (X2time>time[i] and whichGate==0):
                x_2gate1[i]=1
                y_2gate1[i]=0
                measureGate1[i]=0
                activereset1[i]=0
            if (idtime<time[i] and whichGate==2):
                x_2gate1[i]=0
                y_2gate1[i]=1
                measureGate1[i]=0
                activereset1[i]=0
            if (time[i]>Y2time and whichGate==3):
                x_2gate1[i]=0
                y_2gate1[i]=0
                measureGate1[i]=1
                activereset1[i]=0
                #print(time[i], 'res')
            if (whichGate==4):
                x_2gate1[i]=0
                y_2gate1[i]=0
                measureGate1[i]=0
                activereset1[i]=1
                #print(time[i], 'act')
        
    
    
       
    
    
        thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])
        ExcStateProb[i]=(sin(thetaNow/2))**2
        if np.abs(rotime-time[i])<delta_t and time[i]<rotime:
            # n_ro[1]= 1
            
            parDecExcProb.append(ExcStateProb[i])#(sin(thetaNow/2)**2)
            parityAtRO.append(parity[i])
            
            probeout=ExcStateProb[i]#(sin(thetaNow/2))**2
            excprob=ExcStateProb[i]#(sin(thetaNow/2))**2
            # if 0.01< (sin(thetaNow/2))**2 < 0.99:
                #print('(sin(thetaNow/2))**2',(sin(thetaNow/2))**2)
            #print('probeout',probeout)
            if probeout>0.99:
                probeout=1
                if stateVector[i][3]==1:
                    # ev_e[i]=1
                    n_eve[i]= 1
                else:
                    # od_e[i]=1
                    n_ode[i]= 1
            elif probeout<0.01:
                probeout=0
                if stateVector[i][3]==1:
                    n_evg[i]= 1
                    # ev_g[i]=1
                else:
                    # od_g[i]=1
                    n_odg[i]= 1

            readoutErr=False
            if random.uniform(0, 1)<(1-SSF):
                readoutErr=True
                roerr[nmeas]=1
            else:
                readoutErr=False
            
            if probeout==1:
               
                # parDecReadout_correct.append(1)
                # if random.uniform(0, 1)<(1-SSF)/2:
                #parDecReadout.append(np.random.normal(m1,sd,1)[0])
                if readoutErr:
                    
                    # ro[i]=0
                    parDecReadout.append(0)
                    parDecReadoutTime.append(time[i])
                    nerr=nerr+1
                    
                else:
                    # ro[int(time[i]/(4.4*1e-6*1e9))]=1
                    parDecReadout.append(1)
                    parDecReadoutTime.append(time[i])
                    
                    
            elif probeout==0:
                
                # parDecReadout_correct.append(0)
                if readoutErr:
                    nerr=nerr+1
                    parDecReadout.append(1)
                    parDecReadoutTime.append(time[i])
                    
                else:
                    
                    parDecReadout.append(0)
                    parDecReadoutTime.append(time[i])
                    # if stateVector[i][3]==1:
                    #     #ev_e[i]=1
                    #     n_ev0[i]= 1
                    # else:
                    #     #od_e[i]=1
                    #     n_od0[i]= 1
            
            #thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])        
            if  0.01 <= ExcStateProb[i] <= 0.99:
                # print('probeout value',excprob)#, 'time', time[i])
                #thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])
                if random.uniform(0, 1)< ExcStateProb[i]: #exc state
                    # parDecReadout_correct.append(1)
                    if readoutErr:
                        nerr=nerr+1
                    #parDecReadout.append(np.random.normal(m1,sd,1)[0])
                        parDecReadout.append(0)
                        parDecReadoutTime.append(time[i])
                        #parDecReadoutTime.append(time[i])
                    else:
                        parDecReadout.append(1)
                        parDecReadoutTime.append(time[i])
                else: #grd state
                    # parDecReadout_correct.append(0)
                    if readoutErr:
                        nerr=nerr+1
                    #parDecReadout.append(np.random.normal(m1,sd,1)[0])
                        parDecReadout.append(1)
                        parDecReadoutTime.append(time[i])
                        #parDecReadoutTime.append(time[i])
                    else:
                        parDecReadout.append(0)
                        parDecReadoutTime.append(time[i]) 
            
            # if parDecReadout[-1]!=parDecReadout_correct[-1]:
            #     n_ssferr=n_ssferr+1

            # if par==1 and parDecReadout[-1]==1:
            #     #ev_e[i]=1
            #     n_od1[i]= 1
            # elif par==1 and parDecReadout[-1]==0:
            #     #od_e[i]=1
            #     n_od0[i]= 1

            # elif par==-1 and parDecReadout[-1]==0:
            #     #od_e[i]=1
            #     n_ev0[i]= 1

            # elif par==-1 and parDecReadout[-1]==1:
            #     #od_e[i]=1
            #     n_ev1[i]= 1
            nmeas=nmeas+1
    print('(nmeas-nerr)/nmeas',(nmeas-nerr)/nmeas)
    #print('ending sims')
    timetaken=tm.time() - start_time
    #print('timetaken',str(timetaken))
    binwidth=0.5*1e-3*1e9
    binwidth25= 0.25*1e-3*1e9#
    binwidth1=0.1*1e-3*1e9
    binwidth005=0.05*1e-3*1e9
    # decseqwvfm     , numPerbin, btimes = Make_wvfm_1s(dec    , time,  binwidth25,     delta_t)
    # nonqpdecseqwvfm     , numPerbin, btimes = Make_wvfm_1s(nonqpdec    , time,  binwidth25,     delta_t)
    # qpdecseqwvfm     , numPerbin, btimes = Make_wvfm_1s(qpdec    , time,  binwidth25,     delta_t)
    
    # roseqwvfm25, ronumPerbin25, robtimes25 = waveAndLike.Makewaveform( np.array(parDecReadout), np.array(parDecReadoutTime), binwidth25, ParSeqlen )
    roseqwvfm1, ronumPerbin1, robtimes1 = waveAndLike.Makewaveform( np.array(parDecReadout), np.array(parDecReadoutTime), binwidth1, ParSeqlen )

    # flips=waveAndLike.countjumps(np.array(parDecReadout))
    # roerrEdge=0
    # roerrmid=0
    # oneROERR=0
    # bactobackroerr=0
    # if roerr[0]==1:
    #     roerrEdge=roerrEdge+1
    # elif roerr[2999]==1:
    #     roerrEdge=roerrEdge+1

    # for k in range(1,2999):
    #     if roerr[k]==1:
    #         roerrmid=roerrmid+1
    #         if roerr[k-1]!=roerr[k] and roerr[k]!=roerr[k+1]:
    #             oneROERR=oneROERR+1

    #         if roerr[k-1]==roerr[k] or roerr[k]==roerr[k+1]:
    #             bactobackroerr=bactobackroerr+1

            
                
    # stuff=[flips, roerrEdge, roerrmid, oneROERR, bactobackroerr]
    
    

    
    # if SSF==0.95:
    #     avgnflip_base=11.8
    #     subsize25=int(ronumPerbin25 /avgnflip_base)
    #     # subsize005=int(ronumPerbin005 /avgnflip_base)
    # elif SSF==0.99:
    #     avgnflip_base=2.7125
    #     subsize25=int(ronumPerbin25 /avgnflip_base)
    #     # subsize005=int(ronumPerbin005 /avgnflip_base)

    # filtro25 = waveAndLike.movmed(np.array(parDecReadout), subsize25)
    # # # filtro005 = waveAndLike.movmed(np.array(parDecReadout), subsize005)
    # filtroseqwvfm25    ,filtronumPerbin25    ,  filtrobtimes25    = waveAndLike.Makewaveform(np.array(filtro25), np.array(parDecReadoutTime[:len(filtro25)]), binwidth25, ParSeqlen)
    # # filtroseqwvfm005    ,filtronumPerbin005    ,  filtrobtimes005    = waveAndLike.Makewaveform(np.array(filtro005), np.array(parDecReadoutTime[:len(filtro005)]), binwidth005, ParSeqlen)
    # #roseqwvfmOUT , numPerbin, btimes = waveAndLike.Makewaveform(np.array(parDecReadout), np.array(parDecReadoutTime), binwidth, ParSeqlen)
    
    # # print('robtimes25',robtimes25)
    # # print('ronumPerbin25',ronumPerbin25)
    fid = SSF
    # #print('ssf',fid)
    baseParSwtRate = 0
    baseT1us = (1/base_dec_rate)*1e-3
    baseT2us = (1e-3) / (rate_dph[0]  + 0.5*base_dec_rate )
    #print('baseT2us', baseT2us)
    # filtfid=1
    # #print('ssf',fid)
    # filtbaseParSwtRate=0
    # filtbaseT1us= 1e6 #(1/base_dec_rate)*1e-3
    # filtbaseT2us= 1e6 #(1/base_dec_rate)*1e-3
    filtxqp0=1e-12
    #print('baseT1us',baseT1us)
    #btimes=np.linspace(0, 5*1e-3, len(roseqwvfm))
    #Nperbin=113
    
    # mostprobenergydep25 = waveAndLike.likelihoodEdep(fid, baseParSwtRate,  baseT1us, baseT2us,  roseqwvfm25,  robtimes25, ronumPerbin25 , xqp0    )
    # mostprobenergydep1 = waveAndLike.likelihoodEdep(fid, baseParSwtRate,  baseT1us, baseT2us,  roseqwvfm1,  robtimes1, ronumPerbin1 , xqp0   )
    # #print('mostprobenergydep25',mostprobenergydep25)
    # # mostprobenergydephalf25, likes, Pjump = waveAndLike.likelihoodEdep(fid, baseParSwtRate,  baseT1us,   roseqwvfmhalf25, robtimeshalf25, ronumPerbinhalf25 )

    
    # filtmostprobenergydep25 = waveAndLike.likelihoodEdep(filtfid, filtbaseParSwtRate,  filtbaseT1us, filtbaseT2us,   filtroseqwvfm25,         filtrobtimes25    , filtronumPerbin25 filtxqp0    )
   
    
    return parDecReadout, parDecReadoutTime,  roseqwvfm1, (nmeas-nerr)/nmeas#, stuff, roerr, qpdec, nonqpdec, n_par #, mostprobenergydep1 #roseqwvfm25, mostprobenergydep25,, decNonQPtimes, decQPtimes, tpars, (nmeas-nerr)/nmeas  #, decseqwvfm, nonqpdecseqwvfm, qpdecseqwvfm , decNonQPtimes, decQPtimes
    # return parDecReadout, parDecReadoutTime,roseqwvfm25,  parity , mostprobenergydep25 , filtroseqwvfm25  ,  filtmostprobenergydep25 
    


#,, n_par, ngs, n_ph, parity# roseqwvfm005 ,  filtroseqwvfm005 ,     mostprobenergydep005#, filtmostprobenergydep005 #pos, neg, nodd_nonqpdec, neven_nonqpdec, nonqpdec, dec, n_exc #,  mostprobenergydep, mostprobenergydephalf, mostprobenergydep25, mostprobenergydephalf25, mostprobenergydep1, mostprobenergydephalf1  #, parDecReadout_correct, n_ssferr , decseqwvfm    ,dec_idtseqwvfm,dec_roseqwvfm ,parseqwvfm    ,exc_idtseqwvfm,exc_rowvfm    ,excseqwvfm    ,ev0seqwvfm    ,ev1seqwvfm    ,od0seqwvfm    ,od1seqwvfm    , ev0_ro , ev1_ro, od0_ro , od1_ro


    #[n_par       ,n_par_narrow,n_nonqpdec  ,n_nonqpexc  ,n_qpdec     ,n_qpexc     ,n_dec       ,n_exc           ,n_evg       ,n_eve       ,n_ev0       ,n_ev1       ,n_odg       ,n_ode       ,n_od0       ,n_od1        , dec_idt, exc_idt, dec_ro, exc_ro ]#, x_2gate1, y_2gate1, measureGate1, activereset1    
    #, idt, dec_idt  ,exc_idt, dec_ro ,exc_ro,  ev_0 ,od_0 ,ev_1 ,od_1, ev_g ,od_g ,ev_e ,od_e, dec#,  fullprate# ,parDecExcProb,#,   ,roevent , ,ssf_ro_err ,dec_idt_wrg_ro,exc_idt_wrg_ro,dec_ro_wrg_ro ,exc_ro_wrg_ro ,,      
    

################################################################################################################
def Detectsims(para):#(edep):edep,maxtime_ms
    edep=para[0]
    maxtime_ms=para[1]
    hbar_ev=6.5821*1e-16 #eVs
    hbar_js=1.054571817*1e-34 #Js
    Ej=hbar_ev*(2*np.pi)*9.945*1e9 #J
    Ec=hbar_ev*(2*np.pi)*390*1e6 #J
    supcon_gap_Al=180*1e-6 #eV
    Ege=np.sqrt(8*Ej*Ec)-Ec #J 
    wq=Ege/hbar_ev
    fq=wq/(2*np.pi)
    #print('fq=',fq*1e-6,'MHz')
    adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
    bdf=((Ej/Ec/2)**(3/4))
    cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
    ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
    dwq=adf*bdf*cdf*ddf
    dfq=dwq/(2*np.pi)
    delta_f10=dfq
    
    timechunks=1e6 #ns Contains 20 pulse sequences each of which is about 3.1us.
    numRatepts=0.1
    maxtime=maxtime_ms*1e-3*1e9   #numRatepts*timechunks #ns
    num_time_pts=int(maxtime/50) #50ns timestep
    #E_depo_timestream=np.zeros(num_pts) #ns Energy depo into the island
    #E_depo_timestream[0]=EeV #eV
    
    time=np.linspace(0,maxtime,num_time_pts) #ns
    ratefile=np.load(f'{edep}meVrates.npz')
    xqps=ratefile['xqps']
    rate_D=ratefile['rate_D']
    rate_Exc=ratefile['rate_Exc']
    rate_P=ratefile['rate_P']
    fullprate=ratefile['fullprate']
    rate_dph=ratefile['rate_dph']
    rate_P_no_State_change=ratefile['rate_P_no_State_change']
    rate_Exc_qp=ratefile['rate_Exc_qp']
    rate_Exc_nonqp=ratefile['rate_Exc_nonqp']
    rate_Dqp_base=ratefile['rate_Dqp_base']
    rate_Dqp=ratefile['rate_Dqp']
    base_dec_rate=ratefile['base_dec_rate']
    
    delta_t=time[1] - time[0]
    #print('delta_t',delta_t)
    maxPhaseerror=2*delta_t*1e-9*delta_f10
    #parityy=-1
    #print('parity=',parityy, 'delta_t=',delta_t, 'ns')
    
    Pargates=["x_2","idle","y_2","measure","active_reset"]
    
    idlelen= (1/(4*delta_f10))*1e9 #- y_2len
    #print('idlelen',idlelen)
    #print('idlelen s',idlelen*1e-9)
    measurelen=2000 #ns
    active_resetlen=2000#(2*1e-6)*1e9 #ns
    x_2len=100 #ns
    y_2len=100#idlelen#40 #ns
    idlewait=idlelen#-y_2len
    #print('idlewait',idlewait)
    
    Pargatetimes=[x_2len, x_2len+idlewait, x_2len+idlewait+y_2len, x_2len+idlewait+y_2len+measurelen, x_2len+idlewait+y_2len+measurelen+active_resetlen]
    X2time=x_2len
    Y2time=x_2len+idlewait+y_2len
    idtime=x_2len+idlewait
    Actime=x_2len+idlewait+y_2len+measurelen+active_resetlen
    ParSeqlen=x_2len+idlelen+y_2len+measurelen+active_resetlen
    #print('ParSeqlen',ParSeqlen)
    numSeq=int(maxtime/ParSeqlen)
    
    whichSeq=0
    whichGate=0
    #print('whichSeq',whichSeq)
    start_time=tm.time()
    
    readout=np.zeros(len(time))
    readoutTime=np.zeros(len(time))
    pigate1=np.zeros(len(time))
    x_2gate1    =np.zeros(len(time))
    y_2gate1    =np.zeros(len(time))
    measureGate1=np.zeros(len(time))
    activereset1=np.zeros(len(time))
    ExcStateProb=np.zeros(len(time))
    phi=0
    phis=np.zeros(len(time))
    parityAtRO=[]
    x_2gate1=np.zeros(len(time))
    y_2gate1=np.zeros(len(time))
    measureGate1=np.zeros(len(time))
    activereset1=np.zeros(len(time))
    dephasing=np.zeros(len(time))
    decoherence=[]#np.zeros(len(time))
    decoherenceTime=[]
    excitation=[]#np.zeros(len(time))
    excitationTime=[]
    parityflip=[]#np.zeros(len(time))
    parityflipTime=[]
    startgt=0
    start=0
    parDecReadout=[]
    parDecReadoutTime=[]
    parDecExcProb=[]
    fine=[]
    preve=[]
    gt=[]
    rateT1atRead=[]
    rateTpatRead=[]
    rateT1excatRead=[]
    rateTphiRead=[]
    thetaTest=[]#np.zeros(len(time))
    parIdleVec=[]
    stateVector=np.zeros((len(time),4))
    readout=np.zeros((len(time)))
    idt           =np.zeros((len(time)))
    dec_idt       =np.zeros((len(time)))
    exc_idt       =np.zeros((len(time)))
    roevent       =np.zeros((len(time)))
    dec_ro        =np.zeros((len(time)))
    exc_ro        =np.zeros((len(time)))
    ssf_ro_err    =np.zeros((len(time)))
    dec_idt_wrg_ro=np.zeros((len(time)))
    exc_idt_wrg_ro=np.zeros((len(time)))
    dec_ro_wrg_ro =np.zeros((len(time)))
    exc_ro_wrg_ro =np.zeros((len(time)))
    ev_0          =np.zeros((len(time)))
    od_0          =np.zeros((len(time)))
    ev_1          =np.zeros((len(time)))
    od_1          =np.zeros((len(time)))
    
            
    for i in range((len(time))):
        #print(time[i])
        #print('whichGate',whichGate)
        ####################################
        if i==0:
            stateVector[i+startgt][0]=0
            stateVector[i+startgt][1]=0
            stateVector[i+startgt][2]=1
            if random.uniform(0, 1)<0.5:  #randomly start with even or odd parity with 50-50 chance
                stateVector[i+startgt][3]=-1  #Set parity to 1 at t0
            else:
                stateVector[i+startgt][3]=1
            
            #x=0
            #y=0
            #z=1
            #parity=-1
            rotime=Pargatetimes[3]
            
        elif i>0:
            #print(f'1stateVector[{i}][3]=', stateVector[i][3] , f'stateVector[{i-1}][3]=', stateVector[i-1][3])
            stateVector[i][0]=stateVector[i-1][0]
            stateVector[i][1]=stateVector[i-1][1]
            stateVector[i][2]=stateVector[i-1][2]
            stateVector[i][3]=stateVector[i-1][3]  #At t>t0, set Parity to the value on step behind. So Parity at t1 will be set to parity at t0 which is 1 if during t0, the parity did not change
            
           
            
            #print(f'2stateVector[{i}][3]', stateVector[i][3] , f'stateVector[{i-1}][3]', stateVector[i-1][3]) 
            
        else:
            #print('i is negative!!!!!!!!')
            break
        
        #dec_now=False
        #par_now=False
        #print('stateVector[i+startgt][3]',stateVector[i+startgt][3])
        #readout[i]=-0.5
        ############################################################
        
        ############################################################
        if whichSeq<=numSeq:
            gateApplyTime=whichSeq*ParSeqlen + Pargatetimes[whichGate]
            #print('gateApplyTime',gateApplyTime, Pargates[whichGate])
        if whichGate==3:
            rotime=gateApplyTime
        if whichGate==0:
            X2time=gateApplyTime
        if whichGate==1:
            idtime=gateApplyTime
        if whichGate==2:
            Y2time=gateApplyTime
        if whichGate==4:
            Actime=gateApplyTime
    
    
        x=stateVector[i][0]
        y=stateVector[i][1] 
        z=stateVector[i][2]
        parity[i]=stateVector[i][3]
        #print('stateVector[i+startgt]1',stateVector[i+startgt])
        if np.abs(gateApplyTime-time[i])<delta_t:
            if whichGate==1:
                idt[i]=1
                
            if whichGate==3:
                roevent[i]=1
            #if whichGate==3:
            #    print('Y2-id',y2-id, 'whichGate',whichGate, 'whichSeq',whichSeq)
            prevSt, finVector, readoutValue = Gate(Pargates[whichGate], x,y,z,  P10, P01)
            #print('prevSt',prevSt,'finVector',finVector)
            stateVector[i][0]=finVector[0]
            stateVector[i][1]=finVector[1]
            stateVector[i][2]=finVector[2]
            finSt=finVector
    
            whichGate=whichGate+1
            if whichGate==len(Pargates):
                whichGate=0
                whichSeq=whichSeq+1

        thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])
        ExcStateProb[i]=(sin(thetaNow/2))**2
        
        ############################################################################################
        ########################################################################################
        ##################### Non QP Decay or Excitation ################################################
        if np.abs(X2time-time[i])<delta_t or np.abs(Y2time-time[i])<delta_t:
            #if whichGate==1:
            #    isthere_dec_idt=0 ##start from zero to count decay and excitation events during this idle time
            #    isthere_exc_idt=0 ##start from zero to count decay and excitation events during this idle time
#
            #elif whichGate==3:
            #    isthere_dec_ro=0  ##start from zero to count decay and excitation events during this ro time
            #    isthere_exc_ro=0  ##start from zero to count decay and excitation events during this ro time
            ###Start countdown till next nonQP decay after X/2 pulse or Y/2 pulse
            if ExcStateProb[i]>0.05:
                t_event_decNonQP=-1*np.log(1-random.uniform(0, 1))*(1/base_dec_rate)
                NT1NonQP=t_event_decNonQP/(1/base_dec_rate)
            ##Start countdown till next nonQP excitation after X/2 pulse or Y/2 pulse
            if ExcStateProb[i]<0.95:
                #print('Nexc set')
                t_event_excNonQP=(-1)*np.log(random.uniform(0, 1))*(1/rate_Exc_nonqp)
                NExcNonQP=t_event_excNonQP/(1/rate_Exc_nonqp)
    
            #else:
                
        ##Start countdown till next dephasing after X/2 pulse
        if np.abs(X2time-time[i])<delta_t :
            t_event_dph=-1*np.log(1-random.uniform(0, 1))*(1/rate_dph[i])
            NTdph=t_event_dph/(1/rate_dph[i])
    
        if i==0:
            Phase_jitter=0
            #t_event_dph=-1*np.log(1-random.uniform(0, 1))*(1/rate_dph[i])
            #NTdph=t_event_dph/(1/rate_dph[i])
            
        if i!=0:
            
            #################################### 
            ## Restart countdown for dephasing when it's time for dephasing.
            ## Dephasing by adding a phase jitter. Examine later
            
            if NTdph*(1/rate_dph[i])<delta_t:
                t_event_dph=(-1)*np.log(1-random.uniform(0, 1))*(1/rate_dph[i])
                NTdph=t_event_dph/(1/rate_dph[i])
                Phase_jitter=idlelen*np.cos(2*np.pi*0.3)*delta_f10
                dephasing[i]=1
            else:
                ## If it's not time for dephasing, continue the countdown
                NTdph= NTdph - delta_t/(1/rate_dph[i+start])
                dephasing[i]=0
                
        
        
    
    
            if (X2time>time[i] and whichGate==0):
                x_2gate1[i]=1
                y_2gate1[i]=0
                measureGate1[i]=0
                activereset1[i]=0
            if (idtime<time[i] and whichGate==2):
                x_2gate1[i]=0
                y_2gate1[i]=1
                measureGate1[i]=0
                activereset1[i]=0
            if (time[i]>Y2time and whichGate==3):
                x_2gate1[i]=0
                y_2gate1[i]=0
                measureGate1[i]=1
                activereset1[i]=0
                #print(time[i], 'res')
            if (whichGate==4):
                x_2gate1[i]=0
                y_2gate1[i]=0
                measureGate1[i]=0
                activereset1[i]=1
                #print(time[i], 'act')
        
    #if (X2time<time[i] and time[i]<Y2time):
    #    if NTdph

        ########################################################################
        ########################################################################################
        ########################################################################################

        ################# Parity Switching Or Quasiparticle Tunneling #############################
        ############ Start countdown to next tunneling event ##########################
        if i==0:
            t_event_par=-1*np.log(1-random.uniform(0, 1))*(1/fullprate[i])
            NTP=t_event_par/(1/fullprate[i])
            
            
        ############ Tunneling or Switching parity when it's time ##########################
        if i!=0:
            
            #################################### 
            noChangeParratio=rate_P[i]/fullprate[i]
            decParratio=rate_Dqp[i]/fullprate[i]
            excParratio=rate_Exc_qp[i]/fullprate[i]
            
            
            if NTP*(1/fullprate[i])<delta_t:
                #### When it's time to switch parity, restart counntdown and do it ###########
                #print('partity switch!!!!!!!!')
                t_event_par=(-1)*np.log(1-random.uniform(0, 1))*(1/fullprate[i])
                NTP=t_event_par/(1/fullprate[i])
                parityflip.append(1)
                parityflipTime.append(time[i])
                print('time[i]',time[i])
                stateVector[i][3]=stateVector[i-1][3]*(-1)
                
                #### When it's time to switch parity, randomly choose between no state change, QP-qubit decay, or QP-qubit excitation ########
                if random.uniform(0, 1)<noChangeParratio:
                    #print('partity switch-No change!!!!!!!!')
                    pass
                elif random.uniform(0, 1)>noChangeParratio and random.uniform(0, 1) < (noChangeParratio + decParratio):
                    #print('time for QP-decay!!!', 'gate is ', Pargates[whichGate])
                    if (X2time<time[i] and time[i]<Y2time) or (Y2time<time[i] and time[i]<rotime):
                        if ExcStateProb[i]>0.05:
                            decoherence.append(1)
                            decoherenceTime.append(time[i])
                            #print('QP-decay!!!')
                            stateVector[i][0]=0
                            stateVector[i][1]=0
                            stateVector[i][2]=1
                            #if whichGate==1:
                            #    dec_idt[i]=1
                            #elif whichGate==3:
                            #    dec_ro[i]=1

                elif random.uniform(0, 1)>(noChangeParratio + decParratio) and random.uniform(0, 1)<1:
                    #print('time for QP-excitation!!!', 'gate is ', Pargates[whichGate])
                    if (X2time<time[i] and time[i]<Y2time) or (Y2time<time[i] and time[i]<rotime):
                        if ExcStateProb[i]<0.95:
                            excitation.append(1)
                            excitationTime.append(time[i])
                            #print('QP-excitation!!!')
                            stateVector[i][0]=0
                            stateVector[i][1]=0
                            stateVector[i][2]=-1
                            #if whichGate==1:
                            #    dec_idt[i]=1
                            #    isthere_dec_idt=isthere_dec_idt+1
                            #elif whichGate==3:
                            #    dec_ro[i]=1
                            #    isthere_dec_ro=isthere_dec_ro+1
                
            else:
                NTP= NTP - delta_t/(1/fullprate[i])

        ##########################################################################################
    
        ####################################################################
        ######## NonQP  Decay during idle-time or readout time
        if (X2time<time[i] and time[i]<Y2time) or (Y2time<time[i] and time[i]<rotime):
            ## Decay when it's time and restart countdown till next NonQP decay.
            if NT1NonQP*(1/base_dec_rate)<delta_t:
                t_event_decNonQP=-1*np.log(1-random.uniform(0, 1))*(1/base_dec_rate)
                NT1NonQP=t_event_decNonQP/(1/base_dec_rate)
                
                
                    
                    
                if ExcStateProb[i]  <=0.05:
                    pass
                else:
                    #if whichGate==1 or whichGate==3:
                    decoherence.append(1)
                    decoherenceTime.append(time[i])
                    #print('NonQP-decay!!!')
                    stateVector[i][0]=0
                    stateVector[i][1]=0
                    stateVector[i][2]=1
                    #if whichGate==1:
                    #    dec_idt[i]=1
                    #    isthere_dec_idt=isthere_dec_idt+1
                    #elif whichGate==3:
                    #    dec_ro[i]=1
                    #    isthere_dec_ro=isthere_dec_ro+1
    
            else:
                #decoherence[i]=0
                
                NT1NonQP= NT1NonQP - delta_t/(1/base_dec_rate)
                #isthere_dec_idt=False
                #isthere_dec_ro =False
    
        ####################################################################
        ######## NonQP  Excitation during idle-time or readout time
        if (X2time<time[i] and time[i]<Y2time) or (Y2time<time[i] and time[i]<rotime):
            ## Decay when it's time and restart countdown till next NonQP decay.
            if NExcNonQP*(1/rate_Exc_nonqp)<delta_t:
                t_event_excNonQP=(-1)*np.log(random.uniform(0, 1))*(1/rate_Exc_nonqp)
                NExcNonQP=t_event_excNonQP/(1/rate_Exc_nonqp)
                
                    
                    
                if ExcStateProb[i]  >0.95:
                    pass
                else:
                    #if whichGate==1 or whichGate==3:
                    excitation.append(1)
                    excitationTime.append(time[i])
                    #print('NonQP-excitation!!!')
                    stateVector[i][0]=0
                    stateVector[i][1]=0
                    stateVector[i][2]=-1
                    #if whichGate==1:
                    #    exc_idt[i]=1
                    #    isthere_exc_idt=isthere_exc_idt+1
                    #elif whichGate==3:
                    #    exc_ro[i]=1
                    #    isthere_exc_ro=isthere_exc_ro+1
    
            else:
                #decoherence[i]=0
                
                NExcNonQP= NExcNonQP - delta_t/(1/rate_Exc_nonqp)
        
        deltaPhi=delta_t*stateVector[i][3]*delta_f10*1e-9*(2*math.pi) + Phase_jitter #delta_t*stateVector[i+startgt][3]*delta_f10*1e-9*2*math.pi
        if whichGate==1:
        #if stateVector[i+startgt][0]!=0 or stateVector[i+startgt][1]!=0:
            finStPostZrot=rotateStateVector(stateVector[i][0], stateVector[i][1], stateVector[i][2], deltaPhi, 'z') #Rotate state vector about z axis by deltaPhi
            stateVector[i][0]=finStPostZrot[0] #update state vector after z rotation   
            stateVector[i][1]=finStPostZrot[1] #update state vector after z rotation
            stateVector[i][2]=finStPostZrot[2] #update state vector after z rotation
            phi=phi+deltaPhi
            #print('phi',phi)
            if np.abs(phi)>2*np.pi:
                phi=0
                phis[i]=phi
            else:
                deltaPhi=0
                phis[i]=phi
        
        
        
                #parityflip[i]=0
    
        
       
        ##############################################
    
       
    
    
        thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])
        ExcStateProb[i]=(sin(thetaNow/2))**2
        if np.abs(rotime-time[i])<delta_t and time[i]<rotime:
            parDecExcProb.append(ExcStateProb[i])#(sin(thetaNow/2)**2)
            parityAtRO.append(stateVector[i][3])
            #print('delta_t', delta_t, 'rotime-time[i+startgt]',rotime-time[i+startgt])
            #thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])
            probeout=ExcStateProb[i]#(sin(thetaNow/2))**2
            excprob=ExcStateProb[i]#(sin(thetaNow/2))**2
            #print('roT',time[i])
            rateT1atRead.append(rate_D[i])
            rateTpatRead.append(rate_P[i])
            
            
            
            #print('probeout',probeout)
            if probeout>0.99:
                probeout=1
            elif probeout<0.01:
                probeout=0
            #parDecExcProb.append(round((sin(thetaNow/2))**2, 2))
            #if isthere_dec_idt>0: #decay occurs during idle time
            #    if stateVector[i][3]==1 and probeout==0: #odd and 0 because #decay occurs during idle time
            #        dec_idt_wrg_ro[i]=1
            #    elif stateVector[i][3]==-1 and probeout==1: #even and 1 because #decay occurs during idle time
            #        dec_idt_wrg_ro[i]=1
#
            #if isthere_exc_idt>0: #excitation occurs during idle time
            #    if stateVector[i][3]==1 and probeout==0: #odd and 0 because #excitation occurs during idle time
            #        exc_idt_wrg_ro[i]=1
            #    elif stateVector[i][3]==-1 and probeout==1: #even and 1 because #excitation occurs during idle time
            #        exc_idt_wrg_ro[i]=1
#
            #if isthere_dec_ro>0: #decay occurs during idle time
            #    if stateVector[i][3]==1 and probeout==0: #odd and 0 because #decay occurs during ro time
            #        dec_ro_wrg_ro[i]=1
            #    elif stateVector[i][3]==-1 and probeout==1: #even and 1 because #decay occurs during ro time
            #        dec_ro_wrg_ro[i]=1
#
            #if isthere_exc_ro>0: #excitation occurs during idle time
            #    if stateVector[i][3]==1 and probeout==0: #odd and 0 because #excitation occurs during ro time
            #        exc_ro_wrg_ro[i]=1
            #    elif stateVector[i][3]==-1 and probeout==1: #even and 1 because #excitation occurs during ro time
            #        exc_ro_wrg_ro[i]=1
            
            #print("probeout",probeout)
            if probeout==1:
                #if stateVector[i][3]==1: #od 1
                #    od_1[i]=1
                #elif stateVector[i][3]==-1: #ev 1
                #    ev_1[i]=1
                if random.uniform(0, 1)<(1-SSF)/2:
                #parDecReadout.append(np.random.normal(m1,sd,1)[0])
                    parDecReadout.append(0)
                    parDecReadoutTime.append(time[i])
                    #parDecReadoutTime.append(time[i])
                    #ssf_ro_err[i]=1
                else:
                    parDecReadout.append(1)
                    parDecReadoutTime.append(time[i])
                    
            elif probeout==0:
                #if stateVector[i][3]==1: #od 0
                #    od_0[i]=1
                #elif stateVector[i][3]==-1: #ev 0
                #    ev_0[i]=1
                if random.uniform(0, 1)<(1-SSF)/2:
                #parDecReadout.append(np.random.normal(m1,sd,1)[0])
                    parDecReadout.append(1)
                    parDecReadoutTime.append(time[i])
                    #parDecReadoutTime.append(time[i])
                    #ssf_ro_err[i]=1
                else:
                    parDecReadout.append(0)
                    parDecReadoutTime.append(time[i])
            
            #thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])        
            if ExcStateProb[i]>= 0.01 and ExcStateProb[i] <= 0.99:
                #print('probeout value',excprob)#, 'time', time[i])
                #thetaNow=atan2(math.sqrt(stateVector[i][0]**2 + stateVector[i][1]**2),stateVector[i][2])
                if random.uniform(0, 1)< ExcStateProb[i]: #exc state
                    if random.uniform(0, 1)<(1-SSF)/2:
                    #parDecReadout.append(np.random.normal(m1,sd,1)[0])
                        parDecReadout.append(0)
                        parDecReadoutTime.append(time[i])
                        #parDecReadoutTime.append(time[i])
                    else:
                        parDecReadout.append(1)
                        parDecReadoutTime.append(time[i])
                else: #grd state
                    if random.uniform(0, 1)<(1-SSF)/2:
                    #parDecReadout.append(np.random.normal(m1,sd,1)[0])
                        parDecReadout.append(1)
                        parDecReadoutTime.append(time[i])
                        #parDecReadoutTime.append(time[i])
                    else:
                        parDecReadout.append(0)
                        parDecReadoutTime.append(time[i]) 
    #print('ssf=',SSF)
    return parDecReadout, parDecReadoutTime, parityAtRO, parity#,  parDecExcProb    
