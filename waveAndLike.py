import numpy as np
from hmmlearn import hmm
from scipy import integrate
from matplotlib import pyplot as plt, cm
#import matplotlib.pyplot.colorbar
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
import time as tm
import multiprocessing
from decimal import Decimal
from scipy.special import factorial
from functools import reduce
from operator import mul
from math import comb


SSF=0.95
r=1 # P01/P10
P01=r*((1-SSF)/(1+r))
P10=(1-SSF)/(1+r)


# def probs(SSF, T1, basePar, Edep, Nperbin, btimes):
#     ssf=SSF
#     hbar_ev=6.5821*1e-16 #eVs
#     hbar_js=1.054571817*1e-34 #Js
#     Ej=hbar_ev*(2*np.pi)*9.945*1e9 #J
#     Ec=hbar_ev*(2*np.pi)*390*1e6 #J
#     supcon_gap_Al=180*1e-6 #eV
#     Ege=np.sqrt(8*Ej*Ec)-Ec #J 
#     wq=Ege/hbar_ev
#     fq=wq/(2*np.pi)
    
#     adf=16*np.sqrt(2/np.pi)*(Ec/hbar_ev)
#     bdf=((Ej/Ec/2)**(3/4))
#     cdf=np.exp(-1*np.sqrt(8*Ej/Ec))
#     ddf=(16*((Ej/Ec/2)**(1/2)) + 1)
#     dwq=adf*bdf*cdf*ddf
#     dfq=dwq/(2*np.pi)
    
#     kb_ev=8.6173303*1e-5 #eV/K
#     T= 50*1e-3 #20*1e-3 #K
#     r0=0.018/1e-9 #Hz
#     s0=1e-6/1e-9 #Hz
#     ncp=4*1e24 #m^-3
#     V_island=2400*((1e-6)**3) #m^3
#     phonon_to_qp_eff=0.57
#     base_Par_Rate=basePar    #Hz
    
#     Edeposited=Edep*1e-3 #eV
    
#     xinduced=Edeposited*phonon_to_qp_eff / (ncp*V_island*supcon_gap_Al)

#     decayRateCoeff=np.sqrt((2*wq*supcon_gap_Al)/(np.pi*np.pi*hbar_ev))#(16*Ej/(hbar_ev*np.pi)) * np.sqrt(Ec/(8*Ej)) * (supcon_gap_Al/(2*hbar_ev*wq))
#     baseT1=T1*1e-6 #s
#     baseDecayRate=1/baseT1
#     parRateCoeff_v1=(16*Ej / (hbar_ev * np.pi))  *  np.sqrt(kb_ev*T/(2*np.pi*supcon_gap_Al))
    
    
#     half_x_pi_len=100*1e-9
#     half_y_pi_len=100*1e-9
#     rolen=0.2*1e-6
#     idlen=1/(4*dfq)
#     actlen=0.5*1e-6
#     ramseymeas_len=half_x_pi_len + idlen + half_y_pi_len + rolen + actlen
#     #print('ramseymeas_len',ramseymeas_len)
#     N=Nperbin     # int(binwidth/ramseymeas_len)
#     #maxtime=3.5*1e-3

#     time=btimes #*1e-9  #np.linspace(0,maxtime,num_pts) #ns
#     Ptotal  =np.zeros(len(time))
    
   

#     Peven                  =np.zeros(len(time))
#     Podd                  =np.zeros(len(time))

#     Prob_dec_idt        =np.zeros(len(time))
#     Prob_dec_ro         =np.zeros(len(time))
#     Prob_exc_idt        =np.zeros(len(time))
#     Prob_exc_ro         =np.zeros(len(time))
#     Prob_nonQP_dec_ro    = np.zeros(len(time))
#     Prob_nonQP_exc_ro    = np.zeros(len(time))
#     Prob_nonQP_dec_idt   = np.zeros(len(time))
#     Prob_nonQP_exc_idt   = np.zeros(len(time))
    

#     xqp_v1=np.zeros(len(time))
#     ParRate_v1=np.zeros(len(time))
#     decayrate=np.zeros(len(time))
#     decayrate_qp=np.zeros(len(time))
#     excrate       =np.zeros(len(time))
#     excrate_qp    =np.zeros(len(time))
#     ParRate_v1_nostate_change=np.zeros(len(time))
#     flip = np.zeros(len(time))
   
    
#     dephasingrate  =  np.zeros(len(time))
#     Pflip = np.zeros(len(time))
#     Prob_par =  np.zeros(len(time))
#     Prob_nonQP_dec  =  np.zeros(len(time))
#     Prob_nonQP_exc  =  np.zeros(len(time))
#     Prob_nonQP_dec_ro  =  np.zeros(len(time))
#     Prob_nonQP_exc_ro  =  np.zeros(len(time))
#     Prob_nonQP_dec_idt  =  np.zeros(len(time))
#     Prob_nonQP_exc_idt  =  np.zeros(len(time))
#     Prob_par_nochange  =  np.zeros(len(time))
#     Prob_QP_dec_idt     =  np.zeros(len(time))
#     Prob_QP_exc_idt     =  np.zeros(len(time))
#     Prob_QP_dec_ro      =  np.zeros(len(time))
#     Prob_QP_exc_ro      =  np.zeros(len(time))
#     Prob_par_ramsey     =  np.zeros(len(time))
#     Pflipssf =  np.zeros(len(time))
#     Pflipnossf = np.zeros(len(time))
#     Prob_2_par_swts = np.zeros(len(time))
#     Prob_3_par_swts = np.zeros(len(time))
#     Prob_4_par_swts = np.zeros(len(time))
#     Prob_5_par_swts = np.zeros(len(time))
#     nonQPerror = np.zeros(len(time))
    

#     for i in range(len(time)):
#         ti=time[i]
#         x0_v1= base_Par_Rate / (parRateCoeff_v1 + decayRateCoeff +  (decayRateCoeff * np.exp(-1*hbar_ev*wq/(kb_ev*T))))
    
#         g_v1 = ((2*r0*x0_v1 + s0)*(2*r0*x0_v1 + s0) - s0*s0) / (4*r0)
#         tss_v1= 1 / (2*r0*x0_v1 + s0)
#         rprime_v1= r0*tss_v1*xinduced / (1 + r0*tss_v1*xinduced)
        
#         xqp_v1[i]=((xinduced * (1-rprime_v1))/ (np.exp(ti/tss_v1) - rprime_v1)) + x0_v1 # a/exp(t/tss) + b
#         decayrate_qp[i]               =  decayRateCoeff * xqp_v1[i]
#         decayrate[i]                  =  decayrate_qp[i] + baseDecayRate
#         baseExcrate                   =  baseDecayRate*np.exp(-1*hbar_ev*wq/(kb_ev*T))
#         excrate_qp[i]                 =  decayrate_qp[i]  * np.exp(-1*hbar_ev*wq/(kb_ev*T))
#         excrate[i]                    =  baseExcrate + excrate_qp[i] 
#         ParRate_v1_nostate_change[i]  =  parRateCoeff_v1 * xqp_v1[i]
#         ParRate_v1[i]                 =  ParRate_v1_nostate_change[i] +  decayrate_qp[i] + excrate[i]
#         dephasingrate[i]              =  0
#         basePar                       =  (parRateCoeff_v1 + decayRateCoeff +  (decayRateCoeff * np.exp(-1*hbar_ev*wq/(kb_ev*T))))* x0_v1
#         Prob_par_nochange[i]          =  1 - np.exp(-ramseymeas_len*ParRate_v1_nostate_change[i])
#         Prob_nonQP_dec[i]             =  1 - np.exp(-(ramseymeas_len - actlen) * baseDecayRate)
#         Prob_nonQP_exc[i]             =  1 - np.exp(-(ramseymeas_len - actlen) * baseExcrate)
#         Prob_nonQP_dec_ro[i]          =  1 - np.exp(-(rolen) * baseDecayRate)
#         Prob_nonQP_exc_ro[i]          =  1 - np.exp(-(rolen) * baseExcrate)
#         Prob_nonQP_dec_idt[i]         =  1 - np.exp(-(idlen) * baseDecayRate)
#         Prob_nonQP_exc_idt[i]         =  1 - np.exp(-(idlen) * baseExcrate)
#         Prob_QP_dec_idt[i]            =  1   -   np.exp(-(idlen) * decayrate_qp[i] )
#         Prob_QP_exc_idt[i]            =  1   -   np.exp(-(idlen) * excrate_qp[i])
#         Prob_QP_dec_ro[i]             =  1   -   np.exp(-(rolen) * decayrate_qp[i] )
#         Prob_QP_exc_ro[i]             =  1   -   np.exp(-(rolen) * excrate_qp[i])
#         Prob_dec_idt[i]               =  1   -   np.exp( -idlen*decayrate[i] )
#         Prob_dec_ro[i]                =  1   -   np.exp( -rolen*decayrate[i] )
#         Prob_exc_idt[i]               =  1   -   np.exp( -idlen*excrate[i] )
#         Prob_exc_ro[i]                =  1   -   np.exp( -rolen*excrate[i] )
#         Prob_par[i]                   =  1   -   np.exp( -ramseymeas_len*ParRate_v1[i] )
#         Podd[i]                       =  0.5
#         Peven[i]                      =  0.5
#         Prob_2_par_swts[i] = ( (ParRate_v1[i] * ramseymeas_len)**2 ) * np.exp( -1*ParRate_v1[i] * ramseymeas_len ) / 2
#         Prob_3_par_swts[i] = ( (ParRate_v1[i] * ramseymeas_len)**3 ) * np.exp( -1*ParRate_v1[i] * ramseymeas_len ) / 6
#         Prob_4_par_swts[i] = ( (ParRate_v1[i] * ramseymeas_len)**4 ) * np.exp( -1*ParRate_v1[i] * ramseymeas_len ) / 24
#         Prob_5_par_swts[i] = ( (ParRate_v1[i] * ramseymeas_len)**5 ) * np.exp( -1*ParRate_v1[i] * ramseymeas_len ) / 120
#         Prob_par_ramsey[i]= Prob_par_nochange[i] + (Prob_QP_dec_idt[i] + Prob_QP_exc_idt[i]) + Podd[i]*(Prob_QP_dec_ro[i]) + Peven[i]*(Prob_QP_exc_ro[i]) - Prob_2_par_swts[i] - Prob_3_par_swts[i]  - Prob_4_par_swts[i] - Prob_5_par_swts[i]
#         nonQPerror[i] = Podd[i]*(1 - (Prob_nonQP_dec_idt[i] + Prob_nonQP_exc_idt[i]))*(Prob_nonQP_dec_ro[i] )*(1-Prob_par[i]) + Peven[i]*(1 - (Prob_nonQP_dec_idt[i] + Prob_nonQP_exc_idt[i]))*(Prob_nonQP_exc_ro[i] )*(1-Prob_par[i]) + (Prob_nonQP_dec_idt[i] + Prob_nonQP_exc_idt[i])*(1-( Prob_nonQP_dec_ro[i] + Prob_nonQP_exc_ro[i] ))*0.5
#         flip[i] = Prob_par_ramsey[i]*(1+ssf)/2  +  2*nonQPerror[i]*(1+ssf)/2  +  2*(1 - (Prob_par_ramsey[i] + nonQPerror[i]))*(1-ssf)/2

#         ############################################################################################################
#         ############################ Transition Probabilities ################################################
#         ############ Evg to Evg ######################
#         ###### (no parity & no err), (no parity & err in idt * 0.5), (no parity & err in idt & dec in ro)  
        
        
        
        
        
#         # Pflipssf[i]   = flip[i] *(1+ssf)/2 
#         # Pflipnossf[i] = (1-flip[i])*(1-ssf)/2
#         # Pflip[i]      =  Pflipssf[i] + Pflipnossf[i] 
    
#     return  flip






#, Expectedjumps, binwidth, time, ramseydecayrate, Prob_Par_meas,  P_oddg, P_odde, P_eveng, P_evene, ramseymeas_len, N#


def probsss(SSF, T1, T2base, basePar, Edep, Nperbin, btimes, xqp0):
    
    ssf=SSF
    # print('T1 for likely=',T1)
    # print('btimes',btimes)
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
    
    kb_ev=8.6173303*1e-5 #eV/K
    heV_Hz=4.136*(1e-15) #eV/Hz
    h_bareV_Hz=heV_Hz/(2*np.pi)
    kbeV_K=8.617*(1e-5) #eV/K
    T= 20*1e-3 #20*1e-3 #K
    r0= 0.005/(1e-9) #0.018/1e-9 #Hz
    s0=1e-6/(1e-9) #Hz
    ncp=4*1e24 #m^-3
    V_island= 1000*((1e-6)**3) #2400*((1e-6)**3) #m^3
    phonon_to_qp_eff = 0.57
    base_Par_Rate = basePar    #Hz
    charge_jump_rate = 1.35*1e-3 # Hz
    #print('(1/charge_jump_rate)=', 1/charge_jump_rate)
    Edeposited=Edep*1e-3 #eV
    
    xinduced=Edeposited*phonon_to_qp_eff / (ncp*V_island*supcon_gap_Al)

    decayRateCoeff = ((16*Ej)/(hbar_ev*np.pi)) * np.sqrt(Ec/(8*Ej)) * (supcon_gap_Al/(2*hbar_ev*wq))
    baseT1= T1*1e-6 #s
    T2= T2base * 1e-6 #s
    baseDecayRate= 1/baseT1
    dph_rate= (1/T2) - 0.5*baseDecayRate
    parRateCoeff_v1 = ((16*Ej) / (hbar_ev * np.pi))  *  np.sqrt((kb_ev*T)/(2*np.pi*supcon_gap_Al))
    # 16*(Ej/(h_bareV_Hz*np.pi))*np.sqrt(kbeV_K*T/(2*np.pi*gpeV))
    
    half_x_pi_len=100*1e-9
    half_y_pi_len=100*1e-9
    rolen=0.2*1e-6
    idlen=1/(4*dfq)
    actlen=0.5*1e-6
    ramseymeas_len = half_x_pi_len + idlen + half_y_pi_len + rolen + actlen
    # print('ramseymeas_len',ramseymeas_len)
    #print('ramseymeas_len',ramseymeas_len)
    N=Nperbin     # int(binwidth/ramseymeas_len)
    #maxtime=3.5*1e-3

    time=btimes #*1e-9  #np.linspace(0,maxtime,num_pts) #ns
    #print('times for likely=',time)
    Ptotal  =np.zeros(len(time))
    
   

    Peven                  =np.zeros(len(time))
    Podd                  =np.zeros(len(time))

    Prob_dec_idt        =np.zeros(len(time))
    Prob_dec_ro         =np.zeros(len(time))
    Prob_exc_idt        =np.zeros(len(time))
    Prob_exc_ro         =np.zeros(len(time))
    Prob_nonQP_dec_ro    = np.zeros(len(time))
    Prob_nonQP_exc_ro    = np.zeros(len(time))
    Prob_nonQP_dec_idt   = np.zeros(len(time))
    Prob_nonQP_exc_idt   = np.zeros(len(time))
    

    xqp_v1=np.zeros(len(time))
    ParRate_v1=np.zeros(len(time))
    decayrate=np.zeros(len(time))
    decayrate_qp=np.zeros(len(time))
    excrate       =np.zeros(len(time))
    excrate_qp    =np.zeros(len(time))
    ParRate_v1_nostate_change=np.zeros(len(time))
    flip = np.zeros(len(time))
   
    
    dephasingrate  =  np.zeros(len(time))
    Pflip = np.zeros(len(time))
    Prob_par =  np.zeros(len(time))
    Prob_nonQP_dec  =  np.zeros(len(time))
    Prob_nonQP_exc  =  np.zeros(len(time))
    Prob_nonQP_dec_ro  =  np.zeros(len(time))
    Prob_nonQP_exc_ro  =  np.zeros(len(time))
    Prob_nonQP_dec_idt  =  np.zeros(len(time))
    Prob_nonQP_exc_idt  =  np.zeros(len(time))
    Prob_par_nochange  =  np.zeros(len(time))
    Prob_QP_dec_idt     =  np.zeros(len(time))
    Prob_QP_exc_idt     =  np.zeros(len(time))
    Prob_QP_dec_ro      =  np.zeros(len(time))
    Prob_QP_exc_ro      =  np.zeros(len(time))
    Prob_par_ramsey     =  np.zeros(len(time))
    Pflipssf =  np.zeros(len(time))
    Pflipnossf = np.zeros(len(time))
    Prob_2_par_swts = np.zeros(len(time))
    Prob_3_par_swts = np.zeros(len(time))
    Prob_4_par_swts = np.zeros(len(time))
    Prob_5_par_swts = np.zeros(len(time))
    nonQPerror = np.zeros(len(time))
    nonQPerror_odd = np.zeros(len(time))
    nonQPerror_even = np.zeros(len(time))
    Prob_ng_change_after_t = np.zeros(len(time))
    Prob_dph  = np.zeros(len(time))
    flip_from_ng = np.zeros(len(time))
    flip_from_dph = np.zeros(len(time))
    flipwithng = np.zeros(len(time))
    flipwithdph = np.zeros(len(time))

    for i in range(len(time)):
        ti=time[i]
        x0_v1= xqp0 # base_Par_Rate / (parRateCoeff_v1 + decayRateCoeff +  (decayRateCoeff * np.exp(-1*hbar_ev*wq/(kb_ev*T))))
        base_Par_Rate = (parRateCoeff_v1 + decayRateCoeff +  (decayRateCoeff * np.exp(-1*hbar_ev*wq/(kb_ev*T)))) * x0_v1
        g_v1 = (((2*r0*x0_v1 + s0)**2) - s0*s0) / (4*r0)
        tss_v1= 1 / (2*r0*x0_v1 + s0)
        rprime_v1= r0*tss_v1*xinduced / (1 + r0*tss_v1*xinduced)
        
        xqp_v1[i]=((xinduced * (1-rprime_v1))/ (np.exp(ti/tss_v1) - rprime_v1)) + x0_v1 # a/exp(t/tss) + b
        decayrate_qp[i]               =  decayRateCoeff * xqp_v1[i]
        decayrate[i]                  =  decayrate_qp[i] + baseDecayRate
        baseExcrate                   =  baseDecayRate*     np.exp((-1*hbar_ev*wq)/(kb_ev*T))
        excrate_qp[i]                 =  decayrate_qp[i]  * np.exp((-1*hbar_ev*wq)/(kb_ev*T))
        excrate[i]                    =  baseExcrate + excrate_qp[i] 
        ParRate_v1_nostate_change[i]  =  parRateCoeff_v1 * xqp_v1[i]
        ParRate_v1[i]                 =  ParRate_v1_nostate_change[i] +  decayrate_qp[i] + excrate[i]
        #dephasingrate[i]              =  0
        basePar                       =  (parRateCoeff_v1 + decayRateCoeff +  (decayRateCoeff * np.exp(-1*hbar_ev*wq/(kb_ev*T))))* x0_v1
        Prob_par_nochange[i]          =  1 - np.exp(-ramseymeas_len*ParRate_v1_nostate_change[i])
        Prob_nonQP_dec[i]             =  1 - np.exp(-(ramseymeas_len - actlen) * baseDecayRate)
        Prob_nonQP_exc[i]             =  1 - np.exp(-(ramseymeas_len - actlen) * baseExcrate)
        Prob_nonQP_dec_ro[i]          =  1 - np.exp(-(rolen) * baseDecayRate)
        Prob_nonQP_exc_ro[i]          =  1 - np.exp(-(rolen) * baseExcrate)
        Prob_nonQP_dec_idt[i]         =  1 - np.exp(-(idlen) * baseDecayRate)
        Prob_nonQP_exc_idt[i]         =  1 - np.exp(-(idlen) * baseExcrate)
        Prob_QP_dec_idt[i]            =  1   -   np.exp(-(idlen) * decayrate_qp[i] )
        Prob_QP_exc_idt[i]            =  1   -   np.exp(-(idlen) * excrate_qp[i])
        Prob_QP_dec_ro[i]             =  1   -   np.exp(-(rolen) * decayrate_qp[i] )
        Prob_QP_exc_ro[i]             =  1   -   np.exp(-(rolen) * excrate_qp[i])
        Prob_dec_idt[i]               =  1   -   np.exp( -idlen*decayrate[i] )
        Prob_dec_ro[i]                =  1   -   np.exp( -rolen*decayrate[i] )
        Prob_exc_idt[i]               =  1   -   np.exp( -idlen*excrate[i] )
        Prob_exc_ro[i]                =  1   -   np.exp( -rolen*excrate[i] )
        Prob_par[i]                   =  1   -   np.exp( -ramseymeas_len*ParRate_v1[i] )
        Prob_dph[i]                   =  1   -   np.exp(-ramseymeas_len*dph_rate)
        Podd[i]                       =  0.5
        Peven[i]                      =  0.5

        # flipcauses= Prob_par_nochange[i] + Prob_QP_dec_idt[i] + Prob_QP_exc_idt[i] + Prob_nonQP_dec_idt[i] + Prob_nonQP_exc_idt[i] + Prob_nonQP_dec_ro[i] + Prob_nonQP_exc_ro[i] + Prob_QP_dec_ro[i] + Prob_QP_exc_ro[i] #+ Prob_dph[i]*diff 

        qpflipcauses= Prob_par_nochange[i] + 0.5*(Prob_QP_dec_idt[i] + Prob_QP_exc_idt[i]) + (1 - (Prob_QP_dec_idt[i] + Prob_QP_exc_idt[i] + Prob_nonQP_dec_idt[i] + Prob_nonQP_exc_idt[i]))*(Podd[i]*(  Prob_QP_dec_ro[i]) + Peven[i]*(Prob_QP_exc_ro[i] )) #+ Prob_dph[i]#*diff
        nonqpflipcauses=   0.5*( Prob_nonQP_dec_idt[i] + Prob_nonQP_exc_idt[i]) + (1 - (Prob_QP_dec_idt[i] + Prob_QP_exc_idt[i] + Prob_nonQP_dec_idt[i] + Prob_nonQP_exc_idt[i]))*(Podd[i]*(Prob_nonQP_dec_ro[i]) + Peven[i]*(Prob_nonQP_exc_ro[i] ))
        # flipcauses= Prob_par_nochange[i] + Prob_QP_dec_idt[i] + Prob_QP_exc_idt[i] + Prob_nonQP_dec_idt[i] + Prob_nonQP_exc_idt[i] + Prob_nonQP_dec_ro[i] + Prob_nonQP_exc_ro[i] + Prob_QP_dec_ro[i] + Prob_QP_exc_ro[i]
        flipcauses = qpflipcauses + 2*nonqpflipcauses#(2*(1-nonqpflipcauses)*nonqpflipcauses*(1-nonqpflipcauses) + (1-nonqpflipcauses)*(1-nonqpflipcauses)*nonqpflipcauses  + nonqpflipcauses*(1-nonqpflipcauses)*(1-nonqpflipcauses))
        
        # trusum = truflip + trunoflip
        flip= (ssf)*(flipcauses) + (1-flipcauses)*(ssf*(1-ssf)*ssf*2 + (1-ssf)*(1-ssf)*ssf + ssf*(1-ssf)*(1-ssf) )
        
       
        flipwithng[i] = flip 
    
    return  flipwithng #Pflip, ParRate_v1, flip, xqp_v1












###################################################################################################################
###################################################################################################################


def countjumps(output):
    
    flips=0
    laststate=output[0]
    for i in range(len(output)):
        if output[i]!=laststate:
            flips=flips+1
        laststate=output[i]
    return flips


def movmed(output, size):
    i=0
    filtered=np.zeros((len(output)-size+1))

    while i<len(output)-size+1:
        filtered[i]=np.median(output[i:i+size])
        i=i+1
    return filtered

###################################################################################################################
###################################################################################################################
def movmedFilt_Erecon(params):
    filename=params[0]
    index=params[1]
    file=np.load(filename)
    rotimeSeries=file['ro'][index]
    rotimeSeries_time=file['ro_time'][index]
    binwidth=0.5*1e-3
    binwidth025=0.25*1e-3
    binwidth01=0.1*1e-3
    binwidth005=0.05*1e-3
    windows=[0,50,100,200]
    filteredTimeseries=[]
    filteredwvfms_per_window=[]
    likEs_per_window = []
    filtssf=1
    filtbaseT1=1
    timestep=1.1*1e-6
    filtbasePar=0.1
    ssf=1
    baseT1=1
    basePar=0.1
    
    for window in tqdm(windows):
        print(f'on window={window}')
        movmedfilt = movmed(rotimeSeries, window)
        filteredTimeseries.append(movmedfilt)
        
        #wv, numrepsInBin, btimes    =Makewaveform(rotimeSeries, rotimeSeries_time, binwidth, timestep)
        wv025, numrepsInBin025, btimes025    =Makewaveform(movmedfilt, rotimeSeries_time[:len(movmedfilt)], binwidth025, timestep)
        #wv01, numrepsInBin01, btimes01    =Makewaveform(rotimeSeries, rotimeSeries_time, binwidth01, timestep)
        wv005, numrepsInBin005, btimes005    =Makewaveform(movmedfilt, rotimeSeries_time[:len(movmedfilt)], binwidth005, timestep)
        filteredwvfms= [wv025, wv005]
        filteredwvfms_per_window.append(filteredwvfms)

        #mostprobenergydep = likelihoodEdep(ssf, basePar,  baseT1,   wv, btimes*1e-9, numrepsInBin)
        mostprobenergydep025 = likelihoodEdep(filtssf, filtbasePar,  filtbaseT1,   wv025, btimes025*1e-9, numrepsInBin025)
        #mostprobenergydep01 = likelihoodEdep(filtssf, filtbasePar,  filtbaseT1,   wv01, btimes01*1e-9, numrepsInBin01)
        mostprobenergydep005 = likelihoodEdep(filtssf, filtbasePar,  filtbaseT1,   wv005, btimes005*1e-9, numrepsInBin005)
        likEs = [mostprobenergydep025,  mostprobenergydep005]
        likEs_per_window.append(likEs)

    # mostprobenergydep025 = likelihoodEdep(ssf, basePar,  baseT1,   file['roseqwv25'][index] , btimes025*1e-9, numrepsInBin025)
    # mostprobenergydep005 = likelihoodEdep(ssf, basePar,  baseT1,   file['roseqwv005'][index], btimes005*1e-9, numrepsInBin005)

    return filteredTimeseries, filteredwvfms_per_window, likEs_per_window

###################################################################################################################
###################################################################################################################

################################
########Waveform
def MakewaveformStateVector(seq, seqT, binwidth, timestep):
    numrepsInBin=int(binwidth/timestep)
    numBins=int(len(seq)/numrepsInBin)
    seq=seq[:int(numBins * numrepsInBin)].reshape(numBins, numrepsInBin)
    seqT=seqT[:int(numBins * numrepsInBin)].reshape(numBins, numrepsInBin)
    wvfm=np.zeros(numBins)
    btimes=np.zeros(numBins)
    for i in range(numBins):
        wvfm[i]=countjumps(seq[i])
        #wvfm[i]=jumpsWafm
        btimes[i]=np.median(seqT[i])
    return wvfm, numrepsInBin, btimes 
def Makewaveform(seq, seqT, binwidth, ramseymeas_len):
    numrepsInBin= 100 #int(binwidth/ramseymeas_len)
    numBins= 30  #int(len(seq)/numrepsInBin)
    # seq=seq[:int(numBins * numrepsInBin)].reshape(numBins, numrepsInBin)
    # seqT=seqT[:int(numBins * numrepsInBin)].reshape(numBins, numrepsInBin)

    flips=np.zeros((len(seq)-1))
    laststate=seq[0]
    for i in range(1,len(seq)):
        if seq[i]!=seq[i-1]:
            flips[i-1]=1
        # laststate=seq[i]

    
    wvfm=np.zeros(numBins)
    btimes=np.zeros(numBins)
    for i in range(numBins):
        if len(flips[i*100 : (i+1)*100])==100:
            wvfm[i]= sum(flips[i*100 : (i+1)*100])
        else:
            wvfm[i]= sum(flips[i*100 : ])
        #wvfm[i]=jumpsWafm
        btimes[i]=np.median(seqT[i*100 : (i+1)*100])
    return wvfm, numrepsInBin, btimes    


def Makewaveform_from_ones(seq, seqT, binwidth, ramseymeas_len):
    numrepsInBin=int(binwidth/ramseymeas_len)
    numBins=int(len(seq)/numrepsInBin)
    # print('numBins',numBins)
    # print('len(seq)',len(seq))
    seq=seq[:int(numBins * numrepsInBin)].reshape(numBins, numrepsInBin)
    seqT=seqT[:int(numBins * numrepsInBin)].reshape(numBins, numrepsInBin)
    wvfm=np.zeros(numBins)
    btimes=np.zeros(numBins)
    for i in range(numBins):
        wvfm[i]=sum(seq[i])
        #wvfm[i]=jumpsWafm
        btimes[i]=np.median(seqT[i])
    return wvfm, numrepsInBin, btimes    
###################################################################################################################
###################################################################################################################f

############likelihood
def likelihoodEdep(ssf, basePar,  baseT1,  baseT2,  waveform, bintimes, Nperbin, xqp0):
    #print('starting likelihood')
    start_time=tm.time()
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
    half_x_pi_len=100*1e-9
    half_y_pi_len=100*1e-9
    rolen=0.2*1e-6
    idlen=1/(4*dfq)
    actlen=0.5*1e-6
    ramseymeas_len=half_x_pi_len + idlen + half_y_pi_len + rolen + actlen
    #binwidth=bwidth
    #binwidth=0.5*1e-3
    N=Nperbin-1 #int(binwidth/ramseymeas_len)

    
    btimes= bintimes*1e-9#np.linspace(0,5*1e-3, 10)
    # print('len(btimes)',len(btimes))
    # print('len(waveform)',len(waveform))
    energydeps= np.linspace(1e-30,10000,1001)#np.array([0,25,50,100,150,])#
    #energydeps[0]=0.05
    likes=np.zeros(len(energydeps))
    for i in range(len(energydeps)):
        
        # Edeposited=energydeps[i]
        #, Expectedjumps,  time, ramseydecayrate, Prob_Par_meas,  P_oddg, P_odde, P_eveng, P_evene, ramseymeas_len, N
        Pjump = probsss(ssf, baseT1, baseT2, basePar, energydeps[i], Nperbin, btimes, xqp0)
        
        #N=numrepsInBin
        
        #waveform, numrepsInBin=Makewaveform(roseq,binwidth, ramseymeas_len)
        Prob_si_jps=np.zeros(len(waveform))
        #N=numrepsInBin
        log_Prob_si_jps=np.zeros(len(waveform))
        
        for j in range(len(waveform)):
            #print(f'like index={j}')
            ###Prob that we see si jumps at ti
            si=waveform[j]
            Combi_N_si=math.factorial(int(N)) / (math.factorial(int(si)) * math.factorial(int(N-si)))
            Prob_si_jps[j]=Combi_N_si*(pow(Pjump[j], int(si)))*(pow((1-Pjump[j]),(int(N-si))))###Prob that we see si jumps at ti
            log_Prob_si_jps[j]=np.log(Prob_si_jps[j])
        likelihood=sum(log_Prob_si_jps)#reduce(mul, Prob_si_jps)#
        likes[i]=likelihood
    mostprobenergydep=energydeps[np.argmax(likes)]
    #print('finishing likelihood')
    timetaken=tm.time() - start_time
    #print('timetaken',str(timetaken))

    
    return mostprobenergydep#, likes, Pjump

###################################################################################################################
###################################################################################################################


###################################################################################################################f

############likelihood
def filtlikelihoodEdep(para):
    # index= para[0]
    waveform = para[0]
    bintimes = para[1]
    Nperbin = para[2]
    ssf = para[3]
    T1 = para[4]
    xqp0 = para[5]
    avgwvfm = para[6]
    wvfmProb = para[7]
    
    #print('starting likelihood')
    start_time=tm.time()
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
    
    kb_ev=8.6173303*1e-5 #eV/K
    heV_Hz=4.136*(1e-15) #eV/Hz
    h_bareV_Hz=heV_Hz/(2*np.pi)
    kbeV_K=8.617*(1e-5) #eV/K
    T= 20*1e-3 #20*1e-3 #K
    r0= 0.005/(1e-9) #0.018/1e-9 #Hz
    s0=1e-6/(1e-9) #Hz
    ncp=4*1e24 #m^-3
    V_island= 1000*((1e-6)**3) #2400*((1e-6)**3) #m^3
    phonon_to_qp_eff = 0.57
    # base_Par_Rate = basePar    #Hz
    
    half_x_pi_len=100*1e-9
    half_y_pi_len=100*1e-9
    rolen=0.2*1e-6
    idlen=1/(4*dfq)
    actlen=0.5*1e-6
    ramseymeas_len=half_x_pi_len + idlen + half_y_pi_len + rolen + actlen
    #binwidth=bwidth
    #binwidth=0.5*1e-3
    N=Nperbin-1 #int(binwidth/ramseymeas_len)

    
    btimes= bintimes*1e-9#np.linspace(0,5*1e-3, 10)
    # print('len(btimes)',len(btimes))
    # print('len(waveform)',len(waveform))
    energydeps= np.linspace(1e-30,10000,1001)#10000,1001)
    #energydeps[0]=0.05
    likes=np.zeros(len(energydeps))
    baseT1=   T1 # 1e20 # 100
    baseT2=   T1 # 1e20 # 100
    basePar=0
    # ssf= 0.998 # 1
    # xqp0= 1e-9 # 1e-20 #1e-8 #1e-15
    
    for i in range(len(energydeps)):
        
        # Edeposited=energydeps[i]
        #, Expectedjumps,  time, ramseydecayrate, Prob_Par_meas,  P_oddg, P_odde, P_eveng, P_evene, ramseymeas_len, N
        Pjump = probsss(ssf, baseT1, baseT2, basePar, energydeps[i], Nperbin, btimes, xqp0)
        Pjumpfilt = Pjump - wvfmProb 
        # print('Pjump',Pjump)
        #N=numrepsInBin
        
        #waveform, numrepsInBin=Makewaveform(roseq,binwidth, ramseymeas_len)
        Prob_si_jps=np.zeros(len(waveform))
        #N=numrepsInBin
        log_Prob_si_jps=np.zeros(len(waveform))
        
        for j in range(len(waveform)):
            #print(f'like index={j}')
            ###Prob that we see si jumps at ti
            si=waveform[j]
            Combi_N_si=math.factorial(int(N)) / (math.factorial(int(si)) * math.factorial(int(N-si)))
            Prob_si_jps[j]=Combi_N_si*(pow(Pjump[j], int(si)))*(pow((1-Pjump[j]),(int(N-si))))###Prob that we see si jumps at ti
            log_Prob_si_jps[j]=np.log(Prob_si_jps[j])
        likelihood=sum(log_Prob_si_jps)#reduce(mul, Prob_si_jps)#
        likes[i]=likelihood
    mostprobenergydep=energydeps[np.argmax(likes)]
    #print('finishing likelihood')
    timetaken=tm.time() - start_time
    #print('timetaken',str(timetaken))

    
    return mostprobenergydep, Pjumpfilt#, likes, Pjump

###################################################################################################################
###################################################################################################################



def likeanalyse(params):
    
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
    delta_ng=0
    ng=0
    delta_f10=dfq*np.cos(2*np.pi*(ng + delta_ng))
    
    Pargatetimes=[x_2len, x_2len+idlewait, x_2len+idlewait+y_2len, x_2len+idlewait+y_2len+measurelen, x_2len+idlewait+y_2len+measurelen+active_resetlen]
    X2time=x_2len
    Y2time=x_2len+idlewait+y_2len
    idtime=x_2len+idlewait
    Actime=x_2len+idlewait+y_2len+measurelen+active_resetlen
    ParSeqlen=x_2len+idlelen+y_2len+measurelen+active_resetlen
    
    ramseymeas_len=4.4*1e-6
    filename=params[0]
    
    index=params[1]
    ssf= params[2]
    basePar= params[3]
    baseT1 = params[4]
    baseT2 = params[5]
    
    file=np.load(filename)
    #seq=file['resro'][index]
    print(f'Working on index {index}')
    ro = file['ro'][index]
    ro_time = file['ro_time'][index]

    
def PSD(para):
    ro=para[0]
    rot=np.array(para[1])*1e-9
    dt=rot[1]-rot[0]
    n=len(rot)
    # theFFT= np.fft.fft(ro, n) # Computes the FFT of the noisy signal: The input is a time series vector and the output is a vector of compplex numbers in the form of amplitudes and phases. The amplitdue represents how srongly a frequency component contibutes to the signal and the phase represents how much of a sine or cosine that frequency componenet behaves. I don't understand the phases part.
    f, Pxx_den = signal.periodogram(ro, 1/dt)  #theFFT * np.conj(theFFT) / n #The PSD is gotten from the product of the fft and its conjugate, divided by n
    ## So the PSD is a vector that contains the square of the amplitude of each frequency component. So it's a vector of the power of each frequency component
    # freqs= np.fft.fftfreq(n, d=dt) #np.linspace(0, 1/dt, n) # The possible frequency components. From no oscillation(0) to as fast as complete oscilation within dt(1/dt)
    L=np.arange(1, np.floor(n/2), dtype='int')

    return Pxx_den, f
    





def mean_std_calc(params):
    filename=params[0]
    index=params[1]
    file      =np.load(filename)
    ti        =np.linspace(0, 5*1e-3, int(5*1e-3/(50*1e-9)))
    binwidth  =0.5*1e-3
    decseq    =file['dec'][index]
    dec_idtseq=file['nums'][index][-2] + file['nums'][index][-4]
    parseq    =file['parity'][index]
    exc_idtseq=file['nums'][index][-3]
    exc_ro    =file['nums'][index][-1]
    excseq    =file['nums'][index][-3] + file['exc_ro'][index][-1]
    roseq     =file['parDecReadout'][index] 
    roseqT    =file['parDecReadoutTime'][index]
    ev0seq    =file['nums'][index][10]
    ev1seq    =file['nums'][index][11]
    od0seq    =file['nums'][index][14]
    od1seq    =file['nums'][index][15]

    decseqwvfm     , numPerbin, btimes = Makewaveform_from_ones(decseq    , ti, binwidth, ramseylen)
    dec_idtseqwvfm , numPerbin, btimes = Makewaveform_from_ones(dec_idtseq, ti, binwidth, ramseylen)
    parseqwvfm     , numPerbin, btimes = Makewaveform_from_ones(parseq    , ti, binwidth, ramseylen)
    exc_idtseqwvfm , numPerbin, btimes = Makewaveform_from_ones(exc_idtseq, ti, binwidth, ramseylen)
    exc_rowvfm     , numPerbin, btimes = Makewaveform_from_ones(exc_ro    , ti, binwidth, ramseylen)
    excseqwvfm     , numPerbin, btimes = Makewaveform_from_ones(excseq    , ti, binwidth, ramseylen)
    roseqTwvfm     , numPerbin, btimes = Makewaveform_from_ones(roseqT    , ti, binwidth, ramseylen)
    ev0seqwvfm     , numPerbin, btimes = Makewaveform_from_ones(ev0seq    , ti, binwidth, ramseylen)
    ev1seqwvfm     , numPerbin, btimes = Makewaveform_from_ones(ev1seq    , ti, binwidth, ramseylen)
    od0seqwvfm     , numPerbin, btimes = Makewaveform_from_ones(od0seq    , ti, binwidth, ramseylen)
    od1seqwvfm     , numPerbin, btimes = Makewaveform_from_ones(od1seq    , ti, binwidth, ramseylen)
    roseqwvfm      , numPerbin, btimes = Makewaveform(roseq               , roseqT, binwidth, ramseylen)

    return decseqwvfm    ,dec_idtseqwvfm,parseqwvfm    ,exc_idtseqwvfm,exc_rowvfm    ,excseqwvfm    ,roseqTwvfm    ,ev0seqwvfm    ,ev1seqwvfm    ,od0seqwvfm    ,od1seqwvfm    ,roseqwvfm     

    
    
    
    
    
    
    
    