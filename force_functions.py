from tri_functions import *


def get_F(vc_CW,vc_CCW,tA,t_edgedirCW,t_edgedirCCW,tP,tKappa,tGamma,tLambda_CCW,tLambda_CW,tA0,inv_viscosity):
    tdA_dv = (vc_CW-vc_CCW)/2
    tdA_dv = np.dstack((-tdA_dv[:,:,1],+tdA_dv[:,:,0]))
    tdEA_dA = tKappa*(tA-tA0)
    tdEA_dv = np.expand_dims(tdEA_dA,2)*tdA_dv

    tdP_dv = (t_edgedirCW + t_edgedirCCW)
    tdEP_dP = tGamma*tP
    tdEP_dv = np.expand_dims(tdEP_dP,2)*tdP_dv

    tdEl_dv = np.expand_dims(tLambda_CCW,2)*t_edgedirCCW + np.expand_dims(tLambda_CW,2)*t_edgedirCW

    tdE_dv = tdEA_dv + tdEP_dv + tdEl_dv
    tF = -tdE_dv
    F = tF[:,0] + tF[:,1] + tF[:,2]
    return F*inv_viscosity

@jit(nopython=True)
def get_FL(Lx,Ly,vn,t_edgedirCCW,tP,tGamma,tLambda_CCW,Kappa,A,A0,inv_viscosity):
    L = np.array((Lx,Ly))
    A=A.ravel()
    dEA_dL = np.expand_dims(Kappa*(A-A0)*A,1)/L
    dEA_dL = np.array((dEA_dL[:,0].sum(),dEA_dL[:,1].sum()))
    dEl_dL = np.expand_dims(tGamma*tP + tLambda_CCW,2)*(vn*t_edgedirCCW/L)
    dEl_dL = np.array((dEl_dL[:,:,0].sum(),dEl_dL[:,:,1].sum()))
    FL = -(dEA_dL + dEl_dL)
    return FL*inv_viscosity