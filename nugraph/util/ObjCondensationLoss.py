epsilon=1e-8
w1=100
w2=100
w3=10#100
w4=10#500
w5=0.1
q0=0.1
n_constituents = 383

def create_particle_loss_dict(truth, pred):
    '''
    input features as
    B x V x F
    with F = [pt, eta, phi]
    
    pred as
    B x V x F'
    with F' = [pt, eta, phi, beta, x1, ..., xn]  
    - (x1, ..., xn) clustering space 
    
    truth as 
    B x V x F''
    with F'' = [true_pt, true_eta, true_phi, jet_tag] 
    - jet_tag (t_mask) (=0 if noise, >0 jet constituent)
    
    all outputs in B x V x 1/F form except
    n_active: B x 1
    '''
    outdict={}
    
    #make it all lists
    outdict['t_mask'] =  truth[:,:,3:4] # B x V x 1
    outdict['t_pos']  =  truth[:,:,0:3] # B x V x 3

    outdict['p_beta']    =  pred[:,:,3:4] # B x V x 1
    outdict['p_ccoords'] =  pred[:,:,4:] #latent space
    outdict['p_pos']     =  pred[:,:,0:3]

    outdict['t_mask'] = torch.squeeze(outdict['t_mask']) -1*(torch.squeeze(outdict['t_pos'].sum(dim=2))==0)
    outdict['t_mask'] = outdict['t_mask'][:, :, None]
    
    flattened = torch.squeeze(outdict['t_mask']) # B x V
    flattened = (flattened==0).sum(dim=1)
    outdict['n_noise'] = flattened[:, None]
    outdict['n_total'] = 383
    outdict['n_nonoise'] = outdict['n_total'] - outdict['n_noise']    
    return outdict

def calculate_charge(beta, q_min):
    #don't let gradient go to nan
    beta = torch.clamp(beta,0+epsilon,1-epsilon) 
    return atanh_torch(beta)*atanh_torch(beta)+q_min

def atanh_torch(x):
    return 0.5*torch.log((1+x)*1./(1-x))

def sub_object_condensation_loss(d, q_min, Ntotal=n_constituents):
    
    q = calculate_charge(d['p_beta'], q_min) # B x V x 1

    L_att  = torch.zeros_like(q[:,0,0])
    L_rep  = torch.zeros_like(q[:,0,0])
    L_beta = torch.zeros_like(q[:,0,0])
    L_pos  = torch.zeros_like(q[:,0,0])
    Nobj   = torch.zeros_like(q[:,0,0])

    isobj=[]
    alpha=[]

    for k in range(13):  # maximum number of objects: 13-1                                                                                                                                             
        if k==0:
            continue

        Mki     = 1*(torch.abs(d['t_mask']-float(k))<0.1)
        iobj_k  = torch.max(Mki, 1)[0] # B x 1                                                                                 
        Nobj   += torch.squeeze(iobj_k,1)

        kalpha  = torch.max(Mki*d['p_beta'], 1)
        kalpha  = kalpha[1]

        isobj.append(iobj_k)
        alpha.append(kalpha)

        a = kalpha.repeat(1, d['p_ccoords'].shape[2])
        a = a[:, None, :]
        x_kalpha = torch.gather(d['p_ccoords'], 1, a)                                                                                                                        
        
        # Attractive and repulsive terms
        a = kalpha[:, None, :]
        q_kalpha = torch.gather(q, 1, a)
        distance = torch.sqrt(torch.sum( (x_kalpha-d['p_ccoords'])**2,dim=2, keepdims=True)+epsilon) #B x V x 1                                                                               
        F_att    = q_kalpha * q * distance**2 * Mki
        F_rep    = q_kalpha * q * torch.max((1. - distance), torch.zeros_like(distance)) * (1. - Mki)
        L_att  += torch.squeeze(iobj_k * torch.sum(F_att, 1), 1)/(Ntotal)
        L_rep  += torch.squeeze(iobj_k * torch.sum(F_rep, 1), 1)/(Ntotal)
        
        # Beta term
        a = kalpha[:, None, :]
        beta_kalpha = torch.gather(d['p_beta'],1, a)
        L_beta     += torch.squeeze(iobj_k * torch.squeeze((1-beta_kalpha), 2), 1)
        
        # Predictions term
        num = p_Loss(d, 't_pos', 'p_pos') 
        den = torch.sum(torch.squeeze(q*Mki), 1)*torch.squeeze(iobj_k) + (1-torch.squeeze(iobj_k))
        #print(num.shape)         # B x V
        L_pos  += torch.sum(torch.squeeze(q*Mki)*num, 1)*torch.squeeze(iobj_k)*1./den#torch.sum(torch.squeeze(q*Mki), 1)

    #print(Nobj)   
    L_beta/=Nobj
    L_pos /=Nobj
    #L_att/=Nobj
    #L_rep/=Nobj
    
    # Noise term
    L_suppnoise = torch.squeeze(torch.sum( (torch.abs(d['t_mask'])<0.1)*d['p_beta'] , 1) / (d['n_noise'] + epsilon), dim=1)
    
    # Prediction term (moved up)
    #num   = p_Loss(d, 't_pos', 'p_pos') #B x V
    #print(torch.sum(torch.squeeze((d['t_mask']>0.1)*q), 1))
    #L_pos = torch.sum(torch.squeeze((d['t_mask']>0.1)*q)*num, 1)/torch.sum(torch.squeeze((d['t_mask']>0.1)*q), 1)

    # Average 
    reploss            = torch.mean(L_rep)
    attloss            = torch.mean(L_att)
    betaloss           = torch.mean(L_beta)
    supress_noise_loss = torch.mean(L_suppnoise)
    posloss            = torch.mean(L_pos)
    
    return reploss, attloss, betaloss, supress_noise_loss, posloss, Nobj, isobj, alpha
    
def p_Loss(d, tdict, pdict):
    t    = d[tdict]
    p    = d[pdict]
    phii = p[:, :, 1]
    phij = t[:, :, 1]
    etai = p[:, :, 0]
    etaj = t[:, :, 0]
    pti  = p[:, :, 2]
    ptj  = t[:, :, 2]
    delta_phi  = torch.abs(phii-phij)*(1.*(torch.abs(phii-phij)<=math.pi))
    delta_phi += (2*math.pi-torch.abs(phii-phij))*(1.*(torch.abs(phii-phij)>math.pi))*(1.*(torch.abs(phii-phij)<=2*math.pi))
    delta_phi += torch.abs(phii-phij)*(1.*(torch.abs(phii-phij)>2*math.pi))
    delta_eta = torch.abs(etai-etaj)
    delta_pt = torch.abs(pti-ptj)
    loss     = delta_phi+delta_eta+delta_pt
    #print("Dpt: %f, Deta %f, Dphi %f"%(torch.mean(delta_pt).item(), torch.mean(delta_eta).item(), torch.mean(delta_phi).item()))
    return loss

def object_condensation_loss(truth, pred):
    d = create_particle_loss_dict(truth,pred)

    reploss, attloss, betaloss, supress_noise_loss, posloss, Nobj, isobj, alpha = sub_object_condensation_loss(d,q_min=q0)
                                                                                                                                                   
    loss = w1*reploss + w2*attloss + w3*betaloss + w4*supress_noise_loss + w5*posloss #0*supress_noise at the beginning                                                                                  
    #print(w1*reploss.item(), w2*attloss.item(), w3*betaloss.item(), w4*supress_noise_loss.item(), w5*posloss.item())
    return loss

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class ObjCondensation_Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(ObjCondensation_Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, true):
        return object_condensation_loss(true, input)

################################################################################### 
