import torch

def loss_tdCoxSNN_PyTorch(y_pred, y_true):

    y_true = y_true.type(torch.float32) # tstart, tstop, event
    y_pred = y_pred.type(torch.float32)
    y_pred = torch.flatten(y_pred)

    n_sample = y_pred.shape[0]
    
    time0 = torch.flatten(y_true[:,0])
    time = torch.flatten(y_true[:,1])
    event = torch.flatten(y_true[:,2])

    sort_index = torch.argsort(time)
    time0 = time0[sort_index] # ascending order
    time = time[sort_index]
    event = event[sort_index]
    y_pred = y_pred[sort_index]
    
    if torch.sum(event) == 0.: 
        return torch.sum(event)
    else:
        time_event = time * event
        eventtime,tie_count = torch.unique(torch.masked_select(time_event,torch.gt(time_event,0)),
                                           return_counts=True)

        at_risk_index = ((time0 < eventtime[:,None]) & (time >= eventtime[:,None])).type(torch.float32)
        event_index = (time == eventtime[:,None]).type(torch.float32)
        
        # haz = exp(risk)
        tie_haz = torch.mm(event_index, (torch.exp(torch.clip(y_pred,-20,20))*event).unsqueeze(1))
        
        tie_risk = torch.mm(event_index, (y_pred*event).unsqueeze(1))

        cum_haz = torch.mm(at_risk_index, (torch.exp(torch.clip(y_pred,-20,20))).unsqueeze(1))

        mask_tie_haz = torch.arange(torch.max(tie_count)) < (tie_count[:,None]-1)
        mask_tie_risk = torch.arange(torch.max(tie_count)) < (tie_count[:,None])
        out0 = torch.zeros(mask_tie_haz.size(),dtype = torch.float32)
        out1 = (torch.cumsum(torch.ones(mask_tie_haz.size()),1)).type(torch.float32)
        out = torch.where(mask_tie_haz, out1, out0)
        tie_count_matrix = tie_count.type(torch.float32).unsqueeze(1)

        J = torch.divide(out,tie_count_matrix)
        efron_correction = J*tie_haz
        log_sum_haz = torch.where(mask_tie_risk,torch.ones(mask_tie_risk.size(),dtype = torch.float32),out0)*cum_haz
        log_sum_haz = torch.where(mask_tie_risk,torch.log(log_sum_haz-efron_correction+1e-15),out0)
        log_sum_haz = torch.sum(log_sum_haz)
        log_lik = torch.sum(tie_risk)-log_sum_haz
        
        return torch.negative(log_lik)/n_sample
