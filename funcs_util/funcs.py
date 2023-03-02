import pandas as pd
import numpy as np

def baseline_hazard(df_input):
    # baseline_hazard function with Efron correction
    # df_input: n_sample*(time0,time1,event,risk_score)
    # output: n_event_time*(time,h)
    cols = ["time0","time1","event","risk_score"]
    df_input = pd.DataFrame(df_input, columns = cols)
    df_input[cols] = df_input[cols].apply(pd.to_numeric, errors='coerce', axis=1)

    event_time,ties_vector = np.unique(df_input['time1'][df_input['event'] == 1],return_counts=True)
    n_event_time = len(event_time)
    Max_tie = max(ties_vector)

    rs_matrix = np.tile(np.array([df_input['risk_score'].to_numpy()]).transpose(),(1,n_event_time))
    exprs_matrix = np.exp(rs_matrix)
    # n_sample * n_event_time(sorted)

    atrisk_matrix = ((df_input['time0'].to_numpy() < event_time[:,None]) & (df_input['time1'].to_numpy() >= event_time[:,None])).transpose()
    event_matrix = ((df_input['time1'].to_numpy()*df_input['event'].to_numpy()) == event_time[:,None]).transpose()
    # n_sample * n_event_time(sorted)

    # tie sum (for efron correction)
    tie_sum_vector = np.sum(exprs_matrix * atrisk_matrix,axis=0)
    # length: n_event_time(sorted)

    # tie mean (for efron correction)
    tie_mean_vector = np.sum(exprs_matrix * event_matrix,axis=0)/ties_vector

    # efron matrix
    efron_matrix = np.tile(np.array(range(Max_tie)),(n_event_time,1))
    # n_event_time*Max_tie

    # mask matrix to locate the ties
    mask_matrix = ((np.array(range(Max_tie))+1) <= ties_vector[:,None])

    h = np.tile(np.expand_dims(tie_sum_vector, axis=1),(1,Max_tie))-np.tile(np.expand_dims(tie_mean_vector, axis=1),(1,Max_tie))*efron_matrix
    h = 1/h
    h[np.isinf(h)] = 0
    h = h*mask_matrix
    h = np.sum(h,axis = 1)
    
    event_time = np.insert(event_time, 0, 0)
    h = np.insert(h, 0, 0)

    output = pd.DataFrame({'time': event_time, 'h': h})
    return output

def survprob(time_of_interest,haz,test_risk_score):
    # time_of_interest: a list of sorted time points on which the survival probability will be calculated. It is the time window after the last observation time (last_obs_time) of each subject in test_risk_score.
    # haz: n_times_train*(time,h0)
    # test_risk_score: n_obs*(id,last_obs_time,risk_score)
    time_of_interest = np.sort(np.array(time_of_interest))
    haz = pd.DataFrame(haz,columns = ['time','h'])
    test_risk_score = pd.DataFrame(test_risk_score,columns = ['id','last_obs_time','risk_score'])
    
    _,id_count = np.unique(test_risk_score["id"],return_counts=True)
    if sum(id_count>1)>0:
        raise ValueError('Each subject_id should only have one row. Please check test_risk_score')
    
    timeofinterest_matrix = np.add.outer(test_risk_score['last_obs_time'].to_numpy(), time_of_interest)
    all_times = np.unique(np.concatenate((test_risk_score['last_obs_time'].to_numpy(),
                                          timeofinterest_matrix.flatten(),
                                          haz['time'].to_numpy()), 
                                         axis=None))

    haz_alltimes = pd.merge(pd.DataFrame({'time': all_times}), haz, how="outer")
    haz_alltimes.fillna(0, inplace=True)

    survpred = np.outer(np.exp(test_risk_score['risk_score']),haz_alltimes['h'])
    survpred = np.cumsum(survpred,axis=1)
    survpred = np.exp(-survpred)

    index_timeofinterest = np.array([np.searchsorted(all_times, timeofinterest_matrix[i]) for i in range(len(timeofinterest_matrix))])
    index_last_obs_time = np.array([np.searchsorted(all_times, test_risk_score['last_obs_time'][i]) for i in range(len(timeofinterest_matrix))])

    output = np.array([(survpred[ii][index_timeofinterest[ii]])/(survpred[ii][index_last_obs_time[ii]]) for ii in range(len(survpred))])
    output = pd.DataFrame(output,columns = [str(k) for k in time_of_interest])
    output = pd.concat([test_risk_score[['id','last_obs_time']],output],axis = 1)
    return output