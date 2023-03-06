import tensorflow as tf

@tf.function
def loss_tdCoxSNN_Tensorflow(y_true, y_pred):
    # y_true and y_pred should be 2-D
    
    y_true = tf.cast(y_true, tf.float32) # tstart, tstop, event
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.squeeze(y_pred)
    
    time0 = tf.cast(tf.squeeze(y_true[:,0]),tf.float32)
    time = tf.cast(tf.squeeze(y_true[:,1]),tf.float32)
    event = tf.cast(tf.squeeze(y_true[:,2]),tf.float32)
        
    no_event = tf.where(tf.equal(tf.reduce_sum(event),0), tf.constant(1,tf.float32),tf.constant(0,tf.float32))
    
    if no_event == tf.constant(1,tf.float32):
        return tf.constant(0,tf.float32)
    
    sort_index = tf.argsort(time)
    time0 = tf.gather(params = time0, indices = sort_index)
    time = tf.gather(params = time, indices = sort_index)
    event = tf.gather(params = event, indices = sort_index)
    y_pred = tf.gather(params = y_pred, indices = sort_index)
    
    time_event = time * event
    loc = tf.where(time_event>0,tf.ones_like(time_event,dtype=tf.bool),tf.zeros_like(time_event,dtype=tf.bool))
    loc.set_shape([None])
    alleventtime = tf.boolean_mask(time_event,loc)
    eventtime,_,tie_count = tf.unique_with_counts(alleventtime)
    
    at_risk_index = tf.cast((time0 < eventtime[:,None]) & (time >= eventtime[:,None]),
                              dtype=tf.float32)
    event_index = tf.cast((time == eventtime[:,None]),dtype=tf.float32)
    
    # haz = exp(risk)
    tie_haz = tf.matmul(event_index, tf.expand_dims(tf.exp(tf.clip_by_value(y_pred,-20,20))*event,1))
    
    tie_risk = tf.matmul(event_index, tf.expand_dims(y_pred*event,1))
    
    cum_haz = tf.matmul(at_risk_index, tf.expand_dims(tf.exp(tf.clip_by_value(y_pred,-20,20)),1))
 
    mask_tie_haz = tf.range(tf.reduce_max(tie_count)) < (tie_count[:,None]-1)
    mask_tie_risk = tf.range(tf.reduce_max(tie_count)) < (tie_count[:,None])
    out0 = tf.zeros_like(mask_tie_haz,dtype = tf.float32)
    out1 = tf.cumsum(tf.ones_like(mask_tie_haz,dtype = tf.float32),1)
    out = tf.where(mask_tie_haz, out1, out0)
    tie_count_matrix = tf.expand_dims(tf.cast(tie_count, dtype = tf.float32),1)
    
    J = tf.divide(out,tie_count_matrix)
    efron_correction = J*tie_haz
    log_sum_haz = tf.where(mask_tie_risk,
                           tf.ones_like(mask_tie_risk,dtype = tf.float32),
                           out0)*cum_haz
    log_sum_haz = tf.where(mask_tie_risk,
                           tf.math.log(log_sum_haz-efron_correction+1e-15),
                           out0)
    log_sum_haz = tf.math.reduce_sum(log_sum_haz)
    log_lik = tf.math.reduce_sum(tie_risk)-log_sum_haz
    log_lik_output = tf.negative(log_lik)
    
    return log_lik_output
