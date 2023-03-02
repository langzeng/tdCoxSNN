import tensorflow as tf
import tensorflow.keras.backend as tfkb

def loss_tdCoxSNN_Tensorflow(y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.float32) # tstart, tstop, event
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tfkb.flatten(y_pred)
    
    time0 = tf.cast(tfkb.flatten(y_true[:,0]),tf.float32)
    time = tf.cast(tfkb.flatten(y_true[:,1]),tf.float32)
    event = tf.cast(tfkb.flatten(y_true[:,2]),tf.float32)
        
    n = tf.shape(time)[0]
    
    sort_index = tf.argsort(time)
    time0 = tfkb.gather(reference = time0, indices = sort_index)
    time = tfkb.gather(reference = time, indices = sort_index)
    event = tfkb.gather(reference = event, indices = sort_index)
    y_pred = tfkb.gather(reference = y_pred, indices = sort_index)

    time_event = time * event
    eventtime,_,tie_count = tf.unique_with_counts(tf.boolean_mask(time_event, tf.greater(time_event, 0)))
    
    at_risk_index = tfkb.cast((time0 < eventtime[:,None]) & (time >= eventtime[:,None]),
                              dtype=tf.float32)
    event_index = tfkb.cast((time == eventtime[:,None]),dtype=tf.float32)
    
    # haz = exp(risk)
    tie_haz = tfkb.dot(event_index, tfkb.expand_dims(tfkb.exp(tf.clip_by_value(y_pred,-20,20))*event))
    tie_haz = tf.squeeze(tie_haz)
    
    tie_risk = tfkb.dot(event_index, tfkb.expand_dims(y_pred*event))
    tie_risk = tf.squeeze(tie_risk)
    
    cum_haz = tfkb.dot(at_risk_index, tfkb.expand_dims(tfkb.exp(tf.clip_by_value(y_pred,-20,20))))
    cum_haz = tf.squeeze(cum_haz)

    mask_tie_haz = tfkb.arange(tfkb.max(tie_count)) < (tie_count[:,None]-1)
    mask_tie_risk = tfkb.arange(tfkb.max(tie_count)) < (tie_count[:,None])
    out0 = tfkb.zeros(mask_tie_haz.shape,dtype = tf.float32)
    out1 = tfkb.cast(tf.cumsum(tf.ones(mask_tie_haz.shape),1),dtype = tf.float32)
    out = tf.where(mask_tie_haz, out1, out0)
    tie_count_matrix = tfkb.expand_dims(tfkb.cast(tie_count, dtype = tf.float32))
    
    J = tf.divide(out,tie_count_matrix)
    efron_correction = J*tfkb.expand_dims(tie_haz)
    log_sum_haz = tf.where(mask_tie_risk,
                           tf.ones(mask_tie_risk.shape,dtype = tf.float32),
                           out0)*tfkb.expand_dims(cum_haz)
    log_sum_haz = tf.where(mask_tie_risk,
                           tf.math.log(log_sum_haz-efron_correction+1e-15),
                           out0)
    log_sum_haz = tf.math.reduce_sum(log_sum_haz)
    log_lik = tf.math.reduce_sum(tie_risk)-log_sum_haz

    return tf.negative(log_lik)