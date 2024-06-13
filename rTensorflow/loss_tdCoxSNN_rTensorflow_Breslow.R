loss_tdCoxSNN_rTensorflow = function(y_true, y_pred){
  
  y_true = tf$cast(y_true, tf$float32) # tstart, tstop, event
  y_pred = tf$cast(y_pred, tf$float32)
  y_pred = tf$squeeze(y_pred)

  n_sample = tf$cast(tf$size(y_pred),tf$float32)
  
  time0 = tf$cast(tf$squeeze(y_true[,1]),tf$float32)
  time = tf$cast(tf$squeeze(y_true[,2]),tf$float32)
  event = tf$cast(tf$squeeze(y_true[,3]),tf$float32)
  
  no_event = tf$cond(tf$equal(tf$reduce_sum(event),tf$cast(0,tf$float32)), function(){return(tf$cast(1,tf$float32))},function(){return(tf$cast(0,tf$float32))})
  
  event = tf$add(event,no_event)
  sort_index = tf$argsort(time)
  time0 = tf$gather(params = time0, indices = sort_index)
  time = tf$gather(params = time, indices = sort_index)
  event = tf$gather(params = event, indices = sort_index)
  y_pred = tf$gather(params = y_pred, indices = sort_index)
  
  time_event = time * event
  positive_indexes = (tf$where(tf$greater(time_event, tf$zeros_like(time_event))))
  alleventtime = tf$gather(time_event,positive_indexes)
  
  loc = tf$where(time_event>0,tf$ones_like(time_event,dtype=tf$bool),tf$zeros_like(time_event,dtype=tf$bool))
  loc$set_shape(list(NULL))
  alleventtime = tf$boolean_mask(time_event,loc)
  
  unique_w_c = tf$unique_with_counts(alleventtime)
  eventtime = unique_w_c[0]
  tie_count = unique_w_c[2]
  
  at_risk_index = tf$where(tf$logical_and(tf$less(time0,tf$expand_dims(eventtime,tf$cast(1,tf$int32))),tf$greater_equal(time,tf$expand_dims(eventtime,tf$cast(1,tf$int32)))),
                           1., 0.)
  event_index = tf$where(tf$equal(time,tf$expand_dims(eventtime,tf$cast(1,tf$int32))),1.,0.)
  
  # haz = exp(risk)
  tie_haz = tf$matmul(event_index,tf$expand_dims(tf$exp(tf$clip_by_value(y_pred,-20,20))*event,tf$cast(1,tf$int32)))
  
  tie_risk = tf$matmul(event_index,tf$expand_dims(y_pred*event,tf$cast(1,tf$int32)))
  
  cum_haz = tf$matmul(at_risk_index,
                      tf$expand_dims(tf$exp(tf$clip_by_value(y_pred,-20,20)),tf$cast(1,tf$int32)))
  
  mask_tie_risk = tf$less(tf$range(tf$reduce_max(tie_count)),
                          tf$expand_dims(tie_count,tf$cast(1,tf$int32)))
  
  out0 = tf$zeros_like(mask_tie_haz,dtype = tf$float32)

  log_sum_haz0 = tf$where(mask_tie_risk,
                          tf$ones_like(mask_tie_risk,dtype = tf$float32),
                          out0)*cum_haz
  log_sum_haz = tf$where(mask_tie_risk,
                         tf$math$log(log_sum_haz0+tf$cast(1e-15,tf$float32)),
                         out0)
  log_sum_haz_value = tf$reduce_sum(log_sum_haz)
  log_lik = tf$reduce_sum(tie_risk)-log_sum_haz_value
  
  log_lik_output = tf$multiply(tf$negative(log_lik),tf$subtract(1,no_event))/n_sample
  
  return(log_lik_output)
}
