require(dplyr)
require(tidyr)
require(pec)
require(timeROC)

select <- dplyr::select
filter <- dplyr::filter

# baseline hazard function
baseline_hazard <- function(df_input){
  # baseline_hazard function with Efron correction
  # df_input: n_obs*(time0,time1,event,risk_score)
  # output: n_event_time*(time,h)
  df_input <- as.data.frame(as.matrix(df_input))
  names(df_input) <- c("time0","time1","event","risk_score")
  
  # unique event time
  event_time <- sort(unique(df_input$time1[(df_input$event == 1)]))
  n_event_time <- length(event_time)
  
  ties_vector <- unname(table(df_input$time1[(df_input$event == 1)]))
  Max_tie <- max(ties_vector)
  
  # predicted risk score at event time
  rs_matrix <- matrix(df_input$risk_score) %*% base::t(rep(1,n_event_time))
  exprs_matrix <- exp(rs_matrix)
  # n_obs * n_event_time(sorted)
  
  # index matrixs
  atrisk_matrix <- (outer(df_input$time0,event_time,"<") & outer(df_input$time1,event_time,">="))
  event_matrix <- outer(df_input$time1*df_input$event,event_time,"==")
  # n_obs * n_event_time(sorted)
  
  # tie sum (for efron correction)
  tie_sum_vector <- colSums(exprs_matrix * atrisk_matrix)
  # length: n_event_time(sorted)
  
  # tie mean (for efron correction)
  tie_mean_vector <- colSums(exprs_matrix * event_matrix)/ties_vector
  
  # efron matrix
  efron_matrix <- matrix(rep((seq(Max_tie)-1),n_event_time),nrow = n_event_time, byrow = T)
  # n_event_time*Max_tie
  
  # mask matrix to locate the ties
  mask_matrix <- outer(ties_vector,seq(Max_tie),">=")
  
  h <- outer(tie_sum_vector,rep(1,Max_tie),"*")-outer(tie_mean_vector,rep(1,Max_tie),"*")*efron_matrix
  h <- 1/h
  h[is.infinite(h)] <- 0
  h <- h*mask_matrix
  h <- rowSums(h)
  
  output <- data.frame(time = c(0,event_time), h = c(0,h))
  return(output)
}

#####
survprob <- function(time_of_interest=NULL,
                     haz=NULL,
                     test_risk_score=NULL){
  # time_of_interest: a list of sorted time points on which the survival probability will be calculated. 
  # It is the time window after the last observation time (last_obs_time) of each subject in test_risk_score.
  # haz: n_times_train*(time,h0)
  # test_risk_score: n_sample*(id,last_obs_time,risk_score)
  haz <- as.data.frame(haz)
  names(haz) <- c("time","h")
  
  test_risk_score <- as.data.frame(test_risk_score)
  names(test_risk_score) <- c("id","last_obs_time","rs")
  
  id_count <- test_risk_score$id %>% unique %>% table %>% max
  
  if(id_count>1){stop("Each subject_id should only have one row. Please check test_risk_score")}
  
  time_ofinterest_matrix <- outer(test_risk_score$last_obs_time,time_of_interest,"+")
  all_times <- c(as.vector(time_ofinterest_matrix),haz$time,test_risk_score$last_obs_time) %>%
    unique() %>%
    sort()
  
  haz_alltimes <- haz %>%
    complete(time = all_times,fill = list(h=0))
  
  h_tmp <- outer(exp(test_risk_score$rs),haz_alltimes$h)
  cumh_tmp <- t(apply(h_tmp, 1, cumsum)) %>% as.matrix
  
  survpred_output <- sapply(seq(nrow(test_risk_score)),function(j){
    index_timeofinterest_tmp <- (all_times %in% (time_ofinterest_matrix[j,] %>% unlist))
    index_time_tmp <- (all_times == test_risk_score$last_obs_time[j])
    output_tmp <- exp(-cumh_tmp[j, index_timeofinterest_tmp]+cumh_tmp[j,index_time_tmp])
    return(output_tmp)
  }) %>% base::t()
  
  survpred_output <- as.data.frame(survpred_output)
  names(survpred_output) <- paste0('t',time_of_interest)
  survpred_output <- cbind(test_risk_score,survpred_output)
  
  return(survpred_output)
}
