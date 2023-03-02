require(dplyr)
require(pec)
require(timeROC)

select <- dplyr::select
filter <- dplyr::filter

# # baseline cumulative hazard function from linear PI model
# baseline_hazard <- function(df_input){
#   # baseline_hazard function with Efron correction
#   # df_input: n_sample*(time0,time1,event,risk_score)
#   # output: n_event_time*(time,h)
#   df_input <- as.data.frame(df_input)
#   names(df_input) <- c("time0","time1","event","risk_score")
#   
#   # unique event time
#   event_time <- sort(unique(df_input$time1[(df_input$event == 1)]))
#   n_event_time <- length(event_time)
# 
#   ties_vector <- unname(table(df_input$time1[(df_input$event == 1)]))
#   Max_tie <- max(ties_vector)
#   
#   # predicted risk score at event time
#   rs_matrix <- beta_matrix %*% base::t(event_time_van)
#   exprs_matrix <- exp(rs_matrix)
#   # n_sample * n_event_time(sorted)
#   
#   # index matrixs
#   atrisk_matrix <- (outer(df_input$time0,event_time,"<") & outer(df_input$time1,event_time,">="))
#   event_matrix <- outer(df_input$time1*df_input$event,event_time,"==")
#   # n_sample * n_event_time(sorted)
#   
#   # tie sum (for efron correction)
#   tie_sum_vector <- colSums(exppi_matrix * atrisk_matrix)
#   # length: n_event_time(sorted)
#   
#   # tie mean (for efron correction)
#   tie_mean_vector <- colSums(exppi_matrix * event_matrix)/ties_vector
#   
#   # efron matrix
#   efron_matrix <- np.tile(np.array(range(Max_tie)),(n_event_time,1))
#   # n_event_time*Max_tie
#   
#   # mask matrix to locate the ties
#   mask_matrix <- outer(ties_vector,seq(Max_tie),">=")
#   
#   h <- outer(tie_sum_vector,rep(1,Max_tie),"*")-outer(tie_mean_vector,rep(1,Max_tie),"*")*efron_matrix
#   h <- 1/h
#   h[is.infinite(h)] <- 0
#   h <- h*mask_matrix
#   h <- rowSums(h)
#   
#   output <- data.frame(time = event_time, h = h)
#   return(output)
# }
# 
# #####
# survprob_pi <- function(time_of_interest=NULL,
#                         haz=NULL, 
#                         pi=NULL, 
#                         predict_from = "baseline",
#                         extrapolation = 0,
#                         Mix_Extrapolation_nobs = 1){
#   # predict_from = "baseline" or "last obstime"
#   # extrapolation: t^extrapolation (0,1,0.5,...)
#   
#   # pi: n_obs*(id,time0,pi)
#   # haz: n_times_train*(time,h0)
#   haz <- as.data.frame(haz)
#   names(haz) <- c("time","h0")
#   
#   pi <- as.data.frame(pi)
#   names(pi) <- c("id","time0","pi")
#   pi <- pi %>% 
#     group_by(id) %>% 
#     mutate(run = row_number()) %>% 
#     mutate(nobs = max(run)) %>% 
#     ungroup()
#   pi_last <- pi %>% group_by(id) %>% filter(run == nobs) %>% ungroup()
#   uni_id <- pi_last$id
#   
#   # extract "beta"
#   if((extrapolation == 0) | (extrapolation == "LOCF")){
#     beta_tmp <- data.frame(beta0 = pi_last$pi,
#                            beta1 = 0)
#   }else if(is.numeric(extrapolation)){
#     sapply(uni_id,function(x){
#       tmp <- pi %>% filter(id==x) %>% select(pi,time0)
#       if(nrow(tmp) == 1){
#         return(c(tmp$pi,0))
#       }
#       else{
#         tmp$time0 <- (tmp$time0)^extrapolation
#         fit_tmp <- lm(pi~time0,data=tmp)
#         output <- fit_tmp$coefficients %>% unlist %>% unname
#         return(output)
#       }
#     }) %>% base::t() -> beta_tmp
#     names(beta_tmp) <- c("beta0","beta1")
#   }else if(extrapolation == "MEAN"){
#     sapply(uni_id,function(x){
#       tmp <- pi %>% filter(id==x) %>% select(pi,time0)
#       mean_pi <- mean(tmp$pi)
#       return(c(mean_pi,0))
#     }) %>% base::t() -> beta_tmp
#     
#     names(beta_tmp) <- c("beta0","beta1")
#   }else if(extrapolation == "log"){
#     sapply(uni_id,function(x){
#       tmp <- pi %>% filter(id==x) %>% select(pi,time0)
#       tmp_last <- pi_last %>% filter(id==x) %>% select(pi,time0,nobs)
#       if(nrow(tmp) == 1){
#         return(c(tmp$pi,0))
#       } else if(max(tmp_last$nobs) <= Mix_Extrapolation_nobs){
#         return(c(tmp_last$pi,0))
#       } else{
#         tmp$time0 <- log(tmp$time0+1)
#         fit_tmp <- lm(pi~time0,data=tmp)
#         output <- fit_tmp$coefficients %>% unlist %>% unname
#         return(output)
#       }
#     }) %>% base::t() -> beta_tmp
#     names(beta_tmp) <- c("beta0","beta1")
#   }
#   
#   # Time
#   if(predict_from == "baseline"){
#     time_base <- rep(0,length(uni_id))
#   } else if(predict_from == "last obstime"){
#     time_base <- pi_last$time0
#   }
#   time_ofinterest_matrix <- outer(time_base,time_of_interest,"+")
#   all_times <- c(as.vector(time_ofinterest_matrix),haz$time) %>% 
#     unique() %>% 
#     sort()
#   
#   # Predicted PI
#   if(is.numeric(extrapolation)){
#     event_time_van <- matrix(c(rep(1,length(all_times)),(all_times^extrapolation)),ncol = 2)
#   } else if(extrapolation == "log"){
#     event_time_van <- matrix(c(rep(1,length(all_times)),log(all_times+1)),ncol = 2)
#   }else{
#     event_time_van <- matrix(c(rep(1,length(all_times)),(all_times)),ncol = 2)
#   } 
#   pi_matrix <- as.matrix(beta_tmp) %*% base::t(event_time_van)
#   exppi_matrix <- exp(pi_matrix)
#   
#   
#   baseline_tmp <- haz %>% 
#     complete(time = all_times,fill = list(h0=0))
#   
#   survpred_tmp <- exppi_matrix*matrix(rep(baseline_tmp$h0,nrow(exppi_matrix)),ncol = length(baseline_tmp$h0),byrow = T)
#   survpred_tmp <- t(apply(survpred_tmp, 1, cumsum))
#   survpred_tmp <- exp(-survpred_tmp) %>% as.matrix
#   
#   survpred_output <- sapply(seq(length(uni_id)),function(j){
#     timeofinterest_tmp <- time_ofinterest_matrix[j,] %>% unlist
#     if((predict_from == "baseline") | (pi_last$nobs[j] == 1)){
#       output_tmp <- survpred_tmp[j,all_times %in% timeofinterest_tmp]
#     }else{
#       time_tmp = max(all_times[all_times<=pi_last$time0[j]])
#       output_tmp <- survpred_tmp[j, all_times %in% (timeofinterest_tmp/1.0)]/survpred_tmp[j,all_times==time_tmp]
#     }
#     return(output_tmp)
#   }) %>% base::t()
#   
#   return(survpred_output)
# }