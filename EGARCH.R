#install.packages("rugarch")
library(rugarch)

# Risk-free rate
# Fetch command line arguments
if (length(commandArgs(trailingOnly=TRUE))==0){
  rf <- 0.02  # if no rf input, set default rf
} else {
  myArg <- commandArgs(trailingOnly = TRUE)
  # Convert to numerics
  rf <- as.numeric(myArg)
}

# Market returns
ret_mkt_annual <- read.csv('C:\\_Programming\\GitHubRepo\\BL_Allocation_APT_EGARCH\\market.csv', 
                           header=TRUE, row.names="Date")

# Stocks returns
ret_stocks_annual <- read.csv('C:\\_Programming\\GitHubRepo\\BL_Allocation_APT_EGARCH\\asset.csv', 
                              header=TRUE, row.names="Date")

# Factors returns
ret_factors_annual <- read.csv('C:\\_Programming\\GitHubRepo\\BL_Allocation_APT_EGARCH\\factors.csv', 
                               header=TRUE, row.names="Date")

# EGARCH-M to derive investor views and uncertainty levels
pred_res <- list()
for (ticker in colnames(ret_stocks_annual)){
  ret_stock_annual <- ret_stocks_annual[,ticker]
  excess_ret_stock_annual <- data.matrix(ret_stock_annual - rf)
  excess_ret_factors_annual <- data.matrix(ret_factors_annual - rf)
  # build and fit the EGARCH-M model
  spec <- ugarchspec(variance.model=list(model="eGARCH",garchOrder=c(1,1)), 
                     mean.model=list(armaOrder=c(0,0), archm = TRUE, include.mean = TRUE,
                                     external.regressors = excess_ret_factors_annual))
  egarch <- ugarchfit(spec, excess_ret_stock_annual, solver = 'hybrid')
  # forecast based on the EGARCH-M and latest factor returns
  pred <- ugarchforecast(egarch, n.ahead=1, 
                         external.forecasts = list(mregfor=tail(excess_ret_factors_annual, n=1)))
  pred_res[[ticker]] <- c(fitted(pred)[1], sigma(pred)[1])
}
pred_res_df <- t(data.frame(pred_res))
colnames(pred_res_df) <- c('view', 'uncertainty')

# Output to csv
write.csv(pred_res_df, 'C:\\_Programming\\GitHubRepo\\BL_Allocation_APT_EGARCH\\pred.csv')

cat("\nDone.")