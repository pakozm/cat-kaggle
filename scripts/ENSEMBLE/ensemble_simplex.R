options(echo=FALSE) # if you want see commands in output file
args <- commandArgs(trailingOnly = TRUE)

mask <- args[1]

# Uses R-convex-ensemble, available here: https://github.com/pakozm/R-convex-ensemble
source("/home/pako/programas/R-convex-ensemble/ensemble.R")
setwd("/home/pako/Dropbox/CATERPILLAR/WORK")

load_data <- function(pattern) {
    files <- list.files(pattern=pattern)
    print(files)
    matrix <- do.call(cbind,lapply(files, function(x) as.matrix(read.csv(x)$cost)))
    matrix
}

loss <- function(x, target) {
    sqrt(mean((log1p(x) - log1p(target))^2))
}

test_id     <- read.csv("../input/test_set.csv")$id
val_target  <- as.matrix(read.csv("../input/train_set.csv")$cost)
val_matrix  <- load_data(paste("val_",mask,"*", sep=''))
test_matrix <- load_data(paste("test_",mask,"*", sep=''))

test_p <- ensemble(val_matrix, val_target, test_matrix, loss)

write.csv(data.frame(id=test_id, cost=test_p),
          file="test_stage1_ensemble_convex.csv", row.names=FALSE, na="")
