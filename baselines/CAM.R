library(CAM)
library(bnlearn)
library(ggm)

sachs <- read.table("sachs.data.txt", header = TRUE)
sachs.name <- c(1:length(sachs))
names(sachs.name) <- colnames(sachs)
estDAG <- CAM(sachs, scoreName = "SEMGAM", pruning = FALSE)
estOrder <- topOrder(as.matrix(estDAG$Adj))
rank <- match(c(1:length(estOrder)), estOrder)

sachs.modelstring <-
  paste0("[PKC][PKA|PKC][Raf|PKC:PKA][Mek|PKC:PKA:Raf]",
         "[Erk|Mek:PKA][Akt|Erk:PKA][P38|PKC:PKA]",
         "[Jnk|PKC:PKA][Plcg][PIP3|Plcg][PIP2|Plcg:PIP3]")
dag.sachs <- model2network(sachs.modelstring)
cnt_backwards <- apply(dag.sachs$arcs, 1, (function (x) rank[sachs.name[x[1]]] > rank[sachs.name[x[2]]]))

show(sum(cnt_backwards))