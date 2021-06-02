############################Problem 1#####################################
install.packages("recommenderlab")
library(recommenderlab)
install.packages("reshape2")
library(reshape2)

game_list <-read.csv("C:/Users/usach/Desktop/recommendation engine/game.csv",header=TRUE)
head(game_list)
head(game_list)
#game_list <- game_list[1:500, ]
#game_list <- game_list[, c(2,1,3)]
dim(game_list)
#game_list <- transform(game_list, rating = as.integer(rating))
## covert to matrix format
game_matrix <- as.matrix(acast(game_list, userId~game, fun.aggregate = mean))
dim(game_matrix)

## recommendarlab realRatingMatrix format
R <- as(game_matrix, "realRatingMatrix")

# rec1 = Recommender(R, method="UBCF") ## User-based collaborative filtering
# rec2 = Recommender(R, method="IBCF") ## Item-based collaborative filtering
# rec3 = Recommender(R, method="SVD")
rec4 = Recommender(R, method="POPULAR")
#rec5 = Recommender(binarize(R,minRating=2), method="UBCF")

## create n recommendations for a user
game_nam = "Grand Theft Auto V"
games <- subset(game_list, game_list$game==game_nam)
game_id <- which(game_list$game == game_nam)
print("You have rated:")
games
print("recommendations for you:")
prediction <- predict(rec4, R[game_id], n=10) 
as(prediction, "list")

###############################Problem 2############################################
install.packages("recommenderlab")
library(recommenderlab)
install.packages("reshape2")
library(reshape2)

Entertainment_list <-read.csv("C:/Users/usach/Desktop/recommendation engine/Entertainment.csv",header=TRUE)
head(Entertainment_list)
Entertainment_list <- Entertainment_list[, c(2:4)]

dim(Entertainment_list)
colnames(Entertainment_list)
## covert to matrix format
Entertainment_matrix <- as.matrix(acast(Entertainment_list, Category~Titles, fun.aggregate = median))
dim(Entertainment_matrix)

## recommendarlab realRatingMatrix format
R <- as(Entertainment_matrix, "realRatingMatrix")

#rec1 = Recommender(R, method="UBCF") ## User-based collaborative filtering
#rec2 = Recommender(R, method="IBCF") ## Item-based collaborative filtering
#rec3 = Recommender(R, method="SVD")
rec4 = Recommender(R, method="POPULAR")
#rec5 = Recommender(binarize(R,minRating=2.0), method="UBCF") ## binarize all 2+ rating to 1

## create n recommendations for a user
movie_nam <- "Sense and Sensibility (1995)"
movie_id <- which(Entertainment_list$Titles == movie_nam)
movies <- subset(Entertainment_list[,3], Entertainment_list$Titles ==movie_nam)
print(paste("You have rated:",movies))
print("recommendations for you:")
prediction4 <- predict(rec4, R[movie_id], n=5) 
as(prediction4, "list")

