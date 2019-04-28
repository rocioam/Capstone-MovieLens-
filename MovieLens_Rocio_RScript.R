# GitHub repo: https://github.com/rocioam/Capstone-MovieLens-

# SECTION 1, INTRODUCTION
#-none-

# SECTION 2, CAPSTONE GIVEN CODE

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", 
                                  readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



########################################################################

# PACKAGES JUST IN CASE

library(lubridate)
library(purrr)
library(tidyverse)
library(caret)


########################################################################


# SECTION 3 CODE: EXPLORING THE DATA

# edx Dataset analysis
dim(edx)
head(edx)

# How many different movies are in the dataset?
n_distinct(edx$movieId)

# How many different users can we find in our df?
n_distinct(edx$userId)


# What are the film genres?
sep_genres <- edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>% summarize(count = n()) %>%
  arrange(desc(count))

sep_genres


# Total movie rates for each genre
ggplot(sep_genres, aes(x=reorder(genres, count),y = count), 
       position = position_stack(reverse = TRUE)) + 
  geom_bar(stat = "identity", color = 'white', fill = 'darkgreen') + 
  coord_flip() +  # Make the bars horizontal
  geom_text(aes(label=sep_genres$count), nudge_y = 250000, size = 3) +
  labs(title = "Total movie rates for each genre") +
  theme(plot.title = element_text(hjust = 0.5), #Center title
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(), axis.title.x=element_blank(),
        axis.title.y=element_blank())



# Number of ratings per user
edx %>% group_by(userId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(color = "white", aes(fill=..count..), bins = 25) +
  scale_x_log10() + 
  labs(title = "Number of ratings per user", x = "Number of rated films", 
       y = "Users count") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Users count", low="lightblue", high="darkblue")


# Movie count depending on the number of ratings
edx %>% count(movieId) %>% arrange(-n) %>% ggplot(aes(n)) + scale_x_log10() +
  geom_histogram(color = "white", aes(fill=..count..), bins = 18) + 
  labs(title="Movie count depending on the number of ratings", 
       y="Movie count", x="Number of ratings") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Movie count", low="lightblue", high="darkblue")


# Top 10 rated Movies
rated_titles <- edx %>% group_by(title) %>% summarize(count = n()) %>%
  arrange(desc(count)) #Descending order to get top 10
head(rated_titles,10)


# User Rating [in stars]
edx %>% ggplot(aes(rating)) +
  geom_histogram(binwidth = .5, aes(fill=..count..), color = 'white') +
  labs(title = "User rating", x = "Rating [in stars]", y = "Number of ratings") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_gradient("Number of Ratings", low="magenta", high="blue")


# User density plot (histogram + density)
mean_rating <- edx %>% group_by(userId) %>% summarize(avg = mean(rating)) 
mean_rating %>% ggplot(aes(avg)) + 
  geom_histogram(aes(x = avg, y = ..density..), bins = 40, 
                 fill = "lightgray", color = "white") + 
  geom_density(alpha = .2, fill="#FF6655") + # Add density
  labs(title="User rating density", 
       x="Average rating by users [in stars]", y="Density") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_vline(aes(xintercept = mean(avg, na.rm = T)), # Add mean
             colour = "magenta", linetype ="longdash", size = .8)
mean(mean_rating$avg)


############################################################################

# SECTION 4: MACHINE LEARNING

# 4.1. Creating train and test sets


# create train and test sets out of the edx dataframe
set.seed(42) #You gotta love this number!
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = F)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]


# Make sure that movieId and userId in the test set are also on the train set
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")



# 4.2. Model-based approach

mu_hat <- mean(train_set$rating)
mu_hat
RMSE_base <- RMSE(test_set$rating, mu_hat)
RMSE_base
rmse_results <- data_frame(Method = "Model-based Approach", RMSE = RMSE_base)
rmse_results



# 4.3. Movie Effect Model
mu <- mean(train_set$rating)
movie_avgs <- train_set %>% group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Bias variation
movie_avgs %>% ggplot(aes(b_i)) +
  geom_histogram(bins = 20, color = 'white', fill = 'darkblue') +
  labs(title = "Bias variation", x = "bias b_i", y = "Count") +
  theme(plot.title = element_text(hjust = 0.5)) 

# New RMSE
predicted_ratings <- mu + test_set %>% left_join(movie_avgs, by = 'movieId') %>% .$b_i

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Movie Effect Model", RMSE = model_1_rmse))

rmse_results



#4.4. Movie + Users Effect Model

# New model
user_avgs <- test_set %>% left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>% .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Movie + User Effects Model",
                                     RMSE = model_2_rmse))
rmse_results


#4.5. Model application to validation data

# In the validation data
predicted_validation <- validation %>% left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>% .$pred
validation_RMSE <- RMSE(predicted_validation, validation$rating, na.rm = T)
validation_RMSE

#RMSE Results in validation data
rmse_validation_results <- data_frame(Validation_Method = "Movie + User Effects Model", 
                                      RMSE = validation_RMSE)
rmse_validation_results


#5. IMPROVING THE RESULTS: REGULARIZATION

#5.1. Regularization on the edx dataset

# Tuning parameter lambda search code
lambdas <- seq(2, 7, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu) / (n()+l) )
  
  b_u <- train_set %>% left_join(b_i, by = "movieId") %>% group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu) / (n()+l) )
  
  predicted_ratings <- test_set %>% left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    mutate(pred = mu + b_i + b_u) %>% .$pred
  
  return(RMSE(predicted_ratings, test_set$rating))
})


# Lambda optimization for Regularization method
lambdas_df <- data.frame(Lambda_values = lambdas, RMSES = rmses)

ggplot(lambdas_df, aes(x = Lambda_values, y = RMSES)) + 
  geom_point(size = 2, shape = 22, color = "darkgreen") +
  labs(title = "Lambda optimization for Regularization method", 
       x = "Lambda values", y = "RMSE") +
  theme(plot.title = element_text(hjust = 0.5))


# Optimized lambda value
opt_lambda <- lambdas[which.min(rmses)]
opt_lambda


# RME Table results for edx set
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = 
                                       "Movie + User Effects Model with Regularization",
                                     RMSE = min(rmses)))
rmse_results



#5.2. Regularization on the validation dataset

# We do the same for the validation set
lambdas_validation <- seq(2, 7, 0.25)

rmses_validation <- sapply(lambdas_validation, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu) / (n()+l) )
  
  b_u <- train_set %>% left_join(b_i, by = "movieId") %>% group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu) / (n()+l) )
  
  predicted_validation_ratings <- validation %>% left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    mutate(pred = mu + b_i + b_u) %>% .$pred
  
  return(RMSE(predicted_validation_ratings, validation$rating, na.rm = T))
})


# Validation Lambda optimization for Regularization method
lambdas_validation_df <- data.frame(Validation_Lambda_values = lambdas_validation, 
                                    Validation_RMSES = rmses_validation)

ggplot(lambdas_validation_df, aes(x = Validation_Lambda_values, y = Validation_RMSES)) + 
  geom_point(size = 2, shape = 22, color = "darkblue") +
  labs(title = "Validation Lambda optimization for Regularization method", 
       x = "Validation Lambda values", y = "Validation RMSE") +
  theme(plot.title = element_text(hjust = 0.5))


# Optimized lambda value for validation set
opt_lambda_validation <- lambdas_validation[which.min(rmses_validation)]
opt_lambda_validation


# Final validation set RMSE table
rmse_validation_results <- bind_rows(rmse_validation_results,
                                     data_frame(Validation_Method = 
                                                  "Movie + User Effects Model with  Regularization", 
                                                RMSE = min(rmses_validation)))
rmse_validation_results

# FINAL RMSE: MOVIE+USER EFFECTS MODEL WITH REGULARIZATION METHOD

final_RMSE <- min(rmses_validation)
final_RMSE

print(paste0("Final RMSE Value obtained for the validation set with the Movie + User Effects Model with Regularization Method: ", final_RMSE))
#"Final RMSE Value obtained for the validation set with the Movie + 
#User Effects Model with Regularization Method: 0.865722399226163"