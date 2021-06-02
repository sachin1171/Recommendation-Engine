#############################Problem -1########################################
import pandas as pd

my_game = pd.read_csv("C:/Users/usach/Desktop/recommendation engine/game.csv", encoding = 'utf8')

#term frequencey- inverse document frequncy is a numerical statistic that is intended 
#to reflect how important a word is to document in a collecion or corpus
from sklearn.feature_extraction.text import TfidfVectorizer
# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    
# taking stop words from tfid vectorizer 

my_game.isna().sum()
my_game.isnull().sum()

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(my_game.game)   #Transform a count matrix to a 
#normalized tf or tf-idf representation
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of game name to index number 
game_index = pd.Series(my_game.index, index = my_game['game'])
game_index = game_index[~game_index.index.duplicated(keep='first')]
#game_index = pd.Series(game_index.index, index = game_index)

game_id = game_index["Grand Theft Auto IV"]
game_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the game index using its title 
    game_id = game_index[Name]
    
    # Getting the pair wise similarity score for all the game's with that 
    # game
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the game index 
    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    game_similar_show = pd.DataFrame(columns=["name", "Score"])
    game_similar_show["name"] = my_game.iloc[game_idx,1]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  
    print (game_similar_show)
    
# Enter your game and number of game's to be recommended 
get_recommendations("Metal Gear Solid 2: Sons of Liberty", topN = 10)
game_index["Metal Gear Solid 2: Sons of Liberty"]


####################################Problem 2#####################################
import pandas as pd

# import Dataset 
Entertainment = pd.read_csv("C:/Users/usach/Desktop/recommendation engine/Entertainment.csv", encoding = 'utf8')
Entertainment.shape # shape
Entertainment.columns

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse 
#document frequncy is a numerical statistic that is intended to reflect how important 
#a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
# taking stop words from tfid vectorizer
tfidf = TfidfVectorizer(stop_words = "english")     

# replacing the NaN values in overview column with empty string
Entertainment["Category"].isnull().sum() 
Entertainment["Category"] = Entertainment["Category"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming
 #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix = tfidf.fit_transform(Entertainment.Category)  
tfidf_matrix.shape


from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of movie name to index number 
Entertainment_index = pd.Series(Entertainment.index, index = Entertainment['Titles']).drop_duplicates()

#testing
Entertainment_id = Entertainment_index["Grumpier Old Men (1995)"]
Entertainment_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    Entertainment_id = Entertainment_index[Name]
    
    # Getting the pair wise similarity score for all the movies's with that 
    cosine_scores = list(enumerate(cosine_sim_matrix[Entertainment_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    Entertainment_idx  =  [i[0] for i in cosine_scores_N]
    Entertainment_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    Entertainment_similar_show = pd.DataFrame(columns=["name", "Score"])
    Entertainment_similar_show["name"] = Entertainment.loc[Entertainment_idx, "Titles"]
    Entertainment_similar_show["Score"] = Entertainment_scores
    Entertainment_similar_show.reset_index(inplace = True)  
    print (Entertainment_similar_show)
    

    
# Enter your movie and number of movies's to be recommended 
get_recommendations("Lamerica (1994)", topN = 10)
Entertainment_index["Lamerica (1994)"]
