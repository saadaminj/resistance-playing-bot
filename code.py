from player import Bot 
from game import State
import random

# to run this python3 competition.py 1000 bots/beginners.py bots/neuralbot.py
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys


from keras.models import model_from_json
json_file = open('resistance_nn_model_improved.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("resistance_nn_model_improved.h5")

# from loggerbot import LoggerBot # this assumes our loggerbot was in a file called loggerbot.py

class MyBot(Bot):
    
    def predict(self,data):
        # return model.predict(np.reshape(data,(1,np.shape(data)[0])))[0][0]
        return model.predict(data)[0][0] 

    def calc_player_probabilities_of_being_spy(self):
        # This loop could be made much more efficient if we push all player's input patterns
        # through the neural network at once, instead of pushing them through one-by-one
        probabilities={}
        for p in self.game.players:
            input_vector=[self.game.turn, self.game.tries, p.index, p.name, self.missions_been_on[p], self.failed_missions_been_on[p]]+self.num_missions_voted_up_with_total_suspect_count[p]+self.num_missions_voted_down_with_total_suspect_count[p]
            input_vector=input_vector[4:] # remove the first 4 cosmetic details, as we did when training the neural network
            input_vector = pd.DataFrame([input_vector])  
            input_vector=(input_vector-self.df.min())/(self.df.max()-self.df.min())
            output=self.predict(input_vector) # run the neural network# The neural network didn't have a softmax on the final layer, so I'll add the softmax step here manually.
            probabilities[p]=output # this [0,1] pulls off the first row (since there is only one row) and the second column (which corresponds to probability of being a spy; the first column is the probability of being not-spy)
        return probabilities# This returns a dictionary of {player: spyProbability}

    def select(self, players, count):
        spy_probs=self.calc_player_probabilities_of_being_spy()
        sorted_players_by_trustworthiness=[k for k, v in sorted(spy_probs.items(), key=lambda item: item[1])]
        
        if self.spy:
            random_players = random.sample(self.others(), count - 1)
            #select atleast one spy
            for i in range(len(self.spies)):
                if list(self.spies)[i] not in random_players:
                    return random_players + [list(self.spies)[i]]
        elif not self.spy:
            if self in sorted_players_by_trustworthiness[:count]:
                return sorted_players_by_trustworthiness[:count]
            return [self]+sorted_players_by_trustworthiness[:count - 1]
        else:
            return random.sample(self.others(), count)

    def vote(self, team): 
        spy_probs=self.calc_player_probabilities_of_being_spy()
        sorted_players_by_trustworthiness=[k for k, v in sorted(spy_probs.items(), key=lambda item: item[1],reverse=True)]
        if not self.spy:
            for x in team:
                if x in sorted_players_by_trustworthiness[:len(team)]:
                    return False
            return True
        else:
            return len([p for p in self.game.team if p in self.spies]) >= 1

    def sabotage(self):
        return True if self.spy else False

    ''' The 3 methods onVoteComplete, onGameRevealed, onMissionComplete
    will inherit their functionality from ancestor.  We want them to do exactly 
    the same as they did when we captured the training data, so that the variables 
    for input to the NN are set correctly.  Hence we don't override these methods
    '''
    def onMissionComplete(self, num_sabotages):
        """Callback once the players have been chosen.
        @param num_sabotages Integer how many times the mission was sabotaged.
        """
        if num_sabotages:
            for p in self.game.team:
                self.failed_missions_been_on[p] += 1
                self.missions_been_on[p] += 1
        else:
            for p in self.game.team:
                self.missions_been_on[p] += 1
    
    # This function used to output log data to the log file. 
    # We don't need to log any data any more so let's override that function
    # and make it do nothing...
    def onGameComplete(self, win, spies):
        pass
    
    def read_df_and_normalize(self):
        with open('C:/Users/saada/Desktop/essex1/original_game/ce811-the-resistance-main/logs/LoggerBot.log') as f:
            lines = f.readlines()
        
        data = []
        index = 0
        names = []
        for i in lines:
            split = i.split(',')[3:]
            data.append({})
        #     data[index]['name'] = split[0]
        #     names.append(split[0])
            data[index]['missions'] = int(split[1])
            data[index]['failed_missions'] = int(split[2])
            data[index]['num_missions_voted_up_with_total_suspect_count'] =  [int(x) for x in split[3:9]]
            data[index]['num_missions_voted_down_with_total_suspect_count'] = [int(x) for x in split[9:15]]
            data[index]['spy'] = int(split[15][0])
            index = index + 1
        self.df = pd.DataFrame.from_dict(data)
        self.df[['up_0', 'up_1','up_2', 'up_3','up_4', 'up_5']] = pd.DataFrame(self.df["num_missions_voted_up_with_total_suspect_count"].to_list(), columns=['up_0', 'up_1','up_2', 'up_3','up_4', 'up_5'])
        self.df = self.df.drop('num_missions_voted_up_with_total_suspect_count', axis=1)
        self.df[['down_0', 'down_1','down_2', 'down_3','down_4', 'down_5']] = pd.DataFrame(self.df["num_missions_voted_down_with_total_suspect_count"].to_list(), columns=['down_0', 'down_1','down_2', 'down_3','down_4', 'down_5'])
        self.df = self.df.drop('num_missions_voted_down_with_total_suspect_count', axis=1)   
        self.df = self.df.drop('spy',axis = 1)
        index = 0
        df_dict = {}
        for i in self.df.columns:
            df_dict[i] = index
            index = index + 1
        self.df = self.df.rename(columns = df_dict)
# df = (df - df.min()) / df.max() - df.min()

    def onGameRevealed(self, players, spies):
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        @param players List of all players in the game including you.
        @param spies List of players that are spies, or an empty list.
        """
        self.spies = spies
        self.failed_missions_been_on = {}
        self.training_feature_vectors = {}
        self.missions_been_on = {}
        self.num_missions_voted_up_with_total_suspect_count = {}
        self.num_missions_voted_down_with_total_suspect_count = {}
        
        self.read_df_and_normalize()

        for player in players:
            self.missions_been_on[player] = 0
            self.failed_missions_been_on[player] = 0
            self.num_missions_voted_up_with_total_suspect_count[player] = [0, 0, 0, 0, 0, 0]
            self.num_missions_voted_down_with_total_suspect_count[player] = [0, 0, 0, 0, 0, 0]
            self.training_feature_vectors[player] = []
    
    def mission_total_suspect_count(self, team):
        result = 0
        for players in team:
            result += self.failed_missions_been_on[players]

        return result
    
    def onVoteComplete(self, votes):
        """Callback once the whole team has voted.
        @param votes Boolean votes for each player (ordered).
        """
        suspects = min(self.mission_total_suspect_count(self.game.team),5)
        
        j = 0
        for player in self.game.players:
            self.num_missions_voted_up_with_total_suspect_count[player][suspects] += votes[j]
            self.num_missions_voted_down_with_total_suspect_count[player][suspects] += not votes[j]
            j += 1
            
        for player in self.game.players:
            self.training_feature_vectors[player].append(
            [self.game.turn, self.game.tries, player.index, player.name, self.missions_been_on[player],
            self.failed_missions_been_on[player]] + self.num_missions_voted_up_with_total_suspect_count[player] +
            self.num_missions_voted_down_with_total_suspect_count[player])



