# AI for Self Driving Car

#Importing the liberias

import numpy
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) # Setter opp hvor mange hidden layer neurons det er. Her er det altså 30
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state)) #Activate hidden neurons. Relu er aktiveringsfunksjon
        q_values = self.fc2(x) #Får output neurons
        return q_values
    
#Implementing Experience replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []        #Put the transistion into the memory list in push function

    def push(self, event):         #Go to next transistion. 
        self.memory.append(event)   #Put event into memory
        if len(self.memory) > self.capacity: #Delete first memory when larger than capacity
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) #Random.sample tar random sample of the memory that has a fixed size of batch_size. Zip* is a reshape function. Kan lage en ny liste hvor den ganger sammen to tidligere lister. Dette deler opp i 3 batches(state, action, reward) som kreves senere
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #Obtains a list of batches from samples where each batch is a pytorch variable

#implementing Deep Q Learning

class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)  #initianting neural network
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #Danner en tensor som blir pyttet inn i neural network. Denne har en extra dimensjon. Derav unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state): #State == Inputstate. Vil senere være torchtensor. For å gjøre algoritmen bedre tar man state om til en torch variable 
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) #Calculates the probability of on action based on the Q value. Using softmax. T = 7. Temperature is telling us how sure the nn is before taking a action. Closer to zero means less sure
        action = probs.multinomial(1) #Gives random draw from probs 
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): #Implementing back propegation. Det er dette teorien om Deep Q-learning kommer inn
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #Gir actions played. Gir unsqueeze siden vi har det i last_state. Noe som gjør at de får like mange dimensjoner. Fjerner denne fake state med squeeze() for å gi en simple vector ikke tensor
        next_outputs = self.model(batch_next_state).detach().max(1)[0]  #Need next_output to find target
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)     #Temporal difference loss
        #Back propegation
        self.optimizer.zero_grad() #Zero grad reinitialise the optimser at each loop
        td_loss.backward(retain_graph = True) #Back propegates the error into the network
        self.optimizer.step() #Update weights with optimizer

    def update(self, reward, new_signal): #Updates everything we need to update when reaching a new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #Lagrer verdier i memory
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:   #Beginng learning from 100 transistions
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward #Reward updates i map.py
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:  #Begrenser reward-window til 1000 rewards
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)      #Returns the average of the rewardwindow
    
    def save(self):      #Saves model and optimizer
        torch.save({"state_dict" : self.model.state_dict(),
                    "optimizer" : self.optimizer.state_dict(),
                    }, "last_brain.pth")
        
    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("=> loading checkpoint")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Done!")
        else:
            print("No checkpoint found...")


