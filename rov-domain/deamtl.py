import pyximport
pyximport.install()

from rover_domain_core_gym import RoverDomainGym
import teaming.ccea as ccea
import code.agent_domain_2 as domain
import code.reward_2 as rewards
import mods

from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time
import os
import mpu
import sys
import math

import torch
from torch import nn
import copy

class ApproximatorNeuralNetwork(nn.Module):
    def __init__(self):
        super(ApproximatorNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 1),
            nn.Tanh()
        )

    def forward(self, x):
        logits = self.network(x)
        return logits

# device = "cpu"
# model = ApproximatorNeuralNetwork().to(device)
# model2 = copy.deepcopy(model)

# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

# x = torch.randn(8, 8)
# y = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype = torch.float32)

# out = model(x)
# loss = loss_fn(out, y)
# print("Before Training")
# print(x)
# print(out)
# print(loss)
# print()

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print("After Training")
# print(x)
# out = model(x)
# print(out)
# loss = loss_fn(out, y)
# print(loss)


# dataSamplingRate defines how many episodes before more data is tracked about agents
def evolutionaryProcess(processNum, dataSamplingRate):

    filename = "Process" + str(processNum) + "LogDS.pickle"
    device = "cpu"

    episodes = 4000
    nagents = 5
    nagentssub = 4
    nsteps = 30

    poiPos = np.array([[0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [0.0, 0.5],
                    [1.0, 0.0]])

    poiVals = np.array([0.8, 1.0, 0.6, 0.3, 0.2, 0.1])

    agentData = {
        "Number of Agents": nagents,
        "Trains per Episode": 32
    }

    agentData["Agent Approximators"] = [[ApproximatorNeuralNetwork().to(device) for i in range(agentData["Trains per Episode"])] for i in range(nagents)]

    sim = RoverDomainGym(nagentssub, nsteps, poiPos, poiVals)
    obs = sim.reset(fully_resetting = False)
    obsSize = len(obs[0])

    sim.data["Coupling"] = 2
    sim.data["Observation Radius"] = 5
    sim.data["Trains per Episode"] = agentData["Trains per Episode"]
    sim.data["Mode"] = "Train"

    ccea.initCcea(input_shape = obsSize, num_outputs = 2, num_units = 16)(agentData)

    populationSize = agentData["Trains per Episode"]

    processData = {
        "GSigma": [],
        "Sampling Rate": dataSamplingRate,
        "Training Episodes": episodes,
        "Number of Agents": nagents,
        "Team Size": nagentssub,
        "Simulation Time Steps": nsteps,
        "POI Positions": poiPos,
        "POI Values": poiVals,
        "Population Size": agentData["Trains per Episode"],
        "Observation Size": obsSize,
        "Coupling": sim.data["Coupling"],
        "Observation Radius": sim.data["Observation Radius"],
        "Agent Rewards": [],
        "Agent States": [],
        "Agent Actions": [],
        "Position History": [],
        "Team Global Rewards": []
    }

    for episode in range(episodes):
        globalRewards = []

        # Used to track and save agent actions and states
        agentRewards = []
        agentStates = []
        agentActions = []
        positionHistory = []
        teamRewards = []

        for world in range(populationSize):
            agentData["World Index"] = world
            sim.data["World Index"] = world

            ccea.assignCceaPolicies(agentData)

            worldAgentRewards = []
            worldAgentStates = []
            worldAgentActions = []
            worldAgentPositionHistory = []

            subTeamGlobalRewards = []

            GSigma = 0

            # print("     World: " + str(world) + "/" + str(populationSize) + ", Agents: " + str(agentData["Agent Policies"]))

            for comb in itertools.combinations([i for i in range(nagents)], nagentssub):

                startingObs = sim.reset(fully_resetting = False)
                gymData = {
                    "Number of Agents": nagentssub,
                    "Agent Policies": [agentData["Agent Policies"][i] for i in comb],
                    "Agent Observations": startingObs
                }

                teamAgentStates = [startingObs]
                teamAgentActions = []
                teamAgentPositionHistory = []

                done = False
                while not done:
                    domain.doAgentProcess(gymData)
                    teamAgentActions.append(np.array(gymData["Agent Actions"]) * 2)
                    obs, reward, done, info = sim.step(np.array(gymData["Agent Actions"]) * 2)
                    #rewards.assignDifferenceReward(sim.data)
                    gymData["Agent Observations"] = obs
                    teamAgentStates.append(obs)

                #rewards.assignGlobalReward(sim.data)

                subTeamGlobalRewards.append(sim.data["Global Reward"])
                #globalRewards.append(sim.data["Global Reward"])
                
                worldAgentRewards.append(sim.data["Agent Rewards"])
                worldAgentStates.append(teamAgentStates)
                worldAgentActions.append(teamAgentActions)
                worldAgentPositionHistory.append(sim.data["Agent Position History"])

                for i in range(len(gymData["Agent Policies"])):
                    gymData["Agent Policies"][i].fitness += worldAgentRewards[-1][i]
                GSigma += sim.data["Global Reward"]
            
            #for agent in gymData["Agent Policies"]:
            #    agent.fitness = GSigma
            globalRewards.append(GSigma)

            agentRewards.append(worldAgentRewards)
            agentStates.append(worldAgentStates)
            agentActions.append(worldAgentActions)
            positionHistory.append(worldAgentPositionHistory)
            teamRewards.append(subTeamGlobalRewards)

        if(episode % dataSamplingRate == 0 or episode == episodes - 1):
            print("Process " + str(processNum) + " is tracking data at episode " + str(episode))

            processData["Agent Rewards"].append(agentRewards)
            processData["Agent States"].append(agentStates)
            processData["Agent Actions"].append(agentActions)
            processData["Position History"].append(positionHistory)
            processData["Team Global Rewards"].append(teamRewards)

        processData['GSigma'].append(np.max(np.array(globalRewards)))

        ccea.evolveCceaPolicies(agentData)
        ccea.clearFitness(agentData)

        print("Process Number: " +  str(processNum) + ", Episode: " + str(episode) + "/" + str(episodes) + ", Global Reward (Average): " + str(processData['GSigma'][-1]))
    
    #mpu.io.write(filename, processData)

evolutionaryProcess(0, 500)

# testData = {'foo': 'bar'}
# mpu.io.write('test.pickle', testData)
# unserialized_data = mpu.io.read('test.pickle')

# print(unserialized_data)

# if __name__ == '__main__':
#     for i in range(12):
#         p = Process(target=evolutionaryProcess, args=(i, 500,))
#         p.start()

"""
p0d = mpu.io.read("Process0LogDS.pickle")
p1d = mpu.io.read("Process1LogDS.pickle")
p2d = mpu.io.read("Process2LogDS.pickle")
p3d = mpu.io.read("Process3LogDS.pickle")
p4d = mpu.io.read("Process4LogDS.pickle")
p5d = mpu.io.read("Process5LogDS.pickle")
p6d = mpu.io.read("Process6LogDS.pickle")
p7d = mpu.io.read("Process7LogDS.pickle")
p8d = mpu.io.read("Process8LogDS.pickle")
p9d = mpu.io.read("Process9LogDS.pickle")
p10d = mpu.io.read("Process10LogDS.pickle")
p11d = mpu.io.read("Process11LogDS.pickle")

results = [p0d, p1d, p2d, p3d, p4d, p5d, p6d, p7d, p8d, p9d, p10d, p11d]

x = np.linspace(0, 4000, 4000)

print(results[9]["GSigma"][-1])
print(results[9]["Team Global Rewards"][-1][2])
#print(results[9]["Position History"][-1][0][-1])

f, (ax1, ax2) = plt.subplots(1, 2)

xPoi = []
yPoi = []
for poi in results[9]["POI Positions"]:
    xPoi.append(poi[0] * 30)
    yPoi.append(poi[1] * 30)
ax2.scatter(xPoi, yPoi)

for agent in range(results[9]["Team Size"]):
    xPos = []
    yPos = []
    for step in range(results[9]["Simulation Time Steps"]):
        xPos.append(results[9]["Position History"][-1][2][-1][step][agent][0])
        yPos.append(results[9]["Position History"][-1][2][-1][step][agent][1])
        #print(str(results[9]["Position History"][-1][0][-1][step][agent]))
    ax2.scatter(xPos, yPos)
    ax2.plot(xPos, yPos)


GSigmaAverageD = np.array([np.average(np.array([p0d["GSigma"][i],
                                               p1d["GSigma"][i],
                                               p2d["GSigma"][i],
                                               p3d["GSigma"][i],
                                               p4d["GSigma"][i],
                                               p5d["GSigma"][i],
                                               p6d["GSigma"][i],
                                               p7d["GSigma"][i],
                                               p8d["GSigma"][i],
                                               p9d["GSigma"][i],
                                               p10d["GSigma"][i],
                                               p11d["GSigma"][i]])) for i in range(len(p0d["GSigma"]))])

GSigmaStandardErrorD = np.array([(np.std(np.array([p0d["GSigma"][i],
                                                 p1d["GSigma"][i],
                                                 p2d["GSigma"][i],
                                                 p3d["GSigma"][i],
                                                 p4d["GSigma"][i],
                                                 p5d["GSigma"][i],
                                                 p6d["GSigma"][i],
                                                 p7d["GSigma"][i],
                                                 p8d["GSigma"][i],
                                                 p9d["GSigma"][i],
                                                 p10d["GSigma"][i],
                                                 p11d["GSigma"][i]])) / math.sqrt(12)) for i in range(len(p0d["GSigma"]))])

ax1.plot(x, GSigmaAverageD, color="blue")
ax1.fill_between(x, GSigmaAverageD - GSigmaStandardErrorD, GSigmaAverageD + GSigmaStandardErrorD, alpha = 0.35)
#plt.show()
"""