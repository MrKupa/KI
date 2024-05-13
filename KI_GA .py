#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import math
import gzip
from io import BytesIO


gz_dateipfad = "C:/Users/Anwender/OneDrive - TH Köln/Desktop/berlin52.tsp.gz"



class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
#breechne distanz
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return np.round(distance)
#output cities as coordinates
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

#Berechnung der Fitness durch kehrwert der länge 
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
#inverse of route distance. minimizee route distance so a larger fitness score is better
    def routeFitness(self): # fitnessScore
        if self.fitness == 0:
            dis = self.routeDistance()
            self.fitness = dis
        return self.fitness

#random route wie wir die Städte besuchen
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

#Step 1: Start Population und random routes erstellen in höher der Population
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

#Determine fitness: Wir ranken in einer liste die population
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=False)

#3.Parent selection in 2 steps
#determine which routes to select in select function mit Turnier Selections Operator für die restlichen nicht elitären


def selection(popRanked, eliteSize):
    selectionResults = []
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    popRanked_pre = popRanked[:len(popRanked)]
    for i in range(0, len(popRanked) - eliteSize):
        c1 = random.sample(popRanked_pre, 1)#zufällige auswahl aus der population
        c2 = random.sample(popRanked_pre, 1)
        winner = None
        if c1[0][1] > c2[0][1]:
            winner = c1
        else:
            winner = c2
        selectionResults.append(winner[0][0])

    return selectionResults



#selektieren 
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#crossover
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

#genrate offspring population, use elite to retain the best routes from the current population
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

#Mutation durch tausch von 2 städten
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#mutierte population
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#funktion um neue generation zu erstellen
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    return children


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        print(i)
        progress.append(rankRoutes(pop)[0][1])
  
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.title('Progress')
    plt.show()
    print("Final distance: " + str(rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def plotRoute(route):
    x = [cities.x for cities in cityList]
    y = [cities.y for cities in cityList]
    plt.scatter(x, y)
    plt.plot(x, y, color='red')
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title('Rute ')
    plt.show()

cityList = []
with open(r'C:\Users\Anwender\Downloads\TSP_data.txt','rt') as f:
    for line in f:
        a, b, c = read_line(line)
        cityList.append(City(x=b, y=c))


        
best = geneticAlgorithmPlot(population=cityList, popSize=2000, eliteSize=1000, mutationRate=0.01, generations=2000)
plotRoute(best)



# In[ ]:




