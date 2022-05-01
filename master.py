from Iris_TTT4275 import prob1, prob2, histogram
from MNist_ttt4275.classifier import main

print("========= Running Iris tasks =========")
print("Problem 1 running")
prob1.prob1(30, 20)
prob1.prob1(20, 30)

print("Problem 2 running")
histogram.histogram()
prob2.prob2(30,20)

print()
print()
print("========= Running MNist tasks =========")
main.run()
