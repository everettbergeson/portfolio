# %%
import numpy as np
import pandas as pd
# %%
#
# 3 player game
#
#

players = ['A', 'B', 'C']
scores = {i:0 for i in players}
for round in range(10):
    print(f"\n\nRound {round}")
    # Each player has a different threshold:
    # A leaves at 20
    # B leaves at 40
    # C leaves at 100
    round_score = 0
    A_in = True
    B_in = True
    C_in = True
    while True:
        roll = np.random.randint(1, 7)
        if roll == 1:
            if A_in and B_in and C_in:
                round_score = 0
                print("0'ed out but everyone was still in")
            else:
                print("Roll: 1\nRound ends!")
                break
        else:
            round_score += roll
            print(f'Roll: {roll}\tRound score: {round_score}')
            if A_in and round_score > 20:
                print('A out!')
                scores['A'] += round_score 
                A_in = False
            if B_in and round_score > 40:
                print('B out!')
                scores['B'] += round_score 
                B_in = False
            if C_in and round_score > 100:
                print('C out!')
                scores['C'] += round_score 
                C_in = False
    print(scores)
max(scores, key=scores.get)

# %%
#
# N player game
#
#
n = 10
scores = {p:0 for p in range(n)}
# Each player p has a threshold of (p+1)*10
threshold = {p:(p+1) * 10 for p in range(n)}

for round in range(10):
    print(f"\n\nRound {round}")
    
    round_score = 0
    status = {p:True for p in range(10)}
    while True:
        roll = np.random.randint(1, 7)
        if roll == 1:
            if sum(status.values()) == n:
                round_score = 0
                print("0'ed out but everyone was still in")
            else:
                print("Roll: 1\nRound ends!")
                break
        else:
            if roll == 2:
                round_score = round_score * 2
            else:
                round_score += roll
            print(f'Roll: {roll}\tRound score: {round_score}')
            for p in range(n):
                if status[p] and round_score > threshold[p]:
                    scores[p] += round_score
                    status[p] = False
                
    print(scores)
max(scores, key=scores.get)
            

# %%
winners = []
for j in range(100000):
    n = 20
    scores = {p:0 for p in range(n)}
    # Each player p has a threshold of (p+1)*10
    threshold = {p:(p+1) * 30 for p in range(n)}

    for round in range(15):
        # print(f"\n\nRound {round}")
        
        round_score = 0
        status = {p:True for p in range(n)}
        while True:
            roll = np.random.randint(1, 7)
            if roll == 1:
                if sum(status.values()) == n:
                    round_score = 0
                    # print("0'ed out but everyone was still in")
                else:
                    # print("Roll: 1\nRound ends!")
                    break
            else:
                if roll == 2:
                    round_score = round_score * 2
                else:
                    round_score += roll
                # print(f'Roll: {roll}\tRound score: {round_score}')
                for p in range(n):
                    if status[p] and round_score > threshold[p]:
                        scores[p] += round_score
                        status[p] = False
    winners.append(max(scores, key=scores.get))

print(pd.Series(winners).value_counts())
# %%
for i in range(n):
    print((i*10), '-', (i+1)*10, sum([j == i for j in winners]) * 100 / len(winners))
# %%
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, 1)

ax.hist([(w+1) * 30 for w in winners], bins=39, density=True)
ax.set_xlabel("Keep rolling until the round's total is X")
ax.set_ylabel("Win percentage in a 15 round game")
fig.suptitle("Pig Play Strategies")
# %%
scores = []
for j in range(100000):
    for round in range(15):
        # print(f"\n\nRound {round}")
        
        round_score = 0
        while True:
            roll = np.random.randint(1, 7)
            if roll == 1:
                # if round_score < 20:
                #     round_score = 0
                # else:
                break
            else:
                if roll == 2:
                    round_score = round_score * 2
                else:
                    round_score += roll
            if round_score > 500:
                round_score = 500
                break
        scores.append(round_score)
            
# %%
plt.hist([s for s in scores if s > 20], bins=30, density=True)
plt.title("Distribution of rolls > 20")
plt.xlabel("Rolls above 500 capped at 500")
# %%
# Median roll: 14
# Median roll if only keeping above 20: 53
pd.Series([s for s in scores if s > 20]).median()
# %%

