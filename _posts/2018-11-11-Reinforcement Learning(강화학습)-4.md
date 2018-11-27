---
title : Reinforcement learning 4 Q-learning exploit & exploration
category : 
 - Reinforcement learning (강화학습)
 - machine learning
tags : machine learning
---

> 4번째 시간. Exploit & Exploration 방법에 대해 알아보자.

<!-- more -->

## 도입

기존의 Q-learning 방법에서 action의 선택은 무조건 다음선택에서 받을 수 있는 모든 reward값의 합, 즉 maxQ(s',a')+reward를 기준으로 선택했다. 하지만 이렇다면 무조건 하나의 해결방법(길)만 고수하는 결과가 나온다. 그래서 action을 시도할 때 무작위성이 어느 정도 필요하다.
## Exploit, Exploration - decaying E-greedy
![1](https://user-images.githubusercontent.com/28972289/48314857-00fabe80-e613-11e8-8d9e-bbd3db5ec1ee.JPG)  
이 방법은 e값을 정하고 랜덤한 수를 뽑았을 떄 e 보다 작을경우 action을 랜덤으로, 그렇지 않을 경우 argmax를 선택한다.  
## Exploit, Exploration - add random noise
![2](https://user-images.githubusercontent.com/28972289/48314858-01935500-e613-11e8-9e19-671cb8908ba7.JPG)  

이 방법은 action을 선택할 때 각각에 random noise를 추가한다. random noise와 Q(s,a)를 합한 값에서 가장 높은 값이 선택되므로 원래 Q(s,a)의 첫 번쨰로 높은값, 그리고 2, 3번째로 높은값이 차례대로 선택될 확률이 높다.  
고로 E-greedy처럼 아예 무작위가 아닌 확률이 높은 순서대로 랜덤을 고르고 싶다면 random noise가 더 좋은 방법이다.  

## Discount reward
![3](https://user-images.githubusercontent.com/28972289/48314854-00622800-e613-11e8-87dc-d94e2ad2b2fd.JPG)  

Discount reward는 이 게임같은 경우에는 최단거리의 해가 더 좋은 값을 나타내도록 한다. maxQ(s',a')에 r을 곱해 더 멀리 돌아가는 길일 수록 reward를 discount한다.  
![4](https://user-images.githubusercontent.com/28972289/48314855-00fabe80-e613-11e8-82d8-b9c8dda60494.JPG)  

마치 이 그림처럼 아래쪽은 0.9이지만 왼쪽으로 돌아가는 길을 0.72이다.  

### Code 1 - discount, add decaying noise
```python
'''
Created on 2018. 11. 10.

@author: wnlwn
'''
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4' , 'is_slippery' : False}
)
env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000
dis = .99 # discount factor


rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    
    while not done:
        action = np.argmax(Q[state, : ] + np.random.randn(1, env.action_space.n) / (i+1)) # add noise 
 
        new_state, reward, done,_ = env.step(action)
        
        Q[state,action] = reward + dis * np.max(Q[new_state,:])
        
        rAll += reward
        state = new_state
        
    rList.append(rAll)
    

print("Success rate: "+ str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)),rList, color="blue")
plt.show()

```  

### Code 2 - discount, decaying e-greedy 

``` python
'''
Created on 2018. 11. 10.

@author: wnlwn
'''

'''
Created on 2018. 11. 10.

@author: wnlwn
'''
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4' , 'is_slippery' : False}
)
env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000
dis = .99 # discount factor


rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    e = 1. / ((i//100)+1)
    while not done:
        # action = np.argmax(Q[state, : ] + np.random.randn(1, env.action_space.n) / (i+1)) # add noise 
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, : ]) #best way
        
        new_state, reward, done,_ = env.step(action)
        
        Q[state,action] = reward + dis * np.max(Q[new_state,:])
        
        rAll += reward
        state = new_state
        
    rList.append(rAll)
    

print("Success rate: "+ str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)),rList, color="blue")
plt.show()
```  
각각 discount reward 방식이 적용되어 있고 decaying noise, decaying e-greedy 방식이다.

## Conclusion

![5](https://user-images.githubusercontent.com/28972289/48314856-00fabe80-e613-11e8-8f0d-902cca1d4d77.JPG)  

그래서 결국 중요한건 이것이 converge하느냐? 그렇다. 하지만 deterministic한 환경과 finite state인 경우 두 가지를 만족하는 경우에만 이다. 그렇다면 nondeterministic 한 상황은...?
