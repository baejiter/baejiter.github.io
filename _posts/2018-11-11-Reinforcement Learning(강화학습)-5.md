---
title : Reinforcement learning 5 Q-learning exploit & exploration
category : 
 - Reinforcement learning (강화학습)
 - machine learning
tags : machine learning
---

> 5번째 시간. Nondeterministic world에서의 Q-learning은?

<!-- more -->

## 도입

![6](https://user-images.githubusercontent.com/28972289/48315321-01498880-e618-11e8-88ea-07ff3d894678.JPG)  

지금까지는 deterministc, 즉 같은 인풋에 같은 아웃풋을 나타내는 상황에서 Q learning 알고리즘을 공부했다. 이제 nondeterministic, stochastic이라고도 불리는 같은 input에 대해서도 다른 output이 나올 수 있는 상황에서 Q-learning을 살펴보자.  


## Nondeterminitic Environment
![7](https://user-images.githubusercontent.com/28972289/48315322-01e21f00-e618-11e8-9cce-37357a8a4bb3.JPG)  

nondeterministic 상황에서 기존의 q-learning알고리즘이 먹힐까?. 실행해본 결과 0.1도 안되는 정확도가 나온다. 당연한 결과이다. 결국 우리는 조금 다른 알고리즘이 필요하다.  

## Learning rate
![8](https://user-images.githubusercontent.com/28972289/48315318-00b0f200-e618-11e8-9b49-5da68fa0650d.JPG)  

sthochastic 환경에서는 Q function이 정확하지 않다. 내가 의도한 input에 따라 Q function이 작동하지 않는다는 뜻이다. 고로 이 Q function의 말을 조금만 듣고, 내가 가지고 있는 Q(s,a)의 말도 고수한다. 어느 정도의 가중치를 두고 들을 것인가는 learning rate로 결정한다.  
실제 이 결과 0.5~0.7정도의 정확도가 나온다.  
이론적으로는 이해가 되지만 아무리 생각해도 이게 가능한 이유를 모르겠다. 내가 가지고 있는 Q(s,a) 또한 옳은 값이아니다. 내가 준 input의 결과가 이상할 수도, 아닐 수도 있는 값인데 결국엔 이 값을 고수하며 추가로 신뢰도가 그리 높지 않은 값을 또 조금 참고한다. 두 값을 이리저리 참고하지만 결국엔 둘 다 내가 의도할 수 없는 부정확한 값들 아닌가?  
 

### Code
```python
''''
Created on 2018. 11. 10.

@author: wnlwn
'''
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr
from kbhit import KBHit
from colorama import init

# init(autoreset=True)


env = gym.make('FrozenLake-v0')
# env.render();

Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000
dis = .99 # discount factor
learning_rate= .85
key = KBHit()

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    
    
    while not done:
        action = np.argmax(Q[state, : ] + np.random.randn(1, env.action_space.n) / (i+1)) # add noise 
#         action = key.getarrow();
        new_state, reward, done,_ = env.step(action)
#         env.render()
        
        Q[state,action] = (1-learning_rate) * Q[state,action] \
        + learning_rate*(reward + dis * np.max(Q[new_state, :]))
        rAll += reward
        state = new_state
#         print(Q)

        
    rList.append(rAll)
    

print("Success rate: "+ str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)),rList, color="blue")
plt.show()

```  

## Conclusion

![9](https://user-images.githubusercontent.com/28972289/48315319-01498880-e618-11e8-882e-09ca42dff83d.JPG)
 
결국 완벽히 이해하지는 못했지만 이런 방법이 먹힌다는 것을 알게 되었다. 다음은 Q array를 쓰는 방법이 아닌 Q-network에 대해 알아보자
