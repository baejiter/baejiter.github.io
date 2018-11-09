---
title : Reinforcement learning 3 - Q Learning
category : 
 - Reinforcement learning (강화학습)
 - machine learning
tags : machine learning
---

> 3번째 시간. Q learning에 대해 알아보자.

<!-- more -->

##도입

강화학습을 어떻게 구현할 것인가. 전 예제에서 Agent는 다음 step으로 가기전에는 그곳이 어떤 곳인지 알 수 없다. 더구나 게임이 끝나고 나서야 성공한지에대한 reward가 주어지니 agent는 답답할 수 밖에 없다.  
그래서 Q라는 친구에게 물어보며 이동하기로 한다.  
![4](https://user-images.githubusercontent.com/28972289/48269734-a549fd00-e47b-11e8-92b4-d5b7b4c221e5.JPG)  
![5](https://user-images.githubusercontent.com/28972289/48269735-a549fd00-e47b-11e8-871c-0a88e7b2be7c.JPG)  
![6](https://user-images.githubusercontent.com/28972289/48269737-a5e29380-e47b-11e8-8bbc-48ba7b6ffa2b.JPG)  
![7](https://user-images.githubusercontent.com/28972289/48269738-a5e29380-e47b-11e8-85a2-c4ea2577f996.JPG)

Q-function은 argument와 state가 주어졌을 때 Q^(s',a')의 값을 안다고 가정하고 시작한다. 그렇게 가정하면 Optimal Policy는 argmaxQ(s,a)이며 공식은 마지막 그림에서와 같이 Q(s,a) = r + maxQ(s',a')된다.

##예제

```python
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

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    
    while not done:
        action = rargmax(Q[state,:])
        
        new_state, reward, done,_ = env.step(action)
        
        Q[state,action] = reward + np.max(Q[new_state,:])
        
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
이 코드는 결국 몇 번의 시도를 거듭하며  


![8](https://user-images.githubusercontent.com/28972289/48269733-a549fd00-e47b-11e8-9d29-62ebe019d48c.JPG)  
이렇게 된다. 그리고 강화된 learning은 성능을 향상시킨다.


##Conclusion

근데 Q learning Dummy다. 왜? 다음시간에.
