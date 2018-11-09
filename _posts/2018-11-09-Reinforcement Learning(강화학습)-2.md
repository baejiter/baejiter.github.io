---
title : Reinforcement learning 2
category : 
 - Reinforcement learning (강화학습)
 - machine learning
tags : machine learning
---

> 2번째 시간. 설치 및 설정. OpenAI GYM 예제 실행.

<!-- more -->

##도입

리눅스로 하려했지만 일단 윈도우즈에서 진행해본다.
나중에 리눅스 써봐야지.

##간단한 예제
![3](https://user-images.githubusercontent.com/28972289/48269643-75025e80-e47b-11e8-9bc7-41bfc9025068.JPG)  

저번에 말했다시피 Agent는 Action을 취한다. 이 때, Environment속에서 행동을 취하는 것이며 이 Env를 구성하는건 매우 복잡하기도하다. 이번같은 경우엔 OpenAI GYM에서 편리하게도 Atari game 등등을 제공해준다.

##Code
```python
import gym
from gym.envs.registration import register
from colorama import init
from kbhit import KBHit
# import msvcrt

init(autoreset=True)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4' , 'is_slippery' : False}
)

env = gym.make('FrozenLake-v3')
env.render()

key = KBHit()

while True:
    
    action = key.getarrow();
    
    if action not in [0, 1, 2, 3]:
        print("Game aborted!")
        break
    
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ",info)
    
    if done:
        print("Finished with reward", reward)
        break   
```  
대충 환경만들고, 키를 입력받고 액션을 취한다음에 state, reward, done?, info를 받아 끝날 때 까지 반복수행. 

##Conclusion

오랜만에 파이썬 써보니 재밌다.
