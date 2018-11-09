---
title : Reinforcement learning 1
category : 
 - Reinforcement learning (강화학습)
 - machine learning
tags : machine learning
---

> Introduction. 강화학습에 관한 소개.

<!-- more -->

##도입

Atari 게임 중 하나를 골라 강화학습을 해보는 주제로 공부하기로 함.  
[참고](https://hunkim.github.io/ml/)
여기서 강의듣고 공부하였고 자료를 받았다.

##What is RL?
강화학습이란 머신러닝의 한 종류로 행동심리학에서 영감을 받았으며, 어떤 환경 안에서 정의된 에이전트가 현재의 상태를 인식하여, 선택 가능한 행동들 중 보상을 최대화하는 행동 혹은 행동 순서를 선택하는 방법이다. 매우 포괄적이어서 다양한 분야에 쓰인다.(위키참조)  
+오래전부터 있던 개념이며 Alphago의 핵심 알고리즘 중 하나가 RL이다. 이 알고리즘으로 딥마인드 AI Cooling Bill을 40% 감소 시켰다.

###Example
![2](https://user-images.githubusercontent.com/28972289/48269049-f658f180-e479-11e8-81b1-8977e3567300.JPG)  

로보틱스(joint의 사용량), E-commerce(어떤 컨텐츠를 보여줄 것인가.. reward = visit time), Business(유통 등등 인벤토리 재고) 등에 사용가능.

##Overview

![1](https://user-images.githubusercontent.com/28972289/48269040-f22cd400-e479-11e8-9962-dddecf1e90bb.JPG)  
기본적인 개요는 그림과 같이 State와 action 그리고 reward가 있으며 agent의 state는 action에 따라 바뀌며 어떤 action을 취할 때 마다 reward가 있을 수도, 없을 수도 있다.