# Ad Click Prediction: a View from the Trenches
Ad Click Prediction: a View from the Trenches(link)[https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf]

## 1. Introduction
온라인 광고 시장은 기계 학습이 성공한 사례 중 하나이다.
검색 광고, contextual 광고, 디스플레이 광고, 실시간 경매는 ctr을 정확하고, 빠르고, 안정적으로 예측하는 모델을 학습하는 것에 의존성을 가지고 있다.
지난 10년간 시간이 흐름에 따라, 이러한 문제는 거대한 규모의 이슈를 대응하도록 다루어졌다.
일반적인 산업 모델을 매일 10억건의 예측을 제공하고, 엄청나게 큰 feature 공간을 활용하며, 방대한 양의 데이터로 부터 학습한다.

본 논문에서는 구글 검색광고에서 여러가지 세팅의 실험에 대해 분석한다.
또한 메모리 절약, 성능 분석, confidence in predictions, calibration, feature 관리등의 이슈를 다룬다. 

## 2.Brief system overview
사용자가 q를 검색하면, 광고주가 고른 키워드에 따라 q와 "후보 광고" 집합이 매칭된다. 
경매 메커니즘이 사용자에게 광고가 보여질지 아닐지, 어떤 순서로, 어떤 가격(광고주가 클릭이 된 경우 집행할 금액)으로 할지 결정한다.
광고주들의 경매에서는 이 광고가 보여졌을 때 클릭이 될 확률이 중요한 입력이 된다.

feature는 검색 쿼리, 광고 소재, 다양한 광고 관련 메타데이터들에서 활용된다. 
데이터는 주로 엄청 sparse하므로, 0이 아닌 아주 작은 비율의 feature만 가지게 된다.

regularized logistic regression은 이러한 문제에 잘 맞는 방법이다.
모델은 매일 몇 십억건의 예측을 이루어야하고, 새로운 클릭과 클릭이 안된 건들에 대해 빠르게 업데이트 해줘야한다.
당연하게도 학습을 위한 데이터는 방대한 양을 가지고 있습니다.
데이터는 Photon system을 바탕으로한 스트리밍 서비스를 통해 제공받는다.

여러 층으로 쌓은 모델이 아니라 한층짜리 모델을 학습하는 것을 제외하고, 학습 방법은 구글 브레인 팀에서 설명한 Downpour SGD와 유사하다.
이 방법으로 인해 수십 억의 계수를 가진 엄청나게 큰 모델과 데이터를 처리할 수 있다.
학습된 모델들이 서빙을 위해 여러 데이터 센터로 복제되기 때문에, 학습보다는 서빙 시간의 sparsification에 더욱 집중할 수 있다.


## 3. Online learning and sparsity
방대한 크기의 학습에서는 일반화된 선형 모델의 온라인 알고리즘은 많은 장점을 가진다.
feature vector x는 몇 십억 차원을 가지지만 실제로는 몇 백 정도의 0이 아닌 값을 가진다.
이러한 점은 학습 당시 한 번만 고려되는 스트리밍으로 하면 효과적으로 학습할 수 있다.

학습을 위한 gradient는 다음과 같다. 
> ▽lt(w) = (σ(w · xt) − yt)xt = (pt − yt)xt
현실에서는 최종 모델의 크기가 중요한 고려 요소이므로, sparse한 모델의 계수들이 어떻게 저장되느냐에 따라 메모리 사용양이 달라진다.
OGD(Online Gradient Descent)는 이러한 sparse 모델을 생성하는것에 효과적이지 않다.
단순히 L1 penalty를 더하는 것만으로도 zero인 계수를 만들 수 없다. 

FOBOS나 truncate gradient 와 같은 조금 더 정교한 방법들은 sparse한 상태에서 성공했다. 
RDA(Regularized Dual Averaging)알고리즘은 FOBOS 보다 더 나은 정확도 vs sparsity 트레이드 오프를 보였다.
RDA의 sparsity에 강한 점과 OGD의 높은 정확도를 둘다 지닐 수 있는 방식으로 FTRL-Proximally 알고리즘이 있다.
이 알고리즘은 regularization 없이 OGD와 동일한 효과를 보이나, 모델의 계수 w를 alternative lazy representation을 활용해서 더욱 효과적으로 나타낼 수 있다. 

FTRL은 이론적으로 분석하기 편하게 만드는 방법으로 알려져있다. 
수식은 기존 OGD보다 복잡해 보이고 구현이 어려울 것처럼 보이지만, 계수당 하나의 수만 저장하고 있으면 된다.
실험을 통해 FTRL-Proximal은 다른 알고리즘들과 비슷하거나 나은 정확도를 보이면서, sparsity 면에서 상당한 개선을 보인다.

### 3.1 Per-Coordinate Learning Rates
기존의 OGD는 전체적으로 하나의 learning rate를 활용한다. 
그러나 각각의 독립적인 시행에 대해 하나의 학습으로 묶는 것은 좋지 못한 방법이다.
실험을 통해 하나로 묶지 않고 각 시행마다 다른 per-coordinate learning rate를 적용하면 더 좋은 결과를 보인다.

## 4. Saving Memory at Massive Scale
메모리 절약을 위해서 L1 regularization을 활용했다.

### 4.1 Probabilistic feature inclusion
높은 차원의 데이터를 활용하는 많은 도메인에서, 대부분의 feature들은 극도로 희귀(rare)하다. 
이렇게 희귀한 feature의 통계값을 지속적으로 추적하기에는 비용이 많이 들고, 어떤 feature가 희귀할지 알 수 없다.
온라인 환경에서는 희귀한 데이터를 전처리로 지우는 것이 어려운 문제이다. 
데이터의 추가 read와 write는 비싸고, 몇몇 feature가 지워지면 해당 피처를 사용하는 다른 모델을 정확도를 올리기 위한 전처리 비용을 측정하고 비교하는 것에 활용할 수 없게된다.

이러한 희소성 문제를 해결하는 방법 중 하나는 계수가 0인 어떠한 feature의 통계를 추적하지 않아도 되는 L1 regularization이다.
이 방법을 활용하면, 학습 과정에서 정보가 작은 feature는 제거된다.
그러나 이러한 방법은 학습 때 더 많은 feature을 추적하고 서빙때에만 sparsify하는 방법들에 비해 안좋은 정확도를 야기한다. 
다른 방법으로는 hash를 활용한 방법이 있으나, 이 또한 크게 활용할만한 이점을 제공하지 않았다.
또 다른 방법으로는 처음 feature가 발생되었을 떄 확률에 기반하여 모델의 feature로 활용하는 probabilistic feature inclusion이 있다.
(e.g. Poisson Inclusion - 확률에 따라 feature로 활용, Bloom Filter Inclusion - filter에 집어 넣다가 n번 이상 나온 feature 추가)

### 4.2 Encoding values with fewer bits
기본적인 OGD에서는 계수를 담기 위해 32bit 또는 64bit를 활용한다.
그러나 논문에서 활용된 regularized logistic regression의 경우 해당 자료형이 필요하지 않아 너무 많은 공간을 차지한다.
이를 위해 q2.13 encoding을 활용해서 2개의 2진 정수와 소수점 아래 13개의 2진 정수로 나타내고 부호까지 16bit를 활용하도록 변경하였다.
encoding을 위해서는 roundoff error를 만들 수 있다.(그러나 명시적인 rounding을 통해 (regret term을 더 해주어) 이산화에 대한 오류를 평균이 0이 되도록 할 수 있다.)

### 4.3 Traninig many similar models








