# Ad Click Prediction: a View from the Trenches
[Ad Click Prediction: a View from the Trenches](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)

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
하이퍼 파라미터나 feature의 변경을 실험할 때, 하나의 form의 많은 variant을 평가하는 것은 유용하다.
고정된 모델을 활용하면서 다른 variation의 residual error를 평가하는 방법을 활용하면 저렴하다. 
그러나 이러한 방법은 feature의 제거나 alternate learning 세팅에서는 어려움이 있다.

몇몇 계수들은 여러 모델 variants들 사이에서 공유될 수 있지만, 일부 데이터들은 모델의 특수 값들이 때문에 사이에서 공유될 수 없다.
모델의 계수를 hash table을 통해 저장하면, 모든 variant들에 대해 하나의 table을 활용할 수 있다. 
여러 모델을 한번에 학습 시키면 공유할 수 있는 데이터들이 늘어 메모리를 아낄 수 있을 뿐만아니라, 네트워크 대역, CPU, 디스크 공간을 아낄 수 있다. 

### 4.4 A single value structure
일부 feature의 추가나 제거의 차이만 있는 다양한 모델을 평가하고자 할 때, single value structure을 활용할 수 있다.
이는 모든 모델이 공유하는 하나의 계수만 저장하는 방식이다.

### 4.5 Computing learning rates with counts
모든 확률이 일정하다고 할 때, approximation을 활용해서 N과 P의 수만 기록하고 추적하도록 설정할 수 있다.
logistic regression의 경우, negative event의 gradient가 p일 때 positive event에 대한 gradient를 p-1로 나타낼 수 있다. 
이렇게 approximation을 활용하는 경우, 더 작은 공간을 차지하지만 모든 합을 활용해 계산한 learning rate와 비슷한 결과를 보인다.

### 4.6 Subsampling training data
일반적으로 CTR은 50% 이하이므로, click이 된 경우가 상대적으로 희귀하다.
그러므로 클릭에 대한 단순한 통계값들이 CTR을 학습하는 데에 더 높은 가치를 지닌다.
우리는 이러한 특징을 활용해서 정확도에 최소한의 영향을 주며 학습 데이터를 줄일 수 있다.
- 모든 클릭을 활용하고,
- 클릭이 되지 않은 경우 일정 확률 _r_에 의해 활용한다.

이러한 sampling 데이터는 모델의 예측에 bias를 야기할 수 있지만, 단순한 weight를 적용하면서 해결할 수 있다.

## 5 Evaliating Model Performance
모델의 성능을 평가하는 것은 쌓여있는 과거의 로그를 활용하면 저렴하게 할 수 있다. 
AucLoss(1 - AUC), LogLoss, SquaredError를 계산하여 모델의 성능을 평가하였다.

### 5.1 Progressive validation
논문에서는 데이터 셋에서 이루어지는 cross-validation이나 evalution보다는 Progressive validation(online loss)를 활용했다.
online loss는 100%의 데이터를 활용하기 때문에 데이터셋에서 이루어지는 통계들보다 더 좋은 통계이다.
절대적 평가지표 값은 종종 잘못될 수 있다. 그러므로 상대적 변화를 바라보고 baseline보다 몇 % 개선되었는지 측정한다.

### 5.2 Deep understanding through visualization
종합된 성능 평가지표는 데이터 각각의 작은 특정 변화의 효과를 가릴 수 있다.
다양한 슬라이스의 데이터에 따라 다른 결과를 보일 수 있으므로, 데이터의 종합을 효과적으로 볼 수 있는 시각적인 요약은 필수적이다. 
이에 따라 GridViz를 개발하여 확인할 수 있도록 했다.

## 6. Confidence Estimates
많은 어플리케이션에서는 CTR을 계산하는 것도 중요하지만, 예측한 값의 기대 정확도도 중요한 요소이다. 
이에 따라 더 많은 데이터의 수집을 위해 데이터가 부족한 것들을 더 틀어줄 수도 있다.

또한 예측 시 비싸지 않게 측정이 가능해야한다. 
이를 측정하기 위해 예측 정확도를 정량화하는 uncertainty score라는 휴리스틱 기법을 제안한다. 
학습 알고리즘들은 learning rate 조절을 위해 내부적으로 불확실성을 위한 카운터를 유지한다.
해당 값을 이용해 계산하면, 불확실성 점수는 다음과 같이 하나의 내적 식으로 표현할 수 있다. 
> u(x) ≡ αη · x

## 7. Calibrating Predictions
경매를 실행하기 위해서는 정확하고 잘 보정된 예측이 필수적이다.
또한 calibration으로 CTR예측 모델과 경매를 위한 최적화를 구분할 수 있다.
실제 집행된 CTR과 예측값들의 차이는 부적절한 모델 가설, 학습 알고리즘 내에서의 결핍, hidden feature의 제공 차이 등 다양한 이유에서 발생할 수 있다.

이러한 문제를 해결하기 위해 calibration layer를 둘 수 있다. 
간단한 방법으로는 aggregate 된 데이터에서 포아송 회귀를 활용해 아래의 식에서 γ와 κ를 학습하는 방법이 있다. 
> τ(p) = γp^κ

조금 더 일반적으로는 편향 곡선의 복잡함에 대응하기 위해 일부에 조금씩 선형 함수나 상수 수정 함수를 두는 방법이 있다.
τ의 유일한 제약조건은 단조 증가해야 한다는 것이다.
이를 위해 단조 증가 회귀(isotonic re에 가중 최소 제곱을 계산하여 맞는 값을 찾을 수 있고, 이러한 조각별 선형 접근은 예측값의 최소 최대에서 편향치를 감소시킨다. 

## 8. Automated Feature Management
수 많은 모델을 여러 곳에서 활용하는 구조이기 때문에 자동화된 입력 신호 관리 체계가 필요하다.
광고가 표기된 언어나 국가 등과 같은 여러 모델이 활용할 수 있는 많은 입력 신호를 관리하기 위해 metadata index를 개발하여 활용한다.
이로 인해 다양한 입력 신호는 관리 종료, 플랫폼 특화된 사용 가능 여부, 도메인 특성에 따른 적용여부 등이 자동으로 관리된다.

새로운 신호는 자동으로 테스트 되며 심사된다. 
이러한 자동화된 신호 소비 관리는 많은 학습이 한 번에 정확하게 완료될 수 있도록 한다.
또한 이는 중복된 엔지니어링 노력의 낭비를 막을 수 있고, 엔지니어링 시간을 아껴준다.

## 9. Unsuccessful Experiments
### 9.1 agressive feature hashing
최근 대규모 학습을 위해 hashing이 좋은 효과를 발휘하였으나(spam filtering, display advertisement data), 해당 실험에서는 졸은 효과를 보이지 못했다. 
### 9.2 dropout
dropout은 feature 집단에서 bagging을 하는 것과 같이 regularization의 효과를 가져와 DNN에서 좋은 효과를 보였다.
본 연구에서도 0.1에서 0.5까지 dropout을 시도했으나, 효과를 보이지 못했다.
연구자들은 ctr 예측 문제에서 feature의 sparsity로 인해 feature의 분포가 달라서 큰 효과를 얻지 못한 것이라고 생각한다. 

### 9.3 feature bagging
데이터 마이닝 때 decision tree를 활용한 방법에서, 앙상블 방법 중에 하나인 bagging을 활용해 좋은 성과를 얻을 수 있다.
그러나 본 연구에서 이러한 방법을 적용시 예측 성능을 조금씩 떨어뜨리는 결과를 가져왔다. 

### 9.4 feature vector normalization
연구에서 활용된 모델에서는 0이 아닌 feature가 서로 다른 크기를 가지고 있다.
서로 다른 크기의 feature들은 학습 과정에서 수렴을 느리게 할 수 있고, 또 예측 성능에 영향을 줄 수 있다. 
이를 우려하여 여러가지 normalization을 적용하여 테스트한 결과, 학습 초기에는 정확도의 증가를 가져왔으나, positive metric으로 해석되지 못했다.

