## Recommender Systems(추천 시스템)
- 사용자에게 활용할만한 아이템의 제안하는 소프트웨어 툴 또는 기술
- 추천 시스템은 왜 필요할까?
    - 정보가 무수히 많고 끊임없이 폭발적인 확장이 이루어 지고 있으며, 이는 사용자가 이겨내기 어려운 환경을 제공하여, 나쁜 결정을 하도록 만들기 때문이다.
- 추천은 보통 개인화(personalized)를 의미한다.
- 가장 간단한 형태의 추천 시스템.
    - 유저에게서 그들의 선호도를 수집
     (상품의 rating과 같이 명확한 의미의 증거나 특정 페이지 방문 등 내포되어 있는 의미의 증거 등)
    - 제약 조건과 유저의 선호도에 가장 잘 맞는 상품이나 서비스를 계산
    - 그들에게 관련성 높은 랭킹에 맞게 정령된 리스트를 제공

### 추천 시스템이 서비스 제공자(service provider)들에게 필요한 이유는
1. 상품 판매 증가
2. 더욱 다양한 상품의 판매 - long tail
3. 유저의 만족 증대
4. 유저의 충실도 증대
5. 유저의 요구 이해
를 들 수 있다.

> Long tail - 인기도 및 인지도가 높을 수록 개체 수가 많지만 실질적으로 Long tail이 존재한다. 
(인기도가 높은 아이템을 추천하는 것은 어렵지 않으나, 실질적으로 잘된 추천은 Long tail에 맞춰 선택권을 좁혀주는 작업을 의미한다.)


### 추천 시스템에서 익히 알려진 task로는
(1) 좋은 아이템을 찾는 문제, (2) TV 프로그램 추천, (3) 기존 동작과 연속된 추천, (4) 여행 예약과 같은 묶음 추천, (5) 브라우징에서의 추천, (6) 프로필 향상 등이 있다.
추천에서 **items**은 추천되는 객체를 의미하고 **Users**는 다양한 목적과 성향을 가진다.
따라서 비슷한 상황을 묶어주는 Collaborative filtering이나, Demographic 정보를 활용한 추천, 또는 행동 패턴을 활용한 추천 등 다양한 방법으로 모델링 할 수 있다.
**Transactions**은 추천 시스템과 유저 사이의 상호작용에 대한 기록을 의미하고, rating나 tag처럼 로그와 같은 형태로 기록된다. 

### 추천 시스템은 **1.예측** 하거나 **2.랭킹**하는 두 가지 형태의 목적을 지닌다.<br>
추천 방법은 "사용자가 좋아하는 아이템과 비슷한 아이템을 추천하도록 학습하는 **content-based**"와 "취향이 비슷한 사용자가 좋아하는 것을 추천하는 **collaborative filterin**"이 존재한다.
또한 Demographic 정보를 이용하는 **Demographic**, 도메인 지식을 활용해서 사용자에게 맞는 아이템을 추천하는 방식으로 지정한 쿼리에 맞춰 결과를 제공하는 **case-based**나 사전에 결정된 특정 rule에 따른 추천을 하는 **Constraint-based** 방식도 있다.
이외에도 지역이나 시간과 같은 문맥에 따른 추천으로 **Context-based** 방식, 사용자가 속한 그룹을 바탕으로 추천하는 **Community-based**, 다양한 방식을 섞어서 제공하는 **Hybrid** 등이 있다.

추천을 활용하는 Application에는 
음악이나 영화와 같은 "Entertainment", 뉴스나 web-page와 같은 "Content", 책이나 가전 등 상품을 추천하는 "E-commerce", 여행이나 집 렌트와 같이 매칭을 제공하는 "Service" 등이 있다.
이러한 적용을 위해서 추천시스템을 설계하기 위해서는 많은 요소(대상 유저의 종류, 추천의 역할, 추천의 목적, 활용가능한 데이터 등)들이 고려되어야 한다.

평가는 (1)설계할 때, (2)시스템이 적용 및 실행되고 난 뒤 또는 (3)실행이 어려운 경우는 작은 그룹의 유저를 대상으로 집중적인 실험을 통해 평가할 수 있다.
이때, 비지니스의 목표를 위해서 revenue의 증가, 페이지 뷰의 증가, 또는 KPI를 트래킹할 수 있고, 기능적인 목표를 위해, 연관성이나 새로움 또는 seredipity나 다양성을 평가할 수 있다.

자주 활용되는 지표로는 Root Mean Square Error, Precision at K(P@K), Normalized Discounted Cumulative Gain(NDCG), precision&recall, AUC 등이 있다.

단순히 맞는 추천 결과를 보여주는 것을 목적으로 하는 것은 현실적인 시스템에 충분하지 않다.
추천된 결과가 보여졌을 때, 추천된 결과에서 평가하는 것 등 따라오는 것들도 같이 고려해야한다.

