본 문서는 랭킹에 대한 견해 [article](https://bahnsville.tistory.com/1242)을 읽고 정리한 글 입니다.

필자는 독특하게 랭킹을 알파벳 순서에 빗대어 표현하는데 __*OPQR(ST)*__ 로 나타낸다. 

> <br> Order by (Relevance, Popularity, Quality, [situation], [time]) <br>
"랭킹은 (_상황_ 과 _시간_ 을 고려한) 관련도, 인기도, 품질에 따른 순서를 따른다."<br><br>


이외에 서비스에 따라 price와 같은 정보들이 추가 활용될 수 있다고 언급한다. 

검색이나 추천에 있어서 랭킹을 중요한 요소이다.
검색에서는 무엇을 보여줄 지 순서를 정하는 일, 추천에서는 그 중 Top N을 뽑아 제공하는 일
또는 광고 도메인으로 넘어왔을 때, Top 1을 골라 내보내는 일을 생각해보면 랭킹은 아주 중요한 요소임을 알 수 있다.

검색에 있어, Page rank에서도 __BM25와 같이 쿼리와 관련된 정보를 나타내는 지표(R)__, __더 많이 클릭된 또는 참조된 페이지(P)__, __출처의 신뢰도 또는 글쓴이의 명성(Q)__ 등을 통해 순서가 결정되도록 만들 수 있다.

이는 추천에서도 마찬가지로 적용된다. 추천해줄 수 있는 특별한 알고리즘이 존재하지 않는 경우, __most popular(MP)__ 를 추천해주고, 주로 __유저의 프로필이나 아이템 간 유사도나 거리로(R)__ 을 활용하여 추천이 이루어진다. 
또한 화질이 낮은 이미지 대비 __선명한 이미지 또는 잘 설명되어 있는 아이템(Q)__ 이 더 많은 클릭을 이끌어 낸다.(선정적 이미지가 아닌 경우..!)
또한 제품을 만들어낸 회사의 명성에 따라 더 많은 클릭을 유발하기도 한다.

__광고__ 에서는 보통 eCPM을 기준으로 추천을 한다.
> note:<br> eCPM = bid amount(광고주가 지불할 의향이 있는 금액) * pCTR(보여주었을 때 사용자가 선택할 확률) <br>

광고에서는 P와 R이 쉽게 혼동될 수 있다고 언급한다.
사전에 광고주 쪽에서 광고 또는 제품에 반응할 audience를 좁혀두었고, __그들에게 지불할 광고비__ 를 책정한 것이므로 관련도 __(R)__ 를 1/0으로 나타낼 수 있다.
이때 pCTR은 __광고를 보여주었을 때 얼마나 많은 사용자가 클릭을 할 것인가__ 로 놓고 보면 __P__ 를 의미한다고 할 수 있다.
이외에도 __광고 소재의 품질, 광고주의 평판(Q)__ 에 따라 종합적으로 고려되어 추천된다.


개인적 의견: 본문에서 알파벳 순서에 따라 랭킹의 중요 요소를 표현한 것이 새롭게 다가와 인상이 남은 글이었습니다. 
또 광고 추천에서 pCTR과 BA(bid amount)를 다른 관점에서 해석하는 것이 재미있는 글이 었습니다.  

