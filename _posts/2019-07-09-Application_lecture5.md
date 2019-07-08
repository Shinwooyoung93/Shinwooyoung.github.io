---
layout: post
title:  "Application Lecture5"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 1. 텍스트 마이닝

## 1-1. 텍스트 마이닝이란

- 웹페이지, 블로그, 전자저널, 이메일 등 전자문서로 된 텍스트자료로 부터 유용한 정보를 추출해 분석하기 위한 도구

- 텍스트 데이터로부터 새로운 고급정보를 추출하는 과정

## 1-2. 코퍼스

- 코퍼스(corpus) : 언어학에서 구조를 이루고 있는 텍스트 집합
  
  + 신문사의 기사, 세익스피어의 소설 등등
  
- 코퍼스의 저장형식 

    + volatile corpus : 메모리에 저장되는 코퍼스
  
    + permanent corpus : DB 혹은 디렉토리에 저장
  
    + distibuted corpus : 분산저장장치를 이용해 저장
  
## 1-3. 텍스트 마이닝의 절차

1. 분석 문서 자료의 준비

2. 코퍼스의 구성

3. 데이터 전처리(EDA) 

4. 분석

# 2. R-code

## 2-1. Read data

- 데이터의 형태는 다양함(csv, txt, html, ...)

- 위의 파일들을 부르는 데에는 각각의 함수들이 존재한다.

    + csv : read.csv()
    
    + txt : read.table(), scan()
    
    + xlsx : read.table()

- 그러나 우리의 데이터는 정형화된 데이터가 아니라 노래 가사를 다룰 것이기 때문에 다음의 명령어를 사용

```r
rm(list = ls())
lyrics <- scan(file.choose(), what = "character")
head(lyrics)
```
```r
## [1] "Yesterday" "all"       "my"        "troubles"  "seemed"    "so"
```

`file.choose()`는 파일디렉토리를 지정하지 않아도 클릭만으로 불러올 수 있다.

해당 노래가사에 ","이 어디 지점에 찍혀있는지 알려주는 명령어이다.
```r
grep(",", lyrics)
```
```r
##  [1]  18  39  47  50  57  80  89  92  99 122
```

해당 노래가사에 "."이 어디 지점에 찍혀있는지 알려주는 명령어이다.
```r
grep("\\.", lyrics)
```
```r
##  [1]   8  17  22  32  38  42  53  62  71  79  84  95 104 113 121 126
```

해당 노래가사에 "!"이 어디 지점에 찍혀있는지 알려주는 명령어이다.
```r
grep("\\!", lyrics)
```
```r
## integer(0)
```

해당 노래가사에 "?"이 어디 지점에 찍혀있는지 알려주는 명령어이다.
```r
grep("\\?", lyrics)
```
```r
## integer(0)
```

또한 주의할 점은 "Yesterday"와 "yesterday"가 다른 단어가 아니기 때문에 이를 통일시켜줘야하며 분석에 방해될 수 있는 구두점(",", ".", "!", "?")을 제거하는 것이다.

먼저 `gsub()`함수를 살펴보자.
```r
gsub("-", ".", "2019-04-05")
```
```r
## [1] "2019.04.05"
```
```r
gsub("\\!", "", "2019-04-05!")
```
```r
## [1] "2019-04-05"
```

이처럼 3번째의 단어에 구성되어있는 A패턴("-")을 B패턴(".")으로 변경해주는 것이다. 
```r
lyrics1 <- gsub(",", "", lyrics)
lyrics1 <- gsub("\\.", "", lyrics1)
lyrics1 <- gsub("\\!", "", lyrics1)
lyrics1 <- gsub("\\?", "", lyrics1)
```

또한 전체 대문자를 소문자로 변환해주기 위해 다음의 명령어를 사용한다.
```r
for(j in 1:26){ # 26 : number of alphabet
  lyrics1 <- gsub(LETTERS[j], letters[j], lyrics1)
}
head(cbind(lyrics, lyrics1), 10)
```
```r
##       lyrics      lyrics1    
##  [1,] "Yesterday" "yesterday"
##  [2,] "all"       "all"      
##  [3,] "my"        "my"       
##  [4,] "troubles"  "troubles" 
##  [5,] "seemed"    "seemed"   
##  [6,] "so"        "so"       
##  [7,] "far"       "far"      
##  [8,] "away."     "away"     
##  [9,] "Now"       "now"      
## [10,] "it"        "it"
```

먼저 반도표를 구해보자. 이는 `table()`함수를 이용해 그린다.
```r
tab1 <- table(lyrics1)
tab1
```
```r
## lyrics1
##         a       all        an        as      away        be   believe 
##         3         1         2         1         3         1         3 
##      came     don't      easy       far       for      game        go 
##         1         2         2         1         2         2         2 
##       had      half   hanging      here      hide         i       i'm 
##         2         1         1         1         2        12         1 
##        in        it      know      long     looks      love       man 
##         3         1         2         2         1         2         1 
##        me        mm        my      need       not       now        oh 
##         1         7         1         2         1         5         4 
##      over     place      play      said       say    seemed    shadow 
##         1         2         2         2         2         1         1 
##       she        so something      stay      such  suddenly       the 
##         4         1         2         1         2         2         1 
##   there's   they're    though        to  troubles      used       was 
##         1         1         1         8         1         1         2 
##       why  wouldn't     wrong yesterday 
##         2         2         2         9
```

여기에서 테이블의 이름들은 다음의 코드로 볼 수 있다.
```r
names(tab1)
```
```r
##  [1] "a"         "all"       "an"        "as"        "away"     
##  [6] "be"        "believe"   "came"      "don't"     "easy"     
## [11] "far"       "for"       "game"      "go"        "had"      
## [16] "half"      "hanging"   "here"      "hide"      "i"        
## [21] "i'm"       "in"        "it"        "know"      "long"     
## [26] "looks"     "love"      "man"       "me"        "mm"       
## [31] "my"        "need"      "not"       "now"       "oh"       
## [36] "over"      "place"     "play"      "said"      "say"      
## [41] "seemed"    "shadow"    "she"       "so"        "something"
## [46] "stay"      "such"      "suddenly"  "the"       "there's"  
## [51] "they're"   "though"    "to"        "troubles"  "used"     
## [56] "was"       "why"       "wouldn't"  "wrong"     "yesterday"
```

빈도수를 작아지는 순서로 정리하는 코드는 아래와 같다.
```r
tab2 <- sort(tab1, decreasing = T)
tab2
```
```r
## lyrics1
##         i yesterday        to        mm       now        oh       she 
##        12         9         8         7         5         4         4 
##         a      away   believe        in        an     don't      easy 
##         3         3         3         3         2         2         2 
##       for      game        go       had      hide      know      long 
##         2         2         2         2         2         2         2 
##      love      need     place      play      said       say something 
##         2         2         2         2         2         2         2 
##      such  suddenly       was       why  wouldn't     wrong       all 
##         2         2         2         2         2         2         1 
##        as        be      came       far      half   hanging      here 
##         1         1         1         1         1         1         1 
##       i'm        it     looks       man        me        my       not 
##         1         1         1         1         1         1         1 
##      over    seemed    shadow        so      stay       the   there's 
##         1         1         1         1         1         1         1 
##   they're    though  troubles      used 
##         1         1         1         1
```

1개의 빈도수가 너무 많으면 그래프 해석에 어려움을 주는 경우가 종종있다.

따라서 빈도수가 3개 이상인 데이터만을 가지고 그래프를 그린다.
```r
tab3 <- tab2[tab2>2]
barplot(rev(tab3), las = 2, horiz = T, main = "Beatles's Yesterday")
```

<center><img src="/assets/Application_lecture5/3.png"></center>

`wordcloud()`함수를 이용해 그래프를 그려보자.

`wordcloud(words, freq, scale, min.freq, max.words, random.order, colors)`:
  - `words` : 그래프에 표시될 단어
  
  - `freq` : 단어의 빈도수
  
  - `scale` : 그래프 내 문자 크기의 최솟값과 최댓값
  
  - `min.freq` : 최소로 표시될 단어 갯수
  
  - `max.words` : 화면에 표시될 전체 단어 갯수의 최대값
  
  - `random.order` : 문자의 위치를 임의로 할 것인가에 관한 여부
  
  - `colors` : 표시될 단어의 색
  
```r
# install.packages("wordcloud")
library(wordcloud)
set.seed(1)
wordcloud(words = names(tab1), freq = tab1, min.freq = 1, colors = rainbow(10), random.order = F)
```

<center><img src="/assets/Application_lecture5/4.png"></center>

# 3. KoNLP

간단한 웹 크롤링을 통해 수집된 텍스트를 기반으로 연관분석을 실시해보자.

https://movie.daum.net/moviedb/grade?movieId=93252&type=netizen&page=1 를 먼저 방문한다.

<center><img src="/assets/Application_lecture5/1.PNG"></center>

다음의 페이지에서 리뷰에 관한 연관분석을 실시하겠다.

먼저 유용한 library를 설치한다.

```r
rm(list = ls())
set.seed(1)
#install.packages(c("rvest", "stringr", "tm", "qgraph", "KoNLP"))
library(rvest)
library(stringr)
library(tm)
library(qgraph)
library(KoNLP)
```

그 다음에, 내가 클롤링하고자 하는 url을 먼저 설정한다.
```r
url = "https://movie.daum.net/moviedb/grade?movieId=93252&type=netizen&page="
```

만약 내가 `url`변수 마지막에 보고자 하는 page를 넣게 되면
```r
paste0(url, 15, sep='')
```
```r
## [1] "https://movie.daum.net/moviedb/grade?movieId=93252&type=netizen&page=15"
```

<center><img src="/assets/Application_lecture5/2.PNG"></center>

15페이지의 리뷰가 나오게 된다.

따라서 내가 원하는 페이지 수 만큼의 리뷰를 다운 받는 코드는 아래와 같다.
```r
all.reviews <- NULL
all.score <- NULL
for(page in 1:100){
  temp.url <- paste0(url, page, sep='')
  htxt <- read_html(temp.url) # rvest
  temp.score <- html_nodes(htxt, 'ul') %>% html_nodes('em')
  temp.score <- temp.score[seq(2, length(temp.score), by = 2)]
  score <- as.numeric(html_text(temp.score)) 
  comments <- html_nodes(htxt, 'div') %>% html_nodes('p')
  reviews <- html_text(comments)
  reviews <- repair_encoding(reviews, from = 'utf-8')
  if(length(reviews) == 0){break}
  reviews <- str_trim(reviews)
  all.reviews <- c(all.reviews, reviews)
  all.score <- c(all.score, score)
}
htxt
```
```r
## {xml_document}
## <html lang="ko" class="os_unknown none unknown version_0">
## [1] <head>\n<meta http-equiv="Content-Type" content="text/html; charset= ...
## [2] <body class="movie">\n\t                        \t\t\t\t\t\t\n\n     ...
```
```r
tail(score)
```
```r
## [1]  8  0 10  7  3  7
```
```r
tail(comments)
```
```r
## {xml_nodeset (6)}
## [1] <p class="desc_review"> <!-- 모바일에서 더보기 클릭시 style="height:auto" -->\n ...
## [2] <p class="desc_review"> <!-- 모바일에서 더보기 클릭시 style="height:auto" -->\n ...
## [3] <p class="desc_review"> <!-- 모바일에서 더보기 클릭시 style="height:auto" -->\n ...
## [4] <p class="desc_review"> <!-- 모바일에서 더보기 클릭시 style="height:auto" -->\n ...
## [5] <p class="desc_review"> <!-- 모바일에서 더보기 클릭시 style="height:auto" -->\n ...
## [6] <p class="desc_review"> <!-- 모바일에서 더보기 클릭시 style="height:auto" -->\n ...
```
```r
tail(reviews)
```
```r
## [1] "해피앤딩 인지 해피앤드인지\r어중간함 \r등장 캐릭터가 너무 많아서 그런지 액션 볼거리 약간 부좀함 \r차라리 인피니트워가 더 좋았어요"                                                                                                                                                                               
## [2] "액션을 원하십니까?\r그럼 이 영화는 피하십시요\r드라마를 원하십니까?\r그럼 이 영화는 피하십시요\r로맨틱을 원하십니까?\r그럼 이 영화는 피하십시요\r그냥 이 영화가 보고 싶다면\r다음 추석이나 설날에 집에서 해주는\r특선영화를 보세요\r이유를 아시게 될겁니다\r돈 버리지마세요\r돈 벌기 힘들잖아요\r전 버렸지만 ..."
## [3] "굿"                                                                                                                                                                                                                                                                                                              
## [4] "그냥 그렇더라.\r좀 지루함..."                                                                                                                                                                                                                                                                                    
## [5] "납득이 안 되는 장면들이 많아서 스토리 집중이 안 되었어요." 
## [6] ""
```

`read_html`함수는 크롤링에 기본이 되는 `<head>`와 `<body>`부분을 가져오게 된다.

`html_nodes`을 통해 `htxt`에 있는 리뷰테그를 찾을 수 있다.

그 다음 온전한 리뷰만을 가져오기 위해 `html_text`함수를 사용한다.

`repair_encoding`함수를 통해 혹시 모를 한글문자가 깨지는 것을 방지해준다.

그 후에 내가 추출한 `review`들의 명사와 형용사만을 추출해보자.
```r
ko.words <- function(doc){
  d <- as.character(doc)
  pos <- paste(SimplePos09(d))
  extracted <- str_match(pos, '([가-힣]+)/[NP]')
  keyword <- extracted[,2]
  keyword[!is.na(keyword)]
}
reviews[5]
```
```r
## [1] "해피앤딩 인지 해피앤드인지\r어중간함 \r등장 캐릭터가 너무 많아서 그런지 액션 볼거리 약간 부좀함 \r차라리 인피니트워가 더 좋았어요"
```
```r
ko.words(reviews[5])
```
```r
##  [1] "해피앤딩"     "일"           "해피앤드"     "어중간함"    
##  [5] "등장"         "캐릭터"       "많"           "그러"        
##  [9] "액션"         "보"           "부좀함"       "인피니트워가"
## [13] "좋"
```

위의 함수 `ko.words`를 통해 필요한 단어들만을 추출한다.
```r
options(mc.cores=1)
cps <- Corpus(VectorSource(all.reviews))
tdm <- TermDocumentMatrix(cps, 
                          control=list(tokenize=ko.words,
                                       removePunctuation=T,
                                       removeNumbers=T,
                                       wordLengths=c(2, 6),  
                                       weighting=weightBin))
tdm.mat <- as.matrix(tdm)
Encoding(rownames(tdm.mat)) <- "UTF-8"
tdm.mat[1:10,1:10]
```
```r
##       Docs
## Terms  1 2 3 4 5 6 7 8 9 10
##   그   0 1 0 0 0 0 0 0 0  0
##   년   0 1 0 0 0 0 0 0 0  0
##   작품 0 0 0 0 1 0 0 0 0  0
##   한   0 0 0 0 1 0 0 0 0  0
##   굳이 0 0 0 0 0 0 0 0 1  0
##   머   0 0 0 0 0 0 0 0 1  0
##   오와 0 0 0 0 0 0 0 0 0  0
##   가장 0 0 0 0 0 0 0 0 0  0
##   관람 0 0 0 0 0 0 0 0 0  0
##   날이 0 0 0 0 0 0 0 0 0  0
```

내가 처리한 단어들의 빈도수를 확인하는 명령어이다.
```r
rownames(tdm.mat)[1:100]
```
```r
##   [1] "그"   "년"   "작품" "한"   "굳이" "머"   "오와" "가장" "관람" "날이"
##  [11] "내"   "놈"   "다"   "영화" "친구" "ubd"  "그만" "이젠" "보통" "그저"
##  [21] "너무" "라는" "이후" "다음" "아나" "아예" "완전" "장난" "좋냐" "캡틴"
##  [31] "않다" "속상" "진짜" "엉엉" "잊지" "평생" "나가" "나의" "내가" "되는"
##  [41] "될줄" "먹튀" "모든" "바로" "버린" "보고" "본"   "분만" "사기" "위해"
##  [51] "이"   "최고" "탑인" "편에" "d로"  "강추" "다시" "보니" "최강" "ㅎ"  
##  [61] "가서" "길고" "역시" "ㅠ"   "가면" "넘"   "딸"   "마니" "아"   "잘"  
##  [71] "제일" "계속" "더"   "독점" "에휴" "정말" "끝만" "중간" "지루" "대박"
##  [81] "마블" "싸움" "만큼" "와우" "love" "you"  "ㅜㅜ" "그냥" "난다" "두번"
##  [91] "또"   "모두" "없다" "조금" "아주" "ㅠㅠ" "그들" "긴"   "년의" "달려"
```

위의 `tdm.mat`에서 빈도수가 낮은 단어를 사용하지 말고 높은 빈도수만 갖는 단어들을 사용하는 코드는 아래와 같다.
```r
word.count <- apply(tdm.mat, 1, sum)
word.order <- order(word.count, decreasing=T)
freq.words <- tdm.mat[word.order[1:20],]
freq.words[1:10, 1:10]
```
```r
##       Docs
## Terms  1 2 3 4 5 6 7 8 9 10
##   영화 0 0 0 0 0 0 0 0 0  0
##   너무 0 0 0 0 0 0 0 0 0  0
##   다   0 0 0 0 0 0 0 0 0  0
##   마블 0 0 0 0 0 0 0 0 0  0
##   진짜 0 0 0 0 0 0 0 0 0  0
##   보고 0 0 0 0 0 0 0 0 0  0
##   그냥 0 0 0 0 0 0 0 0 0  0
##   이   0 0 0 0 0 0 0 0 0  0
##   본   0 0 0 0 0 0 0 0 0  0
##   만큼 0 0 0 0 0 0 0 0 0  0
```
```r
co.matrix <- freq.words %*% t(freq.words)
co.matrix
```
```r
##       Terms
## Terms  영화 너무 다 마블 진짜 보고 그냥 이 본 만큼 더 정말 시간 잘 최고 그
##   영화   79   10 14   16    7   12    7 10  7    2  7    7    2  5    1  5
##   너무   10   67  6    7    4    3    5  1  1    3  4    1    5  2    1  1
##   다     14    6 49   10    5   12    8  5  9    4  2    0    4  3    2  4
##   마블   16    7 10   44    4   11    4  7  8    2  5    2    2  3    1  1
##   진짜    7    4  5    4   38    4    2  3  4    3  2    0    3  1    1  0
##   보고   12    3 12   11    4   33    4  6  5    4  4    1    2  0    2  3
##   그냥    7    5  8    4    2    4   33  2  1    1  2    2    2  1    0  2
##   이     10    1  5    7    3    6    2 31  7    0  2    1    0  3    1  3
##   본      7    1  9    8    4    5    1  7 28    0  0    0    2  0    3  0
##   만큼    2    3  4    2    3    4    1  0  0   28  0    0    0  3    0  2
##   더      7    4  2    5    2    4    2  2  0    0 26    1    2  4    0  0
##   정말    7    1  0    2    0    1    2  1  0    0  1   25    2  3    0  1
##   시간    2    5  4    2    3    2    2  0  2    0  2    2   24  0    0  1
##   잘      5    2  3    3    1    0    1  3  0    3  4    3    0 23    0  0
##   최고    1    1  2    1    1    2    0  1  3    0  0    0    0  0   21  3
##   그      5    1  4    1    0    3    2  3  0    2  0    1    1  0    3 20
##   좀      1    2  2    3    1    1    2  0  1    0  1    0    3  1    0  1
##   점      1    2  2    1    1    3    3  1  0    1  1    0    0  0    0  0
##   이런    3    0  3    2    0    4    1  0  0    0  2    1    0  1    0  1
##   모든    5    3  8    5    1    3    2  3  4    0  1    1    1  2    1  2
##       Terms
## Terms  좀 점 이런 모든
##   영화  1  1    3    5
##   너무  2  2    0    3
##   다    2  2    3    8
##   마블  3  1    2    5
##   진짜  1  1    0    1
##   보고  1  3    4    3
##   그냥  2  3    1    2
##   이    0  1    0    3
##   본    1  0    0    4
##   만큼  0  1    0    0
##   더    1  1    2    1
##   정말  0  0    1    1
##   시간  3  0    0    1
##   잘    1  0    1    2
##   최고  0  0    0    1
##   그    1  0    1    2
##   좀   20  1    2    1
##   점    1 19    2    0
##   이런  2  2   18    2
##   모든  1  0    2   16
```

연관성을 나타내는 매트릭스이다. 이를 통해 연관성 그래프인 `qgraph`를 이용한다.

```r
qgraph(co.matrix,
       labels=rownames(co.matrix),
       diag=F,  
       layout='spring',
       edge.color='dodgerblue',
       vsize=log(diag(co.matrix))*2)
```

<center><img src="/assets/Application_lecture5/5.png"></center>

이를 통해 영화 리뷰에 대한 부정평가와 긍정평가를 가늠할 수 있다.

# 3. Regression

위의 `tdm.mat`를 이용해 영화평점 예측 모델을 만들어보자. 그러기 위해 예측하고자 하는 리뷰를 생성해보자.
```r
bad <- "쓰레기 같이 재미 없다"
good<- "영화 정말 최고다"
ko.words(bad)
```r
## [1] "쓰레기" "같"     "재미"   "없"
```
```r
ko.words(good)
```
```r
## [1] "영화" "최고"
```

`tdm.mat`에 사용된 모든 단어들을 사용하면 좋지만, 효율성을 위해 상위 100개의 단어만을 사용해보자.
```r
train.mat <- t(tdm.mat[word.order[1:100],])
train.mat[1:10, 1:10]
```
```r
##     Terms
## Docs 영화 너무 다 마블 진짜 보고 그냥 이 본 만큼
##   1     0    0  0    0    0    0    0  0  0    0
##   2     0    0  0    0    0    0    0  0  0    0
##   3     0    0  0    0    0    0    0  0  0    0
##   4     0    0  0    0    0    0    0  0  0    0
##   5     0    0  0    0    0    0    0  0  0    0
##   6     0    0  0    0    0    0    0  0  0    0
##   7     0    0  0    0    0    0    0  0  0    0
##   8     0    0  0    0    0    0    0  0  0    0
##   9     0    0  0    0    0    0    0  0  0    0
##   10    0    0  0    0    0    0    0  0  0    0
```

예측을 할 변수인 평점변수 `all.score`을 함께 묶어준다.
```r
train.data <- data.frame(score = all.score, train.mat)
train.data[1:10, 1:10]
```
```r
##    score 영화 너무 다 마블 진짜 보고 그냥 이 본
## 1      9    0    0  0    0    0    0    0  0  0
## 2     10    0    0  0    0    0    0    0  0  0
## 3      6    0    0  0    0    0    0    0  0  0
## 4     10    0    0  0    0    0    0    0  0  0
## 5     10    0    0  0    0    0    0    0  0  0
## 6     10    0    0  0    0    0    0    0  0  0
## 7     10    0    0  0    0    0    0    0  0  0
## 8      8    0    0  0    0    0    0    0  0  0
## 9     10    0    0  0    0    0    0    0  0  0
## 10    10    0    0  0    0    0    0    0  0  0
```

`train.data`를 이용해 선형회귀분석 모델을 만들어준다.
```r
fit <- lm(score ~., data = train.data)
```

그 다음, 만든 회귀분석 모델을 통해 예측을 해주기 위해 예측 데이터를 만들어준다.
```r
test.bad <- rep(0, ncol(train.mat))
for(i in 1:length(ko.words(bad))){
  test.bad <- ifelse(ko.words(bad)[i] == colnames(train.mat), 1, test.bad)
}
names(test.bad) <- colnames(train.mat)

test.good<- rep(0, ncol(train.mat))
for(i in 1:length(ko.words(good))){
  test.good<- ifelse(ko.words(good)[i] == colnames(train.mat), 1, test.good)
}
names(test.good) <- colnames(train.mat)
head(test.bad, 20)
```
```r
## 영화 너무   다 마블 진짜 보고 그냥   이   본 만큼   더 정말 시간   잘 최고 
##    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0 
##   그   좀   점 이런 모든 
##    0    0    0    0    0
```
```r
head(test.good,20)
```
```r
## 영화 너무   다 마블 진짜 보고 그냥   이   본 만큼   더 정말 시간   잘 최고 
##    1    0    0    0    0    0    0    0    0    0    0    0    0    0    1 
##   그   좀   점 이런 모든 
##    0    0    0    0    0
```

이를 통해 `bad`와 `good`의 평점을 예측해보자.
```r
predict(fit, new = data.frame(t(test.bad)))
```
```r
##        1 
## 5.564403
```
```r
predict(fit, new = data.frame(t(test.good)))
```
```r
##        1 
## 9.245286
```

# 4. Question

다음의 형식을 이용해 영화 데이터를 이용해 분석 보고서를 완성하시오. 영화 데이터의 `wordcloud`분석은 `word.count`변수를 사용하시오.(단 `min.freq` = 4로 지정하시오)

```r
---
title: "Text Mining report"
author: "SSWU, Your Name"
date: \today
output: html_document
---
```

```r
` ```{r, echo = T, out.width = '70%', fig.align='center'}
` (your script)
` ``` 
```
```r
library(wordcloud)
set.seed(1)
wordcloud(words = names(word.count), freq = word.count, min.freq = 4, colors = rainbow(10), random.order = F)
```

<center><img src="/assets/Application_lecture5/6.png"></center>

### 1. Crawling
### 2. Plotting
#### 2-1. Wordcloud
#### 2-2. Qgraph
#### 2-3. Regression
### 3. Conclusion
