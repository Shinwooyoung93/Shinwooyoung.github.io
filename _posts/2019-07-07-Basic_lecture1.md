---
layout: post
title:  "Basic Lecture1"
date:   2019-07-08
use_math: true
tags:
 - R
 - korean
 - Lecture
---

# 0. Lecture intro

- 신우영, sinwy93@korea.ac.kr

- 50분 강의, 10분 휴식, 50분 강의, 10분 휴식, 40분 과제, 10분 질문

- 수업내용은 유동적으로 변경될 수 있음

- 스케줄

|단계 | 주요내용 | 소요시간|
|:--:|:--:|:--:|
|1일차|R 소개, R-studio 설치 및 기초기능 탐색|3시간|
|2일차|R을 이용한 통계데이터 분석 $-$ 데이터 구조|3시간|
|3일차|R을 이용한 통계데이터 분석 $-$ 함수의 생성|3시간|
|4일차|R을 이용한 통계데이터 분석 $-$ 그래프 시각화|3시간|
|5일차|R을 이용한 통계데이터 분석 $-$ 라이브러리 활용 및 고급 시각화|3시간|
|6일차|R을 이용한 통계데이터 분석 $-$ 회귀분석 및 로지스틱 회귀분석|3시간|

# 1. Introduction to R

## 1-1. What is R

- R은 통계 계산과 그래픽을 위한 프로그래밍 언어이자 소프트웨어 환경

- 뉴질랜드 오클랜드 대학의 Robert Gentleman과 Ross Ihaka에 의해 시작됨

- 현재는 R development Core Team이 유지 개선 중

- R은 통계 소프트웨어 개발과 자료 분석에 널리 사용되고 있음

## 1-2. Summary of R

- R 프로그램은 무료
  : 전 세계 많은 사용자가 다양한 함수와 패키지를 만들고 공유
  
- R 프로그램은 오픈소스
  : 수많은 R 사용자가 자유롭게 분석기법들을 추가할 수 있음  

- R 프로그램은 다양한 분야의 통계분석이 가능
  : 통계학자들에 의해 개발된 방법론들이 라이브러리로 추가되어있음
  
- R 프로그램은 도움말 기능이 뛰어남 
  : 새로운 기능을 익히는데 어렵지 않음
  
- R 은 벡터(Vector)기반의 프로그래밍 언어
  : 데이터 전처리에 있어 직관적이고 쉬움
  
- 다만 SPSS에 비해 사용자의 편의성이 부족함 

# 2. Installation

## 2-1. R installation

http://cran.r-project.org/ 에서 다운로드

<center><img src="/assets/Basic_lecture1/1.png" width="500" height="500"></center>

base를 선택

<center><img src="/assets/Basic_lecture1/2.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/3.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/4.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/5.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/6.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/7.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/8.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/9.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/10.png" width="500" height="500"></center>

## 2-2. R-studio

### 2-2-1. What is R-studio

- R 보다 편집작업이 용이한 텍스트 에디터

- R-studion는 R 프로그램을 구동하기 위한 통합개발환경(IDE) S/W

    + 통합개발환경(IDE, Integrated Development Environment) 
    : 코딩, 디버그, 컴파일, 배포 등 프로그램 개발에 관련된 모든 작업을 하나의 프로그램 안에서 처리하는 환경을 제공하는 소프트웨어
  
- R 통합개발환경을 위해 콘솔, 직접 코드를 실행시킬 수 있는 구문강조 기능이 있는 편집기

- 작업공간 관리 기능 수행

- 오픈소스로 개발되어 있어 무료이지만 추가적인 기능을 위한 유료버전도 존재


### 2-2-2. R-studio installation

https://www.rstudio.com 에서 설치

<center><img src="/assets/Basic_lecture1/11.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/12.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/13.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/14.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/15.png" width="500" height="500"></center>

<center><img src="/assets/Basic_lecture1/16.png" width="500" height="500"></center>

### 2-2-3. R-studio

<center><img src="/assets/Basic_lecture1/17.png" width="500" height="500"></center>

- 편집기 : R 명령어를 입력하는 창

- 콘솔 : 명령문을 실행하는 창 

- 워크스페이스 : R 콘솔창에서 작업한 모든 객체(변수, 함수, 데이터파일 등)가 저장되는 곳

**Q. 변수 $x$에 1을 할당하고 싶다.**

<center><img src="/assets/Basic_lecture1/18.png" width="500" height="500"></center>

1. 편집기에 $x \leftarrow 1$명령어를 입력하고 실행한다
    + 실행방법은 원하는 영역을 선택하고 $Run$버튼을 누르거나 Ctrl + Enter를 이용
 
2. 탭에 있는 Untitled1이 저장되지 않은 작업중으로 바뀐다.

3. 콘솔창에서 명령문이 실행된다.

4. 워크스페이스에 변수가 할당되었다고 표시된다.
