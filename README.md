# 도서 추천 시스템

Amazon 도서 리뷰와 평점을 이용해 미평가 도서를 추천하는 개인 캡스톤디자인 프로젝트다. 학부 과정에서 사용자 간 평점 패턴을 비교하는 협업 필터링과 잠재 요인을 학습하는 행렬 분해를 구현했다.

## 데이터

프로젝트 자료에 기록된 출처는 [Kaggle Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)다. 리뷰와 도서 정보를 정리해 만든 `books_review_2.csv`를 두 노트북의 입력으로 사용했다. 이 파일에는 리뷰 21,538건, 책 제목 1,616개, 사용자 5,070명이 들어 있다. 노트북에서는 같은 사용자가 같은 책에 남긴 여러 평점을 평균으로 묶어 19,767개 사용자-책 조합을 만든다.

원본 데이터의 정확한 버전과 라이선스, 현재 전처리 파일을 만든 전체 과정은 남아 있지 않아 CSV는 저장소에서 제외했다. Kaggle 원본을 받는 것만으로 `books_review_2.csv`와 같은 파일을 만들 수 있다고 보장할 수 없다. 로컬에서 실행하려면 준비된 입력 파일을 `capstone/books_review_2.csv`에 둔다. 데이터 구조와 남아 있는 전처리 자료는 [실행 안내](capstone/README.md)와 [복원 기록](docs/reconstruction_notes.md)에 정리했다.

## 추천 방식

- [사용자 기반 협업 필터링](capstone/user_based_cf.ipynb): 사용자별 평균을 기준으로 평점을 중심화하고, 피어슨 유사도가 높은 사용자의 평가를 이용해 미평가 도서의 점수를 계산한다.
- [SGD 행렬 분해](capstone/matrix_factorization_lf.ipynb): 사용자와 책을 잠재 요인 벡터로 표현하고, 관측된 평점을 바탕으로 벡터를 갱신해 미평가 책의 점수를 예측한다. 노트북 설정은 요인 50개, 200회 반복, 학습률 0.01, 정규화 0.01이다.

두 노트북에는 각각 추천 결과 예시가 남아 있다. 사용자 기반 방식의 예시에는 *Ever Since Darwin* 5.00점, *Zen Guitar* 4.33점이 기록돼 있고, 행렬 분해 예시에는 *Riddle-Master* 4.62점, *Black Hole* 4.39점이 나온다. 점수는 해당 노트북의 한 사용자에 대한 추천 예시다.

## 실행

Python 3.10 이상을 사용한다. 저장소 루트에서 환경을 만들고 의존성을 설치한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r capstone/requirements.txt
```

VS Code에서 노트북을 열고 `.venv` 커널을 선택해 실행한다. 두 노트북은 `capstone/` 또는 저장소 루트에서 `books_review_2.csv`를 찾는다. 평가 스크립트는 저장소 루트에서 실행한다.

```bash
python capstone/evaluate_holdout.py
```

## 평가와 정리 기록

노트북에 기록된 RMSE는 학습 데이터 기준이다. 2026년에 추가한 홀드아웃 평가에서는 행렬 분해가 평균 기준선보다 낮은 성능을 보였다. 정확한 RMSE·MAE와 평가 설정은 [복원 기록](docs/reconstruction_notes.md)에서 확인할 수 있다.

2026년에는 Colab 경로를 로컬 경로로 바꾸고 사용자 기반 예측 계산을 수정해 두 노트북을 다시 실행했다. 발표자료에 언급된 아이템 기반·TF-IDF·지식 기반 추천 코드는 공개한 두 노트북에 포함돼 있지 않다.
