# 코드와 데이터

상위 [README](../README.md)는 프로젝트 개요와 추천 방식을 설명한다. 이 문서에는 실행 파일과 입력 데이터의 형태를 정리했다. 데이터 간 관계와 수정 이력은 [복원 기록](../docs/reconstruction_notes.md)을 참고한다.

## 파일

| 파일 | 내용 |
|---|---|
| `user_based_cf.ipynb` | 사용자별 평점 평균과 피어슨 유사도를 이용한 추천 |
| `matrix_factorization_lf.ipynb` | SGD 행렬 분해와 개인별 미평가 도서 추천 |
| `evaluate_holdout.py` | 2026년에 추가한 평점 예측 홀드아웃 평가 |
| `requirements.txt` | 노트북 실행에 필요한 NumPy, pandas, ipykernel |
| `books_review_2.csv` | 두 노트북과 평가 스크립트가 읽는 리뷰·도서 입력 파일. 저장소에는 포함하지 않는다. |

`데이터셋/`에는 전처리 리뷰와 도서 메타데이터 파일이, `originals/`에는 작업 전 노트북 사본이 있다. 이 자료는 로컬 보관용이며 GitHub에는 올리지 않는다.

## 입력 파일

`books_review_2.csv`는 리뷰 한 건을 한 행으로 저장한 CSV다. 노트북은 `user_num`과 `Title`을 기준으로 평점을 피벗 테이블로 만들고, 같은 사용자가 같은 책에 남긴 여러 평점은 평균으로 집계한다.

| 컬럼 | 설명 |
|---|---|
| `Title` | 도서 제목 |
| `description` | 도서 설명 |
| `authors` | 저자 |
| `publisher` | 출판사 |
| `categories` | 도서 분류 |
| `year` | 출판 연도 |
| `Id` | 도서 ID 또는 ISBN 값 |
| `Price` | 가격 |
| `review/helpfulness` | 리뷰 유용성 비율 |
| `review/score` | 1~5점 평점 |
| `review/time` | 리뷰 날짜 |
| `review/summary` | 리뷰 제목 |
| `review/text` | 리뷰 본문 |
| `user_num` | 원천 사용자 ID에 대응시킨 번호 |

## 데이터 준비

프로젝트 자료는 [Kaggle Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)를 출처로 기록하고 있다. 정확한 데이터 버전과 라이선스, 현재 CSV를 만든 전체 전처리 과정은 보존돼 있지 않다. Kaggle에서 원본을 내려받는 것만으로 이 입력 파일을 재생성할 수 있다고 보장할 수 없다. 준비된 `books_review_2.csv`가 있다면 이 폴더에 저장한다.

CSV 파일은 데이터 출처와 재배포 권한을 확인할 때까지 저장소에서 제외한다. 파일이 없으면 노트북과 평가 스크립트는 실행되지 않는다.

## 실행

저장소 루트에서 Python 3.10 이상 가상환경을 만들고 설치한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r capstone/requirements.txt
```

VS Code에서 `user_based_cf.ipynb` 또는 `matrix_factorization_lf.ipynb`를 열고 `.venv` 커널로 셀을 위에서부터 실행한다. 노트북은 현재 작업 디렉터리나 `capstone/`에서 `books_review_2.csv`를 찾는다.

홀드아웃 평가는 저장소 루트에서 실행한다.

```bash
python capstone/evaluate_holdout.py
```

이 스크립트는 Python 표준 라이브러리만 사용한다. 평가 분할과 전체 결과는 [복원 기록](../docs/reconstruction_notes.md)에 정리했다.
