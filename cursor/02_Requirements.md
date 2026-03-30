# 02_requirements.txt 생성

## 변경 사항
- 전체 `.py`, `.ipynb` 파일에서 import된 라이브러리를 자동 추출
- 표준 라이브러리 제외, 서드파티만 필터링하여 `requirements.txt` 생성
- 이미 설치된 패키지는 버전 고정 (예: `numpy>=2.2.3`)

## 이점
- `pip install -r requirements.txt` 한 줄로 환경 재현 가능
- 팀원 또는 새 환경에서 의존성 누락 방지
