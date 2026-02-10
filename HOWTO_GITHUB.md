# CMD에서 GitHub 연결 방법 (가이드)

이 문서는 로컬 폴더를 GitHub 저장소에 연결하는 명령어를 정리한 것입니다.


## 1. 초기화 및 새 프로젝트 연결 (New Project Setup)

**중요:** `C:\Users\user` 같은 홈 폴더에서 `git init`을 실행하셨다면, 해당 폴더에 생긴 숨겨진 `.git` 폴더를 즉시 삭제하세요. 프로젝트 폴더(`d:\빅프로젝트`) 안에서만 git 명령어를 사용해야 합니다.

cmd(명령 프롬프트)를 열고 `d:\빅프로젝트` 폴더로 이동한 뒤 실행합니다.

```cmd
:: 1. 기존 연결 끊기 (HEATTRACK 삭제)
git remote remove origin

:: 2. 새 저장소 연결 (새로 만든 GitHub 주소 입력)
git remote add origin https://github.com/Sunwoo731/NEW_PROJECT_NAME.git

:: 3. 현재 브랜치 이름을 main으로 변경
git branch -M main
```

## 2. 코드 올리기 (업로드)
코드를 수정하거나 파일을 추가한 뒤에는 다음 3단계를 수행합니다.

```cmd
:: 1. 변경된 모든 파일 담기 (Staging)
git add .

:: 2. 설명과 함께 저장 (Commit)
git commit -m "수정된 내용 설명"

:: 3. GitHub로 보내기 (Push)
git push -u origin main
```

---


## 3. 데이터 및 대용량 파일 관리 (Data Management)

**주의:** GitHub에는 **100MB 이상의 대용량 파일**이나 **민감한 정보(비밀번호, API Key)**를 올리면 안 됩니다.
현재 프로젝트의 `.gitignore` 파일이 다음항목들을 자동으로 제외하고 있습니다:

*   **가상환경:** `.venv`, `venv/`, `env/`
*   **대용량 데이터:** `*.tif`, `*.tiff`, `*.geojson` (지도 데이터), `*.h5` (모델 파일) 등
*   **보안 파일:** `.env` (API Key 포함)
*   **시스템 파일:** `__pycache__`, `.DS_Store` 등

따라서 `git add .` 명령어를 실행해도 위 파일들은 자동으로 제외되고 코드 위주로 올라가니 안심하세요.

---

## 4. 지금 바로 올리기 (Quick Start)

현재 프로젝트 설정이 완료되어 있으므로, 다음 명령어를 순서대로 복사해서 실행하면 바로 GitHub에 저장됩니다.

```cmd
:: 1. 변경사항 확인 (어떤 파일이 올라가는지 미리 보기)
git status

:: 2. 모든 변경사항 담기
git add .

:: 3. 커밋 메시지 작성 (메시지는 자유롭게 변경 가능)
git commit -m "프로젝트 데이터 및 코드 업데이트"

:: 4. GitHub로 업로드
git push origin main
```
