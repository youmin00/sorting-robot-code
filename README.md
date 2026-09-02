# 분류로봇

Intel RealSense D435 카메라로 큐브를 인식하고, STM32 기반 양팔 로봇과
컨베이어를 제어하는 자동 분류 프로젝트입니다.

## 실행 파일

- `통합_자동분류_실행.py`: 전체 자동 분류 운전용 메인 프로그램
- `카메라_공압_시험.py`: 컨베이어 없이 카메라·양팔·공압 시험
- `오른팔_동작_시험.py`: 카메라 없이 오른팔 동작 시험
- `통합_자동분류_실행.bat`: Windows 통합 프로그램 실행

이 폴더에서 다음 명령으로 실행합니다.

```powershell
.\통합_자동분류_실행.bat
```

또는 Python을 직접 실행합니다.

```powershell
.\.venv312\Scripts\python.exe .\통합_자동분류_실행.py
```

## 새 PC에서 환경 설치

가상환경 폴더는 GitHub에 올리지 않습니다. 저장소를 새로 내려받은 PC에서는 다음 명령으로 다시 만듭니다.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```
현재 코드의 기본 포트는 로봇팔 `COM3`, 컨베이어 `COM4`, 통신 속도는
모두 115200 baud입니다. 실제 연결 포트가 다르면 메인 프로그램 상단의
`ROBOT_ARM_PORT`와 `STM32_PORT`를 수정합니다.

## 필요한 파일

- `STM32_제어코드/로봇팔_제어_메인.c`: 로봇팔 STM32 활성 펌웨어
- `STM32_제어코드/컨베이어_센서_제어.c`: 컨베이어 STM32 활성 펌웨어
- `STM32_제어코드/카메라_로봇팔_통합_안내.md`: 연결 및 통신 안내
- `도구/통합_좌표_안정성_진단.py`: 통합 카메라 좌표 안정성 진단
- `도구/RealSense_공식도구/`: RealSense 장치 확인·뷰어·펌웨어 도구
- `시뮬레이터/`: 작업영역 및 다중 큐브 전략 시뮬레이터
- `문서/전체_시스템_설명.txt`: 전체 시스템 설명과 운전 절차

필요 Python 패키지는 `opencv-python`, `pyrealsense2`, `numpy`, `Pillow`,
`pyserial`입니다. 실행 전에 CubeIDE Serial Terminal의 COM 연결을 해제해야 합니다.
