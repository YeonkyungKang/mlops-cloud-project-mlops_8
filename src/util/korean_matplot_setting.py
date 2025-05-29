__all__ = ['set_korean_font']
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def set_korean_font(font_path: str=None) -> None:
    # 한글 폰트 설정
    # 사용 가능한 한글 폰트 경로를 지정해야 합니다.
    # 예: 나눔고딕, 맑은 고딕 등
    if os.name == 'nt':  # Windows
        try:
            # 시스템에 설치된 폰트 중 하나 선택 (예: Malgun Gothic)
            font_path = 'c:/Windows/Fonts/malgun.ttf' if not font_path else font_path # Windows 경로 기본 한글 폰트
            font_prop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=font_prop.get_name())
            # plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            print(f"폰트 설정 완료: {font_prop.get_name()}")
        except FileNotFoundError:
            print("지정된 한글 폰트 파일을 찾을 수 없습니다. 기본 폰트로 진행합니다.")
        except Exception as e:
            print(f"폰트 설정 중 오류 발생: {e}")
    elif sys.platform == 'darwin':  # macOS
        try:
            # macOS의 기본 한글 폰트 경로 (AppleGothic)
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf' # mac 경로 기본 한글 폰트
            font_prop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=font_prop.get_name())
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            print(f"폰트 설정 완료: {font_prop.get_name()}")
        except FileNotFoundError:
            print("지정된 한글 폰트 파일을 찾을 수 없습니다. 기본 폰트로 진행합니다.")
        except Exception as e:
            print(f"폰트 설정 중 오류 발생: {e}")
    elif os.name == 'posix':  # Linux -> 아직 안 만듬.
        try:
            # Linux의 기본 한글 폰트 경로 (Noto Sans CJK)
            if font_path:
                font_prop = fm.FontProperties(fname=font_path)
            else:
                font_list = fm.findSystemFonts(fontpaths=['/usr/share/fonts', '/usr/local/share/fonts'], fontext='ttf')
                korean_fonts = [f for f in font_list if 'Nanum' in f or 'Batang' in f or 'Gothic' in f]

                if korean_fonts:
                    font_path = korean_fonts[0]  # 첫 번째로 찾은 한글 폰트 사용
                    font_prop = fm.FontProperties(fname=font_path)
                else:
                    raise FileNotFoundError("사용 가능한 한글 폰트가 없습니다. 직접 설치 후 경로를 지정하세요.")

            plt.rc('font', family=font_prop.get_name())
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            print(f"폰트 설정 완료: {font_prop.get_name()}")
        except FileNotFoundError:
            print("지정된 한글 폰트 파일을 찾을 수 없습니다. 기본 폰트로 진행합니다.")
        except Exception as e:
            print(f"폰트 설정 중 오류 발생: {e}")
    else:
        plt.rc('font', family='sans-serif')
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


if __name__ == "__main__":
    set_korean_font()

    # 예시 그래프
    plt.figure(figsize=(10, 6))
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title('예시 그래프 제목')
    plt.xlabel('X축 레이블')
    plt.ylabel('Y축 레이블')
    plt.grid()
    plt.show()
