import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def compare_episodic_errors(file_paths):
    """
    각 에피소드의 평균 오차(Scalar)들이 저장된 pkl 파일들을 비교 시각화합니다.
    """
    base_path = "../../results/"
    
    all_means = []
    all_stds = []
    all_raw_data = []
    controller_labels = []
    
    print("=" * 60)
    print(" 에피소드별 평균 오차 분석 (Episodic Mean Error Analysis)")
    print("=" * 60)

    for idx, file_path in enumerate(file_paths):
        full_path = os.path.join(base_path, file_path)

        # 1. 파일 확인 및 로드
        if not os.path.exists(full_path):
            print(f"[경고] '{full_path}' 파일을 찾을 수 없어 건너뜜.")
            continue
            
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
            
        data = np.array(data)
        
        # 데이터가 1차원이 아닐 경우를 위한 안전장치
        if data.ndim > 1:
            data = data.flatten()
            
        # 2. 범례(Label) 이름 자동 추출
        filename = os.path.basename(file_path)
        label = filename.replace('eval_err_pos_', '').replace('.pkl', '').upper()
        if not label:
            label = f"CTRL_{idx+1}"
            
        # 3. 통계치 계산 (코드 내부의 수식 $mean \pm std$ 구현용)
        p_mean = np.mean(data)
        p_std = np.std(data)
        
        print(f"-> [{label}] 에피소드 수: {len(data)}개")
        print(f"   공식 통계: {p_mean:.4f} ± {p_std:.4f}")
        print(f"   논문 텍스트용 format (x100): ${p_mean*100:.2f} \\pm {p_std*100:.2f}$")
        print("-" * 60)
        
        all_means.append(p_mean)
        all_stds.append(p_std)
        all_raw_data.append(data)
        controller_labels.append(label)

    if not all_raw_data:
        print("[오류] 로드된 데이터가 없습니다.")
        return

    # 4. 시각화 (서브플롯 2개 생성: 바 차트 + 박스 플롯)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(controller_labels)]

    # Left Plot: Bar Chart with Error Bars (평균값과 표준편차 직관적 비교)
    bars = ax1.bar(controller_labels, all_means, yerr=all_stds, color=colors, alpha=0.8, 
                   capsize=8, edgecolor='black', error_kw={'elinewidth': 2, 'capthick': 2})
    ax1.set_title('Mean Position Error with Std Dev', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Position Error [m]', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 바 상단에 수치 텍스트 표시
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')

    # Right Plot: Box Plot (에피소드 전체 분포 및 아웃라이어 확인)
    box = ax2.boxplot(all_raw_data, labels=controller_labels, patch_artist=True,
                      medianprops={'color': 'black', 'linewidth': 2},
                      boxprops={'color': 'black', 'alpha': 0.7})
    
    # 박스별 색상 채우기
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    ax2.set_title('Total Error Distribution (Across Episodes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Position Error [m]', fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Controller Performance Comparison (Trajectory Tracking)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 이미지 저장
    output_img = 'controller_performance_comparison.png'
    plt.savefig(output_img, dpi=300)
    print(f"-> 비교 그래프가 '{output_img}' 파일로 저장되었습니다.")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="에피소드 오차 데이터를 분석하고 바 차트 및 박스 플롯을 그립니다.")
    parser.add_argument('-f', '--files', nargs='+', required=True, 
                        help="비교할 파일 경로들을 띄어쓰기로 구분하여 입력 (../../results/ 내부 경로 기준)")
    
    args = parser.parse_args()
    compare_episodic_errors(args.files)