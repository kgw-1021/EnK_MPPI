import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt

# --- 설정 및 인자 파싱 ---
parser = argparse.ArgumentParser(description="Quadrotor Trajectory and Error Plotter")
parser.add_argument(
    "-f", "--files",
    nargs="+",
    required=True,
    help="Plot multiple .pkl files from ../../results/ (e.g., file1.pkl file2.pkl)"
)
args = parser.parse_args()

COLOR_LIST = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
base_path = "../../results/"

# 데이터를 저장할 딕셔너리
results_data = {}
target_pos_traj = None  # Reference 궤적은 한 번만 저장하여 그리기 위함

print("Loading files and processing data...")

# --- 데이터 로드 및 연산 ---
for idx, filename in enumerate(args.files):
    algorithm_name = filename.replace("state_seq_", "").replace(".pkl", "")
    file_path = os.path.join(base_path, filename)
    
    try:
        with open(file_path, "rb") as f:
            state_seq = pickle.load(f)
            
        # 상태 변수 추출
        pos = np.array([state["pos"] for state in state_seq])
        pos_tar = np.array([state["pos_tar"] for state in state_seq])
        vel = np.array([state["vel"] for state in state_seq])
        vel_tar = np.array([state.get("vel_tar", np.zeros(3)) for state in state_seq])
        
        if target_pos_traj is None:
            target_pos_traj = pos_tar
            
        # 스텝별 RMSE 오차 계산
        pos_error = np.linalg.norm(pos - pos_tar, axis=1)
        vel_error = np.linalg.norm(vel - vel_tar, axis=1)
        
        # 평균 오차 계산
        avg_pos_rmse = np.mean(pos_error)
        avg_vel_rmse = np.mean(vel_error)
        
        # 결과 딕셔너리에 저장
        results_data[algorithm_name] = {
            "pos": pos,
            "pos_error": pos_error,
            "vel_error": vel_error,
            "avg_pos_rmse": avg_pos_rmse,
            "avg_vel_rmse": avg_vel_rmse,
            "color": COLOR_LIST[idx % len(COLOR_LIST)]
        }
            
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Skipping.")
    except Exception as e:
        print(f"Error parsing {filename}: {e}. Skipping.")


if not results_data:
    print("No valid data loaded. Exiting.")
    exit()

# ==========================================
# Figure 1: 3D Trajectory Plot
# ==========================================
fig1 = plt.figure(num="Figure 1: 3D Trajectory", figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Target 궤적 그리기 (검은 점선)
if target_pos_traj is not None:
    ax1.plot(target_pos_traj[:, 0], target_pos_traj[:, 1], target_pos_traj[:, 2], 
             label="Target Trajectory", color='black', linestyle='--', linewidth=1.5)

# 각 알고리즘별 실제 궤적 그리기
for algo, data in results_data.items():
    pos = data["pos"]
    color = data["color"]
    
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"Actual ({algo})", color=color, linewidth=2)
    ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, marker='o', s=50, edgecolors='k', zorder=5)

ax1.set_title("Quadrotor 3D Trajectory Comparison", fontsize=14, fontweight='bold')
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.set_zlabel("Z Position (m)")
ax1.legend(loc="upper left")
ax1.grid(True)
ax1.view_init(elev=25, azim=-45)

# 축 비율을 동일하게 맞추기 위한 트릭
extents = np.array([ax1.get_xlim3d(), ax1.get_ylim3d(), ax1.get_zlim3d()])
centers = np.mean(extents, axis=1)
max_width = np.max(np.abs(extents[:, 1] - extents[:, 0]))
ax1.set_xlim3d([centers[0] - max_width/2, centers[0] + max_width/2])
ax1.set_ylim3d([centers[1] - max_width/2, centers[1] + max_width/2])
ax1.set_zlim3d([centers[2] - max_width/2, centers[2] + max_width/2])


# ==========================================
# Figure 2: Tracking Error Subplots
# ==========================================
fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, num="Figure 2: Tracking Errors", figsize=(10, 8), sharex=True)
fig2.suptitle("Tracking Error Comparison Over Time", fontsize=16, fontweight='bold')

for algo, data in results_data.items():
    steps = np.arange(len(data["pos_error"]))
    color = data["color"]
    
    ax2_1.plot(steps, data["pos_error"], label=f"{algo}", color=color, linewidth=2)
    ax2_2.plot(steps, data["vel_error"], label=f"{algo}", color=color, linewidth=2)

# Subplot 1 (Position) 설정
ax2_1.set_title("Position Tracking Error (RMSE)", fontsize=13)
ax2_1.set_ylabel("Error Distance (m)")
ax2_1.grid(True, linestyle='--', alpha=0.7)
ax2_1.legend(loc="upper right")

# Subplot 2 (Velocity) 설정
ax2_2.set_title("Velocity Tracking Error (RMSE)", fontsize=13)
ax2_2.set_xlabel("Time Steps")
ax2_2.set_ylabel("Error Speed (m/s)")
ax2_2.grid(True, linestyle='--', alpha=0.7)
ax2_2.legend(loc="upper right")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# ==========================================
# 터미널(CMD) 최종 결과 출력
# ==========================================
print("\n" + "="*50)
print(f"{'Algorithm Analysis Summary':^50}")
print("="*50)
print(f"{'Algorithm':<15} | {'Avg Pos Error (m)':<15} | {'Avg Vel Error (m/s)':<15}")
print("-" * 50)

for algo, data in results_data.items():
    print(f"{algo:<15} | {data['avg_pos_rmse']:<15.4f} | {data['avg_vel_rmse']:<15.4f}")

print("="*50)
print("Displaying plots. Close the windows to exit.")

# 그래프 띄우기 (Figure 1, 2가 동시에 뜸)
plt.show()