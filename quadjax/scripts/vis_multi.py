import meshcat
from meshcat import geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation
import numpy as np
import time
import pickle
import argparse
import os
import io

# 텍스트를 이미지로 만들기 위해 PIL(Pillow) 라이브러리 사용
from PIL import Image, ImageDraw, ImageFont

# --- 설정 및 인자 파싱 ---
parser = argparse.ArgumentParser(description="Quadrotor Trajectory Visualizer")
parser.add_argument(
    "-f", "--files",
    nargs="+",
    required=True,
    help="Visualize multiple .pkl files from ../../results/ (e.g., file1.pkl file2.pkl)"
)
args = parser.parse_args()

COLOR_LIST = [
    0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF
]

# Create a Meshcat visualizer
vis = meshcat.Visualizer()
anim = Animation(default_framerate=50)

# --- 텍스트 텍스처 생성 함수 ---
def create_text_texture(text, color_hex):
    # 1. 투명한 배경의 이미지 생성 (가로 256, 세로 128)
    img = Image.new('RGBA', (512, 256), color=(255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    
    # 헥스 색상(0xFF0000 등)을 RGB 튜플로 변환
    r = (color_hex >> 16) & 255
    g_col = (color_hex >> 8) & 255
    b = color_hex & 255
    
    font = ImageFont.load_default()
    try:
        # 가급적 큰 폰트 시도 (에러시 기본 폰트)
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        pass
        
    d.text((10, 80), text, fill=(r, g_col, b, 255), font=font)

    # 2. 이미지를 PNG 바이트 데이터로 변환
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # 3. Meshcat의 PngImage와 ImageTexture로 묶어서 반환
    return g.ImageTexture(g.PngImage(img_bytes))

# --- 유틸리티 함수 ---
def origin_vec_to_transform(origin, vec, scale=1.0):
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return np.array([
            [1, 0, 0, origin[0]], [0, 1, 0, origin[1]],
            [0, 0, 1, origin[2]], [0, 0, 0, 1],
        ])
    vec = vec / vec_norm
    if vec[0] == 0 and vec[1] == 0:
        vec_1, vec_2 = np.array([1, 0, 0]), np.array([0, 1, 0])
    else:
        vec_1 = np.array([vec[1], -vec[0], 0])
        vec_1 /= np.linalg.norm(vec_1)
        vec_2 = np.cross(vec, vec_1)
    rot_mat = np.eye(4)
    rot_mat[:3, 2] = vec
    rot_mat[:3, 0] = vec_1
    rot_mat[:3, 1] = vec_2
    rot_mat[:3, :3] *= vec_norm * scale
    return tf.translation_matrix(origin)

def pos_quat_to_transform(pos, quat):
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    return tf.translation_matrix(pos) @ tf.quaternion_matrix(quat)

def set_frame(i, name, transform):
    with anim.at_frame(vis, i) as frame:
        frame[name].set_transform(transform)

# --- 파일 로드 및 시각화 요소 생성 ---
base_path = "../../results/"
all_state_seqs = []
max_frames = 0

print("Connecting to Meshcat and loading files...")
for filename in args.files:
    algorithm_name = filename.replace("state_seq_", "").replace(".pkl", "")
    file_path = os.path.join(base_path, filename)
    print(f"Loading: {file_path}")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            all_state_seqs.append((algorithm_name, data))
            max_frames = max(max_frames, len(data))
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Skipping.")

# 데이터가 로드된 알고리즘 개수만큼 시각화 객체 생성
for idx, (algorithm_name, _) in enumerate(all_state_seqs):
    color = COLOR_LIST[idx % len(COLOR_LIST)]
    
    # 1. 알고리즘 이름 텍스트 라벨 표시 (Box + ImageTexture 우회법)
    label_path = f"labels/{idx}_{algorithm_name}"
    
    # Plane이 없으므로 두께가 0.001인 아주 얇은 Box를 평면처럼 사용 [가로, 세로, 두께]
    text_geom = g.Box([1.0, 0.5, 0.001])
    
    # PIL로 만든 텍스트 이미지를 텍스처로 입힘 (transparent=True로 배경 투명화)
    text_mat = g.MeshBasicMaterial(
        map=create_text_texture(f"Drone {idx}: {algorithm_name}", color), 
        transparent=True
    )
    vis[label_path].set_object(text_geom, text_mat)
    
    # 글자가 잘 보이도록 회전 및 위치 조정 (서로 안 겹치게 Y축으로 띄움)
    R_x = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    T = tf.translation_matrix([0.0, idx * 0.6, 0.5]) # 드론 살짝 위쪽에 표시
    vis[label_path].set_transform(T @ R_x)

    # 2. 드론, 프레임, 목표물, 외란, 궤적 생성
    vis[f"drone_{idx}"].set_object(g.StlMeshGeometry.from_file("../assets/crazyflie2.stl"), material=g.MeshLambertMaterial(color=color))
    vis[f"drone_frame_{idx}"].set_object(g.StlMeshGeometry.from_file("../assets/axes.stl"))
    vis[f"obj_tar_{idx}"].set_object(g.Sphere(0.02), material=g.MeshLambertMaterial(color=color))
    vis[f"disturb_{idx}"].set_object(g.StlMeshGeometry.from_file("../assets/arrow.stl"))

    for i in range(0, 300, 2):
        vis[f"traj{i}_{idx}"].set_object(
            g.Sphere(0.01), material=g.MeshLambertMaterial(color=color, opacity=0.5)
        )

# --- 애니메이션 프레임 업데이트 ---
for i in range(max_frames):
    for idx, (algorithm_name, state_seq) in enumerate(all_state_seqs):
        if i >= len(state_seq):
            continue
            
        state = state_seq[i]
        
        if i % 20 == 0:
            for j in range(0, 300, 2):
                set_frame(i, f"traj{j}_{idx}", pos_quat_to_transform(state["pos_traj"][j], np.array([0, 0, 0, 1])))
                
        set_frame(i, f"drone_{idx}", pos_quat_to_transform(state["pos"], state["quat"]))
        set_frame(i, f"drone_frame_{idx}", pos_quat_to_transform(state["pos"], state["quat"]))
        set_frame(i, f"obj_tar_{idx}", pos_quat_to_transform(state["pos_tar"], np.array([0, 0, 0, 1])))
        set_frame(i, f"disturb_{idx}", origin_vec_to_transform(state["pos"], state["f_disturb"], 2.0))

vis.set_animation(anim)
print("Animation ready! Open the Meshcat URL in your browser.")
time.sleep(5)