import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Ghi log huấn luyện ---
def log_training(row, path="training_log.csv"):
    """
    Ghi 1 dòng dữ liệu vào file CSV.
    row = [episode, score, steps, epsilon, reward]
    """
    # Nếu file chưa tồn tại → ghi header
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            writer.writerow(["episode", "score", "steps", "epsilon","reward"])
        writer.writerow(row)

# --- Đọc số episode cuối ---
def get_last_episode(path="training_log.csv"):
    """
    Đọc số episode cuối cùng trong file log.
    Nếu chưa có file → trả về 0.
    """
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path, delimiter=';')
    if len(df) == 0:
        return 0
    return int(df["episode"].iloc[-1])

# --- Ghi tự động ---
def log_training_auto(score, steps, epsilon, reward):
    last_ep = get_last_episode()
    new_ep = last_ep + 1
    log_training([new_ep, score, steps, epsilon, reward])

if __name__ == "__main__":
    # --- PHÂN TÍCH VÀ VẼ BIỂU ĐỒ ---
    if os.path.exists("training_log.csv"):
        df = pd.read_csv("training_log.csv", delimiter=';')

        print("\n Biên epsilon:")
        print("Min:", df["epsilon"].min(), "| Max:", df["epsilon"].max())

        # Vẽ biểu đồ tiến trình huấn luyện
        plt.figure(figsize=(8,5))
        plt.plot(df["episode"], df["score"], marker='o', label="Score per Episode")
        plt.title("Learning Progress of Pacman Agent")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Chưa có file 'training_log.csv' — hãy huấn luyện trước!")
