import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self):
        self.kmeans = None
        self.team_colors = {}  # Lưu màu chủ đạo của 2 đội

    def get_clustering_model(self, image):
        # Reshape ảnh thành danh sách các điểm ảnh (pixel)
        image_2d = image.reshape(-1, 3)
        # Dùng K-Means chia làm 2 cụm màu (ví dụ: Áo và Quần/Da)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Bước 1: Resize nhỏ lại để xử lý cho nhanh
        # Chỉ lấy nửa trên của cầu thủ (áo) để tránh nhầm với tất/giày
        top_half_image = image[0:int(image.shape[0]/2), :] 
        
        # Bước 2: Loại bỏ background xanh lá (nếu cần thiết)
        # (Ở đây mình làm đơn giản là lấy màu trực tiếp, nếu nhiễu quá mới cần mask)
        
        return self.get_dominant_color(top_half_image)

    def get_dominant_color(self, image):
        if image is None or image.size == 0:
            return None
            
        # Chuyển đổi sang RGB (OpenCV mặc định là BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Gom nhóm màu
        kmeans = self.get_clustering_model(image)
        
        # Lấy các tâm cụm (cluster centers) và nhãn (labels)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Đếm xem cụm nào chiếm nhiều pixel nhất
        count_labels = np.bincount(labels)
        dominant_color = centers[np.argmax(count_labels)]
        
        return dominant_color

    def fit_teams(self, frames_crops):
        """
        Hàm này chạy 1 lần đầu trận đấu.
        Nó lấy mẫu ảnh của nhiều cầu thủ để xác định màu áo chuẩn của 2 đội.
        """
        all_colors = []
        for crop in frames_crops:
            color = self.get_dominant_color(crop)
            if color is not None:
                all_colors.append(color)
        
        # Gom tất cả màu áo lại thành 2 nhóm lớn (Đội 1 và Đội 2)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(all_colors)
        
        self.team_colors[0] = kmeans.cluster_centers_[0] # Màu đội 1
        self.team_colors[1] = kmeans.cluster_centers_[1] # Màu đội 2
        
        print(f"Team 1 Color: {self.team_colors[0]}")
        print(f"Team 2 Color: {self.team_colors[1]}")

    def predict(self, frame, bbox):
        player_color = self.get_player_color(frame, bbox)
        
        # Tính khoảng cách từ màu cầu thủ hiện tại đến màu chuẩn của 2 đội
        dist_0 = np.linalg.norm(player_color - self.team_colors[0])
        dist_1 = np.linalg.norm(player_color - self.team_colors[1])
        
        # Gần đội nào hơn thì gán đội đó
        if dist_0 < dist_1:
            return 0 # Team 0
        else:
            return 1 # Team 1