import cv2
import numpy as np
import pickle

# Load video and polygon positions
cap = cv2.VideoCapture('../carPark.mp4')
with open('polygon_spots.pkl', 'rb') as f:
    polygon_spots = pickle.load(f)

# Constants
OCCUPIED_THRESHOLD = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX

# Count white pixels inside polygon
def count_white_pixels_inside_polygon(thresh_img, polygon):
    mask = np.zeros(thresh_img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
    masked_img = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
    white_pixels = cv2.countNonZero(masked_img)
    area = cv2.countNonZero(mask)
    return white_pixels / area if area > 0 else 0

# divide into 3 columns based on polygon X positions
from sklearn.cluster import KMeans

def group_spots_auto(polygons, n_clusters=3):
    avg_x_coords = np.array([[np.mean([pt[0] for pt in poly])] for poly in polygons])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(avg_x_coords)
    labels = kmeans.labels_

    columns = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        columns[label].append(polygons[idx])

    # Optional: sort columns left to right by cluster center X
    sorted_indices = np.argsort(kmeans.cluster_centers_.flatten())
    sorted_columns = [columns[i] for i in sorted_indices]

    return sorted_columns


columns = group_spots_auto(polygon_spots, n_clusters=3)


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    overlay = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)

    free_count = 0
    spot_status = []  # (polygon, ratio, is_free)

    for polygon in polygon_spots:
        ratio = count_white_pixels_inside_polygon(thresh, polygon)
        is_free = ratio < OCCUPIED_THRESHOLD
        if is_free:
            free_count += 1
        spot_status.append((polygon, ratio, is_free))

    # Draw each polygon
    for polygon, ratio, is_free in spot_status:
        color = (0, 255, 0) if is_free else (0, 0, 255)
        pts = np.array(polygon, np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(overlay, f"{ratio:.2f}", tuple(pts[0]), font, 0.4, color, 1)

    # Count free/occupied in each column
    col_stats = []
    for i, column in enumerate(columns):
        col_free = 0
        col_total = len(column)
        for poly in column:
            idx = polygon_spots.index(poly)
            _, _, is_free = spot_status[idx]
            if is_free:
                col_free += 1
        col_stats.append((i + 1, col_free, col_total - col_free))

    # Draw info panel
    panel_height = 20 + 25 * len(col_stats)
    cv2.rectangle(overlay, (0, 0), (220, panel_height), (50, 50, 50), -1)
    cv2.putText(overlay, f"Total Free: {free_count}/{len(polygon_spots)}", (10, 25),
                font, 0.6, (255, 255, 255), 1)

    for i, (col_no, free, occupied) in enumerate(col_stats):
        cv2.putText(overlay, f"Column {col_no}: Free {free}, Occ {occupied}",
                    (10, 50 + i * 25), font, 0.5, (200, 255, 200), 1)

    cv2.imshow("Parking Spot Detection - 3 Columns", overlay)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Quit requested by user .")
        break

cap.release()
cv2.destroyAllWindows()
