import cv2
import numpy as np

def pixel_to_world(x_pixel, y_pixel, H):
    point = np.array([[x_pixel, y_pixel]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point.reshape(1, -1, 2), H)
    return transformed[0][0][0], transformed[0][0][1]

def pixel_to_world_cam1(x_pixel, y_pixel):
    H1 = np.array([[0.037534854106039205, 0.004008511087959193, -20.998422428775157], 
            [0.0018919919389477693, 0.06775448236489855, -27.789219276191865], 
            [0.0003986513362058893, 0.005119166915182983, 1.0]], dtype=np.float32) 
    return pixel_to_world(x_pixel, y_pixel, H1)

def pixel_to_world_cam2(x_pixel, y_pixel):
    H2 = np.array([[0.05435367941782544, -0.007587650688584197, -6.827153299637387], 
            [0.033574494242627456, 0.07604811928025516, -38.43546233738026], 
            [-0.004442127955927964, -0.008821697033771673, 1.0]], dtype=np.float32) 
    return pixel_to_world(x_pixel, y_pixel, H2)

def world_to_pixel(X, Y, H_inv):
    point = np.array([[X, Y]], dtype=np.float32)
    pixel_coords = cv2.perspectiveTransform(point.reshape(1, -1, 2), H_inv)
    return pixel_coords[0][0][0], pixel_coords[0][0][1]

def world_to_pixel_cam1(X, Y):
    H1_inv = np.array(
        [[25.628171941347425, -13.606911523217594, 160.02573251126032], 
         [-1.5827778288930383, 5.601982885565307, 122.4388933264528], 
         [-0.0021142010931780373, -0.02325307198684122, 0.30942039605866434]],
         dtype=np.float32
    )
    return world_to_pixel(X, Y, H1_inv)

def world_to_pixel_cam2(X, Y):
    H2_inv = np.array(
        [[16.837517539166175, -4.3412701183044735, -51.90641070326519], 
         [-8.780568262293409, -1.5381012805978027, -119.0639194261009], 
         [-0.002665105625145974, -0.032853140861445036, -0.28092074290731645]],
         dtype=np.float32
    )
    return world_to_pixel(X, Y, H2_inv)


if __name__ == "__main__":
    cam1_pixel = np.array([
        [512, 400], [607, 450], [508, 451], [602, 397],
        [321, 263], [345, 667], [673, 316], [821, 655]
    ], dtype=np.float32)

    cam2_pixel = np.array([
        [178, 424], [122, 410], [175, 386], [119, 449],
        [608, 560], [220, 281], [59, 584], [65, 351]
    ], dtype=np.float32)

    real_world = np.array([
        [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0],
        [-3.3, -3.7], [-1.1, 4], [2, -1.7], [2.5, 4]
    ], dtype=np.float32)

    # Compute homographies with RANSAC (robust to outliers)
    H1, _ = cv2.findHomography(cam1_pixel, real_world, cv2.RANSAC, 5.0)
    H2, _ = cv2.findHomography(cam2_pixel, real_world, cv2.RANSAC, 5.0)
    # Reproject Camera 1 pixels to real-world
    reprojected_world = cv2.perspectiveTransform(cam1_pixel.reshape(-1, 1, 2), H1)

    # Compute error (Euclidean distance)
    errors = np.linalg.norm(reprojected_world - real_world.reshape(-1, 1, 2), axis=2)
    print("Mean Reprojection Error (Camera 1):", np.mean(errors))