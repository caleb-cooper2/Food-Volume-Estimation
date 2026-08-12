import numpy as np
import open3d as o3d
import trimesh

from geometry import M3_TO_CM3
from logging_config import get_logger

logger = get_logger(__name__)

MAX_FOOD_HEIGHT_M = 0.15 # taller than 15 cm above the table is probably not food
BELOW_PLANE_TOLERANCE_M = 0.005
MESH_VS_PRISM_MAX_RATIO = 2.0 # mesh this much larger than the prism bound => probably inflated
ORIGIN_INVARIANCE_TOL = 0.01 # 1% drift under translation => mesh isn't closed enough
PRISM_CELL_M = 0.003


# Point cloud cleaning up

def clean_food_points(points: np.ndarray, table_plane: tuple[np.ndarray, np.ndarray] | None, max_food_height_m: float = MAX_FOOD_HEIGHT_M) -> np.ndarray:
    """
    Drop non-finite points, points that are physically impossible given the table plane, and statistical / radius outliers
    to theoretically get a more accurate point cloud
    """
    real_points = points[np.isfinite(points).all(axis=1)]

    if table_plane is not None:
        normal, point_on_plane = table_plane
        heights = (real_points - point_on_plane) @ normal
        real_points = real_points[(heights > -BELOW_PLANE_TOLERANCE_M) & (heights < max_food_height_m)]

    if len(real_points) < 50:
        return real_points

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(real_points))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=12, radius=0.006)
    cleaned = np.asarray(pcd.points)

    dropped = len(points) - len(cleaned)
    if dropped:
        logger.info(f"  Clean | dropped {dropped}/{len(points)} pts ({dropped / len(points):.1%}) before meshing")
    return cleaned


# Height-map prism integral just in case

def volume_by_height_map(points: np.ndarray, table_plane: tuple[np.ndarray, np.ndarray], cell_m: float = PRISM_CELL_M) -> float:
    """
    Bin the cloud into cells on the table plane, take the median height per cell, sum height * cell area.
    """
    normal, point_on_plane = table_plane
    heights = np.clip((points - point_on_plane) @ normal, 0.0, None)

    axis_1 = np.cross(normal, [0.0, 0.0, 1.0])
    if np.linalg.norm(axis_1) < 1e-6:
        axis_1 = np.cross(normal, [0.0, 1.0, 0.0])
    axis_1 /= np.linalg.norm(axis_1)
    axis_2 = np.cross(normal, axis_1)

    u = np.floor(((points - point_on_plane) @ axis_1) / cell_m).astype(np.int64)
    w = np.floor(((points - point_on_plane) @ axis_2) / cell_m).astype(np.int64)
    keys = (u - u.min()) * (w.max() - w.min() + 1) + (w - w.min())

    order = np.argsort(keys)
    keys, heights = keys[order], heights[order]
    edges = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    cell_heights = np.array([np.median(heights[a:b]) for a, b in zip(edges[:-1], edges[1:])])

    return float(cell_heights.sum() * cell_m * cell_m * M3_TO_CM3)


# Meshing

def get_surface_normals(points: np.ndarray, camera_location: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location)
    return np.asarray(pcd.normals)


def volume_is_well_posed(mesh: trimesh.Trimesh, tol: float = ORIGIN_INVARIANCE_TOL) -> bool:
    """
    A closed mesh has a translation-invariant volume. An open one does not, because the divergence integral picks
    up a term proportional to the offset times the boundary's vector area.
    """
    volume = mesh.volume
    shifted = trimesh.Trimesh(mesh.vertices + np.array([0.137, -0.291, 0.443]), mesh.faces, process=False)
    denominator = max(abs(volume), abs(shifted.volume), 1e-12)
    return abs(shifted.volume - volume) / denominator < tol


def create_poisson_mesh(
        points: np.ndarray,
        table_plane: tuple[np.ndarray, np.ndarray] | None,
        camera_location: np.ndarray
) -> trimesh.Trimesh:
    # VGGT's world frame is the first camera's frame, so the camera sits at the origin and every reconstructed surface point is visible from it.
    surface_normals = get_surface_normals(points, camera_location)

    if table_plane is None:
        cloud_points, cloud_normals = points, surface_normals
    else:
        normal, point_on_plane = table_plane
        heights = np.clip((points - point_on_plane) @ normal, 0.0, None)
        # Only cap points that are genuinely above the plane.
        # Otherwise cap == point and the cloud picks up exact duplicates right at the rim...
        above = heights > 1e-4
        cap = points[above] - heights[above, None] * normal
        cloud_points = np.vstack([points, cap])
        # The cap is the underside of the solid, so its outward normal is known analytically.
        cloud_normals = np.vstack([surface_normals, np.tile(-normal, (int(above.sum()), 1))])

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_points))
    point_cloud.normals = o3d.utility.Vector3dVector(cloud_normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9, scale=1.05, linear_fit=True)

    # Only keep the largest component, get rid of other components that lead to a "draping" effect
    clusters, triangles_per_cluster, _ = (np.asarray(x) for x in mesh.cluster_connected_triangles())
    if len(triangles_per_cluster):
        mesh.remove_triangles_by_mask(triangles_per_cluster[clusters] < triangles_per_cluster.max())

    mesh = mesh.merge_close_vertices(1e-7)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    trimesh_result = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    trimesh_result.update_faces(trimesh_result.unique_faces())
    trimesh_result.remove_unreferenced_vertices()
    if not trimesh_result.is_watertight:
        trimesh_result.fill_holes()
    return trimesh_result


def compute_volume_from_mesh(
        food_points_metric: np.ndarray,
        table_plane: tuple[np.ndarray, np.ndarray] | None = None,
        camera_location: np.ndarray | None = None,
        debug_tag: str | None = None
) -> float:
    """
    Poisson mesh volume in cm^3, bounded by the volume actually being well posed, with a
    height-map prism integral as both a sanity bound and a fallback.

    Returns 0.0 when there is nothing meshable.
    """
    if camera_location is None:
        camera_location = np.zeros(3)  # VGGT world frame == first camera frame

    points = clean_food_points(food_points_metric, table_plane)
    if len(points) < 50:
        logger.warning(f"  Mesh | only {len(points)} pts after cleaning -> volume 0")
        return 0.0

    prism_cm3 = volume_by_height_map(points, table_plane) if table_plane is not None else None

    try:
        mesh = create_poisson_mesh(points, table_plane, camera_location)
    except Exception as exc:
        logger.error(f"  Mesh | Poisson failed ({exc}) -> using prism fallback {prism_cm3}")
        return prism_cm3 or 0.0

    if debug_tag:
        mesh.export(f"/tmp/debug_food_mesh_{debug_tag}.ply")

    if not volume_is_well_posed(mesh):
        logger.warning(
            f"  Mesh | volume not translation-invariant, mesh is open "
            f"({len(trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1))} boundary edges) "
            f"-> prism fallback {prism_cm3}"
        )
        return prism_cm3 or 0.0

    mesh.fix_normals()
    volume_cm3 = abs(mesh.volume) * M3_TO_CM3

    if prism_cm3 is not None and volume_cm3 > MESH_VS_PRISM_MAX_RATIO * prism_cm3:
        logger.warning(
            f"  Mesh | {volume_cm3:.1f} cm3 exceeds {MESH_VS_PRISM_MAX_RATIO}x the prism bound "
            f"{prism_cm3:.1f} cm3 -> inflated, using prism"
        )
        return prism_cm3

    logger.info(f"  Mesh | volume {volume_cm3:.1f} cm3 (prism cross-check {prism_cm3:.1f} cm3)" if prism_cm3 is not None else f"  Mesh | volume {volume_cm3:.1f} cm3")
    return volume_cm3


def compute_volume_per_instance(
        instance_masks: list[np.ndarray],
        world_points_metric: np.ndarray,
        table_plane: tuple[np.ndarray, np.ndarray] | None = None
) -> float:
    import cv2

    total_volume_cm3 = 0.0
    target_h, target_w = world_points_metric.shape[0], world_points_metric.shape[1]

    for i, mask in enumerate(instance_masks):
        mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        pts = world_points_metric[mask_resized > 0]
        if len(pts) < 50:
            logger.info(f"  Instance {i} | fewer than 50 points, skipped")
            continue
        try:
            vol_cm3 = compute_volume_from_mesh(pts, table_plane, debug_tag=f"instance_{i}")
        except Exception as e:
            logger.error(f"  Instance {i} | volume failed: {e}")
            continue
        logger.info(f"  Instance {i} | {vol_cm3:.2f} cm^3 from {len(pts)} points")
        total_volume_cm3 += vol_cm3

    return total_volume_cm3